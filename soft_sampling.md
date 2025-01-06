# Soft (Latent) Decoding

> Early musings on a new decoding technique.

## 1. Abstract

Modern large language models are typically trained and used in a discrete autoregressive manner. At each step, the model projects its hidden state to vocabulary logits, and a single token is either sampled or chosen greedily. This discrete choice collapses the probability distribution at every timestep, potentially discarding latent or “parallel” reasoning paths.

In contrast, a soft (latent) chain-of-thought approach proposes to retain a mixture of possible tokens by computing a continuous next-token embedding. Rather than collapsing to one token, the model’s next input embedding is the distribution over all possible next tokens, weighted by probability using the softmax. This in effect is _removing_ the sampling step from the model.

## 2. Motivation

### 2.1 Standard Discrete Decoding

```python
tokens = [ BOS ]
embeddings = [ W_E[tokens] ]
for t in range(max_length):
    h = model(embeddings)
    logits = h[-1] @ W_U
    probabilities = softmax(logits)
    next_token = sample(probabilities)
    next_emb = W_E[next_token] # discrete sampling
    tokens.append(next_token)
    embeddings.append(next_emb)
    if next_token == EOS:
        break
```

This method is efficient and well-aligned with the training distribution (discrete). However, it collapses the model’s uncertainty at every timestep, keeping only one token thread and discarding all other plausible tokens.

### 2.2 Soft (Latent) Chain-of-Thought


```python
tokens = [ BOS ]
embeddings = [ W_E[tokens] ]
for t in range(max_length):
    h = model(embeddings)
    logits = h[-1] @ W_U
    probabilities = softmax(logits)
    next_token = sample(probabilities)
    next_emb = proabilities @ W_E.T # use sample distribution
    tokens.append(next_token)
    embeddings.append(next_emb)
    if next_token == EOS:
        break
```

Instead of using a discrete token, use the softmax distribution to use tokens weighted by the probability.

### 2.3 Motivations for Soft Decoding

1. The model may be allowed to maintain it's distribution over each token for each step, so the model's internal state may reflect several distribution simultaneously.
    - (Kind of like BEAM search without the combinatorial blowup).
2. The model remains "end-to-end differentiable" -- gradients from from future tokens can propagate through the sampling.

## 3 Approaches to Creating a Soft Next-Token Embedding

### 3.1 Plain Softmax Mixture

Take the model’s logits, apply a standard softmax, then multiply the resulting distribution by the embedding matrix $W_E$.

- Possible to have "blurry" embedding if distribution is diffuse over many tokens -- doesn't approximate a single token.

### 3.2. Gumbel-Softmax -- LOOK AT

- Paper Reference: Eric Jang et al., “Categorical Reparameterization with Gumbel-Softmax,” ICLR 2017.
- Key Mechanism: Add Gumbel noise $\mathbf{g}$ to the logits and scale by temperature $\tau$, then apply softmax:

$$
\mathbf{y}_t = \text{Softmax}\Bigl(\frac{\mathbf{z}_t + \mathbf{g}_t}{\tau}\Bigr).
$$

For low $\tau$, $\mathbf{y}_t$ becomes close to a one-hot vector—mimicking discrete sampling—yet remains differentiable.

- Helpful for end-to-end gradient flow through "token choices".

## 4. Challenges and Caveats

### 4.1 Out-Of-Distribution Inputs

Pretrained LLMs only saw discrete token embeddings during training. A "soft mix" of embeddings can be drastically different distribution from any single token embedding, causing unpredictable outputs. Finetuning will be needed to adapt the model to these continuous embeddings.

### 4.2 Sequential Fine-Tuning

Each output is dependent on all previous outputs, which can be useful for gradient propagation, but is compute inefficient in training. Kind of like RNNs.

### 4.3 Blurry embeddings / Entropy explosion

If the distribution is too broad, embedding may become uniform over many tokens, which may push the model input off manifold.

Possible solutions:

- Sampling adjustments: temperature, top-k, min-p
- Gumbel-Softmax?
- Entropy regularization: $\mathbf{L} = \mathbf{L}_{\text{xent}} + \beta \cdot \mathbf{H}(P)$
  - $\beta$ is a hyperparameter
  - $\mathbf{H}$ is the entropy of the distribution $P$

### 4.4 Scheduled Sampling

Not obvious exactly how to incorporate the ground truth into the sampling. Discrete decoding handles this simply with "teacher-forcing", replacing the predicted embedding with the correct one.

We may need to mix the ground truth with the soft sample to stabilize training.

### 4.5 Catastrophic Forgetting

- Partial freezing (maybe $W_E$ / first layer only?)
- LoRA
- low LR

### 4.6 Integration with RLHF (PPO)

Standard PPO assume discrete tokens as actions.

- If you feed a soft distribution, you must either discretize before giving it to a reward model.

## 5. Implementation Sketch

### 5.1 Stage 1: SFT for Soft Decoding

```python
import torch
import torch.nn.functional as F
from fastcore.meta import delegates

def mk_proba_dist(
    logits, # (batch_size, d_vocab)
    temperature=1.0,
    top_k=None,
    min_p=None,
):
    batch_size, d_vocab = logits.shape
    device = logits.device
    if top_k:
        logits, idxs = logits.topk(top_k, dim=-1)
    else:
        idxs = (
            torch.arange(d_vocab, device=device)
            .repeat(batch_size)
            .reshape(batch_size, d_vocab)
        )

    # TODO: temperature before or after min_p?
    probs = F.softmax(logits / temperature, dim=-1)

    if min_p is not None:
        max_probs = probs.max(dim=-1, keepdim=True).values
        threshold = max_probs * min_p
        mask = probs >= threshold
        probs = probs * mask
        probs = probs / probs.sum(dim=-1, keepdim=True) # renormalize
        idxs = idxs * mask
    return idxs, probs

@delegates(mk_proba_dist)
def soft_sampling_train_step(
    model,
    batch, # tokens of shape (batch_size, seq_len)
    guidance_alpha, # guidance weighting -- 1 equivalent to discrete sampling
    **kwargs, # passed to mk_proba_dist
):
    "Single train step using soft sampling"
    assert 0 <= guidance_alpha <= 1
    batch_size, seq_len = batch.shape
    device = batch.device

    # cache
    past_key_values = None
    position_ids = torch.zeros(batch_size, 1, dtype=torch.long, device=device)

    W_E = model.get_input_embeddings().weight

    loss = torch.tensor(0., device=device)
    embeds = W_E[batch[:, :1]]  # BOS shape: (batch_size, 1, d_model)
    tokens = [ batch[:, :1].detach().cpu() ]
    for t in range(1, seq_len):
        outputs = model(
            inputs_embeds=embeds,
            past_key_values=past_key_values,
            position_ids=position_ids,
            use_cache=True
        )

        logits_t = outputs.logits[:, -1]
        past_key_values = outputs.past_key_values

        i_t, p_t = mk_proba_dist(logits_t, **kwargs)

        # loss
        loss_t = F.cross_entropy(p_t, batch[:, t])
        loss += loss_t

        # discrete sample -- for logging
        indices = torch.multinomial(p_t, 1) # (batch_size, 1)
        batch_indices = torch.arange(batch_size)[:, None] # (batch_size, 1)
        next_token = i_t[batch_indices, indices].detach().cpu()
        tokens.append(next_token)

        # soft sample
        next_emb_soft = p_t @ W_E      # soft sampling
        next_emb_gt = W_E[batch[:, t]] # guidance sampling

        next_embed = (
            guidance_alpha * next_emb_gt +
            (1 - guidance_alpha) * next_emb_soft
        )
        embeds = torch.cat([embeds, next_embed[:, None, :]], dim=1)
        position_ids += 1

    tokens = torch.cat(tokens, dim=1)
    # normalize gradient: sum batch, mean sequence length
    loss /= seq_len
    return loss, tokens
```

I think I want a initial `guidance_alpha` of 1 to mimic discrete training to warm start the model to get to a stable baseline before we shift the input distribution. Perhaps warmup `lr` here too. Then warmup `guidance_alpha` to some maximum value.

TODO: how do we want to apply `guidance_alpha`? I think we should clamp the correct token at _at least_ `guidance_alpha` proportion in the probability distribution (this is helpful when the model already gives the correct token a high probability).

### 5.2 Stage 2: RLHF / PPO

> TODO

## 6. Additional Considerations

### 6.1 Comparison with BEAM

Beam Search / Self-Consistency: Another way to keep multiple possibilities is to track multiple discrete beams. However, that can explode combinatorially.

If the soft decoding works, we effectively combine multiple tokens into a single “latent” path at each step.

### 6.3 Interpretability

We can record how the model's beliefs evolve by sampling tokens at each step, though it's not a "true" discrete chain.

### 6.4 Selectively use soft decoding

Perhaps only for thinking tokens? And switch back to discrete for the final answer.

### 6.5 Temperature / TopK / MinP as a learnable parameter

As the model is fully differentiable, we could use temperature (and perhaps topk, minp as learnable parameters).

- This might collapse initially where the model greedily samples to keep in old distribution

### 6.6 Soft Beam Search

Instead of combining every step, maintain a beam of plausible tokens, and for each entry produce a soft mixture.

_Or_ we could do a beam search, and collapse the embeddings back down into a soft embedding.

### 6.7 Combine with prefix tuning

Specialized prefix of a set of learnable vectors, that signifies the model does soft decoding.

Alternatively put inside a new `<think>` token or something.

---

- Keep multiple reasoning paths alive in a single forward pass.
- Provide a differentiable approach to token selection, which is valuable in advanced training or RL settings.
- Fully differentiable through sampling steps

## Integration with HF Trainer

## References

1. Gumbel-Softmax
  - Jang, E., Gu, S., & Poole, B. (2017). Categorical Reparameterization with Gumbel-Softmax. ICLR.
  - arXiv:1611.01144
2. Self-Consistency
  - Wang, X., et al. (2022). Self-Consistency Improves Chain of Thought Reasoning in Language Models. arXiv:2203.11171
3. RLHF
  - Christiano, P., et al. (2017). Deep Reinforcement Learning from Human Preferences. NIPS.
  - Bai, Y., et al. (2022). Training a Helpful and Harmless Assistant with RL from Human Feedback. (Anthropic work)
4. Prefix / Prompt Tuning
  - Lester, B., Al-Rfou, R., & Constant, N. (2021). The Power of Scale for Parameter-Efficient Prompt Tuning. EMNLP.
  - Li, X. L., & Liang, P. (2021). Prefix-Tuning: Optimizing Continuous Prompts for Generation. ACL.

