import re
from typing import List, Dict, Any, Optional
import torch
import torch.nn.functional as F
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# ==================== 1) GSM8K DATASET PREP ====================

SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""


def extract_hash_answer(text: str) -> Optional[str]:
    if "####" not in text:
        return None
    return text.split("####", 1)[1].strip()


def get_gsm8k_questions(split="train") -> Dataset:
    """
    Load GSM8K from Hugging Face and produce a dataset in
    a 'prompt' + 'answer' format for chat-like interactions.
    """
    data = load_dataset("openai/gsm8k", "main")[split]  # type: ignore

    def map_func(row):
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": row["question"]},
            ],
            "answer": extract_hash_answer(row["answer"]),
        }

    return data.map(map_func)


# ==================== 2) REWARD FUNCTIONS ====================


def extract_xml_answer(text: str) -> str:
    """Extract text between <answer>...</answer> if present."""
    if "<answer>" not in text or "</answer>" not in text:
        return ""
    return text.split("<answer>", 1)[1].split("</answer>", 1)[0].strip()


def correctness_reward(prompts, completions, answer, **kwargs) -> List[float]:
    """
    +2.0 if the extracted <answer> matches the ground truth 'answer'; else 0.
    """
    responses = [c[0]["content"] for c in completions]
    model_answers = [extract_xml_answer(r) for r in responses]
    return [2.0 if m == a else 0.0 for m, a in zip(model_answers, answer)]


def int_reward_func(completions, **kwargs) -> List[float]:
    """
    +0.5 if <answer> text is purely digits; else 0.
    """
    responses = [c[0]["content"] for c in completions]
    extracted = [extract_xml_answer(r) for r in responses]
    return [0.5 if ans.isdigit() else 0.0 for ans in extracted]


def strict_format_reward(completions, **kwargs) -> List[float]:
    """
    +0.5 if response matches the multiline CoT format exactly.
    """
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [c[0]["content"] for c in completions]
    return [0.5 if re.match(pattern, r, flags=re.DOTALL) else 0.0 for r in responses]


def soft_format_reward(completions, **kwargs) -> List[float]:
    """
    +0.5 if response has <reasoning>.*</reasoning><answer>.*</answer> in any arrangement.
    """
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [c[0]["content"] for c in completions]
    return [0.5 if re.search(pattern, r, flags=re.DOTALL) else 0.0 for r in responses]


# ==================== 3) SOFT-SAMPLING UTILS ====================


def mk_proba_dist(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    min_p: Optional[float] = None,
) -> torch.Tensor:
    """
    Build a probability distribution from logits with top-k + min_p filtering.
    Returns shape (B, vocab_size).
    """
    if top_k is not None:
        vals, idx = logits.topk(top_k, dim=-1)
        mask = torch.full_like(logits, float("-inf"))
        mask.scatter_(1, idx, vals)
        logits = mask

    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)

    if min_p is not None:
        max_p = probs.max(dim=-1, keepdim=True).values
        thresh = max_p * min_p
        keep_mask = probs >= thresh
        probs = probs * keep_mask
        probs = probs / probs.sum(dim=-1, keepdim=True)

    return probs


def soft_decode_batch(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,  # (B, prompt_len)
    max_new_tokens: int = 64,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    min_p: Optional[float] = None,
    eos_token_id: Optional[int] = None,
):
    """
    Perform batched 'soft sampling' decoding:
      1) Run the prompt once to get past_key_values.
      2) For up to `max_new_tokens`, produce distribution from last step's logits,
         sample discrete token for logging, build a 'soft embedding' from distribution,
         feed it into the model, and continue.
    Returns:
      all_tokens: (B, prompt_len + new_tokens)
    """

    device = input_ids.device
    batch_size = input_ids.size(0)

    # 1) Forward pass for the prompt
    out = model(input_ids=input_ids, use_cache=True)
    pkv = out.past_key_values
    logits = out.logits[:, -1, :]  # shape (B, vocab_size)

    # We'll store discrete tokens for final text
    all_tokens = [input_ids]

    # We'll track which sequences are "finished" (hit EOS). Initially all are unfinished
    unfinished = torch.ones(batch_size, dtype=torch.bool, device=device)

    W_E = model.get_input_embeddings().weight  # shape (vocab_size, d_model)

    for step in range(max_new_tokens):
        # Skip forward pass for any sequence that is already finished
        # (We'll just keep appending EOS tokens if you prefer, or do partial updates.)
        # For simplicity, we still do a single forward pass for the entire batch,
        # but we won't update "finished" samples further.
        # If you want partial batch updates, you'd need more advanced logic.

        # 2) Build distribution for each sequence
        probs = mk_proba_dist(logits, temperature=temperature, top_k=top_k, min_p=min_p)
        # 3) Sample discrete tokens
        next_tokens = torch.multinomial(probs, num_samples=1)  # (B,1)

        # 4) If eos_token_id is set, mark those samples as finished
        if eos_token_id is not None:
            eos_mask = next_tokens.squeeze(1) == eos_token_id
            unfinished = unfinished & (~eos_mask)

        # 5) Build 'soft embedding'
        # shape (B, d_model)
        soft_emb = probs @ W_E
        # For finished sequences, we can keep embedding zero or the same— they won't matter
        # if we skip them in future steps, but let's be consistent:
        soft_emb = torch.where(
            unfinished.unsqueeze(1),  # shape (B,1)
            soft_emb,
            torch.zeros_like(soft_emb),
        )
        next_emb = soft_emb.unsqueeze(1)  # shape (B,1,d_model)

        # 6) Store the sampled tokens
        all_tokens.append(next_tokens)

        # If all sequences are finished, we can stop early:
        if not unfinished.any():
            break

        # 7) Another forward pass, feeding next_emb and pkv:
        out = model(inputs_embeds=next_emb, past_key_values=pkv, use_cache=True)
        pkv = out.past_key_values
        logits = out.logits[:, -1, :]

    return torch.cat(all_tokens, dim=1)  # shape (B, prompt_len + actual_new_tokens)


# ==================== 4) CUSTOM GRPO TRAINER ====================

from trl import GRPOTrainer, GRPOConfig


class SoftGRPOTrainer(GRPOTrainer):
    """
    Subclass GRPOTrainer to override the generation with soft-sampling
    in a batched manner.
    """

    def __init__(
        self,
        *args,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        min_p: Optional[float] = None,
        max_new_tokens: int = 128,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.temperature = temperature
        self.top_k = top_k
        self.min_p = min_p
        self.max_new_tokens = max_new_tokens

    def generate_completions(
        self, prompts: List[Any], num_completions: int = 1, **kwargs
    ) -> List[List[Dict[str, str]]]:
        """
        Called by GRPOTrainer to produce completions for each prompt in 'prompts'.
        We do batched decoding for efficiency, replicating each prompt 'num_completions' times.
        Returns a list of chat-style completions: for each prompt + repetition,
        we return a single assistant message with the entire generated text.
        """
        # Convert chat prompts to text, then batch them
        text_prompts = [self._concat_chat(p) for p in prompts]

        # If we want multiple completions per prompt, replicate
        batched_prompts = []
        for p in text_prompts:
            batched_prompts.extend([p] * num_completions)

        # Tokenize
        enc = self.tokenizer(
            batched_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.model.device)

        # Soft decode
        with torch.no_grad():
            all_tokens = soft_decode_batch(
                model=self.model,
                input_ids=enc["input_ids"],
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_k=self.top_k,
                min_p=self.min_p,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode each sample
        decoded_list = [
            self.tokenizer.decode(ids, skip_special_tokens=True) for ids in all_tokens
        ]

        # Reformat back into chat list-of-lists
        # We'll group them in chunks of num_completions for each prompt
        completions = []
        idx = 0
        for _ in text_prompts:
            chunk = []
            for _ in range(num_completions):
                content = decoded_list[idx]
                chunk.append({"role": "assistant", "content": content})
                idx += 1
            completions.extend(chunk)

        # BUT the expected output is:
        #  -> a list with length = # of prompts * num_completions,
        # each item is a list of messages (for that single completion).
        # We'll re-group them properly:
        out = []
        idx = 0
        for _ in range(len(text_prompts)):
            block = []
            for _ in range(num_completions):
                block.append(completions[idx])
                idx += 1
            out.extend(block)

        # The trainer wants a list of length (#prompts * num_completions),
        # each an array of messages, so it's consistent with the rest of TRL.
        # Actually, a simpler approach is to just do it in one pass:
        final_completions = []
        idx = 0
        for _ in range(len(text_prompts)):
            for _ in range(num_completions):
                final_completions.append([completions[idx]])
                idx += 1

        return final_completions

    @staticmethod
    def _concat_chat(chat_messages: List[Dict[str, str]]) -> str:
        """
        Flatten chat messages into a single string.
        Minimal approach; adapt as needed (e.g., roles or newlines).
        """
        return "\n".join(m["content"] for m in chat_messages)


# ==================== 5) TRAINING SCRIPT (MAIN) ====================


def main():
    # 1. Load the dataset
    train_data = get_gsm8k_questions("train").select(
        range(100)
    )  # small subset for demo

    # 2. Combine reward functions (we can sum them)
    def composite_reward(prompts, completions, answer, **kwargs):
        r1 = correctness_reward(prompts, completions, answer, **kwargs)
        r2 = int_reward_func(completions, **kwargs)
        r3 = strict_format_reward(completions, **kwargs)
        r4 = soft_format_reward(completions, **kwargs)
        # sum up
        return [sum(x) for x in zip(r1, r2, r3, r4)]

    # 3. GRPOConfig
    from trl import GRPOConfig

    training_args = GRPOConfig(
        learning_rate=5e-6,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        logging_steps=10,
        max_steps=200,  # shortened for demo
        max_prompt_length=512,
        max_completion_length=256,
        per_device_train_batch_size=1,
        num_generations=2,  # how many completions to sample per prompt
        output_dir="soft-grpo-checkpoints",
        report_to="none",
    )

    # 4. Load model & tokenizer
    model_name = "facebook/galactica-125m"  # or another small model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))

    # 5. Initialize our SoftGRPOTrainer
    trainer = SoftGRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        reward_funcs=composite_reward,
        args=training_args,
        train_dataset=train_data,
        temperature=1.0,  # soft sampling hyperparams
        top_k=None,
        min_p=0.1,
        max_new_tokens=128,
    )

    # 6. Train
    trainer.train()


if __name__ == "__main__":
    main()
