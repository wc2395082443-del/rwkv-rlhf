import argparse
import re
from pathlib import Path

import torch
from datasets import load_from_disk
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
from trl.rewards import accuracy_reward


BOOL_ANSWERS = {"yes", "no", "true", "false"}


def _as_message_completions(completions):
    if completions and isinstance(completions[0], str):
        return [[{"role": "assistant", "content": text}] for text in completions]
    return completions


def _extract_bool_answer(text):
    boxed = re.findall(r"\\boxed\s*{\s*(yes|no|true|false)\s*}", text, flags=re.IGNORECASE)
    if boxed:
        return boxed[-1].lower()
    matches = re.findall(r"\b(yes|no|true|false)\b", text, flags=re.IGNORECASE)
    if matches:
        return matches[-1].lower()
    return None


def compat_accuracy_reward(completions, solution, **kwargs):
    completions = _as_message_completions(completions)
    rewards = [None] * len(solution)
    math_completions = []
    math_solutions = []
    math_indices = []

    for idx, (completion, sol) in enumerate(zip(completions, solution)):
        sol_text = str(sol).strip()
        sol_norm = sol_text.lower()
        if sol_norm in BOOL_ANSWERS:
            pred = _extract_bool_answer(completion[0]["content"])
            rewards[idx] = float(pred == sol_norm) if pred is not None else 0.0
        else:
            math_completions.append(completion)
            math_solutions.append(sol)
            math_indices.append(idx)

    if math_completions:
        math_rewards = accuracy_reward(math_completions, math_solutions, **kwargs)
        for idx, reward in zip(math_indices, math_rewards):
            rewards[idx] = reward

    return rewards


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reproduce the TRL GRPO docs example on Qwen2-0.5B-Instruct + DeepMath-103K."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/dev/shm/official_repro_assets/Qwen2.5-0.5B-Instruct",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/dev/shm/official_repro_assets/DeepMath-103K-trl-hf-official",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/dev/shm/trl-grpo-repro/outputs/qwen2p5_trl_docs_repro",
    )
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_generations", type=int, default=8)
    parser.add_argument("--max_completion_length", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument(
        "--loss_type",
        type=str,
        default="dapo",
        choices=["dapo", "grpo", "dr_grpo", "bnpo"],
    )
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--save_total_limit", type=int, default=1)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="flash_attention_2",
        choices=["sdpa", "flash_attention_2", "eager"],
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_from_disk(args.dataset_path)["train"]
    if args.max_train_samples is not None:
        dataset = dataset.select(range(min(args.max_train_samples, len(dataset))))

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    model_init_kwargs = {
        "torch_dtype": dtype,
        "attn_implementation": args.attn_implementation,
    }

    training_args = GRPOConfig(
        output_dir=str(output_dir),
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        top_p=args.top_p,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        report_to="none",
        remove_unused_columns=False,
        bf16=dtype == torch.bfloat16,
        fp16=dtype == torch.float16,
        gradient_checkpointing=True,
        use_vllm=False,
        beta=args.beta,
        loss_type=args.loss_type,
        seed=args.seed,
        model_init_kwargs=model_init_kwargs,
    )

    trainer = GRPOTrainer(
        model=args.model_path,
        reward_funcs=compat_accuracy_reward,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model()
    print(f"Done. Artifacts written to: {output_dir}")


if __name__ == "__main__":
    main()

