import argparse
from pathlib import Path

import torch
from datasets import load_from_disk
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
from trl.rewards import accuracy_reward


def _as_message_completions(completions):
    if completions and isinstance(completions[0], str):
        return [[{"role": "assistant", "content": text}] for text in completions]
    return completions


def compat_accuracy_reward(completions, solution, **kwargs):
    return accuracy_reward(_as_message_completions(completions), solution, **kwargs)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Paper-aligned GRPO reproduction on Qwen2-0.5B-Instruct with official HF DeepMath formatting."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/root/autodl-tmp/official_repro_assets/Qwen2-0.5B-Instruct",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/root/autodl-tmp/official_repro_assets/DeepMath-103K-trl-hf-official",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/dev/shm/trl-grpo-repro/outputs/qwen2_trl_paper_aligned",
    )
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_generations", type=int, default=64)
    parser.add_argument("--max_completion_length", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.04)
    parser.add_argument("--loss_type", type=str, default="grpo", choices=["dapo", "grpo", "dr_grpo", "bnpo"])
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="sdpa",
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
        save_strategy="no",
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

