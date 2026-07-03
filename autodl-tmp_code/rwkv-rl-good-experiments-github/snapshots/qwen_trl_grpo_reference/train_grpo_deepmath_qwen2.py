import argparse
from pathlib import Path

import torch
from datasets import load_from_disk
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from deepmath_official_rewards import deepmath_r1_reward, deepmath_zero_reward


def parse_args():
    parser = argparse.ArgumentParser(
        description="Single-GPU TRL GRPO reproduction with DeepMath official-style prompt/reward logic."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/root/autodl-tmp/official_repro_assets/Qwen2-0.5B-Instruct",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Path created by prepare_deepmath_trl_dataset.py. If omitted, a mode-specific official path is used.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/root/autodl-tmp/trl-grpo-repro/outputs/qwen2_deepmath_grpo",
    )
    parser.add_argument(
        "--official_mode",
        type=str,
        default="deepmath_zero",
        choices=["deepmath_zero", "deepmath_r1"],
        help=(
            "DeepMath official preset. `deepmath_zero` uses simplerl boxed-format reward; "
            "`deepmath_r1` uses R1-style </think> reward."
        ),
    )
    parser.add_argument("--max_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_generations", type=int, default=8)
    parser.add_argument("--max_completion_length", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=None, help="KL coefficient. If omitted, uses the official preset default.")
    parser.add_argument(
        "--loss_type",
        type=str,
        default="grpo",
        choices=["dapo", "grpo", "dr_grpo", "bnpo"],
    )
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--log_completions", action="store_true")
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

    if args.dataset_path is None:
        dataset_path = (
            f"/root/autodl-tmp/official_repro_assets/DeepMath-103K-trl-{args.official_mode}"
        )
    else:
        dataset_path = args.dataset_path

    dataset = load_from_disk(dataset_path)["train"]
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

    reward_func = deepmath_zero_reward if args.official_mode == "deepmath_zero" else deepmath_r1_reward
    temperature = args.temperature
    if temperature is None:
        temperature = 1.0 if args.official_mode == "deepmath_zero" else 0.6
    beta = args.beta
    if beta is None:
        beta = 0.0 if args.official_mode == "deepmath_zero" else 0.001

    training_args = GRPOConfig(
        output_dir=str(output_dir),
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        temperature=temperature,
        top_p=args.top_p,
        logging_steps=args.logging_steps,
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False,
        bf16=dtype == torch.bfloat16,
        fp16=dtype == torch.float16,
        gradient_checkpointing=True,
        use_vllm=False,
        beta=beta,
        loss_type=args.loss_type,
        log_completions=args.log_completions,
        seed=args.seed,
        model_init_kwargs=model_init_kwargs,
    )

    trainer = GRPOTrainer(
        model=args.model_path,
        reward_funcs=reward_func,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model()
    print(f"Done. Artifacts written to: {output_dir}")


if __name__ == "__main__":
    main()
