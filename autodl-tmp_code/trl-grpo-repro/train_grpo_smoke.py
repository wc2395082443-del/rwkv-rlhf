import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoTokenizer, BitsAndBytesConfig
from trl import GRPOConfig, GRPOTrainer
from trl.rewards import accuracy_reward, think_format_reward


def _as_message_completions(completions):
    if completions and isinstance(completions[0], str):
        return [[{"role": "assistant", "content": text}] for text in completions]
    return completions


def compat_accuracy_reward(completions, solution, **kwargs):
    return accuracy_reward(_as_message_completions(completions), solution, **kwargs)


def compat_think_format_reward(completions, **kwargs):
    return think_format_reward(_as_message_completions(completions), **kwargs)


def parse_args():
    parser = argparse.ArgumentParser(description="Minimal local reproduction of TRL GRPOTrainer.")
    parser.add_argument(
        "--model_path",
        type=str,
        default=r"C:\Users\23950\code\DeepSeek-R1-Distill-Qwen-1.5B",
        help="Local model path or HF model id.",
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default=str(Path(__file__).resolve().parent / "data" / "tiny_math.jsonl"),
        help="JSONL file with `prompt` and `solution` columns.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "outputs" / "smoke"),
        help="Trainer output directory.",
    )
    parser.add_argument("--max_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--num_generations", type=int, default=2)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--max_completion_length", type=int, default=64)
    parser.add_argument(
        "--loss_type",
        type=str,
        default="dapo",
        choices=["dapo", "grpo", "dr_grpo", "bnpo"],
        help="TRL 1.0.0 defaults to `dapo`; use `grpo` for classic GRPO loss.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset("json", data_files={"train": args.train_file})["train"]

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    model_init_kwargs = {
        "quantization_config": quant_config,
        "device_map": "auto",
        "dtype": torch.float16,
    }

    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
    )

    training_args = GRPOConfig(
        output_dir=str(output_dir),
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=1,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False,
        fp16=False,
        bf16=False,
        gradient_checkpointing=True,
        use_vllm=False,
        max_completion_length=args.max_completion_length,
        num_generations=args.num_generations,
        temperature=0.7,
        top_p=1.0,
        beta=0.0,
        loss_type=args.loss_type,
        model_init_kwargs=model_init_kwargs,
        log_completions=True,
        seed=args.seed,
    )

    trainer = GRPOTrainer(
        model=args.model_path,
        reward_funcs=[compat_accuracy_reward, compat_think_format_reward],
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model()
    print(f"Done. Artifacts written to: {output_dir}")


if __name__ == "__main__":
    main()
