import argparse
from pathlib import Path
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
from mcq_reward import mcq_accuracy_reward


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--dataset_path", default="/root/autodl-tmp/stem-rlvr-repro/data/mmlupro_stem_trl")
    ap.add_argument("--output_dir", default="/root/autodl-tmp/stem-rlvr-repro/outputs/qwen25_1p5b_mmlupro_stem_grpo")
    ap.add_argument("--max_steps", type=int, default=100)
    ap.add_argument("--learning_rate", type=float, default=1e-6)
    ap.add_argument("--per_device_train_batch_size", type=int, default=8)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=1)
    ap.add_argument("--num_generations", type=int, default=8)
    ap.add_argument("--max_completion_length", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=0.0)
    ap.add_argument("--loss_type", default="dapo", choices=["dapo", "grpo", "dr_grpo", "bnpo"])
    ap.add_argument("--max_train_samples", type=int, default=None)
    ap.add_argument("--save_steps", type=int, default=1000)
    ap.add_argument("--attn_implementation", default="sdpa")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ds = load_from_disk(args.dataset_path)["train"]
    if args.max_train_samples:
        ds = ds.select(range(min(args.max_train_samples, len(ds))))

    tok = AutoTokenizer.from_pretrained(args.model_path, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    cfg = GRPOConfig(
        output_dir=str(out),
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        top_p=args.top_p,
        logging_steps=1,
        save_steps=args.save_steps,
        save_total_limit=1,
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False,
        bf16=dtype == torch.bfloat16,
        fp16=dtype == torch.float16,
        gradient_checkpointing=True,
        use_vllm=False,
        beta=args.beta,
        loss_type=args.loss_type,
        seed=42,
        model_init_kwargs={"torch_dtype": dtype, "attn_implementation": args.attn_implementation},
    )
    trainer = GRPOTrainer(
        model=args.model_path,
        reward_funcs=mcq_accuracy_reward,
        args=cfg,
        train_dataset=ds,
        processing_class=tok,
    )
    trainer.train()
    trainer.save_model()
    print(f"Done. Artifacts written to: {out}")

if __name__ == "__main__":
    main()

