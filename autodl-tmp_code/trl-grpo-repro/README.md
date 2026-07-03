# TRL GRPO Repro

This folder reproduces the official Hugging Face TRL `GRPOTrainer` workflow from:

- `https://github.com/huggingface/trl/blob/main/docs/source/grpo_trainer.md`

It is a local, low-VRAM reproduction based on the official trainer, not a reimplementation.

## What is included

- `train_grpo_smoke.py`: minimal single-GPU smoke run
- `data/tiny_math.jsonl`: tiny math dataset with `prompt` + `solution`

## Local environment used

- `trl==1.0.0`
- `transformers==5.5.0`
- `torch==2.7.0+cu128`
- `math_verify`
- `bitsandbytes>=0.46.1`

## Important note

In current TRL, `GRPOConfig.loss_type` defaults to `dapo`, not classic `grpo`.

- Use `--loss_type dapo` to match current TRL defaults.
- Use `--loss_type grpo` if you want the classic GRPO loss.
- Use `--loss_type dr_grpo` if you want the Dr.GRPO variant.

## Quick start

The default script is tuned for an 8 GB GPU:

- 4-bit base model
- LoRA adapters
- no vLLM
- tiny batch
- 1 training step by default

Run:

```powershell
cd C:\Users\23950\code\trl-grpo-repro
python train_grpo_smoke.py `
  --model_path C:\Users\23950\code\DeepSeek-R1-Distill-Qwen-1.5B `
  --max_steps 1 `
  --loss_type dapo
```

## Dataset format

Training data is plain JSONL. Each row must contain:

```json
{"prompt":"...", "solution":"\\boxed{4}"}
```

This matches the built-in TRL reward combination used here:

- `trl.rewards.accuracy_reward`
- `trl.rewards.think_format_reward`

## How this maps to the official docs

Official minimal trainer shape:

```python
trainer = GRPOTrainer(
    model="...",
    reward_funcs=accuracy_reward,
    train_dataset=dataset,
)
trainer.train()
```

This local repro keeps the same core pieces, but adds:

- `BitsAndBytesConfig(load_in_4bit=True)`
- `LoRA`
- explicit `GRPOConfig`

Those additions are only to make the example fit a local 8 GB GPU.

## Next step for a larger machine

If you move to a larger GPU or a Linux server, the next changes should be:

1. switch model to `Qwen/Qwen2.5-0.5B-Instruct` or your target model
2. replace `data/tiny_math.jsonl` with GSM8K / math dataset
3. increase `num_generations`
4. increase `max_completion_length`
5. optionally enable `use_vllm=True`

