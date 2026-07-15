# DeepMath GRPO

This directory contains the code used for the clean DeepMath split and RWKV GRPO ablations.

## Data Split

`split_deepmath.py` deduplicates normalized questions, keeps the official test split isolated, stratifies validation by difficulty, and writes JSONL outputs. The reported seed-42 split contains 91,602 train examples, 5,092 validation examples, and 5,145 test examples.

## Training Sources

- `train_rl_baseline.py`: main RWKV GRPO trainer.
- `train_rl_deepmath_resume.py`: continuation trainer used for step-20 to step-300 and step-300 to step-500 runs.
- `train_rl_standard_grpo.py`: standard group-reward advantage ablation with MaxRL-inspired group-correct-count scaling removed.
- `run_deepmath_gsm8k_grpo20.sh`, `run_deepmath_resume_to300.sh`, and `run_deepmath_resume_300_to500.sh`: main DeepMath run templates.
- `run_standard_grpo_neg1_lr5e7_500.sh` and `run_standard_grpo_neg1_lr2e7_200.sh`: `neg_adv_weight=1.0` standard-GRPO ablations.

## Main Configuration

The reported branch uses 8 questions per rollout step, 8 samples per question, `max_new_tokens=1024`, rollout sampling `temperature=1`, `top_p=0.6`, `top_k=0`, K3 KL coefficient `0.05`, full-parameter BF16 ZeRO-3 offload, and no hard-buffer, length, zstd, or n-gram reward.

## Reported Results

The selected `neg_adv_weight=1.0` PPO4 continuation reached `28.123%` full validation accuracy. Removing the MaxRL-inspired scaling while keeping `neg_adv_weight=1.0` learned faster early at `lr=5e-7`, but collapsed after step 200: step-100 accuracy was `25.000%`, step-200 `26.394%`, and step-300 `21.760%` with average response length around 19 tokens.

The follow-up `lr=2e-7` run is the stability test for this standard-GRPO formulation.
