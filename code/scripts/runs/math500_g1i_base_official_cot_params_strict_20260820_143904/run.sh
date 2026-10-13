#!/usr/bin/env bash
set -euo pipefail
CODE=/hy-tmp/RWKV7-g1i-strict-quality-gate-adamw-nooffload-20260810
ALB=/root/eval_transfer/albatross/faster3a_2605
export PYTHONPATH=$ALB:$CODE:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=0
python3 /tmp/eval_math500_official_cot_prompt.py \
  --albatross-dir $ALB \
  --model /hy-tmp/models/rwkv7-g1i_preview5445-1.5b-20260729-ctx16384.pth \
  --dataset /hy-tmp/data/math500/test.jsonl \
  --out-dir /hy-tmp/runs/math500_g1i_base_official_cot_params_strict_20260820_143904 \
  --gpus 0 \
  --rollout 1 \
  --bsz 512 \
  --max-new-tokens 4096 \
  --eval-tail-max-tokens 0 \
  --temperature 0.8 \
  --top-p 0.35 \
  --top-k 40 \
  --prompt-style official_cot \
  --progress-every 25 \
  --verify-workers 16
