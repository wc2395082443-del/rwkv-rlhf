#!/usr/bin/env bash
set -euo pipefail
CODE=/hy-tmp/RWKV7-g1i-strict-quality-gate-adamw-nooffload-20260810
EVAL=$CODE/eval_thinking_cot_strict.py
DATA=/hy-tmp/data/math500/test.jsonl
ALBATROSS=/root/eval_transfer/albatross/faster3a_2605
OUT=/hy-tmp/runs/math500_g1i_ckpt_eval_20260820_114548/step200_rerun
PY=python3
EXT=/hy-tmp/torch_extensions_math500_g1i_eval_20260820
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
export TORCH_EXTENSIONS_DIR=$EXT
export PYTHONPATH=/root/eval_transfer/albatross/faster3a_2605:$CODE:${PYTHONPATH:-}
mkdir -p "$OUT" "$EXT"
echo "[$(date '+%F %T')] START step200 rerun" | tee -a /hy-tmp/runs/math500_g1i_ckpt_eval_20260820_114548/status.log
$PY "$EVAL" \
  --albatross-dir "$ALBATROSS" \
  --model /hy-tmp/runs/g1i_step155_to_dapo_math17k_pass1_16_val500_max4096_micro1_fullmem_resetstep_200_20260819_1503/final_step_200.pth \
  --dataset "$DATA" \
  --out-dir "$OUT" \
  --gpus 0 --rollout 1 --bsz 512 \
  --max-new-tokens 4096 --eval-tail-max-tokens 0 \
  --temperature 0.3 --top-p 0.4 --top-k 500 \
  --seed 42 --prompt-style thinking_cot \
  --torch-extensions-dir "$EXT" \
  --progress-every 20 --verify-workers 8 \
  2>&1 | tee "$OUT/eval.log"
echo "[$(date '+%F %T')] END step200 rerun" | tee -a /hy-tmp/runs/math500_g1i_ckpt_eval_20260820_114548/status.log
