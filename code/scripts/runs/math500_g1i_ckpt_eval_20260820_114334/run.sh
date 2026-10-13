#!/usr/bin/env bash
set -euo pipefail

CODE=/hy-tmp/RWKV7-g1i-strict-quality-gate-adamw-nooffload-20260810
EVAL=$CODE/eval_thinking_cot_strict.py
DATA=/hy-tmp/data/math500/test.jsonl
ALBATROSS=$CODE
OUT=/hy-tmp/runs/math500_g1i_ckpt_eval_20260820_$(date +%H%M%S)
PY=python3
EXT=/hy-tmp/torch_extensions_math500_g1i_eval_20260820

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
export TORCH_EXTENSIONS_DIR=$EXT
mkdir -p "$OUT" "$EXT"
cp "$0" "$OUT/run.sh"
echo "$OUT" > /hy-tmp/runs/current_math500_g1i_ckpt_eval.path

run_one() {
  local name="$1"
  local model="$2"
  mkdir -p "$OUT/$name"
  echo "[$(date '+%F %T')] START $name $model" | tee -a "$OUT/status.log"
  $PY "$EVAL" \
    --albatross-dir "$ALBATROSS" \
    --model "$model" \
    --dataset "$DATA" \
    --out-dir "$OUT/$name" \
    --gpus 0 --rollout 1 --bsz 512 \
    --max-new-tokens 4096 --eval-tail-max-tokens 0 \
    --temperature 0.3 --top-p 0.4 --top-k 500 \
    --seed 42 --prompt-style thinking_cot \
    --torch-extensions-dir "$EXT" \
    --progress-every 20 --verify-workers 8 \
    2>&1 | tee "$OUT/$name/eval.log"
  echo "[$(date '+%F %T')] END $name" | tee -a "$OUT/status.log"
}

run_one step155 /hy-tmp/runs/g1i_gsm8k_boxed_strictcot_continue_from_step105_fixed50_maxnew2048_micro3_dyn512_q32_r16_20260812_142929/final_step_155.pth
run_one step170 /hy-tmp/runs/g1i_step155_to_openr1_dapo_math17k_processed_maxnew6144_micro1_20260813_094132/final_step_170.pth
run_one step200 /hy-tmp/runs/g1i_step155_to_dapo_math17k_pass1_16_val500_max4096_micro1_fullmem_resetstep_200_20260819_1503/final_step_200.pth
