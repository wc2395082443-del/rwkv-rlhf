#!/usr/bin/env bash
set -euo pipefail

BASE=/hy-tmp/evals/official_flower_ckpt_eval_step800_20260829
EVAL=/hy-tmp/evals/official_flower_ckpt_eval_20260827/eval_math500_flower.py
ALBATROSS=/root/eval_transfer/albatross/faster3a_2605
MODEL=/hy-tmp/runs/g1i_step600_to_dapo_math17k_officialprompt_q32x16_cand64_max6144_ob512_continue600to800_eval384_20260827/final_step_800.pth
PY=/usr/local/bin/python3
EXT=/hy-tmp/torch_extensions_official_ckpt_eval_20260827
STATUS=$BASE/status.tsv

mkdir -p "$BASE" "$EXT"
: > "$STATUS"
export PATH=/root/miniconda3/bin:$PATH
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
export TORCH_EXTENSIONS_DIR="$EXT"
export HF_HOME=/hy-tmp/hf_home
export XDG_CACHE_HOME=/hy-tmp/xdg_cache

run_bench() {
  local bench=$1
  local dataset rollout out
  if [[ "$bench" == gsm8k ]]; then
    dataset=/hy-tmp/data/gsm8k/gsm8k_test_albatross.jsonl
    rollout=4
  else
    dataset=/hy-tmp/data/math500/test.jsonl
    rollout=8
  fi
  out="$BASE/step800/$bench"
  mkdir -p "$out"
  echo -e "$(date '+%F %T')\tSTART\t$bench" | tee -a "$STATUS"
  "$PY" "$EVAL" \
    --albatross-dir "$ALBATROSS" \
    --model "$MODEL" \
    --dataset "$dataset" \
    --out-dir "$out" \
    --gpus 0 --rollout "$rollout" --bsz 512 \
    --max-new-tokens 4096 --ctx-limit 8192 \
    --temperature 0.8 --top-p 0.35 --top-k 40 \
    --seed 42 --prompt-style official_flower \
    --torch-extensions-dir "$EXT" \
    --progress-every 100 --verify-workers 8 \
    2>&1 | tee "$out/eval.log"
  echo -e "$(date '+%F %T')\tDONE\t$bench" | tee -a "$STATUS"
}

run_bench gsm8k
run_bench math500
echo -e "$(date '+%F %T')\tALL_DONE" | tee -a "$STATUS"
