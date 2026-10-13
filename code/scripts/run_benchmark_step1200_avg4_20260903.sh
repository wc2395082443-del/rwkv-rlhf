#!/usr/bin/env bash
set -u
BASE=/hy-tmp/evals/benchmark_step1200_avg4_20260903
MODEL=/hy-tmp/runs/g1i_step1000_to_dapo_math17k_officialprompt_q32x16_cand64_max6144_ob512_continue1000to1200_eval384_ckpt50_20260901_1100/checkpoint_step_1200.pth
EVAL_SCRIPT=/tmp/eval_math500_official_cot_prompt.py
ALBATROSS=/root/eval_transfer/albatross/faster3a_2605
export PYTHONPATH=$ALBATROSS:/hy-tmp/RWKV7-g1i-strict-quality-gate-adamw-nooffload-20260810:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
mkdir -p "$BASE"
STATUS=$BASE/status.tsv
touch "$STATUS"
run_one() {
  local name=$1
  local dataset=$2
  local out="$BASE/$name"
  mkdir -p "$out"
  if [[ -s "$out/summary.json" ]]; then
    echo "$(date '+%F %T') SKIP $name" | tee -a "$STATUS"
    return 0
  fi
  echo "$(date '+%F %T') START $name" | tee -a "$STATUS"
  python3 "$EVAL_SCRIPT" \
    --albatross-dir "$ALBATROSS" \
    --model "$MODEL" \
    --dataset "$dataset" \
    --out-dir "$out" \
    --gpus 0 \
    --rollout 4 \
    --bsz 512 \
    --max-new-tokens 8192 \
    --ctx-limit 12288 \
    --eval-tail-max-tokens 0 \
    --temperature 0.8 \
    --top-p 0.35 \
    --top-k 40 \
    --prompt-style official_cot \
    --progress-every 20 \
    --verify-workers 16 2>&1 | tee "$out/eval.log"
  local rc=${PIPESTATUS[0]}
  if [[ $rc -eq 0 && -s "$out/summary.json" ]]; then
    echo "$(date '+%F %T') DONE $name" | tee -a "$STATUS"
  else
    echo "$(date '+%F %T') FAILED $name rc=$rc" | tee -a "$STATUS"
  fi
}
run_one gsm8k /hy-tmp/data/gsm8k/gsm8k_test_albatross.jsonl
run_one math500 /hy-tmp/data/math500/test.jsonl
echo "$(date '+%F %T') ALL_DONE" | tee -a "$STATUS"