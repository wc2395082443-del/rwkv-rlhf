#!/usr/bin/env bash
set -uo pipefail
RUN=/hy-tmp/runs/g1i_step1000_to_dapo_math17k_officialprompt_q32x16_cand64_max6144_ob512_continue1000to1200_eval384_ckpt50_20260901_1100
OUT=/hy-tmp/bench_g1i_1050_1200_gsm8k_math500_official4096_20260903
ALB=/root/eval_transfer/albatross/faster3a_2605
export PYTHONPATH="$ALB:/hy-tmp/RWKV7-g1i-strict-quality-gate-adamw-nooffload-20260810:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
DATA_GSM=/hy-tmp/data/gsm8k/gsm8k_test_albatross.jsonl
DATA_MATH=/hy-tmp/data/math500/test.jsonl
for STEP in 1050 1100 1150 1200; do
  MODEL="$RUN/checkpoint_step_${STEP}.pth"
  for BENCH in gsm8k math500; do
    if [ "$BENCH" = gsm8k ]; then DATASET="$DATA_GSM"; else DATASET="$DATA_MATH"; fi
    DEST="$OUT/$BENCH/step_${STEP}"
    mkdir -p "$DEST"
    if [ -s "$DEST/summary.json" ]; then
      printf "%s SKIP %s step_%s\\n" "$(date +%F-%T)" "$BENCH" "$STEP" | tee -a "$OUT/status.log"
      continue
    fi
    printf "%s START %s step_%s\\n" "$(date +%F-%T)" "$BENCH" "$STEP" | tee -a "$OUT/status.log"
    python3 /tmp/eval_math500_official_cot_prompt.py \
      --albatross-dir "$ALB" --model "$MODEL" --dataset "$DATASET" --out-dir "$DEST" \
      --gpus 0 --rollout 1 --bsz 512 --max-new-tokens 4096 --ctx-limit 8192 \
      --eval-tail-max-tokens 0 --temperature 0.8 --top-p 0.35 --top-k 40 \
      --prompt-style official_cot --progress-every 100 --verify-workers 16 \
      2>&1 | tee "$DEST/eval.log"
    RC=${PIPESTATUS[0]}
    if [ "$RC" -ne 0 ]; then
      printf "%s FAILED %s step_%s rc=%s\\n" "$(date +%F-%T)" "$BENCH" "$STEP" "$RC" | tee -a "$OUT/status.log"
      exit "$RC"
    fi
    printf "%s DONE %s step_%s\\n" "$(date +%F-%T)" "$BENCH" "$STEP" | tee -a "$OUT/status.log"
  done
done
printf "%s ALL_DONE\\n" "$(date +%F-%T)" | tee -a "$OUT/status.log"
