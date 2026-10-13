#!/usr/bin/env bash
set -u
OUT=/hy-tmp/evals/pass32_zero_step1200_20260903
MODEL=/hy-tmp/runs/g1i_step1000_to_dapo_math17k_officialprompt_q32x16_cand64_max6144_ob512_continue1000to1200_eval384_ckpt50_20260901_1100/checkpoint_step_1200.pth
DATA=/hy-tmp/data/openr1_dapo_math17k_processed/pass32_0_step155/train_pass32_zero.jsonl
ALBATROSS=/root/eval_transfer/albatross/faster3a_2605
mkdir -p "$OUT"
export PYTHONPATH=$ALBATROSS:/hy-tmp/RWKV7-g1i-strict-quality-gate-adamw-nooffload-20260810:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
printf '%s START bsz=8 dataset=%s model=%s\n' "$(date '+%F %T')" "$DATA" "$MODEL" | tee "$OUT/status.log"
python3 /tmp/eval_math500_official_cot_prompt.py \
 --albatross-dir "$ALBATROSS" --model "$MODEL" --dataset "$DATA" --out-dir "$OUT" \
 --gpus 0 --rollout 1 --bsz 8 --max-new-tokens 8192 --ctx-limit 12288 \
 --eval-tail-max-tokens 0 --temperature 0.8 --top-p 0.35 --top-k 40 \
 --prompt-style official_cot --progress-every 20 --verify-workers 16 2>&1 | tee "$OUT/eval.log"
rc=${PIPESTATUS[0]}
printf '%s END rc=%s\n' "$(date '+%F %T')" "$rc" | tee -a "$OUT/status.log"