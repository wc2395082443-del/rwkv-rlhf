#!/usr/bin/env bash
set -euo pipefail

BASE=/hy-tmp/evals/extra_math_step850_1000_officialcot_r64_ctx12k_max8192_20260831
RAW=$BASE/raw
RECOVERED=$BASE/recovered_tail128
DATA_ROOT=/hy-tmp/data/official_math_extra
MODEL_ROOT=/hy-tmp/runs/g1i_step800_to_dapo_math17k_officialprompt_q32x16_cand64_max6144_ob512_continue800to1000_eval384_ckpt50_20260829_1810
EVAL_SCRIPT=/tmp/eval_math500_official_cot_prompt.py
RECOVER_SCRIPT=/tmp/recover_existing_tail_step850_1000.py
ALBATROSS=/root/eval_transfer/albatross/faster3a_2605

export PYTHONPATH=$ALBATROSS:/hy-tmp/RWKV7-g1i-strict-quality-gate-adamw-nooffload-20260810:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=0
mkdir -p "$RAW" "$RECOVERED"

run_one() {
  local dataset=$1
  local step=$2
  local out=$RAW/$dataset/step$step
  local model=$MODEL_ROOT/checkpoint_step_$step.pth
  mkdir -p "$out"
  if [[ -s "$out/summary.json" ]]; then
    echo "$(date '+%F %T') SKIP step$step $dataset" | tee -a "$BASE/status.tsv"
    return
  fi
  echo "$(date '+%F %T') START step$step $dataset" | tee -a "$BASE/status.tsv"
  python3 "$EVAL_SCRIPT" \
    --albatross-dir "$ALBATROSS" \
    --model "$model" \
    --dataset "$DATA_ROOT/$dataset.jsonl" \
    --out-dir "$out" \
    --gpus 0 \
    --rollout 64 \
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
  echo "$(date '+%F %T') DONE step$step $dataset" | tee -a "$BASE/status.tsv"
}

for dataset in aime24 aime25 amc23; do
  for step in 850 900 950 1000; do
    run_one "$dataset" "$step"
  done
done

echo "$(date '+%F %T') START recovery_tail128" | tee -a "$BASE/status.tsv"
python3 "$RECOVER_SCRIPT" \
  --old-run "$RAW" \
  --out-run "$RECOVERED" \
  --albatross-dir "$ALBATROSS" \
  --datasets aime24 aime25 amc23 \
  --ckpts step850 step900 step950 step1000 \
  --bsz 512 \
  --tail-max-tokens 128 \
  --ctx-limit 12288 \
  --temperature 0.8 \
  --top-p 0.35 \
  --top-k 40 \
  --tail-prefix 'Therefore, the final answer is \boxed{' \
  --verify-workers 16 \
  --progress-every 20 2>&1 | tee "$RECOVERED/recovery.log"
echo "$(date '+%F %T') ALL_DONE" | tee -a "$BASE/status.tsv"
