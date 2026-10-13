#!/usr/bin/env bash
set -euo pipefail

BASE=/hy-tmp/evals/g1k_pass1_31_missing_benchmarks_20261002
EVAL=/hy-tmp/evals/official_flower_ckpt_eval_20260827/eval_math500_flower.py
ALBATROSS=/root/eval_transfer/albatross/faster3a_2605
EXT=/hy-tmp/torch_extensions_g1k_missing_benchmarks
RUN=/hy-tmp/runs/g1k_dapo_pass1_31_officialprompt_q32x16_cand64_max6144_ob512_step900to1000_20261001_014411
PY=/usr/local/bin/python3

export PATH=/root/miniconda3/bin:$PATH
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
export TORCH_EXTENSIONS_DIR="$EXT"
export HF_HOME=/hy-tmp/hf_home
export XDG_CACHE_HOME=/hy-tmp/xdg_cache
mkdir -p "$BASE" "$EXT"

while [ ! -f "$RUN/final_step_1000.pth" ]; do sleep 60; done

declare -A MODELS=(
  [step600]=/hy-tmp/runs/g1k_dapo_pass1_31_officialprompt_q32x16_cand64_max6144_ob512_step500to600_20260925_211542/final_step_600.pth
  [step700]=/hy-tmp/runs/g1k_dapo_pass1_31_officialprompt_q32x16_cand64_max6144_ob512_step600to700_20260926_224314/final_step_700.pth
  [step800]=/hy-tmp/runs/g1k_dapo_pass1_31_officialprompt_q32x16_cand64_max6144_ob512_step700to800_20260928_071906/final_step_800.pth
  [step900]=/hy-tmp/runs/g1k_dapo_pass1_31_officialprompt_q32x16_cand64_max6144_ob512_step800to900_20260929_172737/final_step_900.pth
  [step1000]="$RUN/final_step_1000.pth"
)
declare -A DATASETS=(
  [gsm8k]=/hy-tmp/data/gsm8k/gsm8k_test_albatross.jsonl
  [math500]=/hy-tmp/data/math500/test.jsonl
  [aime24]=/hy-tmp/data/official_math_extra/aime24.jsonl
  [aime25]=/hy-tmp/data/official_math_extra/aime25.jsonl
  [amc23]=/hy-tmp/data/official_math_extra/amc23.jsonl
)
declare -A ROLLOUTS=([gsm8k]=1 [math500]=1 [aime24]=64 [aime25]=64 [amc23]=1)
declare -A MAXNEWS=([gsm8k]=4096 [math500]=4096 [aime24]=8192 [aime25]=8192 [amc23]=8192)
declare -A CTXS=([gsm8k]=8192 [math500]=8192 [aime24]=12288 [aime25]=12288 [amc23]=12288)

for step in step600 step700 step800 step900 step1000; do
  model=${MODELS[$step]}
  [ -f "$model" ] || { echo "MISSING_MODEL $step $model" | tee -a "$BASE/status.tsv"; continue; }
  for bench in gsm8k math500 aime24 aime25 amc23; do
    out="$BASE/$step/$bench"
    mkdir -p "$out"
    if [ -s "$out/summary.json" ]; then
      echo "SKIP_DONE $step $bench" | tee -a "$BASE/status.tsv"
      continue
    fi
    echo "START $(date '+%F %T') $step $bench rollout=${ROLLOUTS[$bench]}" | tee -a "$BASE/status.tsv"
    "$PY" "$EVAL" --albatross-dir "$ALBATROSS" --model "$model" --dataset "${DATASETS[$bench]}" \
      --out-dir "$out" --gpus 0 --rollout "${ROLLOUTS[$bench]}" --bsz 512 \
      --max-new-tokens "${MAXNEWS[$bench]}" --ctx-limit "${CTXS[$bench]}" \
      --temperature 0.8 --top-p 0.35 --top-k 40 --seed 42 \
      --prompt-style official_flower --torch-extensions-dir "$EXT" --progress-every 100 --verify-workers 8 \
      2>&1 | tee "$out/eval.log"
    "$PY" - "$out/summary.json" "$step" "$bench" <<'PY' | tee -a "$BASE/status.tsv"
import json, sys
from datetime import datetime
path, step, bench = sys.argv[1:]
s = json.load(open(path, encoding="utf-8"))
print(f"DONE {datetime.now():%F %T} {step} {bench} avg={s.get('rollout_accuracy')} pass={s.get('pass_at_rollout_accuracy')} n={s.get('num_tasks')} rollout={s.get('rollout')}")
PY
  done
done
echo "ALL_DONE $(date '+%F %T')" | tee -a "$BASE/status.tsv"
