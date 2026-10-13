#!/usr/bin/env bash
set -euo pipefail

CODE=/hy-tmp/RWKV7-g1i-strict-quality-gate-adamw-nooffload-20260810
EVAL=/hy-tmp/evals/official_flower_ckpt_eval_20260827/eval_math500_flower.py
ALBATROSS=/root/eval_transfer/albatross/faster3a_2605
BASE=/hy-tmp/evals/g1k_pass1_31_all_benchmarks_20260925
EXT=/hy-tmp/torch_extensions_g1k_pass1_31_benchmarks
RUN=/hy-tmp/runs/g1k_dapo_pass1_31_officialprompt_q32x16_cand64_max6144_ob512_step400to500_20260924_051022
PY=/usr/local/bin/python3

export PATH=/root/miniconda3/bin:$PATH
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
export TORCH_EXTENSIONS_DIR="$EXT"
export HF_HOME=/hy-tmp/hf_home
export XDG_CACHE_HOME=/hy-tmp/xdg_cache
mkdir -p "$BASE" "$EXT"

while [ ! -f "$RUN/final_step_500.pth" ]; do
  sleep 60
done

declare -a MODEL_PATHS=()
declare -a MODEL_LABELS=()
declare -A HASH_TO_INDEX=()

add_model() {
  local label=$1
  local path=$2
  [ -f "$path" ] || return 0
  local hash
  hash=$(sha256sum "$path" | awk '{print $1}')
  if [ -n "${HASH_TO_INDEX[$hash]+x}" ]; then
    local idx=${HASH_TO_INDEX[$hash]}
    MODEL_LABELS[$idx]="${MODEL_LABELS[$idx]},${label}"
    return 0
  fi
  HASH_TO_INDEX[$hash]=${#MODEL_PATHS[@]}
  MODEL_PATHS+=("$path")
  MODEL_LABELS+=("$label")
}

add_model step0 /hy-tmp/runs/rwkv-g1k-1b-temp-6003.pth
add_model step100 /hy-tmp/runs/g1k_step155_dapo_pass1_31_officialprompt_q32x16_cand64_max6144_ob512_step0to100_20260920_113659/final_step_100.pth
add_model step200 /hy-tmp/runs/g1k_dapo_pass1_31_officialprompt_q32x16_cand64_max6144_ob512_step100to200_20260921_062417/final_step_200.pth
add_model step300 /hy-tmp/runs/g1k_dapo_pass1_31_officialprompt_q32x16_cand64_max6144_ob512_step200to300_20260922_093913/final_step_300.pth
add_model step400 /hy-tmp/runs/g1k_dapo_pass1_31_officialprompt_q32x16_cand64_max6144_ob512_step300to400_20260923_070337/final_step_400.pth

while IFS= read -r path; do
  name=$(basename "$path" .pth)
  parent=$(basename "$(dirname "$path")")
  if [[ "$name" == final_step_500 ]]; then label=step500; else label="${parent}_${name}"; fi
  add_model "$label" "$path"
done < <(find /hy-tmp/runs -maxdepth 2 -type f -path '*g1k_dapo_pass1_31_officialprompt*' -name '*.pth' | sort)

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

printf 'label\tpath\n' > "$BASE/checkpoints.tsv"
for idx in "${!MODEL_PATHS[@]}"; do
  printf '%s\t%s\n' "${MODEL_LABELS[$idx]}" "${MODEL_PATHS[$idx]}" | tee -a "$BASE/checkpoints.tsv"
done

for idx in "${!MODEL_PATHS[@]}"; do
  label=${MODEL_LABELS[$idx]}
  model=${MODEL_PATHS[$idx]}
  for bench in gsm8k math500 aime24 aime25 amc23; do
    out="$BASE/$label/$bench"
    mkdir -p "$out"
    if [ -s "$out/summary.json" ]; then
      echo "SKIP_DONE $label $bench" | tee -a "$BASE/status.tsv"
      continue
    fi
    echo "START $(date '+%F %T') $label $bench rollout=${ROLLOUTS[$bench]}" | tee -a "$BASE/status.tsv"
    "$PY" "$EVAL" \
      --albatross-dir "$ALBATROSS" --model "$model" --dataset "${DATASETS[$bench]}" \
      --out-dir "$out" --gpus 0 --rollout "${ROLLOUTS[$bench]}" --bsz 512 \
      --max-new-tokens "${MAXNEWS[$bench]}" --ctx-limit "${CTXS[$bench]}" \
      --temperature 0.8 --top-p 0.35 --top-k 40 --seed 42 \
      --prompt-style official_flower --torch-extensions-dir "$EXT" \
      --progress-every 100 --verify-workers 8 \
      2>&1 | tee "$out/eval.log"
    "$PY" - "$out/summary.json" "$label" "$bench" <<'PY' | tee -a "$BASE/status.tsv"
import json
import sys
from datetime import datetime
path, label, bench = sys.argv[1:]
summary = json.load(open(path, encoding='utf-8'))
print(f"DONE {datetime.now():%F %T} {label} {bench} avg={summary.get('rollout_accuracy')} pass={summary.get('pass_at_rollout_accuracy')} n={summary.get('num_tasks')} rollout={summary.get('rollout')}")
PY
  done
done
echo "ALL_DONE $(date '+%F %T')" | tee -a "$BASE/status.tsv"
