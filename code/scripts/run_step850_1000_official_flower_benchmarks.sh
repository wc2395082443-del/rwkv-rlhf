#!/usr/bin/env bash
set -euo pipefail

BASE=/hy-tmp/evals/official_flower_ckpt_eval_step850_1000_20260831
EVAL=/hy-tmp/evals/official_flower_ckpt_eval_20260827/eval_math500_flower.py
ALBATROSS=/root/eval_transfer/albatross/faster3a_2605
RUN=/hy-tmp/runs/g1i_step800_to_dapo_math17k_officialprompt_q32x16_cand64_max6144_ob512_continue800to1000_eval384_ckpt50_20260829_1810
PY=/usr/local/bin/python3
EXT=/hy-tmp/torch_extensions_official_ckpt_eval_20260827
STATUS=$BASE/status.tsv

mkdir -p "$BASE" "$EXT"
touch "$STATUS"
export PATH=/root/miniconda3/bin:$PATH
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
export TORCH_EXTENSIONS_DIR="$EXT"
export HF_HOME=/hy-tmp/hf_home
export XDG_CACHE_HOME=/hy-tmp/xdg_cache

declare -A CHECKPOINTS=(
  [step850]="$RUN/checkpoint_step_850.pth"
  [step900]="$RUN/checkpoint_step_900.pth"
  [step950]="$RUN/checkpoint_step_950.pth"
  [step1000]="$RUN/checkpoint_step_1000.pth"
)

run_bench() {
  local name=$1
  local model=$2
  local bench=$3
  local dataset rollout out

  if [[ "$bench" == gsm8k ]]; then
    dataset=/hy-tmp/data/gsm8k/gsm8k_test_albatross.jsonl
    rollout=4
  else
    dataset=/hy-tmp/data/math500/test.jsonl
    rollout=8
  fi

  out="$BASE/$name/$bench"
  mkdir -p "$out"
  if [[ -s "$out/summary.json" ]]; then
    echo -e "$(date '+%F %T')\tSKIP_DONE\t$name\t$bench" | tee -a "$STATUS"
    return
  fi

  echo -e "$(date '+%F %T')\tSTART\t$name\t$bench\t$model" | tee -a "$STATUS"
  "$PY" "$EVAL" \
    --albatross-dir "$ALBATROSS" \
    --model "$model" \
    --dataset "$dataset" \
    --out-dir "$out" \
    --gpus 0 --rollout "$rollout" --bsz 512 \
    --max-new-tokens 4096 --ctx-limit 8192 \
    --temperature 0.8 --top-p 0.35 --top-k 40 \
    --seed 42 --prompt-style official_flower \
    --torch-extensions-dir "$EXT" \
    --progress-every 100 --verify-workers 8 \
    2>&1 | tee "$out/eval.log"

  "$PY" - "$out/summary.json" "$name" "$bench" <<'PY' | tee -a "$STATUS"
import json
import sys
from datetime import datetime

path, name, benchmark = sys.argv[1:]
with open(path, "r", encoding="utf-8") as handle:
    summary = json.load(handle)
print(
    f"{datetime.now():%F %T}\tDONE\t{name}\t{benchmark}\t"
    f"avg={summary.get('rollout_accuracy')}\t"
    f"pass={summary.get('pass_at_rollout_accuracy')}\t"
    f"mean_tokens={summary.get('mean_generated_tokens')}"
)
PY
}

# Run the newest checkpoint first so its result is available earliest.
for name in step1000 step850 step900 step950; do
  model=${CHECKPOINTS[$name]}
  [[ -f "$model" ]] || { echo -e "$(date '+%F %T')\tMISSING\t$name\t$model" | tee -a "$STATUS"; continue; }
  run_bench "$name" "$model" gsm8k
  run_bench "$name" "$model" math500
done

echo -e "$(date '+%F %T')\tALL_DONE" | tee -a "$STATUS"
