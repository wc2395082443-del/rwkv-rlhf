#!/usr/bin/env bash
set -euo pipefail
BASE=/hy-tmp/evals/g1k_step1000_ctx32k_aime24x8_math500x4_20261003
EVAL=/hy-tmp/evals/official_flower_ckpt_eval_20260827/eval_math500_flower.py
ALBATROSS=/root/eval_transfer/albatross/faster3a_2605
EXT=/hy-tmp/torch_extensions_g1k_ctx32k
PY=/usr/local/bin/python3
MODEL=/hy-tmp/runs/g1k_dapo_pass1_31_officialprompt_q32x16_cand64_max6144_ob512_step900to1000_20261001_014411/final_step_1000.pth
export PATH=/root/miniconda3/bin:$PATH
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
export TORCH_EXTENSIONS_DIR=$EXT
export HF_HOME=/hy-tmp/hf_home
export XDG_CACHE_HOME=/hy-tmp/xdg_cache
mkdir -p "$BASE" "$EXT"
run_eval() {
  local name=$1 dataset=$2 rollout=$3
  local out="$BASE/$name"
  mkdir -p "$out"
  if [ -s "$out/summary.json" ]; then echo "SKIP_DONE $name" | tee -a "$BASE/status.tsv"; return; fi
  echo "START $(date '+%F %T') $name rollout=$rollout ctx=32768 max_new=24576" | tee -a "$BASE/status.tsv"
  "$PY" "$EVAL" --albatross-dir "$ALBATROSS" --model "$MODEL" --dataset "$dataset" --out-dir "$out" --gpus 0 --rollout "$rollout" --bsz 512 --max-new-tokens 24576 --ctx-limit 32768 --temperature 0.8 --top-p 0.35 --top-k 40 --seed 42 --prompt-style official_flower --torch-extensions-dir "$EXT" --progress-every 100 --verify-workers 8 2>&1 | tee "$out/eval.log"
  "$PY" - "$out/summary.json" "$name" <<'PY' | tee -a "$BASE/status.tsv"
import json,sys
from datetime import datetime
s=json.load(open(sys.argv[1],encoding='utf-8'))
print(f"DONE {datetime.now():%F %T} {sys.argv[2]} avg={s.get('rollout_accuracy')} pass={s.get('pass_at_rollout_accuracy')} n={s.get('num_tasks')} rollout={s.get('rollout')} elapsed={s.get('elapsed_sec')}")
PY
}
run_eval aime24 /hy-tmp/data/official_math_extra/aime24.jsonl 8
run_eval math500 /hy-tmp/data/math500/test.jsonl 4
echo "ALL_DONE $(date '+%F %T')" | tee -a "$BASE/status.tsv"