#!/usr/bin/env bash
set -euo pipefail
BASE=/hy-tmp/evals/g1i_baseline_unified_20260906
EVAL=/hy-tmp/evals/official_flower_ckpt_eval_20260827/eval_math500_flower.py
MODEL=/hy-tmp/models/rwkv7-g1i_preview5445-1.5b-20260729-ctx16384.pth
ALBATROSS=/root/eval_transfer/albatross/faster3a_2605
PY=/usr/local/bin/python3
EXT=/hy-tmp/torch_extensions_g1i_baseline_20260906
mkdir -p "$BASE" "$EXT"
export PATH=/root/miniconda3/bin:$PATH PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 TORCH_EXTENSIONS_DIR="$EXT" HF_HOME=/hy-tmp/hf_home XDG_CACHE_HOME=/hy-tmp/xdg_cache
declare -A D=([gsm8k]=/hy-tmp/data/gsm8k/gsm8k_test_albatross.jsonl [math500]=/hy-tmp/data/math500/test.jsonl [aime24]=/hy-tmp/data/official_math_extra/aime24.jsonl [aime25]=/hy-tmp/data/official_math_extra/aime25.jsonl [amc23]=/hy-tmp/data/official_math_extra/amc23.jsonl)
declare -A R=([gsm8k]=4 [math500]=8 [aime24]=64 [aime25]=64 [amc23]=64)
declare -A N=([gsm8k]=4096 [math500]=4096 [aime24]=8192 [aime25]=8192 [amc23]=8192)
declare -A C=([gsm8k]=8192 [math500]=8192 [aime24]=12288 [aime25]=12288 [amc23]=12288)
for bench in gsm8k math500 aime24 aime25 amc23; do
  out="$BASE/$bench"; mkdir -p "$out"
  if [[ -s "$out/summary.json" ]]; then continue; fi
  echo "START $(date '+%F %T') baseline $bench" | tee -a "$BASE/status.tsv"
  if "$PY" "$EVAL" --albatross-dir "$ALBATROSS" --model "$MODEL" --dataset "${D[$bench]}" --out-dir "$out" --gpus 0 --rollout "${R[$bench]}" --bsz 512 --max-new-tokens "${N[$bench]}" --ctx-limit "${C[$bench]}" --temperature 0.8 --top-p 0.35 --top-k 40 --seed 42 --prompt-style official_flower --torch-extensions-dir "$EXT" --progress-every 100 --verify-workers 8 2>&1 | tee "$out/eval.log"; then
    "$PY" - "$out/summary.json" "$bench" <<'PY' | tee -a "$BASE/status.tsv"
import json,sys
from datetime import datetime
p,b=sys.argv[1:]
s=json.load(open(p,encoding='utf-8'))
print(f"DONE {datetime.now():%F %T} baseline {b} avg={s.get('rollout_accuracy')} pass={s.get('pass_at_rollout_accuracy')} n={s.get('num_tasks')} tokens={s.get('mean_generated_tokens')}")
PY
  else
    echo "FAIL $(date '+%F %T') baseline $bench" | tee -a "$BASE/status.tsv"
  fi
done
echo "ALL_BASELINE_DONE $(date '+%F %T')" | tee -a "$BASE/status.tsv"
