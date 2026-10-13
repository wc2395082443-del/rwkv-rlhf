#!/usr/bin/env bash
set -euo pipefail
CODE=/hy-tmp/RWKV7-g1i-strict-quality-gate-adamw-nooffload-20260810
OUT=$(cd "$(dirname "$0")" && pwd)
cd "$CODE"
export CUDA_VISIBLE_DEVICES=0
export TORCH_EXTENSIONS_DIR=/hy-tmp/torch_extensions_g1i_strict_quality_adamw_nooffload
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
( while true; do
  nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw --format=csv,noheader,nounits | awk -v ts=$(date +%s.%N) -F, '{gsub(/ /,"",$2);gsub(/ /,"",$3);gsub(/MiB/,"",$4);gsub(/MiB/,"",$5);gsub(/W/,"",$6); print ts","$2","$3","$4","$5","$6}' >> "$OUT/gpu_monitor.csv"
  sleep 1
done ) &
MON_PID=$!
trap 'kill $MON_PID 2>/dev/null || true' EXIT
python3 eval_gsm8k_boxed_strict_reward.py \
  --model /hy-tmp/runs/g1i_gsm8k_boxed_strictcot_continue_from_step85_maxnew2048_micro3_dyn512_q32_r16_100_20260812_082824/final_step_105.pth \
  --tokenizer /hy-tmp/RWKV-v7/rwkv_vocab_v20230424.txt \
  --eval_jsonl /hy-tmp/data/gsm8k/gsm8k_test_albatross.jsonl \
  --out_dir "$OUT/step105_full" \
  --max_new_tokens 2048 \
  --temperature 0.3 \
  --top_p 0.4 \
  --top_k 500 \
  --batch 128 \
  --seed 42 \
  > "$OUT/stdout.log" 2>&1
