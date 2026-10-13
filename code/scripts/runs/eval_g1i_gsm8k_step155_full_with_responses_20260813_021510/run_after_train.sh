#!/usr/bin/env bash
set -euo pipefail
TRAIN_PID=652238
CODE=/hy-tmp/RWKV7-g1i-strict-quality-gate-adamw-nooffload-20260810
OUT=$(cd "$(dirname "$0")" && pwd)
echo "[wait] waiting for train pid ${TRAIN_PID}" | tee -a "$OUT/launcher.log"
while kill -0 "$TRAIN_PID" 2>/dev/null; do
  date '+[wait] %F %T train still running' >> "$OUT/launcher.log"
  sleep 60
done
echo "[eval] starting step155 full eval $(date)" | tee -a "$OUT/launcher.log"
cd "$CODE"
export CUDA_VISIBLE_DEVICES=0
export TORCH_EXTENSIONS_DIR=/hy-tmp/torch_extensions_g1i_strict_quality_adamw_nooffload
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python3 eval_gsm8k_boxed_strict_reward.py \
  --model /hy-tmp/runs/g1i_gsm8k_boxed_strictcot_continue_from_step105_fixed50_maxnew2048_micro3_dyn512_q32_r16_20260812_142929/final_step_155.pth \
  --tokenizer /hy-tmp/RWKV-v7/rwkv_vocab_v20230424.txt \
  --eval_jsonl /hy-tmp/data/gsm8k/gsm8k_test_albatross.jsonl \
  --out_dir "$OUT/step155_full" \
  --max_new_tokens 2048 \
  --temperature 0.3 \
  --top_p 0.4 \
  --top_k 500 \
  --batch 32 \
  --seed 42 \
  > "$OUT/stdout.log" 2>&1
echo "[eval] done $(date)" | tee -a "$OUT/launcher.log"
