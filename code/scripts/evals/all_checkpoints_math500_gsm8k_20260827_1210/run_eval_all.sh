#!/usr/bin/env bash
set -euo pipefail
OUT=/hy-tmp/evals/all_checkpoints_math500_gsm8k_20260827_1210
SKILLS=/hy-tmp/rwkv-skills-official
UV=/root/.local/bin/uv
VLLM_PATH=/hy-tmp/rwkv-skills-official/vendor/vllm-rwkv
PORT=19082
BASE=http://127.0.0.1:${PORT}/v1
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export JUDGE_MODEL=${JUDGE_MODEL:-deepseek-chat}
export JUDGE_BASE_URL=${JUDGE_BASE_URL:-https://api.deepseek.com}
export JUDGE_MAX_WORKERS=${JUDGE_MAX_WORKERS:-16}
export JUDGE_MAX_TOKENS=${JUDGE_MAX_TOKENS:-256}
cd "$SKILLS"
: > "$OUT/status.tsv"
while read -r CKPT; do
  [ -f "$CKPT" ] || { echo -e "$(date '+%F %T')\tMISSING\t$CKPT" | tee -a "$OUT/status.tsv"; continue; }
  SLUG=$(basename "$(dirname "$CKPT")")__$(basename "$CKPT" .pth)
  MODEL="${SLUG//[^A-Za-z0-9_.-]/_}"
  echo -e "$(date '+%F %T')\tSTART_MODEL\t$MODEL\t$CKPT" | tee -a "$OUT/status.tsv"
  pkill -f "rwkv-skills-infer.*${PORT}" || true
  sleep 5
  export VLLM_PYTHON=/hy-tmp/vllm-rwkv-env/bin/python
  $UV run rwkv-skills-infer \
    --model-path "$CKPT" \
    --model-name "$MODEL" \
    --vllm-rwkv-path "$VLLM_PATH" \
    --port "$PORT" \
    --host 127.0.0.1 \
    --tokenizer-mode rwkv \
    --disable-auto-tool-choice \
    --max-model-len 12288 \
    --max-num-seqs 512 \
    --max-num-batched-tokens 65536 \
    --gpu-memory-utilization 0.92 \
    > "$OUT/logs/${MODEL}.infer.log" 2>&1 &
  IPID=$!
  echo $IPID > "$OUT/pids/${MODEL}.infer.pid"
  ok=0
  for i in $(seq 1 240); do
    if curl -fsS "$BASE/models" >/dev/null 2>&1; then ok=1; break; fi
    sleep 5
  done
  if [ "$ok" != 1 ]; then
    echo -e "$(date '+%F %T')\tINFER_START_FAILED\t$MODEL" | tee -a "$OUT/status.tsv"
    kill $IPID 2>/dev/null || true
    continue
  fi
  for BENCH in math500 gsm8k; do
    DATASET=$BENCH
    AVGK=8
    [ "$BENCH" = gsm8k ] && AVGK=4
    echo -e "$(date '+%F %T')\tSTART_EVAL\t$MODEL\t$BENCH" | tee -a "$OUT/status.tsv"
    set +e
    $UV run python -m src.eval.tasks.maths.runner \
      --dataset "$DATASET" \
      --infer-base-url "$BASE" \
      --infer-model "$MODEL" \
      --infer-protocol completions \
      --infer-seed-policy omit \
      --infer-timeout-s 1200 \
      --infer-max-workers 256 \
      --batch-size 512 \
      --avg-k "$AVGK" \
      --cot-mode cot \
      --strategy-a-single-generation \
      --max-tokens 8192 \
      --judge-mode llm \
      --judge-max-workers "${JUDGE_MAX_WORKERS}" \
      --primary-only \
      > "$OUT/logs/${MODEL}.${BENCH}.eval.log" 2>&1
    RC=$?
    set -e
    echo -e "$(date '+%F %T')\tDONE_EVAL\t$MODEL\t$BENCH\trc=$RC" | tee -a "$OUT/status.tsv"
  done
  kill $IPID 2>/dev/null || true
  wait $IPID 2>/dev/null || true
  sleep 8
done < "$OUT/checkpoints.txt"
echo -e "$(date '+%F %T')\tALL_DONE" | tee -a "$OUT/status.tsv"
