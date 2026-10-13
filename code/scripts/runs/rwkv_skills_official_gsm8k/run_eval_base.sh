#!/usr/bin/env bash
set -euo pipefail
cd /hy-tmp/rwkv-skills-official
RUN=/hy-tmp/runs/rwkv_skills_official_gsm8k
: > "$RUN/gpu_eval_base_official.csv"
( while true; do
  nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total,power.draw --format=csv,noheader,nounits >> "$RUN/gpu_eval_base_official.csv"
  sleep 2
done ) > "$RUN/gpu_monitor_base.log" 2>&1 &
echo MON:$! > "$RUN/eval_base_pids.txt"
PYTHONUNBUFFERED=1 .venv/bin/python -m src.eval.tasks.maths.runner \
  --dataset data/gsm8k/test.jsonl \
  --batch-size 128 \
  --infer-base-url http://127.0.0.1:19082 \
  --infer-model g1i-base \
  --infer-protocol vllm \
  --infer-seed-policy omit \
  --judge-mode exact \
  > "$RUN/eval_base_official.log" 2>&1 &
echo EVAL:$! >> "$RUN/eval_base_pids.txt"
cat "$RUN/eval_base_pids.txt"
