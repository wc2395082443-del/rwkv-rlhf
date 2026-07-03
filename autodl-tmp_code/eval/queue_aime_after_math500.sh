#!/usr/bin/env bash
set -euo pipefail
while pgrep -f "rwkv_llama_official_eval.py.*math500_3b.jsonl" >/dev/null; do
  sleep 60
done
cd /root/autodl-tmp/eval
nohup /root/miniconda3/bin/python3 areal_math_eval.py \
  --benchmark areal_aime24 \
  --model /root/autodl-tmp/rwkv_models/rwkv7-g1e-2.9b-20260312-ctx8192.pth \
  --data /root/M1/rl/verl/deepscaler/data/test/aime.json \
  --out /root/autodl-tmp/eval/aime24_3b.jsonl \
  > /root/autodl-tmp/eval/aime24_3b_run.log 2>&1
