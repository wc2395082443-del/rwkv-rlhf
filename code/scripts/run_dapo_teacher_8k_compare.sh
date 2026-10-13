#!/usr/bin/env bash
set -euo pipefail
BASE=/hy-tmp/evals/dapo_teacher_8k_pilot256_20261004
DATA=/hy-tmp/data/openr1_dapo_math17k_processed/strict_split_seed42/train.jsonl
PY=/usr/local/bin/python3
mkdir -p "$BASE"
echo "START $(date '+%F %T') qwen" | tee -a "$BASE/status.tsv"
$PY /hy-tmp/dapo_teacher_8k_compare.py --model /hy-tmp/models/Qwen2.5-Math-7B-Instruct --name qwen_math_7b --data "$DATA" --out "$BASE" --limit 256 --batch 8 2>&1 | tee "$BASE/qwen.log"
echo "DONE $(date '+%F %T') qwen" | tee -a "$BASE/status.tsv"
while pgrep -f download_math_teachers_mirror.sh >/dev/null; do sleep 30; done
echo "START $(date '+%F %T') deepseek" | tee -a "$BASE/status.tsv"
$PY /hy-tmp/dapo_teacher_8k_compare.py --model /hy-tmp/models/DeepSeek-R1-Distill-Qwen-7B --name deepseek_r1_distill_qwen_7b --data "$DATA" --out "$BASE" --limit 256 --batch 8 2>&1 | tee "$BASE/deepseek.log"
echo "DONE $(date '+%F %T') deepseek" | tee -a "$BASE/status.tsv"