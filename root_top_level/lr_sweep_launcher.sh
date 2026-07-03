#!/usr/bin/env bash
set -euo pipefail
cd /root/RWKV-LM/llama_grpo_baseline_v1
ROOT_DIR="$1"
LRS=(6e-5 1.2e-4 2.4e-4 6e-4 1.2e-3)
for LR in "${LRS[@]}"; do
  SAFE_LR=${LR//./p}
  OUT="$ROOT_DIR/lr_${SAFE_LR}"
  mkdir -p "$OUT"
  echo "[$(date)] start lr=$LR out=$OUT" | tee -a "$ROOT_DIR/sweep.log"
  /root/miniconda3/bin/python3 main.py     --train_jsonl /root/autodl-tmp/data/gsm8k/gsm8k_train_formatted.jsonl     --eval_jsonl /root/autodl-tmp/data/gsm8k/gsm8k_test_formatted.jsonl     --model /root/autodl-tmp/models/Llama-3.2-3B-Instruct     --out_dir "$OUT"     --total_steps 100     --skip_pre_eval     --eval_interval 10     --eval_sample_ratio 0.1     --save_interval 50     --num_questions 12     --samples_per_question 8     --hard_buffer_target_samples 0     --hard_buffer_group_size 8     --max_new_tokens 768     --gen_batch_size 16     --eval_gen_batch_size 64     --pre_eval_gen_batch_size 64     --lr "$LR"     > "$OUT/train_stdout.log" 2>&1
  echo "[$(date)] done lr=$LR out=$OUT" | tee -a "$ROOT_DIR/sweep.log"
done
