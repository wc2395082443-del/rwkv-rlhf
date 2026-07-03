#!/bin/bash
set -euo pipefail
ROOT=/root/RWKV-LM/RWKV7-deepmath_trl_doc_g1f1p5b_fp32kernel
RUN=/root/autodl-tmp/RWKV7-deepmath_trl_doc_g1f1p5b_fp32kernel_20260522_211632
TOKENIZER=/root/RWKV-LM/rwkv_vocab_v20230424.txt
MODEL=/dev/shm/rwkv_models/rwkv7-g1f-1.5b-20260419-ctx8192.pth
GSM=/root/autodl-tmp/data/gsm8k/gsm8k_test_formatted.jsonl
MATH=/root/autodl-tmp/data/math500/test.jsonl
for DS in gsm8k math500; do
  if [ "$DS" = "gsm8k" ]; then
    EVAL_JSONL="$GSM"
  else
    EVAL_JSONL="$MATH"
  fi
  ODIR="$RUN/full_eval_pretrain_20260523_1510/${DS}_base"
  mkdir -p "$ODIR"
  echo "[$(date '+%F %T')] start $DS base" | tee -a "$RUN/full_eval_pretrain_20260523_1510/driver.log"
  /root/miniconda3/bin/python "$ROOT/main.py" \
    --train_jsonl "$EVAL_JSONL" \
    --eval_jsonl "$EVAL_JSONL" \
    --model "$MODEL" \
    --tokenizer "$TOKENIZER" \
    --out_dir "$ODIR" \
    --tune_mode full \
    --reward_mode trl_doc \
    --prompt_mode trl_doc \
    --save_responses 0 \
    --num_questions 8 \
    --samples_per_question 8 \
    --total_steps 0 \
    --max_new_tokens 1024 \
    --temperature 1.0 \
    --top_p 1.0 \
    --top_k 0 \
    --eval_temperature 0.3 \
    --eval_top_p 0.4 \
    --eval_top_k 500 \
    --min_tokens 1 \
    --length_weight 0.0 \
    --zstd_penalty_weight 0.0 \
    --ngram_penalty 0.0 \
    --neg_adv_weight 1.0 \
    --kl_coef 0.0 \
    --lr 1e-6 \
    --micro_batch 1 \
    --rollout_forward_batch 8 \
    --hard_buffer_target_samples 0 \
    --eval_interval 999999 \
    --skip_preeval 0 \
    --skip_posteval 1 \
    --save_interval 999999 \
    --save_last 0 \
    --final_full_eval 0 \
    --preeval_sample_ratio 1.0 \
    --seed 42 > "$ODIR/run.log" 2>&1
  echo "[$(date '+%F %T')] done $DS base" | tee -a "$RUN/full_eval_pretrain_20260523_1510/driver.log"
done
