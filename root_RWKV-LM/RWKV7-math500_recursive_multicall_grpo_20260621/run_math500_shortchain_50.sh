#!/bin/bash
set -euo pipefail
ROOT=/root/RWKV-LM/RWKV7-math500_shortchain_grpo_20260620
MODEL=${MODEL:-/dev/shm/rwkv_models/rwkv7-g1f-1.5b-20260419-ctx8192.pth}
TOKENIZER=${TOKENIZER:-/root/RWKV-LM/rwkv_vocab_v20230424.txt}
TRAIN=${TRAIN:-/root/autodl-tmp/data/math500/train.jsonl}
EVAL=${EVAL:-/root/autodl-tmp/data/math500/test.jsonl}
STAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR=${OUT_DIR:-/root/autodl-tmp/math500_shortchain_grpo_50_${STAMP}}
mkdir -p "$OUT_DIR"
cd "$ROOT"
/root/miniconda3/bin/python main.py \
  --train_jsonl "$TRAIN" \
  --eval_jsonl "$EVAL" \
  --model "$MODEL" \
  --tokenizer "$TOKENIZER" \
  --out_dir "$OUT_DIR" \
  --tune_mode full \
  --model_dtype bf16 \
  --reward_mode trl_doc \
  --prompt_mode short_math \
  --num_questions 8 \
  --samples_per_question 8 \
  --total_steps 50 \
  --max_new_tokens 512 \
  --eval_max_new_tokens 512 \
  --temperature 0.8 \
  --top_p 1.0 \
  --top_k 0 \
  --eval_temperature 0.5 \
  --eval_top_p 0.28 \
  --eval_top_k 32 \
  --min_tokens 1 \
  --length_weight 0.02 \
  --zstd_penalty_weight 0.0 \
  --ngram_penalty 0.05 \
  --neg_adv_weight 1.0 \
  --kl_coef 0.0 \
  --lr 1e-6 \
  --micro_batch 1 \
  --rollout_forward_batch 8 \
  --hard_buffer_target_samples 0 \
  --eval_interval 25 \
  --eval_sample_ratio 0.2 \
  --skip_preeval 1 \
  --skip_posteval 1 \
  --save_interval 50 \
  --save_last 1 \
  --final_full_eval 1 \
  --save_responses 0 \
  --seed 42 2>&1 | tee "$OUT_DIR/train_stdout.log"
