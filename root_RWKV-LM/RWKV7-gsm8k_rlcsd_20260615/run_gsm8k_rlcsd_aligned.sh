#!/usr/bin/env bash
set -euo pipefail

STEPS=${1:-100}
STAMP=$(date +%Y%m%d_%H%M%S)
OUT=${OUT:-/root/autodl-tmp/logs/gsm8k_rlcsd_aligned_${STAMP}/train}
mkdir -p "$OUT"

cd /root/RWKV-LM/RWKV7-gsm8k_rlcsd_20260615
/root/miniconda3/bin/python main.py \
  --train_jsonl /root/autodl-tmp/data/gsm8k/gsm8k_train_formatted.jsonl \
  --eval_jsonl /root/autodl-tmp/data/gsm8k/gsm8k_test_formatted.jsonl \
  --model /root/autodl-tmp/rwkv_models/ms_g1_1p5b/rwkv7-g1f-1.5b-20260419-ctx8192.pth \
  --tokenizer /root/RWKV-LM/RWKV-v5/tokenizer/rwkv_vocab_v20230424.txt \
  --out_dir "$OUT" \
  --method rlcsd \
  --tune_mode full \
  --model_dtype bf16 \
  --teacher_dtype bf16 \
  --ctx_len 8192 \
  --num_questions 2 \
  --samples_per_question 8 \
  --max_new_tokens 1024 \
  --temperature 1.0 \
  --top_p 0.95 \
  --top_k 20 \
  --eval_temperature 1.0 \
  --eval_top_p 0.28 \
  --eval_top_k 32 \
  --eval_max_new_tokens 1024 \
  --lr 1e-6 \
  --warmup_steps 50 \
  --weight_decay 0.01 \
  --micro_batch 1 \
  --rollout_forward_batch 4 \
  --ppo_epochs 1 \
  --grad_clip 1.0 \
  --epsilon 0.2 \
  --kl_coef 0 \
  --kl_mode k3_loss \
  --neg_adv_weight 1.0 \
  --hard_buffer_target_samples 0 \
  --reward_mode trl_doc \
  --prompt_mode trl_doc \
  --min_tokens 50 \
  --length_weight 0 \
  --zstd_penalty_weight 0 \
  --ngram_penalty 0 \
  --rlcsd_tau 0.02 \
  --rlcsd_beta 1.0 \
  --rlcsd_lam 0.5 \
  --rlcsd_delta 0.02 \
  --rlcsd_eta 1.0 \
  --rlcsd_residual_clip_low -2.0 \
  --rlcsd_residual_clip_high 2.0 \
  --rlcsd_k_max 4 \
  --rlcsd_teacher_batch 1 \
  --teacher_mode snapshot \
  --teacher_sync_interval 10 \
  --total_steps "$STEPS" \
  --skip_preeval 1 \
  --skip_posteval 1 \
  --eval_interval 0 \
  --save_interval 0 \
  --save_responses 0 2>&1 | tee "$OUT/train.log"
