#!/usr/bin/env bash
set -euo pipefail
PY=/root/miniconda3/bin/python
SCRIPT=/root/RWKV-LM/RWKV-v7/train_temp/train_rl_baseline.py
MODEL=/root/autodl-tmp/rwkv_models/rwkv7-g1e-1.5b-20260309-ctx8192.pth
TRAIN=/root/RWKV-LM/RWKV7-statetuning/gsm8k_train_formatted.jsonl
SMALL=/root/RWKV-LM/RWKV7-statetuning/gsm8k_test_formatted_1of8.jsonl
FULL=/root/RWKV-LM/RWKV7-statetuning/gsm8k_test_formatted.jsonl
run_one () {
  local out="$1"
  local disable_extra="$2"
  mkdir -p "$out"
  cd /root/RWKV-LM/RWKV-v7/train_temp
  CUDA_VISIBLE_DEVICES=0 "$PY" "$SCRIPT" \
    --load_model "$MODEL" \
    --wandb 0 \
    --proj_dir "$out" \
    --train_jsonl "$TRAIN" \
    --eval_jsonl "$SMALL" \
    --full_eval_jsonl "$FULL" \
    --num_questions 24 \
    --samples_per_question 8 \
    --rollout_forward_batch 192 \
    --micro_batch 18 \
    --max_new_tokens 768 \
    --use_stateful_rollout 1 \
    --lr 5e-7 \
    --neg_adv_weight 0.6 \
    --kl_coef 0.05 \
    --length_weight 0.0 \
    --zstd_penalty_weight 0.0 \
    --ngram_penalty 0.0 \
    --hard_buffer_ttl 10 \
    --hard_buffer_cooldown 5 \
    --hard_buffer_target_samples 192 \
    --hard_buffer_group_size 8 \
    --hard_buffer_extra_lr_scale 0.5 \
    --hard_buffer_adv_clip 2.5 \
    --eval_interval 10 \
    --save_interval 50 \
    --total_steps 500 \
    --save_final_checkpoint 1 \
    --disable_extra_step "$disable_extra"
}
run_one "/root/autodl-tmp/gsm8k_dual_eval_500_20260423_031000/with_extra/run" 0 |& tee "/root/autodl-tmp/gsm8k_dual_eval_500_20260423_031000/with_extra/stdout.log"
run_one "/root/autodl-tmp/gsm8k_dual_eval_500_20260423_031000/no_extra/run" 1 |& tee "/root/autodl-tmp/gsm8k_dual_eval_500_20260423_031000/no_extra/stdout.log"