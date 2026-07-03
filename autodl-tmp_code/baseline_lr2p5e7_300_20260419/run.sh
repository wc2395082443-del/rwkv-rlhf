#!/usr/bin/env bash
set -euo pipefail
cd /root/RWKV-LM/RWKV-v7/train_temp
out="/root/autodl-tmp/baseline_lr2p5e7_300_20260419/run"
mkdir -p "$out"
/root/miniconda3/bin/python train_rl_baseline.py \
  --load_model /root/autodl-tmp/rwkv_models/rwkv7-g1e-1.5b-20260309-ctx8192.pth \
  --proj_dir "$out" \
  --tokenizer /root/RWKV-LM/RWKV-v7/rwkv_vocab_v20230424.txt \
  --train_jsonl /root/RWKV-LM/RWKV7-statetuning/gsm8k_train_formatted.jsonl \
  --eval_jsonl /root/RWKV-LM/RWKV7-statetuning/gsm8k_test_formatted_1of8.jsonl \
  --devices 1 --accelerator gpu --strategy deepspeed_stage_3_offload --precision bf16 \
  --total_steps 300 --num_questions 24 --samples_per_question 8 --max_new_tokens 768 \
  --micro_batch 18 --rollout_forward_batch 192 --use_stateful_rollout 1 \
  --lr 2.5e-7 --neg_adv_weight 0.6 --kl_coef 0.0 \
  --length_weight 0.0 --zstd_penalty_weight 0.0 --ngram_penalty 0.0 \
  --skip_preeval 1 --skip_posteval 1 --eval_interval 0 --save_interval 50 \
  --log_interval 1 --enable_progress_bar 0 \
  --hard_buffer_ttl 10 --hard_buffer_cooldown 5 --hard_buffer_target_samples 192 --hard_buffer_group_size 8 --hard_buffer_extra_lr_scale 0.5 --hard_buffer_adv_clip 2.5 \
  > "$out/stdout.log" 2>&1
