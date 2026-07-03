#!/usr/bin/env bash
set -euo pipefail
cd /root/RWKV-LM/RWKV7-statetuning_hardbuffer_ratio_v1

base_log=/root/RWKV-LM/RWKV7-statetuning_hardbuffer_ratio_v1/log/hardratio_sweep_$(date +%Y%m%d_%H%M%S)
mkdir -p "$base_log"

ratios=(0.25 0.50 0.75 1.00)
for ratio in "${ratios[@]}"; do
  tag=$(printf 'ratio_%s' "$ratio" | tr '.' 'p')
  out="$base_log/$tag"
  mkdir -p "$out"
  echo "[$(date '+%F %T')] start $ratio -> $out"
  /root/miniconda3/bin/python3 main.py \
    --train_jsonl /root/RWKV-LM/RWKV7-statetuning/gsm8k_train_formatted.jsonl \
    --eval_jsonl /root/RWKV-LM/RWKV7-statetuning/gsm8k_test_formatted.jsonl \
    --model /root/RWKV-LM/rwkv7-g1b-1.5b-20251202-ctx8192.pth \
    --tokenizer /root/RWKV-LM/rwkv_vocab_v20230424.txt \
    --out_dir "$out" \
    --total_steps 100 \
    --eval_interval 50 \
    --save_interval 50 \
    --num_questions 24 \
    --samples_per_question 8 \
    --max_new_tokens 1024 \
    --lr 6e-5 \
    --ppo_epochs 1 \
    --kl_coef 0.05 \
    --kl_mode k3_loss \
    --neg_adv_weight 0.6 \
    --time_state_clamp 10 \
    --length_weight 0.0 \
    --zstd_penalty_weight 0.0 \
    --ngram_penalty 0.0 \
    --hard_buffer_ttl 4 \
    --hard_buffer_cooldown 4 \
    --hard_buffer_target_samples 192 \
    --hard_buffer_group_size 8 \
    --hard_buffer_hard_ratio "$ratio" \
    --hard_buffer_extra_lr_scale 0.5 \
    --hard_buffer_adv_clip 2.5 \
    > "$out/train_stdout.log" 2>&1
  echo "[$(date '+%F %T')] done $ratio"
done

echo "[$(date '+%F %T')] sweep done: $base_log"
