#!/usr/bin/env bash
set -euo pipefail
cd /root/RWKV-LM/RWKV7-statetuning_real_v1

base=/root/RWKV-LM/RWKV7-statetuning_real_v1/log/real_correct_sweep_$(date +%Y%m%d_%H%M%S)
mkdir -p "$base"
echo "$base" > /root/RWKV-LM/RWKV7-statetuning_real_v1/log/latest_real_correct_sweep.txt

run_one() {
  local tau="$1"
  local kl="$2"
  local tag="tau_${tau}_kl_${kl}"
  tag=$(printf '%s' "$tag" | tr '.' 'p')
  local out="$base/$tag"
  mkdir -p "$out"
  echo "[$(date '+%F %T')] start tau=$tau kl=$kl -> $out"
  /root/miniconda3/bin/python3 main.py \
    --train_jsonl /root/RWKV-LM/RWKV7-statetuning/gsm8k_train_formatted.jsonl \
    --eval_jsonl /root/RWKV-LM/RWKV7-statetuning/gsm8k_test_formatted.jsonl \
    --model /root/RWKV-LM/rwkv7-g1b-1.5b-20251202-ctx8192.pth \
    --tokenizer /root/RWKV-LM/rwkv_vocab_v20230424.txt \
    --out_dir "$out" \
    --total_steps 100 \
    --eval_interval 5 \
    --save_interval 50 \
    --num_questions 24 \
    --samples_per_question 8 \
    --max_new_tokens 1024 \
    --lr 6e-5 \
    --ppo_epochs 1 \
    --kl_coef "$kl" \
    --kl_mode k3_loss \
    --neg_adv_weight 1.0 \
    --time_state_clamp 10 \
    --length_weight 0.0 \
    --zstd_penalty_weight 0.0 \
    --ngram_penalty 0.0 \
    --hard_buffer_ttl 4 \
    --hard_buffer_cooldown 4 \
    --hard_buffer_target_samples 0 \
    --hard_buffer_group_size 8 \
    --hard_buffer_extra_lr_scale 0.5 \
    --hard_buffer_adv_clip 2.5 \
    --policy_objective real \
    --real_tau "$tau" \
    --real_label_source correct \
    --real_reward_weight_cap 2.0 \
    > "$out/train_stdout.log" 2>&1
  echo "[$(date '+%F %T')] done tau=$tau kl=$kl"
}

for tau in 0.5 0.8 1.0; do
  for kl in 0.01 0.03 0.05; do
    run_one "$tau" "$kl"
  done
done

echo "[$(date '+%F %T')] all done: $base"
