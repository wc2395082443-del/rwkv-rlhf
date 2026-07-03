#!/usr/bin/env bash
set -euo pipefail
cd /root/RWKV-LM/RWKV-v7/train_temp
ROOT_DIR="/root/autodl-tmp/overnight_lr_sweep_20260417_3x10_20260417_001810"
TRAIN_JSON=/root/RWKV-LM/RWKV7-statetuning/gsm8k_train_formatted.jsonl
EVAL_JSON=/root/RWKV-LM/RWKV7-statetuning/gsm8k_test_formatted_1of8.jsonl
MODEL=/root/autodl-tmp/rwkv_models/rwkv7-g1e-1.5b-20260309-ctx8192.pth
TOKENIZER=/root/RWKV-LM/RWKV-v7/rwkv_vocab_v20230424.txt
LRS=(1e-7 2e-7 3e-7 5e-7 8e-7 1e-6 2e-6 3e-6 5e-6 1e-5)

echo "root=$ROOT_DIR"
echo "start=$(date '+%F %T')"

run_one() {
  local algo="$1"
  local lr="$2"
  local script="$3"
  local negw="$4"
  local extra_args="$5"
  local tag=${lr//./p}
  local out="$ROOT_DIR/${algo}/lr_${tag}"
  mkdir -p "$out"
  echo "[$(date '+%F %T')] START algo=$algo lr=$lr out=$out"
  /root/miniconda3/bin/python "$script" \
    --load_model "$MODEL" \
    --proj_dir "$out" \
    --tokenizer "$TOKENIZER" \
    --train_jsonl "$TRAIN_JSON" \
    --eval_jsonl "$EVAL_JSON" \
    --devices 1 --accelerator gpu --strategy deepspeed_stage_3_offload --precision bf16 \
    --total_steps 50 --num_questions 24 --samples_per_question 8 --max_new_tokens 768 \
    --micro_batch 18 --rollout_forward_batch 192 --use_stateful_rollout 1 \
    --lr "$lr" --neg_adv_weight "$negw" --kl_coef 0.0 \
    --length_weight 0.0 --zstd_penalty_weight 0.0 --ngram_penalty 0.0 \
    --skip_preeval 1 --skip_posteval 1 --eval_interval 0 --save_interval 50 \
    --log_interval 1 --enable_progress_bar 0 \
    $extra_args \
    > "$out/stdout.log" 2>&1
  local rc=$?
  echo "[$(date '+%F %T')] END algo=$algo lr=$lr rc=$rc out=$out"
}

for lr in "${LRS[@]}"; do
  run_one baseline "$lr" train_rl_baseline.py 0.6 "--hard_buffer_ttl 10 --hard_buffer_cooldown 5 --hard_buffer_target_samples 192 --hard_buffer_group_size 8 --hard_buffer_extra_lr_scale 0.5 --hard_buffer_adv_clip 2.5"
done
for lr in "${LRS[@]}"; do
  run_one direct "$lr" train_rl_direct.py 1.0 "--clip_eps 0.2 --time_state_l2 0.0 --time_state_clamp 10.0"
done
for lr in "${LRS[@]}"; do
  run_one dynamic "$lr" train_rl_dynamic.py 0.6 "--dynamic_min_valid_groups 16 --dynamic_max_resample_rounds 2 --dynamic_max_keep_groups 24"
done

echo "done=$(date '+%F %T')"
