#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/RWKV-LM/RWKV7-statetuning_hardbuffer_v1
MODEL=/root/RWKV-LM/rwkv7-g1b-1.5b-20251202-ctx8192.pth
TOK=/root/RWKV-LM/rwkv_vocab_v20230424.txt
TRAIN=$ROOT/gsm8k_train_formatted.jsonl
EVAL=$ROOT/gsm8k_test_formatted.jsonl

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TS=$(date +%Y%m%d_%H%M%S)
BASE=$ROOT/log/kl10_restart_${TS}
mkdir -p "$BASE"

echo "[BASE] $BASE" | tee -a "$BASE/sweep.log"
echo "[START] $(date)" | tee -a "$BASE/sweep.log"

declare -a KLS=(0.01 0.03 0.05 0.07 0.09)

run_one () {
  local mode="$1"
  local kl="$2"
  local tag="$3"
  local out="$BASE/${tag}"
  mkdir -p "$out"
  echo "[RUN] $(date) mode=$mode kl=$kl out=$out" | tee -a "$BASE/sweep.log"

  /root/miniconda3/bin/python3 "$ROOT/main.py" \
    --train_jsonl "$TRAIN" \
    --eval_jsonl "$EVAL" \
    --model "$MODEL" \
    --tokenizer "$TOK" \
    --out_dir "$out" \
    --total_steps 100 \
    --eval_interval 5 \
    --save_interval 50 \
    --eval_sample_ratio 0.2 \
    --eval_top_k 500 \
    --max_new_tokens 1024 \
    --lr 1e-4 \
    --ppo_epochs 1 \
    --kl_mode "$mode" \
    --kl_coef "$kl" \
    --neg_adv_weight 0.6 \
    --zstd_threshold 2.8 \
    --zstd_penalty_weight 0.2 \
    --hard_buffer_ttl 2 \
    --hard_buffer_cooldown 4 \
    --hard_buffer_target_samples 192 \
    --hard_buffer_group_size 8 \
    --hard_buffer_extra_lr_scale 0.5 \
    --hard_buffer_adv_clip 2.5 \
    > "$out/train_stdout.log" 2>&1

  echo "[DONE] $(date) mode=$mode kl=$kl" | tee -a "$BASE/sweep.log"
}

for kl in "${KLS[@]}"; do
  run_one k1_reward "$kl" "k1_reward_kl${kl}"
done

for kl in "${KLS[@]}"; do
  run_one k3_loss "$kl" "k3_loss_kl${kl}"
done

echo "[ALL_DONE] $(date)" | tee -a "$BASE/sweep.log"
