#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/RWKV-LM/RWKV7-statetuning_hardbuffer_v1
MODEL=/root/RWKV-LM/rwkv7-g1b-1.5b-20251202-ctx8192.pth
TOK=/root/RWKV-LM/rwkv_vocab_v20230424.txt
TRAIN=$ROOT/gsm8k_train_formatted.jsonl
EVAL=$ROOT/gsm8k_test_formatted.jsonl

TS=$(date +%Y%m%d_%H%M%S)
BASE=$ROOT/log/k3_pair_300_${TS}
mkdir -p $BASE

echo [BASE] $BASE > $BASE/sweep.log
echo [START] $(date) >> $BASE/sweep.log

run_one () {
  local KL=$1
  local OUT=$BASE/k3_loss_kl${KL}
  mkdir -p $OUT
  echo [RUN] $(date) mode=k3_loss kl=$KL out=$OUT >> $BASE/sweep.log

  /root/miniconda3/bin/python3 $ROOT/main.py \
    --train_jsonl $TRAIN \
    --eval_jsonl $EVAL \
    --model $MODEL \
    --tokenizer $TOK \
    --out_dir $OUT \
    --total_steps 300 \
    --eval_interval 5 \
    --save_interval 50 \
    --eval_sample_ratio 0.2 \
    --eval_top_k 500 \
    --max_new_tokens 1024 \
    --lr 1e-4 \
    --ppo_epochs 1 \
    --kl_mode k3_loss \
    --kl_coef $KL \
    --neg_adv_weight 0.6 \
    --zstd_threshold 2.8 \
    --zstd_penalty_weight 0.2 \
    --hard_buffer_ttl 2 \
    --hard_buffer_cooldown 4 \
    --hard_buffer_target_samples 192 \
    --hard_buffer_group_size 8 \
    --hard_buffer_extra_lr_scale 0.5 \
    --hard_buffer_adv_clip 2.5 \
    > $OUT/train_stdout.log 2>&1

  echo [DONE] $(date) mode=k3_loss kl=$KL >> $BASE/sweep.log
}

run_one 0.02
run_one 0.05

echo [ALL_DONE] $(date) >> $BASE/sweep.log
