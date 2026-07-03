#!/usr/bin/env bash
set -euo pipefail
cd /root/RWKV-LM/RWKV7-statetuning_hardbuffer_v1
PY=/root/miniconda3/bin/python3
MODEL=/root/RWKV-LM/rwkv7-g1b-1.5b-20251202-ctx8192.pth
TOKENIZER=/root/RWKV-LM/rwkv_vocab_v20230424.txt
TRAIN_JSONL=/root/RWKV-LM/RWKV7-statetuning_hardbuffer_v1/gsm8k_train_formatted.jsonl
EVAL_JSONL=/root/RWKV-LM/RWKV7-statetuning_hardbuffer_v1/gsm8k_test_formatted.jsonl
BASE_LOG=/root/RWKV-LM/RWKV7-statetuning_hardbuffer_v1/log
TS=$(date +%Y%m%d_%H%M%S)
MASTER_DIR=$BASE_LOG/ttl_cd_sweep_${TS}
mkdir -p "$MASTER_DIR"
echo "master_dir=$MASTER_DIR"
for ttl in 2 4 6; do
  for cdv in 2 4 6; do
    OUT=$MASTER_DIR/hb_k3loss_ttl${ttl}_cd${cdv}
    mkdir -p "$OUT"
    echo "[$(date '+%F %T')] start ttl=$ttl cd=$cdv out=$OUT" | tee -a "$MASTER_DIR/sweep.log"
    "$PY" main.py \
      --train_jsonl "$TRAIN_JSONL" \
      --eval_jsonl "$EVAL_JSONL" \
      --model "$MODEL" \
      --tokenizer "$TOKENIZER" \
      --ctx_len 8192 \
      --total_steps 100 \
      --num_questions 24 \
      --samples_per_question 8 \
      --hard_buffer_ttl "$ttl" \
      --hard_buffer_cooldown "$cdv" \
      --hard_buffer_target_samples 192 \
      --hard_buffer_group_size 8 \
      --hard_buffer_extra_lr_scale 0.5 \
      --hard_buffer_adv_clip 2.5 \
      --max_new_tokens 1024 \
      --temperature 1.0 \
      --top_p 0.6 \
      --top_k 0 \
      --eval_temperature 0.3 \
      --eval_top_p 0.4 \
      --eval_top_k 500 \
      --min_tokens 200 \
      --length_weight 0.0 \
      --zstd_threshold 2.8 \
      --zstd_penalty_weight 0.0 \
      --ngram_penalty 0.0 \
      --ppo_epochs 1 \
      --micro_batch 4 \
      --lr 6e-5 \
      --grad_clip 1.0 \
      --kl_coef 0.05 \
      --kl_mode k3_loss \
      --neg_adv_weight 0.6 \
      --time_state_l2 0 \
      --time_state_clamp 10.0 \
      --out_dir "$OUT" \
      --log_interval 1 \
      --save_interval 50 \
      --eval_interval 5 \
      --eval_sample_ratio 0.2 \
      --seed 42 \
      > "$OUT/train_stdout.log" 2>&1
    echo "[$(date '+%F %T')] done ttl=$ttl cd=$cdv" | tee -a "$MASTER_DIR/sweep.log"
  done
done
