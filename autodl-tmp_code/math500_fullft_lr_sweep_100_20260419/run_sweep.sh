#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR=/root/autodl-tmp/math500_fullft_lr_sweep_100_20260419
CODE_DIR=/root/RWKV-LM/RWKV-v7/train_temp
PY=/root/miniconda3/bin/python
MODEL=/root/autodl-tmp/rwkv_models/rwkv7-g1e-1.5b-20260309-ctx8192.pth
TOKENIZER=/root/RWKV-LM/RWKV-v7/rwkv_vocab_v20230424.txt
TRAIN_JSONL=/root/autodl-tmp/data/math500/train.jsonl
EVAL_JSONL=/root/autodl-tmp/data/math500/test.jsonl
LRS=(5e-8 1e-7 2e-7 3e-7 5e-7 7e-7 1e-6 1.5e-6 2e-6 3e-6)
COMMON_ARGS=(
  --load_model "$MODEL"
  --tokenizer "$TOKENIZER"
  --train_jsonl "$TRAIN_JSONL"
  --eval_jsonl "$EVAL_JSONL"
  --devices 1
  --accelerator gpu
  --strategy deepspeed_stage_3_offload
  --precision bf16
  --total_steps 100
  --num_questions 24
  --samples_per_question 8
  --max_new_tokens 1024
  --micro_batch 12
  --rollout_forward_batch 192
  --use_stateful_rollout 1
  --neg_adv_weight 0.6
  --kl_coef 0.0
  --length_weight 0.0
  --zstd_penalty_weight 0.0
  --ngram_penalty 0.0
  --skip_preeval 0
  --skip_posteval 1
  --eval_interval 0
  --save_interval 50
  --log_interval 1
  --enable_progress_bar 0
  --hard_buffer_ttl 10
  --hard_buffer_cooldown 5
  --hard_buffer_target_samples 192
  --hard_buffer_group_size 8
  --hard_buffer_extra_lr_scale 0.5
  --hard_buffer_adv_clip 2.5
)
run_one () {
  local lr="$1"
  local out="$ROOT_DIR/baseline_fullft_lr_${lr}/run"
  mkdir -p "$out"
  echo "START fullft baseline lr=$lr out=$out" | tee -a "$ROOT_DIR/queue.log"
  (
    cd "$CODE_DIR"
    "$PY" train_rl_baseline.py \
      "${COMMON_ARGS[@]}" \
      --lr "$lr" \
      --proj_dir "$out"
  ) > "$out/stdout.log" 2>&1
  echo "DONE fullft baseline lr=$lr" | tee -a "$ROOT_DIR/queue.log"
}
for lr in "${LRS[@]}"; do
  run_one "$lr"
done
echo ALL_DONE | tee -a "$ROOT_DIR/queue.log"
