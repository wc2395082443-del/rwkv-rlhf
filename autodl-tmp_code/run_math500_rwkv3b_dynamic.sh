#!/usr/bin/env bash
set -euo pipefail
TS=$(date +%Y%m%d_%H%M%S)
OUT_ROOT=/root/autodl-tmp/math500_rwkv3b_dynamic_${TS}
OUT_DIR=${OUT_ROOT}/run
mkdir -p $OUT_DIR
cd /root/RWKV-LM/RWKV-v7/train_temp
export PYTHONUNBUFFERED=1
export RWKV_HEAD_SIZE_A=64
export RWKV_CTXLEN=4096
CMD=(
  /root/miniconda3/bin/python train_rl_dynamic.py
  --load_model /root/autodl-tmp/rwkv_models/rwkv7-g1e-2.9b-20260312-ctx8192.pth
  --tokenizer /root/RWKV-LM/RWKV-v7/rwkv_vocab_v20230424.txt
  --train_jsonl /root/autodl-tmp/data/math500/train.jsonl
  --eval_jsonl /root/autodl-tmp/data/math500/test.jsonl
  --devices 1
  --accelerator gpu
  --strategy deepspeed_stage_3_offload
  --precision bf16
  --total_steps 100
  --num_questions 24
  --samples_per_question 8
  --max_new_tokens 1024
  --micro_batch 6
  --rollout_forward_batch 128
  --use_stateful_rollout 1
  --neg_adv_weight 0.6
  --lr 2e-7
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
  --hard_buffer_target_samples 0
  --dynamic_min_valid_groups 16
  --dynamic_max_resample_rounds 2
  --dynamic_max_keep_groups 24
  --proj_dir $OUT_DIR
)
printf '%s\n' $OUT_ROOT > /root/autodl-tmp/current_math500_rwkv3b_dynamic_path.txt
printf '%q ' ${CMD[@]} > $OUT_ROOT/cmd.sh
printf '\n' >> $OUT_ROOT/cmd.sh
${CMD[@]} > $OUT_DIR/stdout.log 2>&1
