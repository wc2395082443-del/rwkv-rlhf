#!/usr/bin/env bash
set -u
export RWKV_JIT_ON=0
ROOT=/root/RWKV-LM/RWKV-v7/train_temp
PY=/root/miniconda3/bin/python
MODEL=/root/autodl-tmp/rwkv_models/rwkv7-g1e-1.5b-20260309-ctx8192.pth
TOK=/root/RWKV-LM/RWKV-v7/rwkv_vocab_v20230424.txt
TRAIN=/root/autodl-tmp/data/math500/train.jsonl
EVAL=/root/autodl-tmp/data/math500/test.jsonl
LOG=$BASE/sweep.log

LRS=(1e-7 2e-7 3e-7 4e-7 5e-7 6e-7 7e-7 8e-7 9e-7 1e-6)

echo "[START] $(date) BASE=$BASE" | tee -a "$LOG"
for LR in "${LRS[@]}"; do
  NAME=$(printf '%s' "$LR" | sed 's/-//g; s/\./p/g')
  OUT=$BASE/lr_${NAME}
  mkdir -p "$OUT/run"
  cat > "$OUT/run.sh" <<EORUN
#!/usr/bin/env bash
set -euo pipefail
export RWKV_JIT_ON=0
cd $ROOT
exec $PY train_rl_baseline.py \
  --load_model $MODEL \
  --proj_dir $OUT/run \
  --tokenizer $TOK \
  --train_jsonl $TRAIN \
  --eval_jsonl $EVAL \
  --devices 1 --accelerator gpu --strategy deepspeed_stage_3_offload --precision bf16 \
  --total_steps 100 --num_questions 24 --samples_per_question 8 --max_new_tokens 768 \
  --micro_batch 18 --rollout_forward_batch 192 --use_stateful_rollout 1 \
  --lr $LR --neg_adv_weight 0.6 --kl_coef 0.05 --kl_mode k3_loss \
  --length_weight 0.5 --zstd_threshold 2.5 --zstd_penalty_weight 0.2 --ngram_penalty 0.0 \
  --skip_preeval 0 --skip_posteval 1 --eval_interval 0 --save_interval 50 \
  --log_interval 1 --enable_progress_bar 0 \
  --hard_buffer_ttl 10 --hard_buffer_cooldown 5 --hard_buffer_target_samples 192 --hard_buffer_group_size 8 \
  --hard_buffer_extra_lr_scale 0.5 --hard_buffer_adv_clip 2.5 \
  --extra_curriculum pure_hard \
  --save_eval_checkpoint 0 --save_final_checkpoint 0 --final_full_eval 1 \
  > $OUT/run/stdout.log 2>&1
EORUN
  chmod +x "$OUT/run.sh"
  echo "[RUN] $(date) lr=$LR out=$OUT" | tee -a "$LOG"
  bash "$OUT/run.sh"
  CODE=$?
  echo "[DONE] $(date) lr=$LR code=$CODE" | tee -a "$LOG"
  if [ $CODE -ne 0 ]; then
    echo "[WARN] lr=$LR failed, continue" | tee -a "$LOG"
  fi
  sleep 5
done

echo "[ALL_DONE] $(date)" | tee -a "$LOG"
