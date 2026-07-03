#!/usr/bin/env bash
set -euo pipefail
BASE=/root/autodl-tmp/math500_g1e1p5b_purehard_reward_lr_sweep_scaled50_20260428_093730
LRS=(1e-7 2e-7 3e-7 4e-7 5e-7 6e-7 7e-7 8e-7 9e-7 1e-6)
for LR in "${LRS[@]}"; do
  TAG=$(echo "$LR" | sed 's/-//g; s/\.//g')
  OUT="$BASE/lr_${TAG}"
  mkdir -p "$OUT/run"
  cat > "$OUT/run.sh" <<RUNEOF
#!/usr/bin/env bash
set -euo pipefail
export RWKV_JIT_ON=0
cd /root/RWKV-LM/RWKV-v7/train_temp
exec /root/miniconda3/bin/python train_rl_baseline.py \
  --load_model /root/autodl-tmp/rwkv_models/rwkv7-g1e-1.5b-20260309-ctx8192.pth \
  --proj_dir $OUT/run \
  --tokenizer /root/RWKV-LM/RWKV-v7/rwkv_vocab_v20230424.txt \
  --train_jsonl /root/autodl-tmp/data/math500/train.jsonl \
  --eval_jsonl /root/autodl-tmp/data/math500/test.jsonl \
  --devices 1 --accelerator gpu --strategy deepspeed_stage_3_offload --precision bf16 \
  --total_steps 100 --num_questions 24 --samples_per_question 8 --max_new_tokens 768 \
  --micro_batch 9 --rollout_forward_batch 96 --use_stateful_rollout 1 \
  --lr $LR --neg_adv_weight 0.6 --kl_coef 0.05 --kl_mode k3_loss \
  --length_weight 0.5 --zstd_threshold 2.5 --zstd_penalty_weight 0.2 --ngram_penalty 0.0 \
  --skip_preeval 0 --skip_posteval 1 --eval_interval 0 --save_interval 50 \
  --log_interval 1 --enable_progress_bar 0 \
  --hard_buffer_ttl 10 --hard_buffer_cooldown 5 --hard_buffer_target_samples 192 --hard_buffer_group_size 8 \
  --hard_buffer_extra_lr_scale 0.5 --hard_buffer_adv_clip 2.5 \
  --extra_curriculum pure_hard \
  --save_eval_checkpoint 0 --save_final_checkpoint 0 --final_full_eval 1 \
  > $OUT/run/stdout.log 2>&1
RUNEOF
  chmod +x "$OUT/run.sh"
  echo "[RUN] $(date) lr=$LR out=$OUT" | tee -a "$BASE/sweep.log"
  set +e
  "$OUT/run.sh"
  code=$?
  set -e
  echo "[DONE] $(date) lr=$LR code=$code" | tee -a "$BASE/sweep.log"
  if [ $code -ne 0 ]; then
    echo "[WARN] lr=$LR failed, continue" | tee -a "$BASE/sweep.log"
  fi
  sleep 5
done

echo "[ALL_DONE] $(date)" | tee -a "$BASE/sweep.log"
