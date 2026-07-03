#!/usr/bin/env bash
set -euo pipefail
cd /root/RWKV-LM/RWKV-v7/train_temp
PY=/root/miniconda3/bin/python
MODEL=/root/autodl-tmp/rwkv_models/rwkv7-g1e-1.5b-20260309-ctx8192.pth
TOK=/root/RWKV-LM/RWKV-v7/rwkv_vocab_v20230424.txt
TRAIN=/root/autodl-tmp/data/gsm8k_openmath_mathreason_13k/train_formatted_answer_only.jsonl
EVAL=/root/RWKV-LM/RWKV7-statetuning/gsm8k_test_formatted_1of8.jsonl
FULL=/root/RWKV-LM/RWKV7-statetuning/gsm8k_test_formatted.jsonl
ROOT=/root/autodl-tmp/openmath_micro_probe_20260501_0129
mkdir -p "$ROOT"
for MB in 17 16 15 14 13 12; do
  OUT="$ROOT/mb${MB}"
  mkdir -p "$OUT"
  echo "=== TRY mb=${MB} ==="
  set +e
  RWKV_JIT_ON=0 "$PY" train_rl_baseline.py \
    --load_model "$MODEL" \
    --proj_dir "$OUT/run" \
    --tokenizer "$TOK" \
    --train_jsonl "$TRAIN" \
    --eval_jsonl "$EVAL" \
    --full_eval_jsonl "$FULL" \
    --devices 1 --accelerator gpu --strategy deepspeed_stage_3_offload --precision bf16 \
    --total_steps 6 --num_questions 24 --samples_per_question 8 --max_new_tokens 768 \
    --micro_batch "$MB" --rollout_forward_batch 192 --use_stateful_rollout 1 \
    --lr 5e-7 --neg_adv_weight 0.6 --kl_coef 0.05 --kl_mode k3_loss \
    --length_weight 0.5 --zstd_threshold 2.5 --zstd_penalty_weight 0.2 --ngram_penalty 0.0 \
    --skip_preeval 1 --skip_posteval 1 --eval_interval 0 --save_interval 0 \
    --log_interval 1 --enable_progress_bar 0 \
    --hard_buffer_ttl 10 --hard_buffer_cooldown 5 --hard_buffer_target_samples 192 --hard_buffer_group_size 8 \
    --hard_buffer_extra_lr_scale 0.5 --hard_buffer_adv_clip 2.5 \
    --extra_curriculum pure_hard \
    --save_eval_checkpoint 0 --save_final_checkpoint 0 --final_full_eval 0 --disable_extra_step 0 \
    > "$OUT/stdout.log" 2>&1
  RC=$?
  set -e
  echo "$RC" > "$OUT/exit_code.txt"
  if [ "$RC" -eq 0 ]; then
    echo "PASS mb=${MB}"
    exit 0
  else
    echo "FAIL mb=${MB} rc=$RC"
  fi
  if grep -q "OutOfMemoryError" "$OUT/stdout.log"; then
    echo "OOM mb=${MB}" >> "$ROOT/summary.txt"
  else
    echo "ERR mb=${MB}" >> "$ROOT/summary.txt"
  fi
  rm -rf "$OUT/run"
done
exit 1

