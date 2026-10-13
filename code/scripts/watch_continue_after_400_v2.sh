#!/usr/bin/env bash
set -uo pipefail
BASE="/hy-tmp/runs/g1i_step300_to_dapo_math17k_officialprompt_q32x16_cand64_max6144_ob512_continue300to400_eval384_20260824_122246"
TRAIN_SCRIPT="/hy-tmp/RWKV7-g1i-strict-quality-gate-adamw-nooffload-20260810/train_rl_thinking_cot_officialprompt_eval512_schedopt_earlystop_step201to300.py"
LOG="/hy-tmp/watch_continue_after_400_v2.log"
echo "watcher v2 started $(date -Is)" >> "$LOG"
while true; do
  STEP=$(python3 - "$BASE/metrics.jsonl" <<"PY" 2>/dev/null
import json, sys, pathlib
p=pathlib.Path(sys.argv[1])
if not p.exists(): print(-1); raise SystemExit
lines=[l for l in p.read_text().splitlines() if l.strip()]
if not lines: print(-1); raise SystemExit
print(json.loads(lines[-1]).get("step", -1))
PY
)
  if [[ "$STEP" =~ ^[0-9]+$ ]] && (( STEP >= 400 )); then
    echo "saw train step $STEP $(date -Is)" >> "$LOG"
    break
  fi
  sleep 60
done
for i in $(seq 1 180); do
  LOAD_MODEL="$BASE/final_step_400.pth"
  [[ -f "$LOAD_MODEL" ]] && break
  LOAD_MODEL="$BASE/best_full_eval.pth"
  [[ -f "$LOAD_MODEL" ]] && break
  sleep 30
done
if [[ ! -f "$LOAD_MODEL" ]]; then
  echo "missing checkpoint after wait $(date -Is)" >> "$LOG"
  exit 1
fi
NEW_DIR="/hy-tmp/runs/g1i_step400_to_dapo_math17k_officialprompt_q32x16_cand64_max6144_ob512_continue400to500_eval384_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$NEW_DIR"
echo "launching $NEW_DIR from $LOAD_MODEL $(date -Is)" >> "$LOG"
nohup python3 "$TRAIN_SCRIPT" \
  --load_model "$LOAD_MODEL" \
  --proj_dir "$NEW_DIR" \
  --tokenizer /hy-tmp/RWKV-v7/rwkv_vocab_v20230424.txt \
  --train_jsonl /hy-tmp/data/openr1_dapo_math17k_processed/pass32_1_16_seed42_val500/train.jsonl \
  --eval_jsonl /hy-tmp/data/openr1_dapo_math17k_processed/pass32_1_16_seed42_val500/validation.jsonl \
  --full_eval_jsonl /hy-tmp/data/openr1_dapo_math17k_processed/pass32_1_16_seed42_val500/validation.jsonl \
  --devices 1 --accelerator gpu --strategy auto --precision bf16 --grad_cp 1 \
  --total_steps 100 --step_offset 400 --random_seed 42 --num_questions 32 --samples_per_question 16 \
  --max_new_tokens 6144 --micro_batch 1 --rollout_forward_batch 512 --use_stateful_rollout 1 \
  --temperature 1.0 --top_p 1.0 --top_k 0 --ppo_epochs 1 \
  --eval_temperature 0.8 --eval_top_p 0.35 --eval_top_k 40 --eval_tail_max_tokens 0 \
  --lr 2e-7 --beta1 0.9 --beta2 0.99 --weight_decay 0.0 --grad_clip 1.0 \
  --neg_adv_weight 1.0 --kl_coef 0.0 --kl_mode k3_loss --diagnostic_ref_forward 0 \
  --length_weight 0.0 --zstd_penalty_weight 0.0 --ngram_penalty 0.0 \
  --hard_buffer_target_samples 0 --disable_extra_step 1 --extra_curriculum off \
  --eval_interval 50 --eval_sample_ratio 1.0 --dapo_dynamic_sampling 1 \
  --dapo_candidate_questions 64 --dapo_max_gen_batches 4 --dapo_min_valid_groups 32 \
  --dapo_dual_clip_c 10.0 --dapo_overlong_buffer_tokens 512 --dapo_overlong_penalty_factor 1.0 \
  --checkpoint_interval 0 --save_interval 0 --save_eval_checkpoint 1 --log_interval 1 \
  --preeval_sample_ratio 1.0 --skip_preeval 1 --skip_posteval 0 --posteval_sample_ratio 1.0 \
  --final_full_eval 1 --f32_master_optimizer 1 --enable_progress_bar 0 --save_final_checkpoint 1 \
  --log_responses 0 --compute_grad_norm 0 --empty_cache_each_micro 0 \
  --train_early_stop_metric none --train_early_stop_window 0 --train_early_stop_patience 0 \
  --train_early_stop_min_delta 0.0 --train_early_stop_min_steps 999999 --train_early_stop_threshold 0.0 \
  >> "$NEW_DIR/launch.log" 2>&1 &
echo "launched pid $! $(date -Is)" >> "$LOG"
