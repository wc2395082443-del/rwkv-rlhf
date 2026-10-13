#!/usr/bin/env bash
set -euo pipefail
NAME=$1; RFB=$2; MICRO=$3
OUT=/hy-tmp/runs/phasebench_${NAME}_$(date +%Y%m%d_%H%M%S)
CODE=/hy-tmp/RWKV7-g1i-strict-quality-gate-adamw-nooffload-20260810
MODEL=/hy-tmp/models/rwkv7-g1i_preview5445-1.5b-20260729-ctx16384.pth
TRAIN_DATA=/hy-tmp/data/gsm8k/gsm8k_train_formatted.jsonl
EVAL_DATA=/hy-tmp/data/gsm8k/gsm8k_test_albatross.jsonl
export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TMPDIR=/hy-tmp/tmp_g1i_strict_quality_adamw_nooffload
export TORCH_EXTENSIONS_DIR=/hy-tmp/torch_extensions_g1i_strict_quality_adamw_nooffload
export HF_HOME=/hy-tmp/hf_home XDG_CACHE_HOME=/hy-tmp/xdg_cache
mkdir -p "$OUT" "$TMPDIR" "$TORCH_EXTENSIONS_DIR"
MON="$OUT/gpu_monitor.csv"
( while true; do date +%s.%N | tr -d '\n' >> "$MON"; echo -n ',' >> "$MON"; nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,power.draw --format=csv,noheader,nounits >> "$MON" || true; sleep 0.5; done ) &
MPID=$!
set +e
cd "$CODE"
python3 train_rl_thinking_cot_strict_phasebench.py \
  --load_model "$MODEL" --proj_dir "$OUT" \
  --tokenizer /hy-tmp/RWKV-v7/rwkv_vocab_v20230424.txt \
  --train_jsonl "$TRAIN_DATA" --eval_jsonl "$EVAL_DATA" --full_eval_jsonl "$EVAL_DATA" \
  --devices 1 --accelerator gpu --strategy auto --precision bf16 \
  --grad_cp 1 --total_steps 1 --step_offset 0 --random_seed 42 \
  --num_questions 8 --samples_per_question 8 --max_new_tokens 1024 \
  --micro_batch "$MICRO" --rollout_forward_batch "$RFB" --use_stateful_rollout 1 \
  --temperature 1.0 --top_p 1.0 --top_k 0 --ppo_epochs 1 \
  --eval_temperature 0.3 --eval_top_p 0.4 --eval_top_k 500 --eval_tail_max_tokens 0 \
  --lr 2e-7 --beta1 0.9 --beta2 0.99 --weight_decay 0.0 --grad_clip 1.0 \
  --neg_adv_weight 1.0 --kl_coef 0.0 --kl_mode k3_loss --diagnostic_ref_forward 0 \
  --length_weight 0.0 --zstd_penalty_weight 0.0 --ngram_penalty 0.0 \
  --hard_buffer_target_samples 0 --disable_extra_step 1 --extra_curriculum off \
  --eval_interval 100000 --eval_sample_ratio 1.0 \
  --dapo_dynamic_sampling 1 --dapo_candidate_questions 24 --dapo_max_gen_batches 4 \
  --dapo_min_valid_groups 8 --dapo_dual_clip_c 10.0 \
  --dapo_overlong_buffer_tokens 0 --dapo_overlong_penalty_factor 0.0 \
  --checkpoint_interval 0 --save_interval 0 --save_eval_checkpoint 0 --log_interval 1 \
  --preeval_sample_ratio 1.0 --skip_preeval 1 --skip_posteval 1 --final_full_eval 0 \
  --fp32_master_optimizer 1 --enable_progress_bar 0 --save_final_checkpoint 0 > "$OUT/stdout.log" 2>&1
RC=$?
kill $MPID 2>/dev/null || true
printf '%s\t%s\t%s\t%s\t%s\n' "$NAME" "$RFB" "$MICRO" "$RC" "$OUT" >> /hy-tmp/runs/latest_phasebench_runs.tsv
exit $RC
