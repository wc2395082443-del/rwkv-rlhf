#!/usr/bin/env bash
set -euo pipefail
CODE=/hy-tmp/RWKV7-g1i-strict-quality-gate-adamw-nooffload-20260810
OUT=/hy-tmp/runs/g1i_step173_to_dapo_pass32_noearlystop_lr1e7_20260816_120144
cd "$CODE"
export CUDA_VISIBLE_DEVICES=0
export TORCH_EXTENSIONS_DIR=/hy-tmp/torch_extensions_g1i_strict_quality_adamw_nooffload
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
(
  echo "timestamp,name,mem_used_mib,mem_total_mib,gpu_util_pct,mem_util_pct,power_w"
  while true; do
    nvidia-smi --query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu,utilization.memory,power.draw --format=csv,noheader,nounits || true
    sleep 5
  done
) > "$OUT/gpu_monitor.csv" &
MON_PID=$!
echo $MON_PID > "$OUT/monitor.pid"
cleanup() { kill "$MON_PID" 2>/dev/null || true; }
trap cleanup EXIT
python3 train_rl_thinking_cot_strict_boxedprompt_schedopt_earlystop.py \
  --load_model /hy-tmp/runs/g1i_step155_to_dapo_processed_strictcot_max4096_lr1e7_pilot50_20260816_024807/final_step_173.pth \
  --proj_dir "$OUT" \
  --tokenizer /hy-tmp/RWKV-v7/rwkv_vocab_v20230424.txt \
  --train_jsonl /hy-tmp/data/openr1_dapo_math17k_processed/strict_split_seed42/train.jsonl \
  --eval_jsonl /hy-tmp/data/openr1_dapo_math17k_processed/strict_split_seed42/validation.jsonl \
  --full_eval_jsonl /hy-tmp/data/openr1_dapo_math17k_processed/strict_split_seed42/test.jsonl \
  --devices 1 --accelerator gpu --strategy auto --precision bf16 --grad_cp 1 \
  --total_steps 50 --step_offset 173 --random_seed 45 \
  --num_questions 4 --samples_per_question 32 --max_new_tokens 4096 \
  --micro_batch 1 --rollout_forward_batch 512 --use_stateful_rollout 1 \
  --temperature 1.0 --top_p 1.0 --top_k 0 \
  --ppo_epochs 1 \
  --eval_temperature 0.3 --eval_top_p 0.4 --eval_top_k 500 --eval_tail_max_tokens 0 \
  --lr 1e-7 --beta1 0.9 --beta2 0.99 --weight_decay 0.0 --grad_clip 1.0 \
  --neg_adv_weight 1.0 --kl_coef 0.0 --kl_mode k3_loss --diagnostic_ref_forward 0 \
  --length_weight 0.0 --zstd_penalty_weight 0.0 --ngram_penalty 0.0 \
  --hard_buffer_target_samples 0 --disable_extra_step 1 --extra_curriculum off \
  --eval_interval 25 --eval_sample_ratio 0.2 \
  --dapo_dynamic_sampling 1 --dapo_candidate_questions 32 --dapo_max_gen_batches 4 --dapo_min_valid_groups 4 \
  --dapo_dual_clip_c 10.0 --dapo_overlong_buffer_tokens 0 --dapo_overlong_penalty_factor 0.0 \
  --checkpoint_interval 0 --save_interval 0 --save_eval_checkpoint 0 \
  --log_interval 1 --preeval_sample_ratio 0.2 --skip_preeval 0 --skip_posteval 0 --posteval_sample_ratio 0.2 --final_full_eval 0 \
  --fp32_master_optimizer 1 --enable_progress_bar 0 --save_final_checkpoint 1 \
  --log_responses 0 --compute_grad_norm 0 --empty_cache_each_micro 0 \
  --train_early_stop_metric avg_reward --train_early_stop_window 10 \
  --train_early_stop_patience 0 --train_early_stop_min_delta 0.01 \
  --train_early_stop_min_steps 30 --train_early_stop_threshold 0.0
