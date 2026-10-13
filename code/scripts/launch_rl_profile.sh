#!/usr/bin/env bash
set -euo pipefail
CODE=/hy-tmp/RWKV7-g1i-strict-quality-gate-adamw-nooffload-20260810
OUT=/hy-tmp/runs/rl_profile_step_$(date +%Y%m%d_%H%M%S)
MODEL=/hy-tmp/runs/rwkv7-g1i-math-instruct-v1.pth
TRAIN=/hy-tmp/data/openr1_dapo_math17k_processed/pass32_0_step155/train_pass32_zero.jsonl
cd "$CODE"
mkdir -p "$OUT"
export CUDA_VISIBLE_DEVICES=0
export TORCH_EXTENSIONS_DIR=/hy-tmp/torch_extensions_g1i_strict_quality_adamw_nooffload
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
(
  echo timestamp,name,mem_used_mib,mem_total_mib,gpu_util_pct,mem_util_pct,power_w
  while true; do
    nvidia-smi --query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu,utilization.memory,power.draw --format=csv,noheader,nounits || true
    sleep 5
  done
) > "$OUT/gpu_monitor.csv" &
MON_PID=$!
echo "$MON_PID" > "$OUT/monitor.pid"
cleanup() { kill "$MON_PID" 2>/dev/null || true; }
trap cleanup EXIT
python3 /hy-tmp/RWKV7-g1i-strict-quality-gate-adamw-nooffload-20260810/rl_profile_step.py \
  --load_model "$MODEL" --proj_dir "$OUT" --tokenizer /hy-tmp/RWKV-v7/rwkv_vocab_v20230424.txt \
  --train_jsonl "$TRAIN" --eval_jsonl "" --full_eval_jsonl "" \
  --devices 1 --accelerator gpu --strategy auto --precision bf16 --grad_cp 1 \
  --total_steps 1 --step_offset 0 --random_seed 42 \
  --num_questions 32 --samples_per_question 16 --max_new_tokens 6144 \
  --micro_batch 1 --rollout_forward_batch 512 --use_stateful_rollout 1 \
  --temperature 1.0 --top_p 1.0 --top_k 0 --ppo_epochs 1 \
  --lr 2e-7 --beta1 0.9 --beta2 0.99 --weight_decay 0.0 --grad_clip 1.0 \
  --neg_adv_weight 1.0 --kl_coef 0.0 --kl_mode k3_loss --diagnostic_ref_forward 0 \
  --length_weight 0.0 --zstd_penalty_weight 0.0 --ngram_penalty 0.0 \
  --hard_buffer_target_samples 0 --disable_extra_step 1 --extra_curriculum off \
  --eval_interval 0 --save_interval 0 --checkpoint_interval 0 \
  --log_interval 1 --skip_preeval 1 --skip_posteval 1 --final_full_eval 0 \
  --fp32_master_optimizer 1 --enable_progress_bar 0 --save_final_checkpoint 0 \
  --log_responses 0 --compute_grad_norm 0 --empty_cache_each_micro 0 \
  > "$OUT/stdout.log" 2>&1