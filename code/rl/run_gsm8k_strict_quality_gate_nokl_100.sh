#!/usr/bin/env bash
set -euo pipefail

OUT=/root/autodl-tmp/gsm8k_g1h_strict_quality_gate_nokl_t1_topp06_max1024_100_20260801
CODE=/root/RWKV-LM/RWKV7-DAPO-strict-quality-gate-20260801
MODEL=/root/autodl-tmp/rwkv_models/ms_g1_1p5b/rwkv7-g1h-1.5b-20260710-ctx10240.pth
TRAIN_DATA=/root/autodl-tmp/data/gsm8k/gsm8k_train_formatted.jsonl
EVAL_DATA=/root/autodl-tmp/data/gsm8k/gsm8k_test_albatross.jsonl
ALBATROSS=/root/autodl-tmp/Albatross_ref_tmp/faster3a_2605
RESCORE=/root/autodl-tmp/evals/dapo_math17k_strict_cot_20260719/qwen_comparison/rescore_dapo_generations.py
PY=/root/miniconda3/bin/python
EXT=/root/autodl-tmp/torch_extensions_deepmath_thinking_cot_strict

export PATH=/root/miniconda3/bin:$PATH
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TMPDIR=/root/autodl-tmp/tmp
export TORCH_EXTENSIONS_DIR=$EXT
export HF_HOME=/root/autodl-tmp/hf_home
export XDG_CACHE_HOME=/root/autodl-tmp/xdg_cache

mkdir -p "$OUT" "$TMPDIR" "$EXT"
cp "$CODE/train_rl_thinking_cot_strict.py" "$OUT/"
cp "$CODE/train.py" "$OUT/"
cp "$CODE/reward.py" "$OUT/"
cp "$CODE/eval_thinking_cot_strict.py" "$OUT/"
cp "$CODE/infer.py" "$OUT/"
cp "$CODE/plot_metrics.py" "$OUT/"
cp "$0" "$OUT/run.sh"

echo "[$(date '+%F %T')] TRAIN_START model=$MODEL benchmark=gsm8k steps=1-100" | tee "$OUT/status.log"
cd "$CODE"
"$PY" train_rl_thinking_cot_strict.py \
  --load_model "$MODEL" --proj_dir "$OUT" \
  --tokenizer /root/RWKV-LM/RWKV-v7/rwkv_vocab_v20230424.txt \
  --train_jsonl "$TRAIN_DATA" --eval_jsonl "$EVAL_DATA" --full_eval_jsonl "$EVAL_DATA" \
  --devices 1 --accelerator gpu --strategy deepspeed_stage_3_offload --precision bf16 \
  --grad_cp 1 --total_steps 100 --step_offset 0 --random_seed 42 \
  --num_questions 8 --samples_per_question 8 --max_new_tokens 1024 \
  --micro_batch 8 --rollout_forward_batch 64 --use_stateful_rollout 1 \
  --temperature 1.0 --top_p 0.6 --top_k 0 --ppo_epochs 1 \
  --eval_temperature 0.3 --eval_top_p 0.4 --eval_top_k 500 --eval_tail_max_tokens 0 \
  --lr 2e-7 --grad_clip 1.0 --neg_adv_weight 1.0 \
  --kl_coef 0.0 --kl_mode k3_loss \
  --length_weight 0.0 --zstd_penalty_weight 0.0 --ngram_penalty 0.0 \
  --hard_buffer_target_samples 0 --disable_extra_step 1 --extra_curriculum off \
  --eval_interval 20 --eval_sample_ratio 1.0 \
  --dapo_dynamic_sampling 1 --dapo_candidate_questions 24 --dapo_max_gen_batches 4 \
  --dapo_min_valid_groups 8 --dapo_dual_clip_c 10.0 \
  --dapo_overlong_buffer_tokens 0 --dapo_overlong_penalty_factor 0.0 \
  --checkpoint_interval 0 --save_interval 20 --save_eval_checkpoint 1 --log_interval 1 \
  --preeval_sample_ratio 1.0 --skip_preeval 0 --skip_posteval 1 --final_full_eval 0 \
  --enable_progress_bar 0 --save_final_checkpoint 1 2>&1 | tee "$OUT/stdout.log"

"$PY" "$CODE/plot_metrics.py" --metrics "$OUT/metrics.jsonl" \
  --out "$OUT/metrics_plot.png" 2>&1 | tee "$OUT/plot.log"

echo "[$(date '+%F %T')] END" | tee -a "$OUT/status.log"
