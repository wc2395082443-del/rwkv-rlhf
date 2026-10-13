#!/usr/bin/env bash
set -euo pipefail

OUT=/root/autodl-tmp/gsm8k_g1h_formatb_nonempty_jointreward100_20260720
CODE=/root/RWKV-LM/RWKV7-gsm8k-g1h-formatb-nonempty-joint-reward-20260720
MODEL=/root/autodl-tmp/rwkv_models/ms_g1_1p5b/rwkv7-g1h-1.5b-20260710-ctx10240.pth
TOKENIZER=/root/RWKV-LM/RWKV-v7/rwkv_vocab_v20230424.txt
TRAIN_DATA=/root/autodl-tmp/data/gsm8k/gsm8k_train_formatted.jsonl
EVAL_DATA=/root/autodl-tmp/data/gsm8k/gsm8k_test_albatross.jsonl
ALBATROSS=/root/autodl-tmp/Albatross_ref_tmp/faster3a_2605
RESCORE=/root/autodl-tmp/evals/dapo_math17k_strict_cot_20260719/qwen_comparison/rescore_dapo_generations.py
PY=/root/miniconda3/bin/python
EXT=/root/autodl-tmp/torch_extensions_deepmath_thinking_cot_strict

export PATH=/root/miniconda3/bin:$PATH
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
export TMPDIR=/root/autodl-tmp/tmp
export TORCH_EXTENSIONS_DIR=$EXT
export HF_HOME=/root/autodl-tmp/hf_home
export XDG_CACHE_HOME=/root/autodl-tmp/xdg_cache

mkdir -p "$OUT" "$OUT/preeval_gsm8k" "$OUT/posteval_gsm8k_step100" "$TMPDIR" "$EXT"
cp "$CODE/train_rl_thinking_cot_strict.py" "$OUT/"
cp "$CODE/eval_thinking_cot_strict.py" "$OUT/"
cp "$CODE/infer.py" "$OUT/"
cp "$CODE/reward.py" "$OUT/"
cp "$CODE/plot_metrics.py" "$OUT/"
cp "$0" "$OUT/run.sh"

if [[ ! -f "$OUT/preeval_gsm8k/summary.json" ]]; then
  echo "[$(date '+%F %T')] PRE_EVAL_START g1h_base" | tee "$OUT/status.log"
  "$PY" "$CODE/eval_thinking_cot_strict.py" \
    --albatross-dir "$ALBATROSS" \
    --model "$MODEL" \
    --dataset "$EVAL_DATA" \
    --out-dir "$OUT/preeval_gsm8k" \
    --gpus 0 --rollout 1 --bsz 512 \
    --max-new-tokens 1024 --eval-tail-max-tokens 0 \
    --temperature 0.3 --top-p 0.4 --top-k 500 \
    --seed 42 --prompt-style thinking_cot \
    --torch-extensions-dir "$EXT" \
    --progress-every 100 --verify-workers 8 \
    2>&1 | tee "$OUT/preeval_gsm8k/eval.log"
else
  echo "[$(date '+%F %T')] PRE_EVAL_REUSE g1h_base" | tee "$OUT/status.log"
fi

"$PY" "$RESCORE" \
  --input "$OUT/preeval_gsm8k/generations.jsonl" \
  --output "$OUT/preeval_gsm8k/rescore_summary.json" \
  --workers 8 | tee "$OUT/preeval_gsm8k/rescore.log"

echo "[$(date '+%F %T')] TRAIN_START g1h_step0_to100" | tee -a "$OUT/status.log"
cd "$CODE"
"$PY" train_rl_thinking_cot_strict.py \
  --load_model "$MODEL" \
  --proj_dir "$OUT" \
  --tokenizer "$TOKENIZER" \
  --train_jsonl "$TRAIN_DATA" \
  --eval_jsonl "$EVAL_DATA" \
  --full_eval_jsonl "$EVAL_DATA" \
  --devices 1 --accelerator gpu --strategy deepspeed_stage_3_offload --precision bf16 \
  --grad_cp 1 --total_steps 100 --step_offset 0 --random_seed 42 \
  --num_questions 8 --samples_per_question 8 --max_new_tokens 1024 \
  --micro_batch 8 --rollout_forward_batch 64 --use_stateful_rollout 1 \
  --temperature 1.0 --top_p 0.6 --top_k 0 --ppo_epochs 1 \
  --eval_temperature 0.3 --eval_top_p 0.4 --eval_top_k 500 --eval_tail_max_tokens 0 \
  --lr 2e-7 --grad_clip 1.0 --neg_adv_weight 1.0 \
  --kl_coef 0.05 --kl_mode k3_loss \
  --length_weight 0.0 --zstd_penalty_weight 0.0 --ngram_penalty 0.0 \
  --hard_buffer_target_samples 0 --disable_extra_step 1 --extra_curriculum off \
  --eval_interval 0 --eval_sample_ratio 0.02 \
  --checkpoint_interval 0 --save_interval 0 --save_eval_checkpoint 0 --log_interval 1 \
  --preeval_sample_ratio 0.02 --skip_preeval 1 --skip_posteval 1 --final_full_eval 0 \
  --enable_progress_bar 0 --save_final_checkpoint 1 \
  2>&1 | tee "$OUT/stdout.log"

"$PY" "$CODE/plot_metrics.py" \
  --metrics "$OUT/metrics.jsonl" \
  --out "$OUT/metrics_plot.png" \
  2>&1 | tee "$OUT/plot.log"

echo "[$(date '+%F %T')] POST_EVAL_START g1h_step100" | tee -a "$OUT/status.log"
"$PY" "$CODE/eval_thinking_cot_strict.py" \
  --albatross-dir "$ALBATROSS" \
  --model "$OUT/final_step_100.pth" \
  --dataset "$EVAL_DATA" \
  --out-dir "$OUT/posteval_gsm8k_step100" \
  --gpus 0 --rollout 1 --bsz 512 \
  --max-new-tokens 1024 --eval-tail-max-tokens 0 \
  --temperature 0.3 --top-p 0.4 --top-k 500 \
  --seed 42 --prompt-style thinking_cot \
  --torch-extensions-dir "$EXT" \
  --progress-every 100 --verify-workers 8 \
  2>&1 | tee "$OUT/posteval_gsm8k_step100/eval.log"

"$PY" "$RESCORE" \
  --input "$OUT/posteval_gsm8k_step100/generations.jsonl" \
  --output "$OUT/posteval_gsm8k_step100/rescore_summary.json" \
  --workers 8 | tee "$OUT/posteval_gsm8k_step100/rescore.log"

echo "[$(date '+%F %T')] END g1h_step100" | tee -a "$OUT/status.log"
