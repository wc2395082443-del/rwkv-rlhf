#!/usr/bin/env bash
set -euo pipefail

OUT=/root/autodl-tmp/deepmath_clean_gsm8k_grpo_g1f1p5b_step500_20260713
MODEL=/root/autodl-tmp/deepmath_clean_gsm8k_grpo_g1f1p5b_step300_20260713/final_step_300.pth
TOKENIZER=/root/RWKV-LM/RWKV-v7/rwkv_vocab_v20230424.txt
DATA_ROOT=/root/autodl-tmp/data/deepmath_clean_split_seed42_20260713

export PATH=/root/miniconda3/bin:$PATH
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
export TMPDIR=/root/autodl-tmp/tmp
export TORCH_EXTENSIONS_DIR=/root/autodl-tmp/torch_extensions_deepmath_clean
export HF_HOME=/root/autodl-tmp/hf_home
export XDG_CACHE_HOME=/root/autodl-tmp/xdg_cache

mkdir -p "$OUT" "$TMPDIR" "$TORCH_EXTENSIONS_DIR"
cd /root/RWKV-LM/RWKV-v7/train_temp

cp train_rl_deepmath_resume.py "$OUT/train_rl_deepmath_resume.py"
cp /root/autodl-tmp/run_deepmath_resume_300_to500_20260713.sh "$OUT/run.sh"
echo "[$(date '+%F %T')] START global_step=300 target_step=500" | tee "$OUT/status.log"

/root/miniconda3/bin/python train_rl_deepmath_resume.py \
  --load_model "$MODEL" \
  --proj_dir "$OUT" \
  --tokenizer "$TOKENIZER" \
  --train_jsonl "$DATA_ROOT/train.jsonl" \
  --eval_jsonl "$DATA_ROOT/validation.jsonl" \
  --full_eval_jsonl "$DATA_ROOT/validation.jsonl" \
  --devices 1 --accelerator gpu --strategy deepspeed_stage_3_offload --precision bf16 \
  --grad_cp 1 --total_steps 200 --step_offset 300 --random_seed 44 \
  --num_questions 8 --samples_per_question 8 --max_new_tokens 1024 \
  --micro_batch 8 --rollout_forward_batch 64 --use_stateful_rollout 1 \
  --temperature 1.0 --top_p 0.6 --top_k 0 --ppo_epochs 1 \
  --eval_temperature 0.3 --eval_top_p 0.4 --eval_top_k 500 \
  --lr 5e-7 --grad_clip 1.0 --neg_adv_weight 0.6 \
  --kl_coef 0.05 --kl_mode k3_loss \
  --length_weight 0.0 --zstd_penalty_weight 0.0 --ngram_penalty 0.0 \
  --hard_buffer_target_samples 0 --disable_extra_step 1 --extra_curriculum off \
  --eval_interval 50 --eval_sample_ratio 0.1 \
  --save_interval 100 --save_eval_checkpoint 1 --log_interval 1 \
  --skip_preeval 1 --skip_posteval 1 --final_full_eval 0 \
  --enable_progress_bar 0 --save_final_checkpoint 1 \
  2>&1 | tee "$OUT/stdout.log"

rc=${PIPESTATUS[0]}
echo "[$(date '+%F %T')] END rc=$rc" | tee -a "$OUT/status.log"
exit "$rc"
