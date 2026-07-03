#!/usr/bin/env bash
set -euo pipefail
cd /root/RWKV-LM/RWKV7-math500_teacher_trace_sft_20260613
BASE=/root/autodl-tmp/logs/gsm8k_opd_sweep_20260614
mkdir -p "$BASE"
TOK=/root/RWKV-LM/RWKV-v5/tokenizer/rwkv_vocab_v20230424.txt
STUDENT=/root/autodl-tmp/rwkv_models/ms_g1_1p5b/rwkv7-g1f-1.5b-20260419-ctx8192.pth
TEACHER=/root/autodl-tmp/rwkv_models/ms_g1_7p2b/rwkv7-g1f-7.2b-20260414-ctx8192.pth
TRAIN=/root/autodl-tmp/data/gsm8k/gsm8k_train_formatted.jsonl
EVAL=/root/autodl-tmp/data/gsm8k/gsm8k_test_formatted.jsonl
PY=/root/miniconda3/bin/python
COEFS=(0.05 0.1 0.2 0.5 1.0)
for C in "${COEFS[@]}"; do
  TAG=$(printf 'coef_%s' "$C" | tr '.' 'p')
  OUT="$BASE/$TAG"
  TRAINOUT="$OUT/train_run"
  EVALOUT="$OUT/eval_step50"
  rm -rf "$TRAINOUT" "$EVALOUT"
  mkdir -p "$TRAINOUT" "$EVALOUT"
  echo "===== train opd_coef=$C =====" | tee "$OUT/status.log"
  $PY main.py \
    --train_jsonl "$TRAIN" --eval_jsonl "$EVAL" \
    --model "$STUDENT" --tokenizer "$TOK" \
    --teacher_model "$TEACHER" --teacher_dtype bf16 \
    --tune_mode full --ctx_len 8192 --model_dtype bf16 \
    --reward_mode trl_doc --prompt_mode trl_doc \
    --num_questions 8 --samples_per_question 8 \
    --max_new_tokens 1024 --eval_max_new_tokens 1024 \
    --temperature 0.8 --top_p 1.0 --top_k 0 \
    --eval_temperature 1.0 --eval_top_p 0.28 --eval_top_k 32 \
    --total_steps 50 --micro_batch 1 --rollout_forward_batch 8 \
    --lr 1e-6 --grad_clip 1.0 --neg_adv_weight 1.0 \
    --kl_coef 0 --hard_buffer_target_samples 0 \
    --opd_coef "$C" --opd_temp 1.0 --logit_chunk_tokens 32 \
    --skip_preeval 1 --skip_posteval 1 \
    --save_interval 50 --save_last 1 --eval_interval 0 \
    --save_responses 0 --out_dir "$TRAINOUT" \
    > "$OUT/train_stdout.log" 2>&1
  CKPT="$TRAINOUT/ckpt_step50.pth"
  echo "===== eval opd_coef=$C ckpt=$CKPT =====" | tee -a "$OUT/status.log"
  $PY main.py \
    --train_jsonl "$TRAIN" --eval_jsonl "$EVAL" \
    --model "$STUDENT" --tokenizer "$TOK" \
    --full_init_ckpt "$CKPT" \
    --tune_mode full --ctx_len 8192 --model_dtype bf16 \
    --reward_mode trl_doc --prompt_mode trl_doc \
    --num_questions 8 --samples_per_question 8 \
    --max_new_tokens 1024 --eval_max_new_tokens 1024 \
    --temperature 0.8 --top_p 1.0 --top_k 0 \
    --eval_temperature 1.0 --eval_top_p 0.28 --eval_top_k 32 \
    --total_steps 0 --micro_batch 1 --rollout_forward_batch 8 \
    --lr 1e-6 --grad_clip 1.0 --neg_adv_weight 1.0 \
    --kl_coef 0 --hard_buffer_target_samples 0 \
    --opd_coef 0 --opd_temp 1.0 --logit_chunk_tokens 32 \
    --skip_preeval 1 --skip_posteval 0 --posteval_sample_ratio 1.0 \
    --save_interval 0 --eval_interval 0 --eval_chunk_size 32 \
    --save_responses 0 --out_dir "$EVALOUT" \
    > "$OUT/eval_stdout.log" 2>&1
  echo "===== done opd_coef=$C =====" | tee -a "$OUT/status.log"
done