#!/usr/bin/env bash
set -euo pipefail
cd /root/RWKV-LM/RWKV7-math500_teacher_trace_sft_20260613
BASE=/root/autodl-tmp/logs/gsm8k_grpo_vs_opd02_500_20260614
mkdir -p "$BASE"
TOK=/root/RWKV-LM/RWKV-v5/tokenizer/rwkv_vocab_v20230424.txt
STUDENT=/root/autodl-tmp/rwkv_models/ms_g1_1p5b/rwkv7-g1f-1.5b-20260419-ctx8192.pth
TEACHER=/root/autodl-tmp/rwkv_models/ms_g1_7p2b/rwkv7-g1f-7.2b-20260414-ctx8192.pth
TRAIN=/root/autodl-tmp/data/gsm8k/gsm8k_train_formatted.jsonl
EVAL=/root/autodl-tmp/data/gsm8k/gsm8k_test_formatted.jsonl
PY=/root/miniconda3/bin/python
COMMON_ARGS=(
  --train_jsonl "$TRAIN" --eval_jsonl "$EVAL"
  --model "$STUDENT" --tokenizer "$TOK"
  --tune_mode full --ctx_len 8192 --model_dtype bf16
  --reward_mode trl_doc --prompt_mode trl_doc
  --num_questions 8 --samples_per_question 8
  --max_new_tokens 1024 --eval_max_new_tokens 1024
  --temperature 0.8 --top_p 1.0 --top_k 0
  --eval_temperature 1.0 --eval_top_p 0.28 --eval_top_k 32
  --total_steps 500 --micro_batch 1 --rollout_forward_batch 8
  --lr 1e-6 --grad_clip 1.0 --neg_adv_weight 1.0
  --kl_coef 0 --hard_buffer_target_samples 0
  --opd_temp 1.0 --logit_chunk_tokens 32
  --skip_preeval 1 --skip_posteval 1
  --save_interval 100 --save_last 1 --eval_interval 0
  --save_responses 0
)
run_one() {
  local name="$1"; shift
  local out="$BASE/$name"
  mkdir -p "$out"
  echo "===== START $name $(date '+%F %T') =====" | tee -a "$BASE/master.log"
  "$PY" main.py "${COMMON_ARGS[@]}" "$@" --out_dir "$out/train" > "$out/stdout.log" 2>&1
  echo "===== DONE $name $(date '+%F %T') =====" | tee -a "$BASE/master.log"
}
run_one pure_grpo --opd_coef 0
run_one opd_coef_0p2 --teacher_model "$TEACHER" --teacher_dtype bf16 --opd_coef 0.2
echo "===== ALL DONE $(date '+%F %T') =====" | tee -a "$BASE/master.log"