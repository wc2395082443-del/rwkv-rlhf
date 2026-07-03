#!/usr/bin/env bash
set -euo pipefail
cd /root/RWKV-LM/RWKV7-math500_teacher_trace_sft_20260613_noinlineeval
BASE=/root/autodl-tmp/logs/gsm8k_grpo_vs_opd02_500_20260614/opd_full_eval
mkdir -p "$BASE"
TOK=/root/RWKV-LM/RWKV-v5/tokenizer/rwkv_vocab_v20230424.txt
STUDENT=/root/autodl-tmp/rwkv_models/ms_g1_1p5b/rwkv7-g1f-1.5b-20260419-ctx8192.pth
TRAIN=/root/autodl-tmp/data/gsm8k/gsm8k_train_formatted.jsonl
EVAL=/root/autodl-tmp/data/gsm8k/gsm8k_test_formatted.jsonl
PY=/root/miniconda3/bin/python
run_eval() {
  local tag="$1"
  local ckpt="$2"
  local out="$BASE/$tag"
  rm -rf "$out"
  mkdir -p "$out"
  echo "===== START eval $tag $(date '+%F %T') =====" | tee -a "$BASE/master.log"
  "$PY" main.py \
    --train_jsonl "$TRAIN" --eval_jsonl "$EVAL" \
    --model "$STUDENT" --tokenizer "$TOK" \
    --full_init_ckpt "$ckpt" \
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
    --save_responses 0 --out_dir "$out/train" \
    > "$out/stdout.log" 2>&1
  echo "===== DONE eval $tag $(date '+%F %T') =====" | tee -a "$BASE/master.log"
}
run_eval total100 /root/autodl-tmp/logs/gsm8k_grpo_vs_opd02_500_20260614/opd_coef_0p2/train/ckpt_step100.pth
run_eval total200 /root/autodl-tmp/logs/gsm8k_grpo_vs_opd02_500_20260614/opd_coef_0p2_resume_lowmem/train/ckpt_step100.pth
run_eval total300 /root/autodl-tmp/logs/gsm8k_grpo_vs_opd02_500_20260614/opd_coef_0p2_resume_lowmem/train/ckpt_step200.pth
run_eval total400 /root/autodl-tmp/logs/gsm8k_grpo_vs_opd02_500_20260614/opd_coef_0p2_resume_lowmem/train/ckpt_step300.pth
run_eval total500 /root/autodl-tmp/logs/gsm8k_grpo_vs_opd02_500_20260614/opd_coef_0p2_resume_lowmem/train/ckpt_step400.pth
echo "===== ALL DONE $(date '+%F %T') =====" | tee -a "$BASE/master.log"

