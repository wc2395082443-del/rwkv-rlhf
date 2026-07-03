#!/usr/bin/env bash
set -euo pipefail
BASE=/root/RWKV-LM/RWKV7-mmlupro_stem_g1f1p5b_boxedslot_20260626
MODEL=/root/autodl-tmp/rwkv_models/ms_g1_1p5b/rwkv7-g1f-1.5b-20260419-ctx8192.pth
TOKENIZER=/root/RWKV-LM/rwkv_vocab_v20230424.txt
SUITE=/root/autodl-tmp/rl_other_datasets/suite_20260627_100936
run_one() {
  local name="$1"
  local train_jsonl="$2"
  local eval_jsonl="$3"
  local steps="$4"
  local out="$SUITE/$name"
  mkdir -p "$out"
  echo "===== START $name $(date) =====" | tee "$out/launcher.log"
  cd "$BASE"
  CUDA_VISIBLE_DEVICES=0 /root/miniconda3/bin/python main.py \
    --train_jsonl "$train_jsonl" \
    --eval_jsonl "$eval_jsonl" \
    --model "$MODEL" \
    --tokenizer "$TOKENIZER" \
    --tune_mode full \
    --ctx_len 8192 \
    --model_dtype bf16 \
    --reward_mode trl_doc \
    --prompt_mode trl_doc \
    --num_questions 8 \
    --samples_per_question 8 \
    --max_new_tokens 1024 \
    --temperature 1.0 \
    --top_p 0.28 \
    --top_k 32 \
    --eval_temperature 1.0 \
    --eval_top_p 0.28 \
    --eval_top_k 32 \
    --total_steps "$steps" \
    --ppo_epochs 1 \
    --lr 1e-6 \
    --grad_clip 1.0 \
    --micro_batch 1 \
    --rollout_forward_batch 4 \
    --kl_coef 0 \
    --neg_adv_weight 1.0 \
    --hard_buffer_target_samples 0 \
    --min_tokens 1 \
    --length_weight 0.0 \
    --zstd_penalty_weight 0.0 \
    --ngram_penalty 0.0 \
    --save_interval 0 \
    --save_last 1 \
    --eval_interval 50 \
    --eval_sample_ratio 0.2 \
    --final_full_eval 0 \
    --preeval_sample_ratio 0.05 \
    --posteval_sample_ratio 0.2 \
    --skip_preeval 1 \
    --skip_posteval 0 \
    --save_responses 1 \
    --out_dir "$out" 2>&1 | tee -a "$out/train_stdout.log"
  echo "===== DONE $name $(date) =====" | tee -a "$out/launcher.log"
}
run_one deepmath103k /root/RWKV-LM/RWKV7-mmlupro_stem_g1f1p5b_boxedslot_20260626/data_deepmath_trl_doc/deepmath_train_rwkv.jsonl /root/RWKV-LM/RWKV7-mmlupro_stem_g1f1p5b_boxedslot_20260626/data_deepmath_trl_doc/deepmath_test_rwkv.jsonl 100
run_one openmath_gsm8k_13k /root/autodl-tmp/data/gsm8k_openmath_mathreason_13k/train_formatted_answer_only.jsonl /root/autodl-tmp/data/gsm8k/gsm8k_test_formatted.jsonl 100
