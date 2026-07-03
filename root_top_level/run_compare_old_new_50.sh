#!/bin/bash
set -euo pipefail
STAMP=$(date +%Y%m%d_%H%M%S)
BASE=/root/autodl-tmp/compare_old_new_50_${STAMP}
OLD_OUT=${BASE}_old
NEW_OUT=${BASE}_new
mkdir -p "$OLD_OUT" "$NEW_OUT"
COMMON_ARGS=(
  --train_jsonl /root/autodl-tmp/data/math500/train.jsonl
  --eval_jsonl /root/autodl-tmp/data/math500/test.jsonl
  --model /dev/shm/rwkv_models/rwkv7-g1f-1.5b-20260419-ctx8192.pth
  --tokenizer /root/RWKV-LM/rwkv_vocab_v20230424.txt
  --tune_mode full
  --reward_mode trl_doc
  --prompt_mode trl_doc
  --num_questions 8
  --samples_per_question 8
  --total_steps 50
  --max_new_tokens 1500
  --temperature 1.0
  --top_p 0.28
  --top_k 32
  --min_tokens 1
  --length_weight 0.0
  --zstd_penalty_weight 0.0
  --ngram_penalty 0.0
  --neg_adv_weight 1.0
  --kl_coef 0.0
  --lr 1e-6
  --micro_batch 1
  --rollout_forward_batch 8
  --hard_buffer_target_samples 0
  --eval_interval 999999
  --skip_preeval 1
  --skip_posteval 1
  --save_interval 999999
  --save_last 0
  --seed 42
)
cd /root/RWKV-LM/RWKV7-deepmath_trl_doc_g1f1p5b_fp32kernel_noentropy_gradcp
/root/miniconda3/bin/python main.py "${COMMON_ARGS[@]}" --out_dir "$OLD_OUT" > "$OLD_OUT/stdout.log" 2>&1
cd /root/RWKV-LM/RWKV7-deepmath_trl_doc_g1f1p5b_fastrollout
/root/miniconda3/bin/python main.py "${COMMON_ARGS[@]}" --out_dir "$NEW_OUT" > "$NEW_OUT/stdout.log" 2>&1
echo BASE=$BASE
