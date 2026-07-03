#!/bin/bash
set -euo pipefail
ROOT=/root/RWKV-LM/RWKV7-math500_recursive_multicall_grpo_20260621
MODEL=${MODEL:-/root/autodl-tmp/rwkv_models/ms_g1_1p5b/rwkv7-g1f-1.5b-20260419-ctx8192.pth}
TOKENIZER=${TOKENIZER:-/root/RWKV-LM/rwkv_vocab_v20230424.txt}
TRAIN=${TRAIN:-/root/autodl-tmp/data/math500/train.jsonl}
EVAL=${EVAL:-/root/autodl-tmp/data/math500/test.jsonl}
STAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR=${OUT_DIR:-/root/autodl-tmp/math500_recursive_multicall_grpo_20_${STAMP}}
mkdir -p "$OUT_DIR"
echo "$OUT_DIR" > /root/autodl-tmp/current_math500_recursive_multicall.path
cd "$ROOT"
/root/miniconda3/bin/python main.py \
  --train_jsonl "$TRAIN" \
  --eval_jsonl "$EVAL" \
  --model "$MODEL" \
  --tokenizer "$TOKENIZER" \
  --out_dir "$OUT_DIR" \
  --tune_mode full \
  --model_dtype bf16 \
  --reward_mode trl_doc \
  --prompt_mode recursive_math \
  --num_questions 6 \
  --samples_per_question 6 \
  --total_steps 20 \
  --max_new_tokens 1024 \
  --eval_max_new_tokens 1024 \
  --temperature 0.8 \
  --top_p 1.0 \
  --top_k 0 \
  --eval_temperature 0.5 \
  --eval_top_p 0.28 \
  --eval_top_k 32 \
  --min_tokens 1 \
  --length_weight 0.005 \
  --zstd_penalty_weight 0.0 \
  --ngram_penalty 0.05 \
  --neg_adv_weight 1.0 \
  --kl_coef 0.0 \
  --lr 1e-6 \
  --micro_batch 1 \
  --rollout_forward_batch 8 \
  --hard_buffer_target_samples 0 \
  --eval_interval 10 \
  --eval_sample_ratio 0.2 \
  --skip_preeval 1 \
  --skip_posteval 1 \
  --save_interval 20 \
  --save_last 1 \
  --final_full_eval 1 \
  --save_responses 1 \
  --seed 42 2>&1 | tee "$OUT_DIR/train_stdout.log"
