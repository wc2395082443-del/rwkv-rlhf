#!/usr/bin/env bash
set -euo pipefail
cd /root/RWKV-LM/RWKV7-math500_7b_best1p5_noref_20260623
OUT="/root/autodl-tmp/math500_7b_g1f_full_best1p5_sgd_20_gradclear_20260624_034419"
export PATH=/root/miniconda3/bin:$PATH
export TORCH_EXTENSIONS_DIR=/root/autodl-tmp/torch_ext
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
/root/miniconda3/bin/python main.py \
  --train_jsonl /root/autodl-tmp/data/math500/train.jsonl \
  --eval_jsonl /root/autodl-tmp/data/math500/test.jsonl \
  --model /root/autodl-tmp/rwkv_models/ms_g1_7p2b/rwkv7-g1f-7.2b-20260414-ctx8192.pth \
  --tokenizer /root/RWKV-LM/rwkv_vocab_v20230424.txt \
  --out_dir "$OUT" \
  --tune_mode full --model_dtype bf16 --reward_mode trl_doc --prompt_mode trl_doc \
  --num_questions 8 --samples_per_question 8 --total_steps 20 \
  --max_new_tokens 1024 --eval_max_new_tokens 1024 \
  --temperature 1.0 --top_p 1.0 --top_k 0 \
  --eval_temperature 1.0 --eval_top_p 0.28 --eval_top_k 32 \
  --min_tokens 1 --length_weight 0.0 --zstd_penalty_weight 0.0 --ngram_penalty 0.0 \
  --neg_adv_weight 1.0 --kl_coef 0.0 --lr 1e-6 --optimizer sgd \
  --micro_batch 1 --rollout_forward_batch 1 --hard_buffer_target_samples 0 \
  --eval_interval 0 --skip_preeval 1 --skip_posteval 1 \
  --save_interval 999999 --save_last 1 --final_full_eval 0 --save_responses 0 \
  --seed 47 2>&1 | tee "$OUT/train_stdout.log"
