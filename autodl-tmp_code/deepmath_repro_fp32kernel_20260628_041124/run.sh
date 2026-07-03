#!/usr/bin/env bash
set -euo pipefail
cd /root/RWKV-LM/RWKV7-deepmath_trl_doc_g1f1p5b_fp32kernel
CUDA_VISIBLE_DEVICES=0 /root/miniconda3/bin/python main.py \
  --train_jsonl data_deepmath_trl_doc/deepmath_train_rwkv.jsonl \
  --eval_jsonl data_deepmath_trl_doc/deepmath_test_rwkv.jsonl \
  --model /root/autodl-tmp/rwkv_models/ms_g1_1p5b/rwkv7-g1f-1.5b-20260419-ctx8192.pth \
  --tokenizer /root/RWKV-LM/rwkv_vocab_v20230424.txt \
  --out_dir /root/autodl-tmp/deepmath_repro_fp32kernel_20260628_041124 \
  --tune_mode full \
  --reward_mode trl_doc \
  --prompt_mode trl_doc \
  --num_questions 8 \
  --samples_per_question 8 \
  --total_steps 2000 \
  --max_new_tokens 1024 \
  --temperature 1.0 \
  --top_p 1.0 \
  --top_k 0 \
  --eval_temperature 0.3 \
  --eval_top_p 0.4 \
  --eval_top_k 500 \
  --min_tokens 1 \
  --length_weight 0.0 \
  --zstd_penalty_weight 0.0 \
  --ngram_penalty 0.0 \
  --neg_adv_weight 1.0 \
  --kl_coef 0.0 \
  --lr 1e-6 \
  --micro_batch 1 \
  --rollout_forward_batch 8 \
  --hard_buffer_target_samples 0 \
  --eval_interval 500 \
  --eval_sample_ratio 1.0 \
  --skip_preeval 1 \
  --skip_posteval 1 \
  --save_interval 500 \
  --save_last 1 \
  --save_responses 1 \
  --seed 42 2>&1 | tee /root/autodl-tmp/deepmath_repro_fp32kernel_20260628_041124/train_stdout.log
