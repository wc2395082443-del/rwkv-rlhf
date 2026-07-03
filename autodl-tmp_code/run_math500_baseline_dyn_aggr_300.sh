#!/usr/bin/env bash
set -euo pipefail
TS=
ROOT=/root/autodl-tmp/log/math500_llama_baseline_dyn_aggr_
mkdir -p " \
echo \root_log_dir=\
cd /root/RWKV-LM/llama_grpo_baseline_v1
/root/miniconda3/bin/python3 main.py \
 --train_jsonl /root/autodl-tmp/data/math500/train.jsonl \
 --eval_jsonl /root/autodl-tmp/data/math500/test.jsonl \
 --model /root/autodl-tmp/models/Llama-3.2-3B-Instruct \
 --answer_judge math_verify \
 --total_steps 300 \
 --save_interval 50 \
 --eval_interval 0 \
 --skip_pre_eval \
 --num_questions 24 \
 --samples_per_question 8 \
 --hard_buffer_target_samples 0 \
 --max_new_tokens 768 \
 --gen_batch_size 32 \
 --eval_gen_batch_size 64 \
 --pre_eval_gen_batch_size 64 \
 --micro_batch 8 \
 --lr 6e-5 \
 --neg_adv_weight 0.6 \
 --dynamic_resample_enable 1 \
 --dynamic_min_effective_groups 4 \
 --dynamic_max_rounds 2 \
 --dynamic_unique_questions 1 \
 --out_dir \\
