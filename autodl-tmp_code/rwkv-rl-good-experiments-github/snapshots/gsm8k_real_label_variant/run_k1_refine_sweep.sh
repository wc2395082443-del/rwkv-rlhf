#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/RWKV-LM/RWKV7-statetuning_hardbuffer_v1
MODEL=/root/RWKV-LM/rwkv7-g1b-1.5b-20251202-ctx8192.pth
TOK=/root/RWKV-LM/rwkv_vocab_v20230424.txt
TRAIN=/gsm8k_train_formatted.jsonl
EVAL=/gsm8k_test_formatted.jsonl

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TS=
BASE=/log/k1_refine_
mkdir -p  

echo [BASE]  | tee -a /sweep.log
echo [START] 03/08/2026 19:35:25  | tee -a  /sweep.log

declare -a KLS=(0.08 0.09 0.10 0.11 0.12)

for kl in ; do
  out=/k1_reward_kl
  mkdir -p 
  echo [RUN] 03/08/2026 19:35:25 mode=k1_reward kl= out= | tee -a /sweep.log

  /root/miniconda3/bin/python3 /main.py     --train_jsonl      --eval_jsonl      --model      --tokenizer      --out_dir      --total_steps 100     --eval_interval 5     --save_interval 50     --eval_sample_ratio 0.2     --eval_top_k 500     --max_new_tokens 1024     --lr 1e-4     --ppo_epochs 1     --kl_mode k1_reward     --kl_coef      --neg_adv_weight 0.6     --zstd_threshold 2.8     --zstd_penalty_weight 0.2     --hard_buffer_ttl 2     --hard_buffer_cooldown 4     --hard_buffer_target_samples 192     --hard_buffer_group_size 8     --hard_buffer_extra_lr_scale 0.5     --hard_buffer_adv_clip 2.5     > /train_stdout.log 2>&1

  echo [DONE] 03/08/2026 19:35:25 mode=k1_reward kl= | tee -a /sweep.log
done

echo [ALL_DONE] 03/08/2026 19:35:25  | tee -a  /sweep.log
