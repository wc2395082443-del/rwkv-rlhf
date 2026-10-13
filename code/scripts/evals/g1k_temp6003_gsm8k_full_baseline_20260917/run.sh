#!/usr/bin/env bash
set -euo pipefail
cd /hy-tmp/RWKV7-g1i-strict-quality-gate-adamw-nooffload-20260810
export CUDA_VISIBLE_DEVICES=0
export TORCH_EXTENSIONS_DIR=/hy-tmp/torch_extensions_g1k_temp6003_gsm8k
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
python3 eval_gsm8k_boxed_strict_reward.py \
  --model /hy-tmp/runs/rwkv-g1k-1b-temp-6003.pth \
  --tokenizer /hy-tmp/RWKV-v7/rwkv_vocab_v20230424.txt \
  --eval_jsonl /hy-tmp/data/gsm8k/gsm8k_test_albatross.jsonl \
  --out_dir /hy-tmp/evals/g1k_temp6003_gsm8k_full_baseline_20260917 \
  --max_new_tokens 2048 --temperature 0.3 --top_p 0.4 --top_k 500 \
  --batch 64 --seed 424242
