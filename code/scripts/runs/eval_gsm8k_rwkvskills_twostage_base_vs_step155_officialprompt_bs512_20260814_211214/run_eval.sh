#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SCRIPT=/root/eval_transfer/eval_gsm8k_rwkvskills_twostage.py
ALB=/root/eval_transfer/albatross/faster3a_2605
DATA=/root/eval_transfer/data/gsm8k_test_formatted.jsonl
BASE=/root/eval_transfer/models/rwkv7-g1i_preview5445-1.5b-20260729-ctx16384.pth
STEP155=/hy-tmp/runs/g1i_gsm8k_boxed_strictcot_continue_from_step105_fixed50_maxnew2048_micro3_dyn512_q32_r16_20260812_142929/final_step_155.pth
COMMON="--dataset $DATA --albatross-dir $ALB --batch-size 512 --max-new-stage1 4096 --max-new-stage2 128 --temp1 0.8 --topp1 0.35 --topk1 40 --temp2 0.8 --topp2 0.30 --topk2 8 --ctx-limit 16384"
python "$SCRIPT" --model "$BASE" --out-dir "$PWD/base" $COMMON 2>&1 | tee "$PWD/base.log"
python "$SCRIPT" --model "$STEP155" --out-dir "$PWD/step155" $COMMON 2>&1 | tee "$PWD/step155.log"
