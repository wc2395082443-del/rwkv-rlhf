#!/usr/bin/env bash
set -euo pipefail

BASE=/hy-tmp/evals/g1k_step1000_train_prompt_20261007
MODEL=/hy-tmp/runs/g1k_dapo_pass1_31_officialprompt_q32x16_cand64_max6144_ob512_step900to1000_20261001_014411/final_step_1000.pth
EVAL=/hy-tmp/evals/eval_rwkv_prompt.py
EXT=/hy-tmp/torch_extensions_train_prompt_eval_20261007

mkdir -p "$BASE" "$EXT"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
export TORCH_EXTENSIONS_DIR="$EXT"
export HF_HOME=/hy-tmp/hf_home

run_one() {
    bench="$1"
    data="$2"
    rollout="$3"
    maxnew="$4"
    ctx="$5"
    out="$BASE/$bench"

    rm -rf "$out"
    mkdir -p "$out"
    echo "START $bench $(date)"

    /usr/local/bin/python3 "$EVAL" \
      --albatross-dir /root/eval_transfer/albatross/faster3a_2605 \
      --model "$MODEL" \
      --dataset "$data" \
      --out-dir "$out" \
      --gpus 0 \
      --rollout "$rollout" \
      --bsz 512 \
      --max-new-tokens "$maxnew" \
      --ctx-limit "$ctx" \
      --temperature 0.8 \
      --top-p 0.35 \
      --top-k 40 \
      --seed 42 \
      --prompt-style train_prompt \
      --torch-extensions-dir "$EXT" \
      --progress-every 100 \
      --verify-workers 8 > "$out/eval.log" 2>&1

    test -s "$out/summary.json"
    echo "DONE $bench $(date)"
}

run_one gsm8k  /hy-tmp/data/gsm8k/gsm8k_test_albatross.jsonl 4 4096 8192
run_one math500 /hy-tmp/data/math500/test.jsonl 8 4096 8192
run_one aime24 /hy-tmp/data/official_math_extra/aime24.jsonl 64 8192 12288
run_one aime25 /hy-tmp/data/official_math_extra/aime25.jsonl 64 8192 12288
run_one amc23  /hy-tmp/data/official_math_extra/amc23.jsonl 64 8192 12288
