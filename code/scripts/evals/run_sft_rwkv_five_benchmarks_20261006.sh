#!/usr/bin/env bash
set -euo pipefail
BASE=/hy-tmp/evals/g1k_openmath_sft_rwkv_prompt_five_20261006
MODEL=/hy-tmp/runs/g1k_rl1000_openmath7752_sft_lr2e6_cosine_1ep_20261005/model_step3682_infer.pth
EVAL=/hy-tmp/evals/eval_rwkv_prompt.py
EXT=/hy-tmp/torch_extensions_rwkv_prompt_eval_20261006
mkdir -p "$BASE" "$EXT"
export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 TORCH_EXTENSIONS_DIR="$EXT" HF_HOME=/hy-tmp/hf_home
run_one() {
  bench="$1"; data="$2"; rollout="$3"; maxnew="$4"; ctx="$5"
  out="$BASE/$bench"; mkdir -p "$out"
  echo "START $bench $(date)"
  /usr/local/bin/python3 "$EVAL" --albatross-dir /root/eval_transfer/albatross/faster3a_2605 --model "$MODEL" --dataset "$data" --out-dir "$out" --gpus 0 --rollout "$rollout" --bsz 128 --max-new-tokens "$maxnew" --ctx-limit "$ctx" --temperature 0.8 --top-p 0.35 --top-k 40 --seed 42 --prompt-style rwkv --torch-extensions-dir "$EXT" --progress-every 100 --verify-workers 8 > "$out/eval.log" 2>&1
  echo "DONE $bench $(date)"
}
run_one gsm8k /hy-tmp/data/gsm8k/gsm8k_test_albatross.jsonl 4 4096 8192
run_one math500 /hy-tmp/data/math500/test.jsonl 8 4096 8192
run_one aime24 /hy-tmp/data/official_math_extra/aime24.jsonl 64 8192 12288
run_one aime25 /hy-tmp/data/official_math_extra/aime25.jsonl 64 8192 12288
run_one amc23 /hy-tmp/data/official_math_extra/amc23.jsonl 64 8192 12288
