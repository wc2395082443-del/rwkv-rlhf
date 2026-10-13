#!/usr/bin/env bash
set -u
BASE=/hy-tmp/evals/g1k_openmath_sft_final_benchmarks_20261005
MODEL=/hy-tmp/runs/g1k_rl1000_openmath7752_sft_lr2e6_cosine_1ep_20261005/model_step3682_infer.pth
EVAL=/hy-tmp/evals/official_flower_ckpt_eval_20260827/eval_math500_flower.py
EXT=/hy-tmp/torch_extensions_openmath_sft_eval_20261005
mkdir -p "$BASE" "$EXT" "$BASE/dapo_validation"
export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 TORCH_EXTENSIONS_DIR="$EXT" HF_HOME=/hy-tmp/hf_home
run_one() {
  bench="$1"; data="$2"; rollout="$3"; maxnew="$4"; ctx="$5"
  out="$BASE/$bench"; mkdir -p "$out"
  if [ -s "$out/summary.json" ]; then echo "SKIP $bench"; return; fi
  echo "START $bench $(date)"
  /usr/local/bin/python3 "$EVAL" --albatross-dir /root/eval_transfer/albatross/faster3a_2605 --model "$MODEL" --dataset "$data" --out-dir "$out" --gpus 0 --rollout "$rollout" --bsz 512 --max-new-tokens "$maxnew" --ctx-limit "$ctx" --temperature 0.8 --top-p 0.35 --top-k 40 --seed 42 --prompt-style official_flower --torch-extensions-dir "$EXT" --progress-every 100 --verify-workers 8 > "$out/eval.log" 2>&1
  echo "DONE $bench $(date)"
}
run_one gsm8k /hy-tmp/data/gsm8k/gsm8k_test_albatross.jsonl 4 4096 8192
run_one math500 /hy-tmp/data/math500/test.jsonl 8 4096 8192
run_one aime24 /hy-tmp/data/official_math_extra/aime24.jsonl 64 8192 12288
run_one aime25 /hy-tmp/data/official_math_extra/aime25.jsonl 64 8192 12288
run_one amc23 /hy-tmp/data/official_math_extra/amc23.jsonl 64 8192 12288
if [ ! -s "$BASE/dapo_validation/summary.json" ]; then
  echo "START dapo $(date)"
  /usr/local/bin/python3 "$EVAL" --albatross-dir /root/eval_transfer/albatross/faster3a_2605 --model "$MODEL" --dataset /hy-tmp/data/openr1_dapo_math17k_processed/strict_split_seed42/validation.jsonl --out-dir "$BASE/dapo_validation" --gpus 0 --rollout 1 --bsz 512 --max-new-tokens 8192 --ctx-limit 8192 --temperature 0.8 --top-p 0.35 --top-k 40 --seed 42 --prompt-style official_flower --torch-extensions-dir "$EXT" --progress-every 100 --verify-workers 8 > "$BASE/dapo_validation/eval.log" 2>&1
  echo "DONE dapo $(date)"
fi
