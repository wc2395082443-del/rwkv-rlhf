#!/usr/bin/env bash
set -u
BASE=/hy-tmp/evals/g1k_openmath_sft_rwkv_prompt_compare_20261006
EVAL=/hy-tmp/evals/eval_rwkv_prompt.py
EXT=/hy-tmp/torch_extensions_rwkv_prompt_eval_20261006
mkdir -p "$BASE" "$EXT"
export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 TORCH_EXTENSIONS_DIR="$EXT" HF_HOME=/hy-tmp/hf_home
run_one() {
  tag="$1"; model="$2"; bench="$3"; data="$4"; rollout="$5"; maxnew="$6"; ctx="$7"
  out="$BASE/$tag/$bench"; mkdir -p "$out"
  echo "START $tag $bench $(date)"
  /usr/local/bin/python3 "$EVAL" --albatross-dir /root/eval_transfer/albatross/faster3a_2605 --model "$model" --dataset "$data" --out-dir "$out" --gpus 0 --rollout "$rollout" --bsz 512 --max-new-tokens "$maxnew" --ctx-limit "$ctx" --temperature 0.8 --top-p 0.35 --top-k 40 --seed 42 --prompt-style rwkv --torch-extensions-dir "$EXT" --progress-every 100 --verify-workers 8 > "$out/eval.log" 2>&1
  echo "DONE $tag $bench $(date)"
}
BASEMODEL=/hy-tmp/runs/g1k_dapo_pass1_31_officialprompt_q32x16_cand64_max6144_ob512_step900to1000_20261001_014411/final_step_1000.pth
SFTMODEL=/hy-tmp/runs/g1k_rl1000_openmath7752_sft_lr2e6_cosine_1ep_20261005/model_step3682_infer.pth
run_one baseline "$BASEMODEL" gsm8k /hy-tmp/data/gsm8k/gsm8k_test_albatross.jsonl 4 4096 8192
run_one baseline "$BASEMODEL" math500 /hy-tmp/data/math500/test.jsonl 8 4096 8192
run_one sft "$SFTMODEL" gsm8k /hy-tmp/data/gsm8k/gsm8k_test_albatross.jsonl 4 4096 8192
run_one sft "$SFTMODEL" math500 /hy-tmp/data/math500/test.jsonl 8 4096 8192
