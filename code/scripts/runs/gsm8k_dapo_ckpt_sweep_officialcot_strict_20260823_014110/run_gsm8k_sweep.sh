#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=/root/eval_transfer/albatross/faster3a_2605:/hy-tmp/RWKV7-g1i-strict-quality-gate-adamw-nooffload-20260810:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=0
OUT_DIR=$(cd $(dirname $0) && pwd)
DATA=/hy-tmp/data/gsm8k/gsm8k_test_albatross.jsonl
ALB=/root/eval_transfer/albatross/faster3a_2605
run_one() {
  local name=$1
  local ckpt=$2
  local dst=$OUT_DIR/$name
  mkdir -p $dst
  echo [$(date '+%F %T')] START $name $ckpt | tee -a $OUT_DIR/sweep.log
  python3 /tmp/eval_math500_official_cot_prompt.py \
    --albatross-dir $ALB \
    --model $ckpt \
    --dataset $DATA \
    --out-dir $dst \
    --gpus 0 \
    --rollout 1 \
    --bsz 512 \
    --max-new-tokens 4096 \
    --eval-tail-max-tokens 0 \
    --temperature 0.8 \
    --top-p 0.35 \
    --top-k 40 \
    --prompt-style official_cot \
    --progress-every 50 \
    --verify-workers 16 2>&1 | tee $dst/eval.log
  echo [$(date '+%F %T')] DONE $name | tee -a $OUT_DIR/sweep.log
}
run_one gsm8k_step155 /hy-tmp/runs/g1i_gsm8k_boxed_strictcot_continue_from_step105_fixed50_maxnew2048_micro3_dyn512_q32_r16_20260812_142929/final_step_155.pth
run_one dapo_step100_final /hy-tmp/runs/g1i_step155_to_dapo_math17k_officialprompt_q32x16_cand64_max6144_ob512_100_20260820_154937/final_step_100.pth
run_one dapo_step100_best /hy-tmp/runs/g1i_step155_to_dapo_math17k_officialprompt_q32x16_cand64_max6144_ob512_100_20260820_154937/best_full_eval.pth
run_one dapo_step200_final /hy-tmp/runs/g1i_step155_to_dapo_math17k_officialprompt_q32x16_cand64_max6144_ob512_continue100to200_eval512_20260821_131710/final_step_200.pth
run_one dapo_step200_best /hy-tmp/runs/g1i_step155_to_dapo_math17k_officialprompt_q32x16_cand64_max6144_ob512_continue100to200_eval512_20260821_131710/best_full_eval.pth
run_one dapo_step300_best /hy-tmp/runs/g1i_step155_to_dapo_math17k_officialprompt_q32x16_cand64_max6144_ob512_continue200to300_eval512_20260822_073045/best_full_eval.pth
