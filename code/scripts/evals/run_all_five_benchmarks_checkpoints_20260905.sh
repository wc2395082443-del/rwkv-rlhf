#!/usr/bin/env bash
set -euo pipefail
BASE=/hy-tmp/evals/all_five_benchmarks_checkpoints_20260905
EVAL=/hy-tmp/evals/official_flower_ckpt_eval_20260827/eval_math500_flower.py
ALBATROSS=/root/eval_transfer/albatross/faster3a_2605
PY=/usr/local/bin/python3
EXT=/hy-tmp/torch_extensions_all_five_benchmarks_20260905
mkdir -p "$BASE" "$EXT"
export PATH=/root/miniconda3/bin:$PATH
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
export TORCH_EXTENSIONS_DIR="$EXT"
export HF_HOME=/hy-tmp/hf_home
export XDG_CACHE_HOME=/hy-tmp/xdg_cache

declare -A MODELS=(
  [step85]=/hy-tmp/runs/g1i_gsm8k_boxed_strictcot_maxnew2048_micro3_dyn512_q32_r16_200_earlystop_20260811_212644/final_step_85.pth
  [step100]=/hy-tmp/runs/g1i_step155_to_dapo_math17k_officialprompt_q32x16_cand64_max6144_ob512_100_20260820_154937/final_step_100.pth
  [step105]=/hy-tmp/runs/g1i_gsm8k_boxed_strictcot_continue_from_step85_maxnew2048_micro3_dyn512_q32_r16_100_20260812_082824/final_step_105.pth
  [step155]=/hy-tmp/runs/g1i_gsm8k_boxed_strictcot_continue_from_step105_fixed50_maxnew2048_micro3_dyn512_q32_r16_20260812_142929/final_step_155.pth
  [step200]=/hy-tmp/runs/g1i_step155_to_dapo_math17k_officialprompt_q32x16_cand64_max6144_ob512_continue100to200_eval512_20260821_131710/final_step_200.pth
  [step300]=/hy-tmp/runs/g1i_step155_to_dapo_math17k_officialprompt_q32x16_cand64_max6144_ob512_continue200to300_eval512_20260822_073045/best_full_eval.pth
  [step400]=/hy-tmp/runs/g1i_step300_to_dapo_math17k_officialprompt_q32x16_cand64_max6144_ob512_continue300to400_eval384_20260824_122246/final_step_400.pth
  [step500]=/hy-tmp/runs/g1i_step400_to_dapo_math17k_officialprompt_q32x16_cand64_max6144_ob512_continue400to500_eval384_20260825_1355_fixed/final_step_500.pth
  [step600]=/hy-tmp/runs/g1i_step500_to_dapo_math17k_officialprompt_q32x16_cand64_max6144_ob512_continue500to600_eval384_20260826_0000_fixed/final_step_600.pth
  [step800]=/hy-tmp/runs/g1i_step600_to_dapo_math17k_officialprompt_q32x16_cand64_max6144_ob512_continue600to800_eval384_20260827/final_step_800.pth
  [step850]=/hy-tmp/runs/g1i_step800_to_dapo_math17k_officialprompt_q32x16_cand64_max6144_ob512_continue800to1000_eval384_ckpt50_20260829_1810/checkpoint_step_850.pth
  [step900]=/hy-tmp/runs/g1i_step800_to_dapo_math17k_officialprompt_q32x16_cand64_max6144_ob512_continue800to1000_eval384_ckpt50_20260829_1810/checkpoint_step_900.pth
  [step950]=/hy-tmp/runs/g1i_step800_to_dapo_math17k_officialprompt_q32x16_cand64_max6144_ob512_continue800to1000_eval384_ckpt50_20260829_1810/checkpoint_step_950.pth
  [step1000]=/hy-tmp/runs/g1i_step800_to_dapo_math17k_officialprompt_q32x16_cand64_max6144_ob512_continue800to1000_eval384_ckpt50_20260829_1810/checkpoint_step_1000.pth
  [step1050]=/hy-tmp/runs/g1i_step1000_to_dapo_math17k_officialprompt_q32x16_cand64_max6144_ob512_continue1000to1200_eval384_ckpt50_20260901_1100/checkpoint_step_1050.pth
  [step1100]=/hy-tmp/runs/g1i_step1000_to_dapo_math17k_officialprompt_q32x16_cand64_max6144_ob512_continue1000to1200_eval384_ckpt50_20260901_1100/checkpoint_step_1100.pth
  [step1150]=/hy-tmp/runs/g1i_step1000_to_dapo_math17k_officialprompt_q32x16_cand64_max6144_ob512_continue1000to1200_eval384_ckpt50_20260901_1100/checkpoint_step_1150.pth
  [step1200]=/hy-tmp/runs/g1i_step1000_to_dapo_math17k_officialprompt_q32x16_cand64_max6144_ob512_continue1000to1200_eval384_ckpt50_20260901_1100/checkpoint_step_1200.pth
  [step1250]=/hy-tmp/runs/g1i_step1200_to_dapo_math17k_officialprompt_q32x16_cand64_max6144_ob512_continue1200to1300_eval384_ckpt50_20260903_1600/checkpoint_step_1250.pth
  [step1300]=/hy-tmp/runs/g1i_step1200_to_dapo_math17k_officialprompt_q32x16_cand64_max6144_ob512_continue1200to1300_eval384_ckpt50_20260903_1600/checkpoint_step_1300.pth
)

declare -A DATASETS=(
  [gsm8k]=/hy-tmp/data/gsm8k/gsm8k_test_albatross.jsonl
  [math500]=/hy-tmp/data/math500/test.jsonl
  [aime24]=/hy-tmp/data/official_math_extra/aime24.jsonl
  [aime25]=/hy-tmp/data/official_math_extra/aime25.jsonl
  [amc23]=/hy-tmp/data/official_math_extra/amc23.jsonl
)
declare -A ROLLOUTS=([gsm8k]=4 [math500]=8 [aime24]=64 [aime25]=64 [amc23]=64)
declare -A MAXNEWS=([gsm8k]=4096 [math500]=4096 [aime24]=8192 [aime25]=8192 [amc23]=8192)
declare -A CTXS=([gsm8k]=8192 [math500]=8192 [aime24]=12288 [aime25]=12288 [amc23]=12288)

for step in step85 step100 step105 step155 step200 step300 step400 step500 step600 step800 step850 step900 step950 step1000 step1050 step1100 step1150 step1200 step1250 step1300; do
  model=${MODELS[$step]}
  [[ -f "$model" ]] || { echo "MISSING_MODEL $step $model" | tee -a "$BASE/status.tsv"; continue; }
  for bench in gsm8k math500 aime24 aime25 amc23; do
    out="$BASE/$step/$bench"
    mkdir -p "$out"
    if [[ -s "$out/summary.json" ]]; then echo "SKIP_DONE $step $bench" | tee -a "$BASE/status.tsv"; continue; fi
    echo "START $(date '+%F %T') $step $bench" | tee -a "$BASE/status.tsv"
    "$PY" "$EVAL" --albatross-dir "$ALBATROSS" --model "$model" --dataset "${DATASETS[$bench]}" --out-dir "$out" --gpus 0 --rollout "${ROLLOUTS[$bench]}" --bsz 512 --max-new-tokens "${MAXNEWS[$bench]}" --ctx-limit "${CTXS[$bench]}" --temperature 0.8 --top-p 0.35 --top-k 40 --seed 42 --prompt-style official_flower --torch-extensions-dir "$EXT" --progress-every 100 --verify-workers 8 2>&1 | tee "$out/eval.log"
    "$PY" - "$out/summary.json" "$step" "$bench" <<'PY' | tee -a "$BASE/status.tsv"
import json, sys
from datetime import datetime
path, step, bench = sys.argv[1:]
s = json.load(open(path, encoding="utf-8"))
print(f"DONE {datetime.now():%F %T} {step} {bench} avg={s.get('rollout_accuracy')} pass={s.get('pass_at_rollout_accuracy')} n={s.get('num_tasks')} tokens={s.get('mean_generated_tokens')}")
PY
  done
done
echo "ALL_DONE $(date '+%F %T')" | tee -a "$BASE/status.tsv"