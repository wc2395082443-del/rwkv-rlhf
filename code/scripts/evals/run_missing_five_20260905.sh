#!/usr/bin/env bash
set -euo pipefail
BASE=/hy-tmp/evals/missing_five_benchmarks_checkpoints_20260905
EVAL=/hy-tmp/evals/official_flower_ckpt_eval_20260827/eval_math500_flower.py
ALBATROSS=/root/eval_transfer/albatross/faster3a_2605
PY=/usr/local/bin/python3
EXT=/hy-tmp/torch_extensions_missing_five_20260905
mkdir -p "$BASE" "$EXT"
export PATH=/root/miniconda3/bin:$PATH PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 TORCH_EXTENSIONS_DIR="$EXT" HF_HOME=/hy-tmp/hf_home XDG_CACHE_HOME=/hy-tmp/xdg_cache
declare -A M=(
[85]=/hy-tmp/runs/g1i_gsm8k_boxed_strictcot_maxnew2048_micro3_dyn512_q32_r16_200_earlystop_20260811_212644/final_step_85.pth
[100]=/hy-tmp/runs/g1i_step155_to_dapo_math17k_officialprompt_q32x16_cand64_max6144_ob512_100_20260820_154937/final_step_100.pth
[105]=/hy-tmp/runs/g1i_gsm8k_boxed_strictcot_continue_from_step85_maxnew2048_micro3_dyn512_q32_r16_100_20260812_082824/final_step_105.pth
[155]=/hy-tmp/runs/g1i_gsm8k_boxed_strictcot_continue_from_step105_fixed50_maxnew2048_micro3_dyn512_q32_r16_20260812_142929/final_step_155.pth
[200]=/hy-tmp/runs/g1i_step155_to_dapo_math17k_officialprompt_q32x16_cand64_max6144_ob512_continue100to200_eval512_20260821_131710/final_step_200.pth
[300]=/hy-tmp/runs/g1i_step155_to_dapo_math17k_officialprompt_q32x16_cand64_max6144_ob512_continue200to300_eval512_20260822_073045/best_full_eval.pth
[1050]=/hy-tmp/runs/g1i_step1000_to_dapo_math17k_officialprompt_q32x16_cand64_max6144_ob512_continue1000to1200_eval384_ckpt50_20260901_1100/checkpoint_step_1050.pth
[1100]=/hy-tmp/runs/g1i_step1000_to_dapo_math17k_officialprompt_q32x16_cand64_max6144_ob512_continue1000to1200_eval384_ckpt50_20260901_1100/checkpoint_step_1100.pth
[1150]=/hy-tmp/runs/g1i_step1000_to_dapo_math17k_officialprompt_q32x16_cand64_max6144_ob512_continue1000to1200_eval384_ckpt50_20260901_1100/checkpoint_step_1150.pth
[1200]=/hy-tmp/runs/g1i_step1000_to_dapo_math17k_officialprompt_q32x16_cand64_max6144_ob512_continue1000to1200_eval384_ckpt50_20260901_1100/checkpoint_step_1200.pth
[1250]=/hy-tmp/runs/g1i_step1200_to_dapo_math17k_officialprompt_q32x16_cand64_max6144_ob512_continue1200to1300_eval384_ckpt50_20260903_1600/checkpoint_step_1250.pth
[1300]=/hy-tmp/runs/g1i_step1200_to_dapo_math17k_officialprompt_q32x16_cand64_max6144_ob512_continue1200to1300_eval384_ckpt50_20260903_1600/checkpoint_step_1300.pth
)
declare -A D=([gsm8k]=/hy-tmp/data/gsm8k/gsm8k_test_albatross.jsonl [math500]=/hy-tmp/data/math500/test.jsonl [aime24]=/hy-tmp/data/official_math_extra/aime24.jsonl [aime25]=/hy-tmp/data/official_math_extra/aime25.jsonl [amc23]=/hy-tmp/data/official_math_extra/amc23.jsonl)
declare -A R=([gsm8k]=4 [math500]=8 [aime24]=64 [aime25]=64 [amc23]=64)
declare -A N=([gsm8k]=4096 [math500]=4096 [aime24]=8192 [aime25]=8192 [amc23]=8192)
declare -A C=([gsm8k]=8192 [math500]=8192 [aime24]=12288 [aime25]=12288 [amc23]=12288)
run_one() {
  local step=$1 bench=$2 out="$BASE/step$1/$2"; mkdir -p "$out"
  if [[ -s "$out/summary.json" ]]; then echo "SKIP_NEW_DONE $step $bench" >> "$BASE/status.tsv"; return; fi
  echo "START $(date '+%F %T') $step $bench" | tee -a "$BASE/status.tsv"
  if ! "$PY" "$EVAL" --albatross-dir "$ALBATROSS" --model "${M[$step]}" --dataset "${D[$bench]}" --out-dir "$out" --gpus 0 --rollout "${R[$bench]}" --bsz 512 --max-new-tokens "${N[$bench]}" --ctx-limit "${C[$bench]}" --temperature 0.8 --top-p 0.35 --top-k 40 --seed 42 --prompt-style official_flower --torch-extensions-dir "$EXT" --progress-every 100 --verify-workers 8 2>&1 | tee "$out/eval.log"; then
    echo "FAIL $(date '+%F %T') $step $bench" | tee -a "$BASE/status.tsv"
    return 0
  fi
  "$PY" - "$out/summary.json" "$step" "$bench" <<'PY' | tee -a "$BASE/status.tsv"
import json,sys
from datetime import datetime
p,step,bench=sys.argv[1:]
s=json.load(open(p,encoding="utf-8"))
print(f"DONE {datetime.now():%F %T} {step} {bench} avg={s.get('rollout_accuracy')} pass={s.get('pass_at_rollout_accuracy')} n={s.get('num_tasks')} tokens={s.get('mean_generated_tokens')}")
PY
}
for step in 1050 1100 1150 1250; do run_one "$step" gsm8k; done
for step in 1050 1100 1150 1200 1250; do run_one "$step" math500; done
for step in 85 100 105 155 200 300 1050 1100 1150 1200 1250 1300; do for bench in aime24 aime25 amc23; do run_one "$step" "$bench"; done; done
echo "ALL_MISSING_DONE $(date '+%F %T')" | tee -a "$BASE/status.tsv"
