#!/usr/bin/env bash
set -euo pipefail
while pgrep -f '/hy-tmp/run_g1k_gsm8k_avg4_20261002.sh' >/dev/null; do sleep 30; done
BASE=/hy-tmp/evals/g1k_math500_avg8_20261002
EVAL=/hy-tmp/evals/official_flower_ckpt_eval_20260827/eval_math500_flower.py
ALBATROSS=/root/eval_transfer/albatross/faster3a_2605
EXT=/hy-tmp/torch_extensions_g1k_math500_avg8
PY=/usr/local/bin/python3
export PATH=/root/miniconda3/bin:$PATH
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
export TORCH_EXTENSIONS_DIR=$EXT
export HF_HOME=/hy-tmp/hf_home
export XDG_CACHE_HOME=/hy-tmp/xdg_cache
mkdir -p "$BASE" "$EXT"
declare -A MODELS=(
[step0]=/hy-tmp/runs/rwkv-g1k-1b-temp-6003.pth
[step100]=/hy-tmp/runs/g1k_step155_dapo_pass1_31_officialprompt_q32x16_cand64_max6144_ob512_step0to100_20260920_113659/final_step_100.pth
[step200]=/hy-tmp/runs/g1k_dapo_pass1_31_officialprompt_q32x16_cand64_max6144_ob512_step100to200_20260921_062417/final_step_200.pth
[step300]=/hy-tmp/runs/g1k_dapo_pass1_31_officialprompt_q32x16_cand64_max6144_ob512_step200to300_20260922_093913/final_step_300.pth
[step400]=/hy-tmp/runs/g1k_dapo_pass1_31_officialprompt_q32x16_cand64_max6144_ob512_step300to400_20260923_070337/final_step_400.pth
[step500]=/hy-tmp/runs/g1k_dapo_pass1_31_officialprompt_q32x16_cand64_max6144_ob512_step400to500_20260924_051022/final_step_500.pth
[step600]=/hy-tmp/runs/g1k_dapo_pass1_31_officialprompt_q32x16_cand64_max6144_ob512_step500to600_20260925_211542/final_step_600.pth
[step700]=/hy-tmp/runs/g1k_dapo_pass1_31_officialprompt_q32x16_cand64_max6144_ob512_step600to700_20260926_224314/final_step_700.pth
[step800]=/hy-tmp/runs/g1k_dapo_pass1_31_officialprompt_q32x16_cand64_max6144_ob512_step700to800_20260928_071906/final_step_800.pth
[step900]=/hy-tmp/runs/g1k_dapo_pass1_31_officialprompt_q32x16_cand64_max6144_ob512_step800to900_20260929_172737/final_step_900.pth
[step1000]=/hy-tmp/runs/g1k_dapo_pass1_31_officialprompt_q32x16_cand64_max6144_ob512_step900to1000_20261001_014411/final_step_1000.pth
)
for step in step0 step100 step200 step300 step400 step500 step600 step700 step800 step900 step1000; do
 model=${MODELS[$step]}; out=$BASE/$step; mkdir -p "$out"
 [ -s "$out/summary.json" ] && continue
 echo "START $(date '+%F %T') $step rollout=8" | tee -a "$BASE/status.tsv"
 "$PY" "$EVAL" --albatross-dir "$ALBATROSS" --model "$model" --dataset /hy-tmp/data/math500/test.jsonl --out-dir "$out" --gpus 0 --rollout 8 --bsz 512 --max-new-tokens 4096 --ctx-limit 8192 --temperature 0.8 --top-p 0.35 --top-k 40 --seed 42 --prompt-style official_flower --torch-extensions-dir "$EXT" --progress-every 100 --verify-workers 8 2>&1 | tee "$out/eval.log"
 "$PY" - "$out/summary.json" "$step" <<'PY' | tee -a "$BASE/status.tsv"
import json,sys
from datetime import datetime
s=json.load(open(sys.argv[1],encoding='utf-8'))
print(f"DONE {datetime.now():%F %T} {sys.argv[2]} avg={s.get('rollout_accuracy')} pass={s.get('pass_at_rollout_accuracy')} n={s.get('num_tasks')} rollout={s.get('rollout')}")
PY
done
echo "ALL_DONE $(date '+%F %T')" | tee -a "$BASE/status.tsv"