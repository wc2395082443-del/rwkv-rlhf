#!/usr/bin/env bash
set -euo pipefail
CODE=/hy-tmp/RWKV7-g1i-strict-quality-gate-adamw-nooffload-20260810
ALB=/root/eval_transfer/albatross/faster3a_2605
export PYTHONPATH=$ALB:$CODE:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=0
run_one() {
  name=$1
  model=$2
  out=/hy-tmp/runs/math500_g1i_step155_step200_official_cot_params_strict_20260820_145238/$name
  mkdir -p $out
  echo === RUN $name === $(date) | tee -a /hy-tmp/runs/math500_g1i_step155_step200_official_cot_params_strict_20260820_145238/progress.log
  python3 /tmp/eval_math500_official_cot_prompt.py \
    --albatross-dir $ALB \
    --model $model \
    --dataset /hy-tmp/data/math500/test.jsonl \
    --out-dir $out \
    --gpus 0 \
    --rollout 1 \
    --bsz 512 \
    --max-new-tokens 4096 \
    --eval-tail-max-tokens 0 \
    --temperature 0.8 \
    --top-p 0.35 \
    --top-k 40 \
    --prompt-style official_cot \
    --progress-every 25 \
    --verify-workers 16 2>&1 | tee $out/run.log
  echo === DONE $name === $(date) | tee -a /hy-tmp/runs/math500_g1i_step155_step200_official_cot_params_strict_20260820_145238/progress.log
}
run_one step155 /hy-tmp/runs/g1i_gsm8k_boxed_strictcot_continue_from_step105_fixed50_maxnew2048_micro3_dyn512_q32_r16_20260812_142929/final_step_155.pth
run_one step200 /hy-tmp/runs/g1i_step155_to_dapo_math17k_pass1_16_val500_max4096_micro1_fullmem_resetstep_200_20260819_1503/final_step_200.pth
python3 - <<PY
import json, pathlib
base=pathlib.Path('/hy-tmp/runs/math500_g1i_step155_step200_official_cot_params_strict_20260820_145238')
rows=[]
for name in ['step155','step200']:
    p=base/name/'summary.json'
    if p.exists():
        d=json.loads(p.read_text())
        rows.append({k:d.get(k) for k in ['num_tasks','rollout','correct_generations','rollout_accuracy','truncated_rate','mean_generated_tokens','elapsed_sec','token_per_sec'] } | {'name':name})
(base/'summary_compare.json').write_text(json.dumps(rows,indent=2))
print(json.dumps(rows,indent=2))
PY
