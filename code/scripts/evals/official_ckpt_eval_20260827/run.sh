#!/usr/bin/env bash
set -euo pipefail
BASE=/hy-tmp/evals/official_ckpt_eval_20260827
EVAL=/root/eval_transfer/albatross/faster3a_2605/eval_math500.py
ALBATROSS=/root/eval_transfer/albatross/faster3a_2605
PY=/usr/local/bin/python3
EXT=/hy-tmp/torch_extensions_official_ckpt_eval_20260827
STATUS=$BASE/status.tsv
mkdir -p "$BASE" "$EXT"
: > "$STATUS"
export PATH=/root/miniconda3/bin:$PATH
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
export TORCH_EXTENSIONS_DIR="$EXT"
export HF_HOME=/hy-tmp/hf_home
export XDG_CACHE_HOME=/hy-tmp/xdg_cache

checkpoints=(
  /hy-tmp/runs/g1i_gsm8k_boxed_strictcot_maxnew2048_micro3_dyn512_q32_r16_200_earlystop_20260811_212644/final_step_85.pth
  /hy-tmp/runs/g1i_gsm8k_boxed_strictcot_continue_from_step85_maxnew2048_micro3_dyn512_q32_r16_100_20260812_082824/final_step_105.pth
  /hy-tmp/runs/g1i_gsm8k_boxed_strictcot_continue_from_step105_fixed50_maxnew2048_micro3_dyn512_q32_r16_20260812_142929/final_step_155.pth
  /hy-tmp/runs/g1i_step155_to_dapo_math17k_officialprompt_q32x16_cand64_max6144_ob512_100_20260820_154937/final_step_100.pth
  /hy-tmp/runs/g1i_step155_to_dapo_math17k_pass1_16_val500_max4096_micro1_fullmem_resetstep_200_20260819_1503/final_step_200.pth
  /hy-tmp/runs/g1i_step155_to_dapo_math17k_officialprompt_q32x16_cand64_max6144_ob512_continue200to300_eval512_20260822_073045/best_full_eval.pth
  /hy-tmp/runs/g1i_step300_to_dapo_math17k_officialprompt_q32x16_cand64_max6144_ob512_continue300to400_eval384_20260824_122246/final_step_400.pth
  /hy-tmp/runs/g1i_step400_to_dapo_math17k_officialprompt_q32x16_cand64_max6144_ob512_continue400to500_eval384_20260825_1355_fixed/final_step_500.pth
  /hy-tmp/runs/g1i_step500_to_dapo_math17k_officialprompt_q32x16_cand64_max6144_ob512_continue500to600_eval384_20260826_0000_fixed/final_step_600.pth
)

for ckpt in "${checkpoints[@]}"; do
  [ -f "$ckpt" ] || { echo "$(date '+%F %T')\tMISSING\t$ckpt" | tee -a "$STATUS"; continue; }
  name=$(basename "$(dirname "$ckpt")")__$(basename "$ckpt" .pth)
  for bench in math500 gsm8k; do
    if [[ "$bench" == math500 ]]; then
      dataset=/hy-tmp/data/math500/test.jsonl
      max_new=1500
      rollout=4
      bsz=512
      ext_tag=math500
    else
      dataset=/hy-tmp/data/gsm8k/gsm8k_test_albatross.jsonl
      max_new=1500
      rollout=4
      bsz=512
      ext_tag=gsm8k
    fi
    out="$BASE/$name/$bench"
    mkdir -p "$out"
    echo "$(date '+%F %T')\tSTART\t$name\t$bench\t$ckpt" | tee -a "$STATUS"
    "$PY" "$EVAL" \
      --albatross-dir "$ALBATROSS" \
      --model "$ckpt" \
      --dataset "$dataset" \
      --out-dir "$out" \
      --gpus 0 --rollout "$rollout" --bsz "$bsz" \
      --max-new-tokens "$max_new" --ctx-limit 8192 \
      --temperature 1.0 --top-p 0.28 --top-k 32 \
      --seed 42 --prompt-style fake_think \
      --torch-extensions-dir "$EXT" \
      --progress-every 100 --verify-workers 8 \
      2>&1 | tee "$out/eval.log"
    if [ -f "$out/summary.json" ]; then
      summary=$(python3 - <<'PY' "$out/summary.json"
import json,sys
p=sys.argv[1]
with open(p,'r',encoding='utf-8') as f:
    s=json.load(f)
print(f"strict={s.get('strict_accuracy')} loose={s.get('rollout_accuracy')} mean_tokens={s.get('mean_generated_tokens')}")
PY
)
      echo "$(date '+%F %T')\tDONE\t$name\t$bench\t$summary" | tee -a "$STATUS"
    else
      echo "$(date '+%F %T')\tDONE_NO_SUMMARY\t$name\t$bench" | tee -a "$STATUS"
    fi
  done
done

echo "$(date '+%F %T')\tALL_DONE" | tee -a "$STATUS"
