#!/usr/bin/env bash
set -euo pipefail

TRAIN_PID=80987
G1K_STAGE5=/hy-tmp/runs/g1k_dapo_pass1_31_officialprompt_q32x16_cand64_max6144_ob512_step400to500_20260924_051022
while kill -0 "$TRAIN_PID" 2>/dev/null; do sleep 30; done
while [ ! -f "$G1K_STAGE5/final_step_500.pth" ]; do sleep 30; done

CODE=/hy-tmp/RWKV7-g1i-strict-quality-gate-adamw-nooffload-20260810
EVAL="$CODE/eval_thinking_cot_officialprompt.py"
DATA_ROOT=/hy-tmp/data/openr1_dapo_math17k_processed/g1i_g1k_shared_questions_20260924
OUT_ROOT=/hy-tmp/evals/g1i_g1k_shared_checkpoint_sweep_20260924
mkdir -p "$OUT_ROOT"
export CUDA_VISIBLE_DEVICES=0
export TORCH_EXTENSIONS_DIR=/hy-tmp/torch_extensions_g1i_strict_quality_adamw_nooffload
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

labels=(
  g1i_step0 g1i_step100 g1i_step200 g1i_step300 g1i_step400 g1i_step500
  g1k_step0 g1k_step100 g1k_step200 g1k_step300 g1k_step400 g1k_step500
)
models=(
  /hy-tmp/runs/g1i_gsm8k_boxed_strictcot_continue_from_step105_fixed50_maxnew2048_micro3_dyn512_q32_r16_20260812_142929/final_step_155.pth
  /hy-tmp/runs/g1i_step155_to_dapo_math17k_officialprompt_q32x16_cand64_max6144_ob512_100_20260820_154937/final_step_100.pth
  /hy-tmp/runs/g1i_step155_to_dapo_math17k_officialprompt_q32x16_cand64_max6144_ob512_continue100to200_eval512_20260821_131710/final_step_200.pth
  /hy-tmp/runs/g1i_step155_to_dapo_math17k_officialprompt_q32x16_cand64_max6144_ob512_continue200to300_eval512_20260822_073045/best_full_eval.pth
  /hy-tmp/runs/g1i_step300_to_dapo_math17k_officialprompt_q32x16_cand64_max6144_ob512_continue300to400_eval384_20260824_122246/final_step_400.pth
  /hy-tmp/runs/g1i_step400_to_dapo_math17k_officialprompt_q32x16_cand64_max6144_ob512_continue400to500_eval384_20260825_1355_fixed/final_step_500.pth
  /hy-tmp/runs/g1k_temp6003_gsm8k_boxed_strictcot_q8x16_cand32_max2048_step0to155_20260917_020430/final_step_155.pth
  /hy-tmp/runs/g1k_step155_dapo_pass1_31_officialprompt_q32x16_cand64_max6144_ob512_step0to100_20260920_113659/final_step_100.pth
  /hy-tmp/runs/g1k_dapo_pass1_31_officialprompt_q32x16_cand64_max6144_ob512_step100to200_20260921_062417/final_step_200.pth
  /hy-tmp/runs/g1k_dapo_pass1_31_officialprompt_q32x16_cand64_max6144_ob512_step200to300_20260922_093913/final_step_300.pth
  /hy-tmp/runs/g1k_dapo_pass1_31_officialprompt_q32x16_cand64_max6144_ob512_step300to400_20260923_070337/final_step_400.pth
  "$G1K_STAGE5/final_step_500.pth"
)

cd "$CODE"
for dataset_name in heldout_by_both_42 train_by_both_2655; do
  dataset="$DATA_ROOT/${dataset_name}.jsonl"
  for index in "${!labels[@]}"; do
    label="${labels[$index]}"
    model="${models[$index]}"
    out="$OUT_ROOT/$dataset_name/$label"
    if [ -f "$out/summary.json" ]; then
      continue
    fi
    mkdir -p "$out"
    echo "[$(date '+%F %T')] START dataset=$dataset_name model=$label" | tee -a "$OUT_ROOT/progress.log"
    python3 "$EVAL" \
      --model "$model" --dataset "$dataset" --out-dir "$out" --gpus 0 \
      --rollout 1 --bsz 512 --max-new-tokens 6144 --eval-tail-max-tokens 0 --ctx-limit 8192 \
      --temperature 0.8 --top-p 0.35 --top-k 40 --seed 42 --prompt-style official \
      --torch-extensions-dir "$TORCH_EXTENSIONS_DIR" --progress-every 20 --verify-workers 8 \
      > "$out/stdout.log" 2>&1
    echo "[$(date '+%F %T')] DONE dataset=$dataset_name model=$label" | tee -a "$OUT_ROOT/progress.log"
  done
done
