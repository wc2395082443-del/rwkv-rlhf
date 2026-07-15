#!/usr/bin/env bash
set -euo pipefail

ROOT_OUT=${1:-/root/autodl-tmp/gsm8k_progress_advantage_paper_$(date +%Y%m%d_%H%M%S)}
WAIT_PID=${WAIT_PID:-}
BASE_MODEL=${BASE_MODEL:-/root/autodl-tmp/rwkv_models/ms_g1_1p5b/rwkv7-g1f-1.5b-20260419-ctx8192.pth}
GRPO_RUNNER=${GRPO_RUNNER:-/root/autodl-tmp/run_gsm8k_best_grpo_dynamic_rescreen_g1f.sh}
PA_SCRIPT=${PA_SCRIPT:-/root/autodl-tmp/gsm8k_progress_advantage_eval.py}
PY=${PY:-/root/miniconda3/bin/python}
EVAL_JSONL=${EVAL_JSONL:-/root/RWKV-LM/RWKV7-statetuning/gsm8k_test_formatted.jsonl}
TRAIN_JSONL=${TRAIN_JSONL:-/root/RWKV-LM/RWKV7-statetuning/gsm8k_train_formatted.jsonl}

mkdir -p "$ROOT_OUT"
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$ROOT_OUT/master.log"; }

log root_out="$ROOT_OUT"
log base_model="$BASE_MODEL"
log grpo_runner="$GRPO_RUNNER"
log pa_script="$PA_SCRIPT"
log eval_jsonl="$EVAL_JSONL"

for f in "$BASE_MODEL" "$GRPO_RUNNER" "$PA_SCRIPT" "$EVAL_JSONL" "$TRAIN_JSONL"; do
  if [[ ! -e "$f" ]]; then
    log "missing required file: $f"
    exit 1
  fi
done

if [[ -n "$WAIT_PID" ]]; then
  while kill -0 "$WAIT_PID" 2>/dev/null; do
    log "waiting for previous experiment pid=$WAIT_PID"
    sleep 60
  done
fi

while true; do
  ACTIVE=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -E '^[0-9]+' | wc -l || true)
  if [[ "$ACTIVE" -eq 0 ]]; then
    break
  fi
  log "waiting for gpu idle active_cuda_processes=$ACTIVE"
  sleep 60
done

GRPO_OUT="$ROOT_OUT/grpo_gsm8k_dynamic_rescreen"
log "starting gsm8k grpo: $GRPO_OUT"
STAGES=2 WAIT_PID= "$GRPO_RUNNER" "$GRPO_OUT" |& tee "$ROOT_OUT/grpo_stdout.log"

FINAL_CKPT="$GRPO_OUT/stage_2/run/final_step_100.pth"
if [[ ! -f "$FINAL_CKPT" ]]; then
  log "missing final checkpoint: $FINAL_CKPT"
  exit 1
fi
log "final checkpoint: $FINAL_CKPT"

PA_OUT="$ROOT_OUT/pa_full_stage2_final_vs_g1f"
log "starting progress advantage full gsm8k: $PA_OUT"
$PY "$PA_SCRIPT" \
  --policy_model "$FINAL_CKPT" \
  --ref_model "$BASE_MODEL" \
  --train_jsonl "$TRAIN_JSONL" \
  --eval_jsonl "$EVAL_JSONL" \
  --out_dir "$PA_OUT" \
  --group_size 8 \
  --max_new_tokens 768 \
  --temperature 1.0 \
  --top_p 0.6 \
  --top_k 0 \
  --chunk_questions 4 \
  --score_micro_batch 2 \
  --save_text 1 \
  |& tee "$ROOT_OUT/pa_stdout.log"

log "done"
