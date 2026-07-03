#!/usr/bin/env bash
set -euo pipefail

TS=$(date +%Y%m%d_%H%M%S)
ROOT=/root/autodl-tmp/log/math500_llama_threeway_dyn_fast_${TS}
mkdir -p "$ROOT"

echo "root_log_dir=$ROOT"

COMMON_ARGS=(
  --train_jsonl /root/autodl-tmp/data/math500/train.jsonl
  --eval_jsonl /root/autodl-tmp/data/math500/test.jsonl
  --model /root/autodl-tmp/models/Llama-3.2-3B-Instruct
  --answer_judge math_verify
  --total_steps 300
  --save_interval 50
  --eval_interval 0
  --skip_pre_eval
  --num_questions 12
  --samples_per_question 8
  --hard_buffer_target_samples 0
  --max_new_tokens 768
  --gen_batch_size 32
  --eval_gen_batch_size 64
  --pre_eval_gen_batch_size 64
  --micro_batch 2
  --lr 6e-5
  --neg_adv_weight 0.6
  --dynamic_resample_enable 1
  --dynamic_min_effective_groups 6
  --dynamic_max_rounds 3
  --dynamic_unique_questions 1
)

run_one () {
  local name="$1"
  local workdir="$2"
  shift 2
  local out="$ROOT/$name"
  mkdir -p "$out"
  echo "[$(date '+%F %T')] start $name"
  (
    cd "$workdir"
    /root/miniconda3/bin/python3 main.py \
      "${COMMON_ARGS[@]}" \
      --out_dir "$out" \
      "$@"
  ) > "$out/train_stdout.log" 2>&1
  echo "[$(date '+%F %T')] done $name"
}

run_one baseline /root/RWKV-LM/llama_grpo_baseline_v1
run_one real /root/RWKV-LM/llama_math500_real_v1 --policy_objective real
run_one gradalign /root/RWKV-LM/llama_math500_gradalign_v1 --ga_enable

echo "[$(date '+%F %T')] all_done"
