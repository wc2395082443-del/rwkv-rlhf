#!/usr/bin/env bash
set -u
ROOT=/hy-tmp/evals/pass32_zero_step1200_20260903
CHUNKS=$ROOT/chunks
MODEL=/hy-tmp/runs/g1i_step1000_to_dapo_math17k_officialprompt_q32x16_cand64_max6144_ob512_continue1000to1200_eval384_ckpt50_20260901_1100/checkpoint_step_1200.pth
ALBATROSS=/root/eval_transfer/albatross/faster3a_2605
export PYTHONPATH=$ALBATROSS:/hy-tmp/RWKV7-g1i-strict-quality-gate-adamw-nooffload-20260810:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$CHUNKS"
printf '%s CHUNKED_START\n' "$(date '+%F %T')" | tee "$ROOT/status.log"
for d in "$CHUNKS"/chunk_*; do
  [ -d "$d" ] || continue
  if [ -s "$d/summary.json" ]; then
    printf '%s SKIP %s\n' "$(date '+%F %T')" "$(basename "$d")" | tee -a "$ROOT/status.log"
    continue
  fi
  printf '%s START %s\n' "$(date '+%F %T')" "$(basename "$d")" | tee -a "$ROOT/status.log"
  python3 /tmp/eval_math500_official_cot_prompt.py \
   --albatross-dir "$ALBATROSS" --model "$MODEL" --dataset "$d/dataset.jsonl" --out-dir "$d" \
   --gpus 0 --rollout 1 --bsz 512 --max-new-tokens 8192 --ctx-limit 12288 \
   --eval-tail-max-tokens 0 --temperature 0.8 --top-p 0.35 --top-k 40 \
   --prompt-style official_cot --progress-every 100 --verify-workers 16 2>&1 | tee "$d/eval.log"
  rc=${PIPESTATUS[0]}
  if [ "$rc" -ne 0 ]; then printf '%s FAILED %s rc=%s\n' "$(date '+%F %T')" "$(basename "$d")" "$rc" | tee -a "$ROOT/status.log"; exit "$rc"; fi
  printf '%s DONE %s\n' "$(date '+%F %T')" "$(basename "$d")" | tee -a "$ROOT/status.log"
done
python3 - <<'PY'
import json,glob,os
root='/hy-tmp/evals/pass32_zero_step1200_20260903'
paths=sorted(glob.glob(root+'/chunks/chunk_*/generations.jsonl'))
out=root+'/generations.jsonl'
total=correct=0
with open(out,'w',encoding='utf-8') as fo:
 for p in paths:
  for line in open(p,encoding='utf-8'):
   x=json.loads(line); fo.write(line); total+=1; correct+=int(bool(x.get('correct')))
summary={'num_tasks':total,'rollout':1,'total_generations':total,'correct_generations':correct,'rollout_accuracy':correct/max(1,total),'pass_at_rollout_accuracy':correct/max(1,total),'chunks':len(paths)}
json.dump(summary,open(root+'/summary.json','w',encoding='utf-8'),ensure_ascii=False,indent=2)
print('RESULT',json.dumps(summary))
PY
printf '%s ALL_DONE\n' "$(date '+%F %T')" | tee -a "$ROOT/status.log"