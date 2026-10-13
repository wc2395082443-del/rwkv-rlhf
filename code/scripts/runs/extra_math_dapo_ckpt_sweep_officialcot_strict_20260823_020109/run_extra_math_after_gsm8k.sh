#!/usr/bin/env bash
set -euo pipefail
GSM=/hy-tmp/runs/gsm8k_dapo_ckpt_sweep_officialcot_strict_20260823_014110
OUT_DIR=$(cd $(dirname $0) && pwd)
DATA_ROOT=/hy-tmp/data/official_math_extra
mkdir -p $DATA_ROOT $OUT_DIR
if [ -f $GSM/pid ]; then
  gp=$(cat $GSM/pid)
  echo [$(date '+%F %T')] waiting gsm8k pid=$gp | tee -a $OUT_DIR/sweep.log
  while ps -p $gp >/dev/null 2>&1; do sleep 30; done
fi
python3 - <<'PY'
import json, pathlib, math
root=pathlib.Path('/hy-tmp/data/official_math_extra')
root.mkdir(parents=True, exist_ok=True)
# AIME official static rows from rwkv-skills -> eval script schema.
for name, src in {
    'aime24': '/hy-tmp/rwkv-skills-official/data/aime24/test.jsonl',
    'aime25': '/hy-tmp/rwkv-skills-official/data/aime25/test.jsonl',
}.items():
    out=root/f'{name}.jsonl'
    n=0
    with open(src, encoding='utf-8') as f, out.open('w', encoding='utf-8') as g:
        for i,line in enumerate(f):
            if not line.strip(): continue
            o=json.loads(line)
            q=o.get('problem') or o.get('question')
            a=o.get('answer') or o.get('expected_answer')
            g.write(json.dumps({'problem': str(q), 'answer': str(a), 'subject': name, 'unique_id': str(o.get('id', i))}, ensure_ascii=False)+'\n')
            n+=1
    print(name, n, out)
# AMC23 from Qwen2.5-Math evaluation data uploaded to /tmp.
out=root/'amc23.jsonl'
n=0
with open('/tmp/amc23_test_qwen.jsonl', encoding='utf-8') as f, out.open('w', encoding='utf-8') as g:
    for i,line in enumerate(f):
        if not line.strip(): continue
        o=json.loads(line)
        a=o.get('answer')
        if isinstance(a, float) and a.is_integer(): a=int(a)
        g.write(json.dumps({'problem': str(o.get('problem') or o.get('question')), 'answer': str(a), 'subject': 'amc23', 'unique_id': str(o.get('id', i))}, ensure_ascii=False)+'\n')
        n+=1
print('amc23', n, out)
PY
export PYTHONPATH=/root/eval_transfer/albatross/faster3a_2605:/hy-tmp/RWKV7-g1i-strict-quality-gate-adamw-nooffload-20260810:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=0
ALB=/root/eval_transfer/albatross/faster3a_2605
run_one() {
  local dataset=$1; local data=$2; local name=$3; local ckpt=$4
  local dst=$OUT_DIR/$dataset/$name
  mkdir -p $dst
  echo [$(date '+%F %T')] START dataset=$dataset ckpt=$name | tee -a $OUT_DIR/sweep.log
  python3 /tmp/eval_math500_official_cot_prompt.py \
    --albatross-dir $ALB \
    --model $ckpt \
    --dataset $data \
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
    --progress-every 10 \
    --verify-workers 16 2>&1 | tee $dst/eval.log
  echo [$(date '+%F %T')] DONE dataset=$dataset ckpt=$name | tee -a $OUT_DIR/sweep.log
}
for dataset in aime24 aime25 amc23; do
  data=$DATA_ROOT/$dataset.jsonl
  run_one $dataset $data gsm8k_step155 /hy-tmp/runs/g1i_gsm8k_boxed_strictcot_continue_from_step105_fixed50_maxnew2048_micro3_dyn512_q32_r16_20260812_142929/final_step_155.pth
  run_one $dataset $data dapo_step100_final /hy-tmp/runs/g1i_step155_to_dapo_math17k_officialprompt_q32x16_cand64_max6144_ob512_100_20260820_154937/final_step_100.pth
  run_one $dataset $data dapo_step100_best /hy-tmp/runs/g1i_step155_to_dapo_math17k_officialprompt_q32x16_cand64_max6144_ob512_100_20260820_154937/best_full_eval.pth
  run_one $dataset $data dapo_step200_final /hy-tmp/runs/g1i_step155_to_dapo_math17k_officialprompt_q32x16_cand64_max6144_ob512_continue100to200_eval512_20260821_131710/final_step_200.pth
  run_one $dataset $data dapo_step200_best /hy-tmp/runs/g1i_step155_to_dapo_math17k_officialprompt_q32x16_cand64_max6144_ob512_continue100to200_eval512_20260821_131710/best_full_eval.pth
  run_one $dataset $data dapo_step300_best /hy-tmp/runs/g1i_step155_to_dapo_math17k_officialprompt_q32x16_cand64_max6144_ob512_continue200to300_eval512_20260822_073045/best_full_eval.pth
done
