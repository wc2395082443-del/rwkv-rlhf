#!/usr/bin/env bash
set -euo pipefail
cd /hy-tmp/RWKV7-g1i-strict-quality-gate-adamw-nooffload-20260810
OUT=$(readlink -f /hy-tmp/runs/latest_eval_g1i_gsm8k_with_responses)
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CUDA_VISIBLE_DEVICES=0 python3 eval_gsm8k_boxed_strict_reward.py \
  --model /hy-tmp/models/rwkv7-g1i_preview5445-1.5b-20260729-ctx16384.pth \
  --tokenizer /hy-tmp/RWKV-v7/rwkv_vocab_v20230424.txt \
  --eval_jsonl /hy-tmp/data/gsm8k/gsm8k_test_albatross.jsonl \
  --out_dir "$OUT/base_full" --limit 0 \
  --max_new_tokens 2048 --temperature 0.3 --top_p 0.4 --top_k 500 --batch 64 \
  2>&1 | tee "$OUT/base_full.log"
CUDA_VISIBLE_DEVICES=0 python3 eval_gsm8k_boxed_strict_reward.py \
  --model /hy-tmp/runs/g1i_gsm8k_boxed_strictcot_maxnew2048_micro3_dyn512_q32_r16_200_earlystop_20260811_212644/final_step_85.pth \
  --tokenizer /hy-tmp/RWKV-v7/rwkv_vocab_v20230424.txt \
  --eval_jsonl /hy-tmp/data/gsm8k/gsm8k_test_albatross.jsonl \
  --out_dir "$OUT/step85_full" --limit 0 \
  --max_new_tokens 2048 --temperature 0.3 --top_p 0.4 --top_k 500 --batch 64 \
  2>&1 | tee "$OUT/step85_full.log"
python3 - <<'PY'
import json, pathlib
out=pathlib.Path('/hy-tmp/runs/latest_eval_g1i_gsm8k_with_responses').resolve()
b=json.loads((out/'base_full/summary.json').read_text())
c=json.loads((out/'step85_full/summary.json').read_text())
keys=['loose_accuracy','strict_accuracy','strict_format_rate','eod_rate','truncated_rate','repeat_rate','multilingual_rate','degeneration_rate','mean_generated_tokens']
res={'base':b,'checkpoint':c,'gain':{k:c[k]-b[k] for k in keys}}
(out/'compare_summary.json').write_text(json.dumps(res,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(res,ensure_ascii=False,indent=2))
PY
