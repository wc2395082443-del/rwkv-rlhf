#!/usr/bin/env bash
set -euo pipefail
cd /root/eval_transfer
/hy-tmp/runs/eval_gsm8k_rwkvskills_official_avg4_base_vs_step155_20260814_223739/gpu_monitor.sh /hy-tmp/runs/eval_gsm8k_rwkvskills_official_avg4_base_vs_step155_20260814_223739/gpu_monitor.csv & echo $! > /hy-tmp/runs/eval_gsm8k_rwkvskills_official_avg4_base_vs_step155_20260814_223739/gpu_monitor.pid
cleanup(){ if [ -f /hy-tmp/runs/eval_gsm8k_rwkvskills_official_avg4_base_vs_step155_20260814_223739/gpu_monitor.pid ]; then kill $(cat /hy-tmp/runs/eval_gsm8k_rwkvskills_official_avg4_base_vs_step155_20260814_223739/gpu_monitor.pid) 2>/dev/null || true; fi; }
trap cleanup EXIT
python3 /root/eval_transfer/eval_gsm8k_rwkvskills_official_avg4.py   --model /root/eval_transfer/models/rwkv7-g1i_preview5445-1.5b-20260729-ctx16384.pth   --dataset /root/eval_transfer/data/gsm8k_test_formatted.jsonl   --out-dir /hy-tmp/runs/eval_gsm8k_rwkvskills_official_avg4_base_vs_step155_20260814_223739/base   --albatross-dir /root/eval_transfer/albatross/faster3a_2605   --batch-size 512 --num-samples 4 --seed 42   > /hy-tmp/runs/eval_gsm8k_rwkvskills_official_avg4_base_vs_step155_20260814_223739/base.log 2>&1
python3 /root/eval_transfer/eval_gsm8k_rwkvskills_official_avg4.py   --model /hy-tmp/runs/g1i_gsm8k_boxed_strictcot_continue_from_step105_fixed50_maxnew2048_micro3_dyn512_q32_r16_20260812_142929/final_step_155.pth   --dataset /root/eval_transfer/data/gsm8k_test_formatted.jsonl   --out-dir /hy-tmp/runs/eval_gsm8k_rwkvskills_official_avg4_base_vs_step155_20260814_223739/step155   --albatross-dir /root/eval_transfer/albatross/faster3a_2605   --batch-size 512 --num-samples 4 --seed 42   > /hy-tmp/runs/eval_gsm8k_rwkvskills_official_avg4_base_vs_step155_20260814_223739/step155.log 2>&1
python3 - <<'PY' > /hy-tmp/runs/eval_gsm8k_rwkvskills_official_avg4_base_vs_step155_20260814_223739/gpu_summary.json
import csv,json,statistics as st, pathlib
p=pathlib.Path('/hy-tmp/runs/eval_gsm8k_rwkvskills_official_avg4_base_vs_step155_20260814_223739/gpu_monitor.csv')
rows=[]
for r in csv.DictReader(p.open()):
    try:
        rows.append({k:float(v.strip()) if k!='ts' else v for k,v in r.items()})
    except Exception: pass
def stat(k):
    vals=[x[k] for x in rows]
    return {'avg':sum(vals)/len(vals) if vals else 0,'peak':max(vals) if vals else 0}
print(json.dumps({'samples':len(rows),'util_gpu':stat('util_gpu'),'mem_used_mb':stat('mem_used_mb'),'power_w':stat('power_w')},indent=2))
PY
