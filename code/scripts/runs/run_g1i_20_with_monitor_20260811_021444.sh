#!/usr/bin/env bash
set -euo pipefail
MON=/hy-tmp/runs/g1i_20_gpu_monitor_$(date +%Y%m%d_%H%M%S).csv
echo "$MON" > /hy-tmp/runs/latest_g1i_20.monitor
( while true; do
    date +%s.%N | tr -d '\n' >> "$MON"; echo -n ',' >> "$MON";
    nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,power.draw --format=csv,noheader,nounits >> "$MON" || true;
    sleep 1;
  done ) &
MPID=$!
set +e
bash /hy-tmp/runs/launch_g1i_strict_quality_fp32master_nooffload_histalign_micro1_20_v3_l40s.sh
RC=$?
kill $MPID 2>/dev/null || true
OUT=$(cat /hy-tmp/runs/latest_g1i_strict_l40s.outdir 2>/dev/null || true)
echo "$OUT" > /hy-tmp/runs/latest_g1i_20.outdir
python3 - <<PY > /hy-tmp/runs/latest_g1i_20_summary.txt
import pathlib, statistics, json
mon=pathlib.Path('$MON')
rows=[]
if mon.exists():
    for line in mon.read_text().splitlines():
        parts=[p.strip() for p in line.split(',')]
        if len(parts)>=6:
            try:
                rows.append(dict(t=float(parts[0]), gpu=float(parts[1]), memutil=float(parts[2]), mem_used=float(parts[3]), mem_total=float(parts[4]), power=float(parts[5])))
            except Exception: pass
print('rc', $RC)
print('monitor', '$MON')
print('out', '$OUT')
print('samples', len(rows))
for key,label in [('gpu','gpu_util_pct'),('memutil','mem_controller_util_pct'),('mem_used','mem_used_mib'),('power','power_w')]:
    vals=[r[key] for r in rows]
    if vals:
        print(label, 'avg', round(statistics.mean(vals),2), 'peak', round(max(vals),2))
out=pathlib.Path('$OUT')
met=out/'metrics.jsonl'
if met.exists():
    lines=met.read_text(errors='replace').splitlines()
    print('metrics_lines', len(lines))
    for line in lines[-5:]: print(line[:1000])
std=out/'stdout.log'
if std.exists():
    print('stdout_tail')
    for line in std.read_text(errors='replace').splitlines()[-30:]: print(line[:1000])
PY
exit $RC
