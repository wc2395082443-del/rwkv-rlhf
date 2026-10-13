#!/usr/bin/env bash
set -euo pipefail
MON=/hy-tmp/runs/g1i_20_rfb256_micro4_gpu_monitor_$(date +%Y%m%d_%H%M%S).csv
echo "$MON" > /hy-tmp/runs/latest_g1i_20_opt.monitor
( while true; do date +%s.%N | tr -d '\n' >> "$MON"; echo -n ',' >> "$MON"; nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,power.draw --format=csv,noheader,nounits >> "$MON" || true; sleep 1; done ) &
MPID=$!
set +e
bash "/hy-tmp/runs/launch_g1i_strict_quality_fp32master_nooffload_rfb256_micro4_20_l40s.sh"
RC=$?
kill $MPID 2>/dev/null || true
OUT=$(cat /hy-tmp/runs/latest_g1i_strict_l40s.outdir 2>/dev/null || true)
echo "$OUT" > /hy-tmp/runs/latest_g1i_20_opt.outdir
echo "rc=$RC out=$OUT mon=$MON" > /hy-tmp/runs/latest_g1i_20_opt.status
exit $RC
