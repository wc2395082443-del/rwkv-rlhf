#!/usr/bin/env bash
out=$1
echo ts,util_gpu,mem_used_mb,mem_total_mb,power_w > $out
while true; do
  nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total,power.draw --format=csv,noheader,nounits >> $out || true
  sleep 5
done
