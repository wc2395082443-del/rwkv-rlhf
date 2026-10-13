#!/bin/bash
while true; do
  date '+%F %T' | tr '\n' ','
  nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,power.draw --format=csv,noheader,nounits
  sleep 5
done
