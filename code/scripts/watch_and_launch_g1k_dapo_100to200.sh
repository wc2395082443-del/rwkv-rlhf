#!/usr/bin/env bash
set -euo pipefail

PID=982339
CHECKPOINT=/hy-tmp/runs/g1k_step155_dapo_pass1_31_officialprompt_q32x16_cand64_max6144_ob512_step0to100_20260920_113659/final_step_100.pth
LOG=/hy-tmp/g1k_dapo_pass1_31_100to200_watcher.log

echo "$(date -Is) waiting for pid=$PID" >> "$LOG"
while kill -0 "$PID" 2>/dev/null; do
  sleep 30
done

if [[ ! -s "$CHECKPOINT" ]]; then
  echo "$(date -Is) ERROR missing checkpoint: $CHECKPOINT" >> "$LOG"
  exit 1
fi

echo "$(date -Is) stage1 finished; launching stage2" >> "$LOG"
nohup bash /hy-tmp/launch_g1k_dapo_pass1_31_stage2_100to200.sh \
  >/hy-tmp/g1k_dapo_pass1_31_stage2_launcher.log 2>&1 </dev/null &
echo "$(date -Is) launcher_pid=$!" >> "$LOG"
