#!/usr/bin/env bash
set -euo pipefail

PID=730
CHECKPOINT=/hy-tmp/runs/g1k_dapo_pass1_31_officialprompt_q32x16_cand64_max6144_ob512_step200to300_20260922_093913/final_step_300.pth
LOG=/hy-tmp/g1k_dapo_pass1_31_300to400_watcher.log

echo "$(date -Is) waiting for pid=$PID" >> "$LOG"
while kill -0 "$PID" 2>/dev/null; do
  sleep 30
done

if [[ ! -s "$CHECKPOINT" ]]; then
  echo "$(date -Is) ERROR missing checkpoint: $CHECKPOINT" >> "$LOG"
  exit 1
fi

echo "$(date -Is) stage3 finished; launching stage4" >> "$LOG"
nohup bash /hy-tmp/launch_g1k_dapo_pass1_31_stage4_300to400.sh \
  >/hy-tmp/g1k_dapo_pass1_31_stage4_launcher.log 2>&1 </dev/null &
echo "$(date -Is) launcher_pid=$!" >> "$LOG"
