#!/usr/bin/env bash
set -euo pipefail
chmod +x /root/autodl-tmp/gsm8k_openmath_rewardrestore_20260430_153012/run.sh
nohup bash /root/autodl-tmp/gsm8k_openmath_rewardrestore_20260430_153012/run.sh > /root/autodl-tmp/gsm8k_openmath_rewardrestore_20260430_153012/stdout.log 2>&1 < /dev/null &
echo $! > /root/autodl-tmp/gsm8k_openmath_rewardrestore_20260430_153012/pid.txt
cat /root/autodl-tmp/gsm8k_openmath_rewardrestore_20260430_153012/pid.txt

