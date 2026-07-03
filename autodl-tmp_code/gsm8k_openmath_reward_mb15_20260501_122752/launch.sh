#!/usr/bin/env bash
set -euo pipefail
sed -i 's/\r$//' /root/autodl-tmp/gsm8k_openmath_reward_mb15_20260501_122752/run.sh
chmod +x /root/autodl-tmp/gsm8k_openmath_reward_mb15_20260501_122752/run.sh
nohup bash /root/autodl-tmp/gsm8k_openmath_reward_mb15_20260501_122752/run.sh > /root/autodl-tmp/gsm8k_openmath_reward_mb15_20260501_122752/stdout.log 2>&1 < /dev/null &
echo $! > /root/autodl-tmp/gsm8k_openmath_reward_mb15_20260501_122752/pid.txt
cat /root/autodl-tmp/gsm8k_openmath_reward_mb15_20260501_122752/pid.txt

