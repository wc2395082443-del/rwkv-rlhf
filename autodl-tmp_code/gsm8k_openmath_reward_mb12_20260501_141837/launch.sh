#!/usr/bin/env bash
set -euo pipefail
sed -i 's/\r$//' /root/autodl-tmp/gsm8k_openmath_reward_mb12_20260501_141837/run.sh
chmod +x /root/autodl-tmp/gsm8k_openmath_reward_mb12_20260501_141837/run.sh
nohup bash /root/autodl-tmp/gsm8k_openmath_reward_mb12_20260501_141837/run.sh > /root/autodl-tmp/gsm8k_openmath_reward_mb12_20260501_141837/stdout.log 2>&1 < /dev/null &
echo $! > /root/autodl-tmp/gsm8k_openmath_reward_mb12_20260501_141837/pid.txt
cat /root/autodl-tmp/gsm8k_openmath_reward_mb12_20260501_141837/pid.txt

