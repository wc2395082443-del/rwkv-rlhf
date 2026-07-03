pkill -f run_llama_math500_full_vllm_stable.sh || true
pkill -f run_llama_math500_full_vllm_tuned.sh || true
pkill -f "python -m verl.trainer.main_ppo" || true
ray stop --force || true
sleep 2
rm -rf /root/autodl-tmp/tmp/ray
rm -f /root/autodl-tmp/log/launch_math500_*.log
rm -f /root/autodl-tmp/log/launch_math500_full_*.log
rm -f /root/autodl-tmp/log/launch_math500_full_stable_*.log
mkdir -p /root/autodl-tmp/tmp
echo CLEANED
echo ---
df -h /root /root/autodl-tmp
echo ---
du -sh /root/autodl-tmp/tmp 2>/dev/null || true
du -sh /root/autodl-tmp/log 2>/dev/null || true
