rm -rf /root/autodl-tmp/log/*
rm -rf /root/autodl-tmp/tmp/*
mkdir -p /root/autodl-tmp/log /root/autodl-tmp/tmp
echo CLEANED_ALL_LOGS
echo ---
df -h /root /root/autodl-tmp
echo ---
du -sh /root/autodl-tmp/log 2>/dev/null || true
du -sh /root/autodl-tmp/tmp 2>/dev/null || true
