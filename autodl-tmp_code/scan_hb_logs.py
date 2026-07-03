import os, json
roots = ['/root/autodl-tmp', '/root/RWKV-LM']
keywords = ['hb_', 'hardbuffer', 'hard_buffer', 'baseline_bf16rollout']
count = 0
for root in roots:
    for dirpath, dirnames, filenames in os.walk(root):
        if any(k in dirpath for k in keywords):
            for fn in filenames:
                if fn in ('metrics.jsonl', 'train.log') or fn.startswith('metrics') and fn.endswith('.jsonl'):
                    path = os.path.join(dirpath, fn)
                    print(path)
                    count += 1
                    if count >= 120:
                        raise SystemExit

