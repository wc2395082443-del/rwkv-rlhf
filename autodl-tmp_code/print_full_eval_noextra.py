import json, os
p='/root/autodl-tmp/baseline_hardbuffer_kl005_noextra_20260423_001234/run/metrics.jsonl'
if not os.path.exists(p):
    print('no metrics')
    raise SystemExit
rows=[]
with open(p,encoding='utf-8') as f:
    for line in f:
        if line.strip():
            rows.append(json.loads(line))
for r in rows:
    if r.get('split')!='train':
        print(json.dumps(r, ensure_ascii=False))

