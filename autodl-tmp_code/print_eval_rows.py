import json
p='/root/autodl-tmp/baseline_nobuffer_bestcfg_20260422_174502/run/metrics.jsonl'
for line in open(p,encoding='utf-8'):
    r=json.loads(line)
    if r.get('split')!='train':
        print(json.dumps(r,ensure_ascii=False))

