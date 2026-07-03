import json, os
runs = {
  'with_extra': '/root/autodl-tmp/baseline_hardbuffer_kl005_bestcfg_20260422_194041/run',
  'no_extra': '/root/autodl-tmp/baseline_hardbuffer_kl005_noextra_20260423_001234/run',
}
for name, run in runs.items():
    p=os.path.join(run,'metrics.jsonl')
    rows=[]
    if os.path.exists(p):
        with open(p,encoding='utf-8') as f:
            for line in f:
                if line.strip(): rows.append(json.loads(line))
    train=[r for r in rows if r.get('split')=='train']
    evals=[r for r in rows if r.get('split')!='train']
    print('===', name, run)
    print('last_step', train[-1].get('step') if train else None)
    for r in evals:
        print('eval', r.get('step'), r.get('accuracy'))
    if train:
        tail=train[-5:]
        print('tail_train')
        for r in tail:
            print(r.get('step'), r.get('step_type'), round(float(r.get('accuracy',0)),4), r.get('avg_kl'), r.get('extra_step_ran'), r.get('hard_buffer_triggered'))

