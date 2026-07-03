import json, os
run='/root/autodl-tmp/baseline_hardbuffer_kl005_noextra_20260423_001234/run'
p=os.path.join(run,'metrics.jsonl')
rows=[]
if os.path.exists(p):
    with open(p,encoding='utf-8') as f:
        for line in f:
            if line.strip(): rows.append(json.loads(line))
print('run',run)
train=[r for r in rows if r.get('split')=='train']
evals=[r for r in rows if r.get('split')!='train']
print('last_step', train[-1].get('step') if train else None)
print('full_eval')
for r in evals:
    print('step',r.get('step'),'acc',r.get('accuracy'),'trunc',r.get('trunc_rate'),'repeat',r.get('repeat_rate'),'noans',r.get('no_answer_rate'),'len',r.get('avg_length'),'zstd',r.get('avg_zstd_ratio'))
if train:
    print('last8 train')
    for r in train[-8:]:
        print('step',r.get('step'),'acc',round(float(r.get('accuracy',0)),4),'avg_kl',r.get('avg_kl'),'trunc',r.get('trunc_rate'),'repeat',r.get('repeat_rate'),'noans',r.get('no_answer_rate'),'len',round(float(r.get('avg_length',0)),1),'all0',r.get('groups_all_wrong'),'all1',r.get('groups_all_correct'),'used',r.get('groups_used'))

