import json, os, glob
run='/root/autodl-tmp/baseline_nobuffer_bestcfg_20260422_174502/run'
mp=os.path.join(run,'metrics.jsonl')
rows=[]
if os.path.exists(mp):
    with open(mp,encoding='utf-8') as f:
        for line in f:
            if line.strip(): rows.append(json.loads(line))
train=[r for r in rows if r.get('split')=='train']
evals=[r for r in rows if r.get('split')!='train']
print('run',run)
print('train_steps',len(train),'last_step',train[-1].get('step') if train else None)
print('full_eval')
for r in evals:
    print('step',r.get('step'),'split',r.get('split'),'acc',r.get('accuracy'),'trunc',r.get('trunc_rate'),'repeat',r.get('repeat_rate'),'no_answer',r.get('no_answer_rate'),'avg_len',r.get('avg_length'),'zstd',r.get('avg_zstd_ratio'))
if train:
    print('last10')
    for r in train[-10:]:
        print('step',r.get('step'),'acc',round(float(r.get('accuracy',0)),4),'avg_kl',r.get('avg_kl'),'trunc',r.get('trunc_rate'),'repeat',r.get('repeat_rate'),'noans',r.get('no_answer_rate'),'len',round(float(r.get('avg_length',0)),1),'all0',r.get('groups_all_wrong'),'all1',r.get('groups_all_correct'),'used',r.get('groups_used'))

