import json, os, statistics, glob
run='/root/autodl-tmp/baseline_nobuffer_bestcfg_20260422_174502/run'
mp=os.path.join(run,'metrics.jsonl')
rows=[]
if os.path.exists(mp):
    with open(mp,'r',encoding='utf-8') as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
train=[r for r in rows if r.get('split')=='train']
evals=[r for r in rows if r.get('split')!='train']
print('run',run)
print('train_steps', len(train), 'last_step', train[-1].get('step') if train else None)
print('eval_rows', len(evals))
if train:
    for start in [1,25,50,75,100,125]:
        block=[r for r in train if start <= int(r.get('step',0)) < start+25]
        if not block: continue
        def avg(k): return sum(float(r.get(k,0) or 0) for r in block)/len(block)
        print('block',f'{start}-{start+24}', 'n',len(block), 'acc',round(avg('accuracy'),4), 'kl_avg',round(avg('avg_kl'),6), 'trunc',round(avg('trunc_rate'),4), 'repeat',round(avg('repeat_rate'),4), 'noans',round(avg('no_answer_rate'),4), 'len',round(avg('avg_length'),1), 'groups_used',round(avg('groups_used'),2), 'all0',round(avg('groups_all_wrong'),2), 'all1',round(avg('groups_all_correct'),2))
    print('last5')
    for r in train[-5:]:
        print('step',r.get('step'),'acc',round(float(r.get('accuracy',0)),4),'avg_kl',r.get('avg_kl'),'trunc',r.get('trunc_rate'),'repeat',r.get('repeat_rate'),'noans',r.get('no_answer_rate'),'len',round(float(r.get('avg_length',0)),1),'all0',r.get('groups_all_wrong'),'all1',r.get('groups_all_correct'))
print('files')
for p in sorted(glob.glob(run+'/*')):
    print(os.path.basename(p))

