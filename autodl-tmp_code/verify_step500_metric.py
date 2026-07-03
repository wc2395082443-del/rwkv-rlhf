import json, os
run='/root/autodl-tmp/baseline_hardbuffer_kl005_bestcfg_20260422_194041/run'
metrics=os.path.join(run,'metrics.jsonl')
print('metrics_exists', os.path.exists(metrics))
if os.path.exists(metrics):
    with open(metrics,encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            r=json.loads(line)
            if r.get('split')!='train' and int(r.get('step',-1))==500:
                print('metric_step500', json.dumps(r, ensure_ascii=False))
for cand in [
    os.path.join(run,'eval_by_step','full_eval_step_500.jsonl'),
    os.path.join(run,'eval_by_step','eval_step_500.jsonl'),
    os.path.join(run,'eval_by_step','post_eval_step_500.jsonl'),
    os.path.join(run,'eval.jsonl')
]:
    print('file', cand, os.path.exists(cand), os.path.getsize(cand) if os.path.exists(cand) else None)
    if os.path.exists(cand):
        cnt=0; correct=0
        with open(cand,encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                obj=json.loads(line)
                if 'step' in obj and int(obj.get('step',-1))!=500 and cand.endswith('eval.jsonl'):
                    continue
                cnt += 1
                correct += int(bool(obj.get('is_correct', False)))
        print('count',cnt,'correct',correct,'acc', (correct/cnt if cnt else None))

