import json, re, pathlib, statistics
runs = {
    'qwen100': pathlib.Path('/root/autodl-tmp/stem-rlvr-repro/outputs/qwen25_1p5b_mmlupro_stem_brief_grpo_100_20260624_053316'),
    'rwkv100': pathlib.Path('/root/autodl-tmp/stem-rlvr-repro/outputs/rwkv_g1f1p5b_mmlupro_stem_boxprefix_grpo_100_20260624_061955'),
    'rwkv20': pathlib.Path('/root/autodl-tmp/stem-rlvr-repro/outputs/rwkv_g1f1p5b_mmlupro_stem_boxprefix_grpo_smoke20_20260624_061436'),
    'rwkv_preeval_answeronly': pathlib.Path('/root/autodl-tmp/stem-rlvr-repro/outputs/rwkv_g1f1p5b_mmlupro_stem_answeronly_preeval_20260624_060236'),
    'rwkv_preeval_boxprefix': pathlib.Path('/root/autodl-tmp/stem-rlvr-repro/outputs/rwkv_g1f1p5b_mmlupro_stem_boxprefix_stopfix_preeval_20260624_061206'),
}

def read_jsonl(p):
    if not p.exists(): return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]

def summarize_rwkv(path):
    rows = read_jsonl(path/'metrics.jsonl')
    out = {}
    for r in rows:
        sp = r.get('split')
        if sp in ('pre_eval','eval','full_eval'):
            out[f"{sp}_{r.get('step')}"] = {
                'acc': r.get('accuracy'), 'delta': r.get('eval_acc_delta'), 'n': r.get('eval_count'),
                'trunc': r.get('trunc_rate'), 'no_answer': r.get('no_answer_rate'), 'fmt': r.get('format_rate'),
            }
    train = [r for r in rows if r.get('split') == 'train']
    if train:
        for w in (10,20,50,100):
            xs = train[-w:] if len(train) >= w else train
            out[f'train_ma{w}'] = {
                'steps': len(xs),
                'acc': sum(r['accuracy'] for r in xs)/len(xs),
                'reward': sum(r['avg_reward'] for r in xs)/len(xs),
                'all0_ratio': sum(r['groups_all_wrong'] for r in xs)/max(1,sum(r['groups_total'] for r in xs)),
                'all1_ratio': sum(r['groups_all_correct'] for r in xs)/max(1,sum(r['groups_total'] for r in xs)),
                'sec_step': sum(r['time'] for r in xs)/len(xs),
            }
    return out

def qwen_from_logs(path):
    out = {}
    # summary file
    sf = path/'post_eval1176.jsonl.summary.json'
    if sf.exists():
        s=json.loads(sf.read_text())
        out['post_full']={'acc':s.get('acc'), 'n':s.get('n'), 'no_parse':s.get('no_parse')}
    txt=(path/'train.log').read_text(errors='ignore') if (path/'train.log').exists() else ''
    # extract final train runtime and maybe pre eval lines if present
    accs=[]
    for line in txt.splitlines():
        if 'eval' in line.lower() or 'pre' in line.lower():
            m=re.search(r'acc[=:] ?([0-9.]+)', line)
            if m: accs.append((line[:160], float(m.group(1))))
    out['log_eval_lines']=accs[-20:]
    rewards=[]; steps=[]
    for line in txt.splitlines():
        if "'reward':" in line or '"reward":' in line:
            try:
                # logs use python dict repr
                m=re.search(r"'reward': '?(\d*\.?\d+)'?", line)
                if m: rewards.append(float(m.group(1)))
            except Exception: pass
    if rewards:
        out['train_reward_ma10']=sum(rewards[-10:])/min(10,len(rewards))
    return out

result={}
for name,path in runs.items():
    result[name]=summarize_rwkv(path) if name.startswith('rwkv') else qwen_from_logs(path)
print(json.dumps(result, indent=2, ensure_ascii=False))
