import json, pathlib, collections
base = pathlib.Path('/root/RWKV-LM/RWKV7-mmlupro_stem_g1f1p5b_20260624/data_mmlupro_stem_boxprefix/mmlupro_stem_eval_rwkv.jsonl')
catmap = {}
for line in base.read_text().splitlines():
    r=json.loads(line); catmap[(r['problem'], r['answer'])]=r.get('category','unknown')
for name,p in {
 'rwkv_pre': pathlib.Path('/root/autodl-tmp/stem-rlvr-repro/outputs/rwkv_g1f1p5b_mmlupro_stem_boxprefix_grpo_100_20260624_061955/eval_by_step/pre_eval_step_0.jsonl'),
 'rwkv_post100': pathlib.Path('/root/autodl-tmp/stem-rlvr-repro/outputs/rwkv_g1f1p5b_mmlupro_stem_boxprefix_grpo_100_20260624_061955/eval_by_step/full_eval_step_100.jsonl'),
}.items():
    d=collections.defaultdict(lambda:[0,0])
    for line in p.read_text().splitlines():
        r=json.loads(line)
        cat=catmap.get((r['problem'], r['ground_truth']),'unknown')
        d[cat][0]+=int(bool(r.get('is_correct'))); d[cat][1]+=1
    print('\n', name)
    for cat,(c,n) in sorted(d.items()): print(cat, c, n, c/n if n else 0)
