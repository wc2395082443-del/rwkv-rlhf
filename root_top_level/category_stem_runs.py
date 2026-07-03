import json, pathlib, collections
paths = {
 'qwen_base': pathlib.Path('/root/autodl-tmp/stem-rlvr-repro/outputs/qwen25_1p5b_mmlupro_stem_brief_base_eval_20260624_0637/eval1176.jsonl'),
 'qwen_post100': pathlib.Path('/root/autodl-tmp/stem-rlvr-repro/outputs/qwen25_1p5b_mmlupro_stem_brief_grpo_100_20260624_053316/post_eval1176.jsonl'),
 'rwkv_pre': pathlib.Path('/root/autodl-tmp/stem-rlvr-repro/outputs/rwkv_g1f1p5b_mmlupro_stem_boxprefix_grpo_100_20260624_061955/eval_by_step/pre_eval_step_0.jsonl'),
 'rwkv_post100': pathlib.Path('/root/autodl-tmp/stem-rlvr-repro/outputs/rwkv_g1f1p5b_mmlupro_stem_boxprefix_grpo_100_20260624_061955/eval_by_step/full_eval_step_100.jsonl'),
}
for name,p in paths.items():
    d=collections.defaultdict(lambda:[0,0])
    if not p.exists():
        print(name, 'missing', p); continue
    for line in p.read_text().splitlines():
        if not line.strip(): continue
        r=json.loads(line)
        cat=r.get('category') or r.get('problem_category') or 'unknown'
        ok = bool(r.get('correct') if 'correct' in r else r.get('is_correct'))
        d[cat][0]+=int(ok); d[cat][1]+=1
    print('\n', name)
    for cat,(c,n) in sorted(d.items()): print(cat, c, n, c/n if n else 0)
