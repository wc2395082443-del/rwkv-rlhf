import json
from pathlib import Path
out = Path('/root/autodl-tmp/stem-rlvr-repro/outputs/rwkv_g1f1p5b_mmlupro_stem_boxprefix_grpo_continue500_20260624_132808')
p = out / 'metrics.jsonl'
rows = []
if p.exists():
    for line in p.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
train = [r for r in rows if r.get('split') == 'train']
evals = [r for r in rows if r.get('split') in ('pre_eval', 'eval', 'full_eval')]
print('train_steps', len(train), 'last_step', train[-1]['step'] if train else None)
for w in (10, 20, 50, 100):
    xs = train[-w:] if len(train) >= w else train
    if not xs:
        continue
    denom_groups = max(1, sum(x.get('groups_total', 0) for x in xs))
    print('ma%d acc=%.4f reward=%.4f all0=%.3f all1=%.3f sec=%.2f kl=%.6f' % (
        w,
        sum(x.get('accuracy', 0) for x in xs) / len(xs),
        sum(x.get('avg_reward', 0) for x in xs) / len(xs),
        sum(x.get('groups_all_wrong', 0) for x in xs) / denom_groups,
        sum(x.get('groups_all_correct', 0) for x in xs) / denom_groups,
        sum(x.get('time', 0) for x in xs) / len(xs),
        sum(x.get('avg_kl', 0) for x in xs) / len(xs),
    ))
print('evals')
for r in evals[-20:]:
    print('%s step=%s acc=%.4f delta=%s n=%s trunc=%.4f noans=%.4f fmt=%.4f' % (
        r.get('split'), r.get('step'), float(r.get('accuracy') or 0),
        'None' if r.get('eval_acc_delta') is None else '%.4f' % float(r.get('eval_acc_delta')),
        r.get('eval_count'), float(r.get('trunc_rate') or 0),
        float(r.get('no_answer_rate') or 0), float(r.get('format_rate') or 0),
    ))
