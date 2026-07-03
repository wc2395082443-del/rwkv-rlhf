import os, json, math
roots = [
    '/root/RWKV-LM/RWKV7-statetuning_hardbuffer_v1/log',
    '/root/RWKV-LM/RWKV7-statetuning_hardbuffer_ratio_v1/log',
]
rows = []
for root in roots:
    for dirpath, dirnames, filenames in os.walk(root):
        if 'metrics.jsonl' not in filenames:
            continue
        path = os.path.join(dirpath, 'metrics.jsonl')
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    obj = json.loads(line)
                    if obj.get('split') != 'train':
                        continue
                    if int(obj.get('extra_step_ran', 0)) != 1:
                        continue
                    rows.append({
                        'path': path,
                        'step': obj.get('step'),
                        'kl': float(obj.get('kl', 0.0) or 0.0),
                        'avg_kl': float(obj.get('avg_kl', 0.0) or 0.0),
                        'extra_avg_kl': float(obj.get('extra_avg_kl', 0.0) or 0.0),
                        'extra_loss': float(obj.get('extra_loss', 0.0) or 0.0),
                        'grad_norm': float(obj.get('grad_norm', 0.0) or 0.0),
                        'extra_grad_norm': float(obj.get('extra_grad_norm', 0.0) or 0.0),
                        'extra_groups_used': int(obj.get('extra_groups_used', 0) or 0),
                        'extra_groups_total': int(obj.get('extra_groups_total', 0) or 0),
                        'extra_groups_all_wrong': int(obj.get('extra_groups_all_wrong', 0) or 0),
                        'extra_groups_all_correct': int(obj.get('extra_groups_all_correct', 0) or 0),
                        'accuracy': float(obj.get('accuracy', 0.0) or 0.0),
                    })
        except Exception:
            pass
print('extra_rows', len(rows))
if not rows:
    raise SystemExit
for key in ['extra_avg_kl','extra_grad_norm','avg_kl','kl']:
    vals = [r[key] for r in rows]
    vals_sorted = sorted(vals)
    def q(p):
        idx = min(len(vals_sorted)-1, max(0, int(round((len(vals_sorted)-1)*p))))
        return vals_sorted[idx]
    print(key, 'min', min(vals), 'p50', q(0.5), 'p90', q(0.9), 'p95', q(0.95), 'max', max(vals))
print('--- top extra_avg_kl ---')
for r in sorted(rows, key=lambda x: x['extra_avg_kl'], reverse=True)[:12]:
    print(r)

