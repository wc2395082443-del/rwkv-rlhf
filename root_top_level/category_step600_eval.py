import json
from pathlib import Path
from collections import defaultdict
base = Path('/root/RWKV-LM/RWKV7-mmlupro_stem_g1f1p5b_20260624/data_mmlupro_stem_boxprefix/mmlupro_stem_eval_rwkv.jsonl')
evalp = Path('/root/autodl-tmp/stem-rlvr-repro/outputs/rwkv_g1f1p5b_mmlupro_stem_step600_benchmark_eval_20260624_171753/eval.jsonl')
catmap = {}
for line in base.read_text().splitlines():
    r = json.loads(line)
    catmap[(r['problem'], r['answer'])] = r.get('category', 'unknown')
stats = defaultdict(lambda: [0, 0])
pred_stats = defaultdict(int)
for line in evalp.read_text().splitlines():
    r = json.loads(line)
    cat = catmap.get((r['problem'], r['ground_truth']), 'unknown')
    stats[cat][0] += int(bool(r.get('is_correct')))
    stats[cat][1] += 1
    pred_stats[r.get('pred_extracted')] += 1
print('category')
for cat, (c, n) in sorted(stats.items()):
    print(cat, c, n, c / n if n else 0)
print('pred_dist')
for k, v in sorted(pred_stats.items(), key=lambda x: (str(x[0]))):
    print(k, v)
