import glob
import json
import os
import re

roots = [
    '/hy-tmp/evals/official_flower_ckpt_eval_20260827',
    '/hy-tmp/evals/official_flower_ckpt_eval_step800_20260829',
    '/hy-tmp/evals/official_flower_ckpt_eval_step850_1000_20260831',
    '/hy-tmp/evals/official_flower_ckpt_eval_step1300_20260904',
    '/hy-tmp/evals/missing_five_benchmarks_checkpoints_20260905',
    '/hy-tmp/evals/extra_math_step400_800_officialcot_r64_ctx12k_max8192_20260901',
    '/hy-tmp/evals/extra_math_step850_1000_officialcot_r64_ctx12k_max8192_20260831',
]
steps = [85, 100, 105, 155, 200, 300, 400, 500, 600, 800, 850, 900, 950, 1000, 1050, 1100, 1150, 1200, 1250, 1300]
benchmarks = ['gsm8k', 'math500', 'aime24', 'aime25', 'amc23']
found = {step: {} for step in steps}
for root in roots:
    for path in glob.glob(root + '/**/summary.json', recursive=True):
        try:
            summary = json.load(open(path, encoding='utf-8'))
        except Exception:
            continue
        parent = os.path.basename(os.path.dirname(path))
        grandparent = os.path.basename(os.path.dirname(os.path.dirname(path)))
        bench = parent if parent in benchmarks else grandparent
        matches = re.findall(r'(?:step|final_step_)(\d+)', path)
        if not matches or bench not in benchmarks:
            continue
        step = int(matches[-1])
        if step not in found:
            continue
        rollout = summary.get('rollout')
        if bench in ['gsm8k', 'math500'] and rollout not in [4, 8]:
            continue
        if bench in ['aime24', 'aime25', 'amc23'] and rollout != 64:
            continue
        score = (1 if 'missing_five' in path else 0, 1 if 'official_flower' in path or 'extra_math' in path else 0, os.path.getmtime(path))
        if bench not in found[step] or score > found[step][bench][0]:
            found[step][bench] = (score, summary)
for step in steps:
    values = []
    for bench in benchmarks:
        item = found[step].get(bench)
        if not item:
            values.append('MISSING')
            continue
        summary = item[1]
        rollout = summary['rollout']
        values.append(f"{bench}=avg@{rollout}:{summary.get('rollout_accuracy', 0) * 100:.2f}% pass@{rollout}:{summary.get('pass_at_rollout_accuracy', 0) * 100:.2f}% ({summary.get('correct_generations')}/{summary.get('total_generations')})")
    print(f'{step}\t' + '\t'.join(values))
