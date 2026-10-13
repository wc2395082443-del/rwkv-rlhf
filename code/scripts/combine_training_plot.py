import json
import os
import matplotlib.pyplot as plt

paths = [
    '/hy-tmp/runs/g1i_step155_to_dapo_math17k_officialprompt_q32x16_cand64_max6144_ob512_100_20260820_154937/metrics.jsonl',
    '/hy-tmp/runs/g1i_step155_to_dapo_math17k_officialprompt_q32x16_cand64_max6144_ob512_continue100to200_eval512_20260821_131710/metrics.jsonl',
    '/hy-tmp/runs/g1i_step155_to_dapo_math17k_officialprompt_q32x16_cand64_max6144_ob512_continue200to300_eval512_20260822_073045/metrics.jsonl',
    '/hy-tmp/runs/g1i_step300_to_dapo_math17k_officialprompt_q32x16_cand64_max6144_ob512_continue300to400_eval384_20260824_122246/metrics.jsonl',
    '/hy-tmp/runs/g1i_step400_to_dapo_math17k_officialprompt_q32x16_cand64_max6144_ob512_continue400to500_eval384_20260825_1355_fixed/metrics.jsonl',
    '/hy-tmp/runs/g1i_step500_to_dapo_math17k_officialprompt_q32x16_cand64_max6144_ob512_continue500to600_eval384_20260826_0000_fixed/metrics.jsonl',
    '/hy-tmp/runs/g1i_step600_to_dapo_math17k_officialprompt_q32x16_cand64_max6144_ob512_continue600to800_eval384_20260827/metrics.jsonl',
    '/hy-tmp/runs/g1i_step800_to_dapo_math17k_officialprompt_q32x16_cand64_max6144_ob512_continue800to1000_eval384_ckpt50_20260829_1810/metrics.jsonl',
    '/hy-tmp/runs/g1i_step1000_to_dapo_math17k_officialprompt_q32x16_cand64_max6144_ob512_continue1000to1200_eval384_ckpt50_20260901_1100/metrics.jsonl',
    '/hy-tmp/runs/g1i_step1200_to_dapo_math17k_officialprompt_q32x16_cand64_max6144_ob512_continue1200to1300_eval384_ckpt50_20260903_1600/metrics.jsonl',
]
rows = {}
for path in paths:
    for line in open(path, encoding='utf-8'):
        if line.strip():
            item = json.loads(line)
            if item.get('split', item.get('tag')) == 'train':
                rows[item['step']] = item
rows = [rows[k] for k in sorted(rows)]
evals = []
for path in paths:
    validation = path.replace('/metrics.jsonl', '/validation_metrics.jsonl')
    if os.path.exists(validation):
        for line in open(validation, encoding='utf-8'):
            if line.strip():
                item = json.loads(line)
                if item.get('tag') == 'eval':
                    evals.append(item)
evals.sort(key=lambda x: x['step'])
spec = [
    ('avg_reward', 'avg_reward', 'Avg reward (%)', 100),
    ('accuracy', 'accuracy', 'Accuracy (%)', 100),
    ('selected_acc', 'selected_accuracy', 'Selected accuracy (%)', 100),
    ('selected_strict', 'selected_strict_accuracy', 'Selected strict (%)', 100),
    ('repeat_rate', 'repeat_rate', 'Repeat rate (%)', 100),
    ('trunc_rate', 'trunc_rate', 'Truncation rate (%)', 100),
]
plt.style.use('seaborn-v0_8-whitegrid')
fig, axes = plt.subplots(3, 2, figsize=(18, 13), sharex=True)
axes = axes.ravel()
for ax, (title, key, ylabel, scale) in zip(axes.ravel(), spec):
    x = [r['step'] for r in rows]
    y = [r.get(key) for r in rows]
    y = [None if value is None else value * scale for value in y]
    ax.plot(x, y, color='#9ecae1', linewidth=0.8, label='raw')
    valid = [(xx, yy) for xx, yy in zip(x, y) if yy is not None]
    ma = []
    for i, (_, value) in enumerate(valid):
        ma.append(sum(v for _, v in valid[max(0, i - 9):i + 1]) / len(valid[max(0, i - 9):i + 1]))
    ax.plot([p[0] for p in valid], ma, color='#d62728', linewidth=2.0, label='MA10')
    for item in evals:
        value = item.get('strict_accuracy', item.get('loose_accuracy'))
        if value is not None and key in ('accuracy', 'selected_acc', 'selected_strict'):
            ax.scatter(item['step'], value * 100, color='black', s=22, zorder=5)
            ax.annotate(f"eval {item['step']}\n{value * 100:.1f}", (item['step'], value * 100), xytext=(3, 5), textcoords='offset points', fontsize=6)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend(loc='best', fontsize=8)
    if key in ('repeat_rate', 'trunc_rate'):
        ax.set_ylim(bottom=0)
axes[-2].set_xlabel('global step')
axes[-1].set_xlabel('global step')
fig.suptitle('DAPO RL Training Curves | step1-step1300 | official prompt | q32 x 16 | candidates64 | max_new6144 | overlong buffer512 | eval batch384', fontsize=14, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig('/hy-tmp/rwkv-rlhf-gsm8k-project/DAPO-gsm8k/results/g1i_dapo_global_1_1300_sixpanel.jpg', dpi=180)
print(len(rows), rows[0]['step'], rows[-1]['step'], len(evals))
