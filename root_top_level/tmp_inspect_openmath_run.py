import json
from pathlib import Path
for step in [1,2]:
    p = Path(f'/root/autodl-tmp/gsm8k_openmath_rewardrestore_20260430_153012/run/responses_by_step/step_{step}.jsonl')
    if not p.exists():
        continue
    rows = [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]
    lens = [len(r.get('completion_tokens', [])) for r in rows]
    rewards = [float(r.get('reward', 0.0)) for r in rows]
    correct = sum(1 for r in rows if r.get('is_correct'))
    trunc = sum(1 for r in rows if r.get('truncated'))
    print('step', step, 'n', len(rows), 'avg_len', sum(lens)/len(lens), 'max_len', max(lens), 'trunc', trunc, 'correct', correct, 'avg_reward', sum(rewards)/len(rewards))

