import json
from pathlib import Path
for step in [1,2]:
    p = Path(f'/root/autodl-tmp/gsm8k_openmath_rewardrestore_20260430_153012/run/responses_by_step/step_{step}.jsonl')
    if not p.exists():
        continue
    rows = [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]
    lens = [int(r.get('reward_details', {}).get('token_length', r.get('gen_len', 0))) for r in rows]
    print('step', step, 'n', len(rows), 'avg_len', sum(lens)/len(lens), 'max_len', max(lens), 'trunc', sum(1 for r in rows if r.get('truncated')), 'correct', sum(1 for r in rows if r.get('is_correct')))

