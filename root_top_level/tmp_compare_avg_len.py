import json
from pathlib import Path
runs = {
  'openmath_mb12': Path('/root/autodl-tmp/gsm8k_openmath_reward_mb12_20260501_141837/run/metrics.jsonl'),
  'openmath_mb15': Path('/root/autodl-tmp/gsm8k_openmath_reward_mb15_20260501_122752/run/metrics.jsonl'),
  'openmath_mb17_10step': Path('/root/autodl-tmp/openmath_micro_probe_20260501_0129/confirm_mb17/run/metrics.jsonl'),
  'gsm8k_orig_10step': Path('/root/autodl-tmp/gsm8k_orig_reward_smoke10_20260430_154200/run/metrics.jsonl'),
}
for name, path in runs.items():
    if not path.exists():
        continue
    rows = [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]
    train = [r for r in rows if r.get('split')=='train']
    normal = [r for r in train if r.get('step_type')=='normal']
    extra = [r for r in train if r.get('step_type')=='extra']
    evals = [r for r in rows if r.get('split') in ('eval','full_eval')]
    print('===', name)
    if train:
        print('last_train_step', train[-1]['step'], 'last_train_len', round(train[-1]['avg_length'],1), 'last_train_acc', round(train[-1]['accuracy'],4))
        print('train_mean_len', round(sum(r['avg_length'] for r in train)/len(train),1))
    if normal:
        print('normal_mean_len', round(sum(r['avg_length'] for r in normal)/len(normal),1))
        print('normal_last3_mean_len', round(sum(r['avg_length'] for r in normal[-3:])/min(3,len(normal)),1))
    if extra:
        print('extra_mean_len', round(sum(r['avg_length'] for r in extra)/len(extra),1))
        print('extra_last3_mean_len', round(sum(r['avg_length'] for r in extra[-3:])/min(3,len(extra)),1))
    if evals:
        print('eval_tags', [(r['split'], r['step'], round(r['avg_length'],1), round(r['accuracy'],4)) for r in evals])

