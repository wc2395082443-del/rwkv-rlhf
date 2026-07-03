import json
from pathlib import Path

def stats(path, key):
    vals = []
    with Path(path).open('r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            vals.append(len(str(obj.get(key, ''))))
    vals.sort()
    def q(p):
        idx = min(len(vals)-1, int((len(vals)-1)*p))
        return vals[idx]
    return {'n': len(vals), 'avg': sum(vals)/len(vals), 'p50': q(0.5), 'p90': q(0.9), 'p99': q(0.99), 'max': vals[-1]}
for name, path, key in [
    ('gsm8k_orig', '/root/RWKV-LM/RWKV7-statetuning/gsm8k_train_formatted.jsonl', 'problem'),
    ('openmath_13k', '/root/autodl-tmp/data/gsm8k_openmath_mathreason_13k/train_formatted_answer_only.jsonl', 'problem'),
]:
    print(name, stats(path, key))

