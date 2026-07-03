import json
from pathlib import Path
for name, path, key in [
    ('gsm8k_orig', '/root/RWKV-LM/RWKV7-statetuning/gsm8k_train_formatted.jsonl', 'problem'),
    ('openmath_13k', '/root/autodl-tmp/data/gsm8k_openmath_mathreason_13k/train_formatted_answer_only.jsonl', 'problem'),
]:
    vals=[]
    with Path(path).open('r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                obj=json.loads(line); vals.append(len(str(obj.get(key,''))))
    for th in [800,1000,1500,2000,4000]:
        c=sum(v>=th for v in vals)
        print(name, th, c)

