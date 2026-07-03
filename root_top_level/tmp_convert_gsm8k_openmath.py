import json
from pathlib import Path
src = Path('/root/autodl-tmp/data/gsm8k_openmath_mathreason_13k/train.jsonl')
out = Path('/root/autodl-tmp/data/gsm8k_openmath_mathreason_13k/train_formatted_answer_only.jsonl')
keep = 0
with src.open('r', encoding='utf-8') as fin, out.open('w', encoding='utf-8') as fout:
    for line in fin:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        q = str(obj.get('question', '')).strip()
        a = str(obj.get('final_answer', '')).strip()
        cot = str(obj.get('cot', '')).strip()
        if not q or not a:
            continue
        rec = {'problem': q, 'answer': a, 'original_answer': cot, 'source': 'HAD653/GSM8K-OpenMath-MathReason-13k'}
        fout.write(json.dumps(rec, ensure_ascii=False) + '\n')
        keep += 1
print(keep)

