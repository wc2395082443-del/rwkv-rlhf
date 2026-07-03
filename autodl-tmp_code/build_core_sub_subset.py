import json
import argparse
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--source_train_jsonl', required=True)
ap.add_argument('--pass8_jsonl', required=True)
ap.add_argument('--out_jsonl', required=True)
args = ap.parse_args()

selected = set()
selected_rows = 0
with open(args.pass8_jsonl, 'r', encoding='utf-8') as f:
    for line in f:
        if not line.strip():
            continue
        r = json.loads(line)
        k = int(r.get('num_correct', 0) or 0)
        samples = r.get('samples', [])
        first = None
        for s in samples:
            if s.get('sample_idx') == 0:
                first = s
                break
        if first is None and samples:
            first = samples[0]
        first_wrong = not (first and bool(first.get('is_correct')))
        if ((2 <= k <= 6) or (k == 1)) and first_wrong:
            selected.add((r.get('problem', ''), r.get('ground_truth', '')))
            selected_rows += 1

out_path = Path(args.out_jsonl)
out_path.parent.mkdir(parents=True, exist_ok=True)
written = 0
with open(args.source_train_jsonl, 'r', encoding='utf-8') as fin, open(out_path, 'w', encoding='utf-8') as fout:
    for line in fin:
        if not line.strip():
            continue
        obj = json.loads(line)
        ans = obj.get('solution', obj.get('ground_truth', obj.get('answer', obj.get('original_answer', ''))))
        if (obj.get('problem', ''), ans) in selected:
            fout.write(json.dumps(obj, ensure_ascii=False) + '\n')
            written += 1

summary = {
    'source_train_jsonl': args.source_train_jsonl,
    'pass8_jsonl': args.pass8_jsonl,
    'out_jsonl': args.out_jsonl,
    'selected_keys': len(selected),
    'written': written,
}
with open(str(out_path) + '.summary.json', 'w', encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print(json.dumps(summary, ensure_ascii=False))
