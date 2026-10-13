#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from datasets import load_from_disk


def main():
    ap = argparse.ArgumentParser(description="Convert DeepMath TRL HF dataset to RWKV jsonl format")
    ap.add_argument('--dataset_path', type=str, default='/dev/shm/official_repro_assets/DeepMath-103K-trl-hf-official')
    ap.add_argument('--out_dir', type=str, required=True)
    args = ap.parse_args()

    ds = load_from_disk(args.dataset_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in ('train', 'test'):
        rows = ds[split]
        out_path = out_dir / f'deepmath_{split}_rwkv.jsonl'
        with out_path.open('w', encoding='utf-8') as f:
            for idx, row in enumerate(rows):
                prompt = row['prompt']
                problem = prompt[0]['content'] if isinstance(prompt, list) and prompt else str(prompt)
                answer = str(row['solution']).strip()
                rec = {
                    'id': idx,
                    'problem': problem,
                    'answer': answer,
                    'solution': answer,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')
        print(f'wrote {len(rows)} -> {out_path}')


if __name__ == '__main__':
    main()
