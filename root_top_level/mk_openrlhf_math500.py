import json
from pathlib import Path
import pandas as pd

out_dir = Path('/root/autodl-tmp/data/openrlhf_math500')
out_dir.mkdir(parents=True, exist_ok=True)
for split in ['train', 'test']:
    df = pd.read_parquet(f'/root/autodl-tmp/data/verl_math500/{split}.parquet')
    out = out_dir / f'{split}.jsonl'
    with out.open('w', encoding='utf-8') as f:
        for p, rm in zip(df['prompt'], df['reward_model']):
            rec = {
                'prompt': p.tolist() if hasattr(p, 'tolist') else p,
                'label': rm.get('ground_truth', ''),
                'datasource': 'math500',
            }
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    print(split, len(df), out)
