# AntiDoom on MATH500

`antidoom_rwkv.py` implements degeneration-prefix mining, preference-pair training, and evaluation for RWKV.

## Commands

```bash
python antidoom_rwkv.py mine --model MODEL --dataset MATH500.jsonl --out_dir mine_out
python antidoom_rwkv.py train --model MODEL --pairs mine_out/antidoom_pairs.jsonl --out_dir ftpo_out --steps 50 --lr 5e-7
python antidoom_rwkv.py eval --model ftpo_out/final_step_50.pth --dataset MATH500.jsonl --out_dir eval_out --max_samples 1024 --group_size 4 --max_new_tokens 768 --temperature 1 --top_p 0.28 --top_k 32
```

## Reported Results

After 50 FTPO steps, accuracy changed from `40.04%` to `38.67%`, truncation from `50.00%` to `43.46%`, and average length from 586 to 542 tokens. Pass-at-group changed from `41.41%` to `41.80%`.

The method reduced degeneration and truncation, but did not improve single-sample accuracy.
