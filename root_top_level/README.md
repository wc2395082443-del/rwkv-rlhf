# `root_top_level/`

Top-level helper scripts copied from `/root` on the experiment server.

## Main Contents

- Evaluation helpers for MMLU-Pro STEM, MMMU text-only, and selected RWKV checkpoints.
- Inspection scripts for completions, parquet data, logits/logprob consistency, and model shape/state issues.
- Patch scripts used to modify old training/eval code during debugging.
- Cleanup scripts and cleanup summaries from disk-space management.
- Small launchers and summarizers used during learning-rate sweeps or experiment comparisons.

These files are mostly operational utilities rather than reusable library code. Read them together with the run directory or log they were created for.
