# `autodl-tmp_code/`

This is the main experiment workspace copied from `/root/autodl-tmp`. It contains the scripts and code snapshots that reflect most of the project history.

## Main Contents

- GSM8K RWKV GRPO/RLVR runs: hard buffer, no buffer, pure hard, dynamic re-screening, pass@8 analysis, and checkpoint eval helpers.
- MATH500 RWKV runs: 1.5B/3B/7B experiments, Albatross-style eval alignment, max-new-token changes, stop/truncation diagnostics, and rollout parameter sweeps.
- DeepMath/OpenMath reproduction attempts: reward variants, length reward ablations, memory-fitting runs, and TRL-document-style training.
- OPD/OPSD work: teacher-student distillation, 7B teacher attempts, token-level distillation diagnostics, and pure-GRPO parity checks.
- RLM/REPL-style experiments: recursive reasoning protocol tests and short-chain reasoning variants.
- STEM benchmark utilities: MMLU-Pro STEM and MMMU text-only evaluation/training scripts.

## How To Use

- `run_*.sh` files usually preserve the exact launch command for an experiment.
- `current_*.path` files often point to the latest or selected historical run directory from that line of work.
- `eval/` contains the more recent evaluation code, including the Albatross-aligned MATH500 eval variant.
- Scripts named `patch_*`, `fix_*`, `scan_*`, `inspect_*`, and `summ*` are one-off debugging or migration utilities.

## Comparison Warning

Many runs are not directly comparable unless prompt, verifier, rollout sampling, max-new-tokens, stop logic, reward scale, and checkpoint step are aligned. Prefer full eval over small eval when reporting final accuracy.
