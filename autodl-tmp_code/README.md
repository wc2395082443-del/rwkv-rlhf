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

## Verified Result Files

Several compact result summaries were preserved in this export. The most useful ones are:

| File | What it records |
| --- | --- |
| `base_gsm8k_real_pass8_20260422/summary.json` | G1E 1.5B base GSM8K test-subset pass@8 under conservative eval sampling. |
| `base_gsm8k_real_pass8_rolloutparams_20260422/summary.json` | Same base model with rollout-style sampling, showing much higher pass@8. |
| `train_gsm8k_real_pass8_rolloutparams_20260425/summary.json` | Full GSM8K train-set pass@8 baseline before data filtering. |
| `pass8_step150_full_20260425/summary.json` | Step-150 checkpoint pass@8 on full GSM8K train set. |
| `pass8_step150_subset_20260425/summary.json` | Same checkpoint evaluated on the filtered core subset. |
| `g1e_dynamic_rescreen_20260426_020719/stage_*/pass8_full/summary.json` | Stage-wise dynamic re-screen pass@8 distribution on GSM8K train. |
| `g1e_dynamic_rescreen_20260426_020719/math500_pass8_compare/{pre,post}/summary.json` | MATH500 pre/post transfer check for the dynamic re-screen line. |
| `eval_math500_rwkv7_g1f_1p5b_full_20260607/summary.json` | Albatross-style MATH500 full eval for RWKV7 G1F 1.5B. |
| `eval_math500_rwkv7_g1f_7p2b_full_20260607/summary.json` | Albatross-style MATH500 full eval for RWKV7 G1F 7.2B. |
| `albatross_math500_run6/summary.json` | Albatross-style MATH500 rollout-4 eval for RWKV7 G1F 1.5B. |
| `stem-rlvr-repro/evals/*.summary.json` | Qwen2.5-1.5B pre-eval summaries for STEM-style benchmark prompts. |

See the root `README.md` for the summarized metrics table.

## Curated Snapshot Results

The curated snapshot folder `rwkv-rl-good-experiments-github/` records several historical result notes:

- GSM8K dynamic re-screen: best comparable full eval `0.6611`, n=1319, stage 2 step 100.
- GSM8K hard-buffer/K3/zstd/length family: best comparable full eval about `0.6262`, n=1319.
- The `0.7500` hard-buffer result is a small eval with n=164, not a full GSM8K result.
- GSM8K real-label variant: best full eval around `0.6065`, n=1319.
- Math500 direct RL snapshots include a full n=500 reference around `0.4000`, but protocol details should be checked before comparing with Albatross-aligned evals.
