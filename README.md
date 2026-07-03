# RWKV RLHF Experiments

This repository is a cleaned code export from the RWKV/RLHF experiment server. It preserves the code, launch scripts, configs, evaluation utilities, and framework snapshots used across the RWKV-7 math RLVR experiments.

## What Is Included

| Path | Contents |
| --- | --- |
| `autodl-tmp_code/` | Experiment-specific scripts and configs from `/root/autodl-tmp`, including GSM8K, MATH500, DeepMath, OPD/OPSD, RLM, MaxRL-related helpers, and sweep launchers. |
| `root_RWKV-LM/` | Local RWKV-LM tree and RWKV-7/Albatross-related model code. |
| `root_Albatross/` | Albatross reference implementation used to align MATH500 prompt, verifier, sampling, and stop behavior. |
| `root_OpenRLHF/` | OpenRLHF framework snapshot used for early PPO/GRPO/RLHF reproduction attempts. |
| `root_verl/` | veRL framework snapshot used as a reference for GRPO/RLVR algorithm implementations. |
| `root_top_level/` | Top-level server helper scripts for evaluation, inspection, cleanup, patching, and reporting. |

## What Is Not Included

Large or sensitive runtime artifacts were intentionally excluded:

- model weights: `*.pth`, `*.pt`, `*.ckpt`, `*.bin`, `*.safetensors`
- datasets, HF cache, conda/venv/cache directories
- checkpoint/model/output/response directories
- generated rollout/eval response files
- large logs and temporary generated artifacts

## Code Index

GitHub does not support arbitrary per-file descriptions in the repository file list. The text shown next to a filename is the latest commit message for that file. For maintainable file-level descriptions, use:

- [`CODE_INDEX.md`](CODE_INDEX.md) - generated index of code/config/script files with short descriptions.

## Main Experiment Areas

- **RWKV GRPO/RLVR on GSM8K and MATH500**: full-parameter RL training, hard-buffer variants, dynamic resampling, pass@k analysis, and full-eval scripts.
- **DeepMath/TRl document reproduction**: Qwen/RWKV-aligned GRPO runs, reward/verifier fixes, and rollout parameter sweeps.
- **OPD / OPSD / teacher distillation**: online policy distillation and self-distillation experiments using RWKV 1.5B/7B teacher-student variants.
- **RLM / REPL-style experiments**: recursive-language-model inspired prompt/protocol experiments for short-chain and long-chain reasoning.
- **Evaluation alignment**: MATH500 evaluation aligned with BlinkDL Albatross settings, including fake-think prompt style, `math_verify`, rollout accuracy, and pass@rollout metrics.
- **Framework references**: OpenRLHF, veRL, Helicopter, and Albatross snapshots kept to preserve implementation context.

## Reproducing From This Export

This repository is code-only. To rerun experiments, restore the corresponding external assets manually:

1. Download or mount the required RWKV/Qwen model weights.
2. Restore datasets such as GSM8K, MATH500, DeepMath, MMLU-Pro, or MMMU to the expected local paths.
3. Install the matching Python/CUDA environment for the selected experiment stack.
4. Use the launch scripts under `autodl-tmp_code/` as the source of truth for historical hyperparameters.
5. Use `autodl-tmp_code/eval/` and Albatross-aligned evaluation scripts for comparable metrics.

## Notes

- Some paths inside scripts still reflect the original server layout, e.g. `/root/autodl-tmp/...`.
- This is an experiment archive, not a polished package. Many scripts are one-off debugging, patching, or sweep utilities.
- Prefer reading the launch script and matching evaluation script together; prompt, verifier, sampling temperature/top-p/top-k, and max-new-token settings materially change results.
