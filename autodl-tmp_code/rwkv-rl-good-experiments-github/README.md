# RWKV RL Experiment Code Snapshots

Curated source-code snapshots for RWKV RL / GRPO-style math-reasoning experiments. This repository intentionally excludes model weights, checkpoints, generated replies, datasets, and large binary artifacts.

## Included snapshots

| Snapshot | Source path | Why included |
|---|---|---|
| `gsm8k_dynamic_rescreen_best_full` | `/root/RWKV-LM/RWKV7-statetuning_dynresample_v1` | Dynamic rescreen / dynamic sampling. Best comparable full GSM8K eval observed: 0.6611, n=1319, stage_2 step=100. Files: 27 |
| `gsm8k_hardbuffer_k3_zstd_length` | `/root/RWKV-LM/RWKV7-statetuning_hardbuffer_v1` | Hard buffer family with K3/negative-weight/zstd/length reward variants. Full GSM8K best in summary: 0.6262 n=1319; small eval 0.7500 n=164 exists but is not full eval. Files: 27 |
| `gsm8k_state_tuning_mainline` | `/root/RWKV-LM/RWKV7-statetuning` | Early GRPO/state-tuning mainline. Included because many early high small-eval and baseline experiments point here. Files: 71 |
| `gsm8k_real_label_variant` | `/root/RWKV-LM/RWKV7-statetuning_real_v1` | Real-label / label-correct variant. Full GSM8K best in summary around 0.6065 n=1319. Files: 31 |
| `math500_hardbuffer` | `/root/RWKV-LM/RWKV7-statetuning_math500_hb_v1` | Math500 hard-buffer code family; also used by several GSM8K hard-buffer result dirs through copied code lineage. Files: 27 |
| `math500_dynamic_resample` | `/root/RWKV-LM/RWKV7-statetuning_math500_dynresample_v1` | Math500 dynamic resampling version. Files: 27 |
| `rwkv_deepmath_trl_doc_baseline` | `/root/RWKV-LM/RWKV7-deepmath_trl_doc_g1f1p5b` | RWKV implementation aligned to TRL GRPO doc / DeepMath-style runs. Files: 17 |
| `rwkv_deepmath_fp32_noentropy` | `/root/RWKV-LM/RWKV7-deepmath_trl_doc_g1f1p5b_fp32kernel_noentropy` | FP32-kernel/no-entropy variant used after precision and entropy-memory debugging. Files: 17 |
| `rwkv_deepmath_fastrollout_traintempv3` | `/root/RWKV-LM/RWKV7-deepmath_trl_doc_g1f1p5b_fastrollout_traintempv3` | Fast rollout / train_temp v3 version. Included as the useful speed-optimized branch. Files: 17 |
| `rwkv_gsm8k_rlcsd` | `/root/RWKV-LM/RWKV7-gsm8k_rlcsd_20260615` | RLCSD reproduction branch for GSM8K. Included as a paper-aligned comparable algorithm branch. Files: 19 |
| `qwen_trl_grpo_reference` | `/root/autodl-tmp/trl-grpo-repro` | Qwen/TRL reference reproduction scripts; useful control for comparing RWKV against TRL doc settings. Files: 18 |

## Result context

Important caveat: old experiments used mixed eval sizes. `n=1319` is full GSM8K, `n=500` is full Math500, while `n=164`, `n=263`, `n=32`, etc. are small evals.

Key references from the collected summaries:

- GSM8K dynamic rescreen: best comparable full eval `0.6611`, `n=1319`, stage 2 step 100.
- GSM8K hard-buffer/K3 family: best comparable full eval around `0.6262`, `n=1319`; the `0.7500` run is small eval `n=164`, not full GSM8K.
- Math500 direct RL runs in these summaries are much lower; one full `n=500` reference is around `0.4000`.
- Qwen/TRL reference scripts are included as control code, not as RWKV model code.

## Manifests

- `manifests/snapshot_file_manifest.tsv`: exact copied files and SHA1 hashes.
- `manifests/code_versions_manifest_20260620.tsv`: experiment-result to code-dir mapping.
- `manifests/code_core_file_hashes_20260620.tsv`: core file hashes from the original server.
- `results/`: selected small metric files copied for reproducibility context.

## Excluded

- `*.pth`, `*.pt`, `*.safetensors`, checkpoints, model weights.
- Generated replies / rollouts / response dumps.
- Datasets and binary artifacts.
- Logs except compact summary/metrics files.
