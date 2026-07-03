# RWKV RLHF Experiments

This repository is a code archive for the RWKV reinforcement-learning and reasoning experiments run on the server. The project focuses on making RWKV-7 models work under RLVR/GRPO-style training, then comparing the behavior against Qwen/TRL reference recipes and Albatross evaluation standards.

The repository is code-only. Model weights, datasets, checkpoints, generated responses, large logs, caches, and virtual environments are intentionally excluded.

## Project Goals

- Reproduce public GRPO/RLVR recipes on small reasoning models, especially the TRL GRPO documentation and Qwen baselines.
- Port the same training and evaluation ideas to RWKV-7 models, mainly G1D/G1E/G1F/G1G variants around 0.4B, 1.5B, 3B, and 7B.
- Understand why RWKV math RL differs from Qwen: truncation, stop-token behavior, verifier mismatch, rollout sampling, KL spikes, entropy, length drift, and all0/all1 group collapse.
- Test data-engineering and curriculum ideas: hard buffer, pure hard, dynamic re-screening by pass@8, staged schedules, mixed datasets, and larger DeepMath/OpenMath-style training sets.
- Explore teacher-guided variants: OPD, OPSD/self-distillation, 7B-to-1.5B distillation, token-level teacher loss, and pure teacher-SFT baselines.
- Extend beyond math-only tasks into STEM-style benchmarks such as MMLU-Pro STEM and MMMU text-only experiments.

## Repository Layout

| Path | Role |
| --- | --- |
| `autodl-tmp_code/` | Main experiment workspace copied from `/root/autodl-tmp`: launch scripts, one-off patches, evaluation tools, sweep scripts, and per-run code snapshots. |
| `root_RWKV-LM/` | Local RWKV-LM source tree and RWKV-7/Albatross-related model code used as the main backend. |
| `root_Albatross/` | Albatross reference implementation used to align RWKV inference, fast rollout, and MATH500 evaluation behavior. |
| `root_OpenRLHF/` | OpenRLHF framework snapshot used during early PPO/RLHF reproduction attempts. |
| `root_verl/` | veRL framework snapshot used as a reference for GRPO-style RL implementations and advantage estimators. |
| `root_top_level/` | Top-level server helper scripts for evaluation, cleanup, inspection, patching, and reporting. |
| `CODE_INDEX.md` | Generated file-level index for code/config/script files. |

Each major directory also has a local README describing how it fits into the experiment history.

## Main Experiment Lines

### 1. GSM8K RWKV Full-Parameter RLVR

The earliest stable line used GSM8K with full-parameter RWKV training. The experiments compared no-buffer, hard-buffer, pure-hard, KL-regularized hard-buffer, mixed-extra-data, and dynamic re-screening variants. The key diagnostics were full eval accuracy, small eval accuracy, pass@1, pass@8, all0/all1 proportions, group `num_correct`, KL, entropy, zstd, average length, and truncation rate.

Important themes in this line:

- full-parameter tuning rather than state-only tuning;
- hard examples can help, but persistent hard-buffer pressure can cause mid-training regression;
- pass@8-based filtering is only useful when enough groups are mixed instead of all0/all1;
- reward details such as k3 loss, zstd reward, length reward, and KL coefficient materially changed behavior;
- checkpoint saving policy became important because many promising runs were only visible after full eval.

### 2. MATH500 RWKV RLVR

The MATH500 line moved the same GRPO/RLVR machinery to harder math. This exposed RWKV-specific problems: high truncation, poor stop behavior, repeated reasoning after solving, verifier sensitivity, and much lower pass@8 than Qwen under comparable prompts.

The evaluation standard was later aligned to BlinkDL Albatross `faster3a_2605/eval_math500.py`:

- prompt style:

```text
User: {problem}\n\nAssistant: <think></think
```
- rollout count: 4 for evaluation
- max new tokens: 1500
- temperature: 1.0
- top-p: 0.28
- top-k: 32
- verifier: `math_verify.parse` plus `math_verify.verify(strict=False)`
- no stop-on-boxed shortcut

Any result produced with older regex/last-number verification or different sampling should be treated as a separate evaluation protocol.

### 3. TRL / Qwen GRPO Reproduction

Several runs reproduced the Hugging Face TRL GRPO documentation and Qwen small-model behavior before porting the method to RWKV. This line was used to debug reward curves, verifier parsing, rollout length, `math_verify` behavior, and the gap between Qwen and RWKV on MATH500-style reasoning.

The main engineering lesson was that prompt, verifier, and sampling alignment matter as much as the optimizer: small changes in max-new-tokens, stop logic, top-p/top-k, and boxed-answer parsing can dominate the apparent RL gain.

### 4. DeepMath / OpenMath / GSM8K Mixed Data

The DeepMath/OpenMath line tested larger and more diverse math data against the smaller GSM8K-only setup. These experiments include reward-on/off variants, length reward ablations, microbatch changes for memory fitting, max-new-token changes, and attempts to reproduce stronger public small-model math RL curves.

This line is useful for studying whether RWKV benefits from broader math data or whether the bottleneck is model/prompt/rollout behavior rather than data volume.

### 5. OPD / OPSD / Teacher-Guided Training

The OPD line tested teacher-guided RL using RWKV teacher/student variants, including 7B teacher to 1.5B student attempts, token-level distillation ideas, chunked teacher forward, reduced max-new-tokens, and pure-GRPO parity checks inside the OPD codepath.

The main debugging target was making `distill_coef=0` reproduce pure GRPO. Without that parity check, OPD gains cannot be interpreted. Related scripts also explore OPSD/self-distillation and teacher-trace SFT baselines.

### 6. RLM / REPL-Style Reasoning

The RLM line explored recursive or REPL-style reasoning protocols inspired by Recursive Language Models and related long-context/compressed-memory papers. The purpose was to see whether RWKV's short-chain strengths could be converted into iterative reasoning behavior instead of relying on very long uninterrupted CoT generations.

### 7. STEM Benchmarks Outside Math500

Additional experiments targeted MMLU-Pro STEM and MMMU text-only evaluation/training. These runs tested prompt formats, boxed-slot variants, CoT vs direct-answer behavior, truncation, and verifier consistency outside the math-only RLVR setup.


## Verified Experiment Results

The table below records results that are present in the exported repository files, mainly `summary.json` files. Metrics from different rows are not always comparable because prompt, verifier, rollout count, and sampling settings changed across experiment lines.

| Experiment / source file | Model / setting | Eval protocol | Key result | Notes |
| --- | --- | --- | --- | --- |
| `autodl-tmp_code/rwkv-rl-good-experiments-github/README.md` | GSM8K dynamic re-screen snapshot | Full GSM8K eval, n=1319 | best comparable full eval `0.6611`, stage 2 step 100 | Curated historical result from collected summaries. |
| `autodl-tmp_code/rwkv-rl-good-experiments-github/README.md` | GSM8K hard-buffer/K3/zstd/length family | Full GSM8K eval, n=1319 | best comparable full eval about `0.6262` | Same note records small-eval `0.7500`, but that was n=164 and should not be reported as full GSM8K. |
| `autodl-tmp_code/rwkv-rl-good-experiments-github/README.md` | GSM8K real-label variant | Full GSM8K eval, n=1319 | best full eval around `0.6065` | Useful lower reference for label-correct/real-label branch. |
| `autodl-tmp_code/rwkv-rl-good-experiments-github/README.md` | Math500 direct RL snapshots | Full MATH500 eval, n=500 | reference around `0.4000` | Curated note; exact protocol should be checked before comparing to Albatross-aligned eval. |
| `autodl-tmp_code/base_gsm8k_real_pass8_20260422/summary.json` | RWKV7 G1E 1.5B base | GSM8K test subset, 164 questions, group size 8, temp 0.3, top-p 0.4, top-k 500 | sample avg acc `0.5724`, pass@8 `0.6159` | Conservative eval sampling produced lower pass@8. |
| `autodl-tmp_code/base_gsm8k_real_pass8_rolloutparams_20260422/summary.json` | RWKV7 G1E 1.5B base | GSM8K test subset, 164 questions, group size 8, temp 1.0, top-p 0.6, top-k 0 | sample avg acc `0.5381`, pass@8 `0.8354` | Rollout-style sampling greatly increased pass@8 while lowering average per-sample acc. |
| `autodl-tmp_code/train_gsm8k_real_pass8_rolloutparams_20260425/summary.json` | RWKV7 G1E 1.5B base | GSM8K train full, 7473 questions, group size 8, temp 1.0, top-p 0.6 | sample avg acc `0.5839`, pass@8 `0.8834` | Pre-RL train-set pass@8 baseline used for data filtering discussion. |
| `autodl-tmp_code/pass8_step150_full_20260425/summary.json` | G1E staged core-sub checkpoint step 150 | GSM8K train full, 7473 questions, group size 8 | sample avg acc `0.6877`, pass@8 `0.8571` | Training raised sample avg acc but reduced pass@8 relative to the base train-set pass@8 baseline. |
| `autodl-tmp_code/pass8_step150_subset_20260425/summary.json` | Same step-150 checkpoint | GSM8K filtered core-subset, 2093 questions, group size 8 | sample avg acc `0.5770`, pass@8 `0.8528` | Used to inspect train-subset transfer and cases that fell to 0/8. |
| `autodl-tmp_code/g1e_dynamic_rescreen_20260426_020719/stage_1/pass8_full/summary.json` | Dynamic re-screen stage 1 | GSM8K train full, 7473 questions | sample avg acc `0.6955`, pass@8 `0.8840` | Best train-set sample avg in the exported dynamic re-screen summaries. |
| `autodl-tmp_code/g1e_dynamic_rescreen_20260426_020719/stage_2/pass8_full/summary.json` | Dynamic re-screen stage 2 | GSM8K train full | sample avg acc `0.6998`, pass@8 `0.8620` | Further sample avg gain, but pass@8 declined. |
| `autodl-tmp_code/g1e_dynamic_rescreen_20260426_020719/stage_4/pass8_full/summary.json` | Dynamic re-screen stage 4 | GSM8K train full | sample avg acc `0.6900`, pass@8 `0.8339` | Later stages show pass@8 erosion despite high sample avg acc. |
| `autodl-tmp_code/g1e_dynamic_rescreen_20260426_020719/math500_pass8_compare/pre/summary.json` | RWKV7 G1E 1.5B base | MATH500 test, group size 8, temp 1.0, top-p 0.6 | sample avg acc `0.3293`, pass@8 `0.5860` | Older MATH500 evaluation protocol, not Albatross-aligned. |
| `autodl-tmp_code/g1e_dynamic_rescreen_20260426_020719/math500_pass8_compare/post/summary.json` | Dynamic re-screen final stage checkpoint | MATH500 test, same older pass@8 protocol | sample avg acc `0.2430`, pass@8 `0.4560` | Negative transfer to MATH500 under this protocol. |
| `autodl-tmp_code/eval_math500_rwkv7_g1f_1p5b_full_20260607/summary.json` | RWKV7 G1F 1.5B base | Albatross-style MATH500, rollout 1, max new 1500, temp 1.0, top-p 0.28, top-k 32 | acc `0.3960`, truncation `0.174`, mean generated tokens `603.8` | Full MATH500 baseline under the later aligned evaluator. |
| `autodl-tmp_code/eval_math500_rwkv7_g1f_7p2b_full_20260607/summary.json` | RWKV7 G1F 7.2B base | Same Albatross-style MATH500, rollout 1 | acc `0.6400`, truncation `0.094`, mean generated tokens `526.8` | Shows large base-model gap between 1.5B and 7.2B RWKV. |
| `autodl-tmp_code/albatross_math500_run6/summary.json` | RWKV7 G1F 1.5B base in `/dev/shm` | Albatross-style MATH500, rollout 4 | rollout accuracy `0.4015`, pass@4 `0.5040`, truncation `0.180` | Multi-rollout eval improved pass@k but kept per-sample acc near the rollout-1 baseline. |
| `autodl-tmp_code/stem-rlvr-repro/evals/qwen25_1p5b_pre_brief_eval1176.jsonl.summary.json` | Qwen2.5-1.5B-Instruct | STEM eval split, brief prompt, 1176 items | acc `0.1922`, no-parse `0.1352` | Pre-training reference for non-math RLVR exploration. |
| `autodl-tmp_code/stem-rlvr-repro/evals/qwen25_1p5b_pre_eval200.jsonl.summary.json` | Qwen2.5-1.5B-Instruct | STEM eval subset, non-brief prompt, 200 items | acc `0.1050`, no-parse `0.4500` | Prompt format strongly affected parseability. |
| `autodl-tmp_code/stem-rlvr-repro/evals/qwen25_1p5b_pre_brief_eval200.jsonl.summary.json` | Qwen2.5-1.5B-Instruct | STEM eval subset, brief prompt, 200 items | acc `0.1850`, no-parse `0.1350` | Brief prompt improved parse rate and accuracy on the same-size pre-eval subset. |

### Main Takeaways From The Verified Results

- Best curated full-GSM8K result in the exported snapshot index is dynamic re-screen stage 2 step 100: `0.6611` on n=1319; the often-noted `0.7500` is explicitly a small eval with n=164, not full GSM8K.
- RWKV G1F 7.2B is much stronger than G1F 1.5B on Albatross-style MATH500: `0.6400` vs `0.3960` rollout-1 accuracy.
- Increasing rollout count helps pass@k: G1F 1.5B has rollout-1 acc `0.3960`, while rollout-4 pass@4 reaches `0.5040` with similar per-sample accuracy.
- GSM8K train-set pass@8 was already high before RL (`0.8834`), so many data-engineering runs mainly changed pass distribution rather than creating new solvable mass.
- Dynamic re-screening improved GSM8K train sample average (`0.5839` base to about `0.70` in stages), but pass@8 declined in later stages, consistent with narrowing/diversity loss.
- GSM8K-oriented dynamic re-screening did not transfer to MATH500 in the exported comparison: MATH500 sample avg fell from `0.3293` to `0.2430` under the older pass@8 protocol.
- STEM multiple-choice experiments showed that prompt/format parseability can dominate score: Qwen2.5-1.5B brief prompt had far lower no-parse and higher accuracy than the non-brief prompt.

## Evaluation Discipline

When comparing experiments, keep these fields fixed or explicitly label them as changed:

- dataset split and whether eval is full or subset;
- prompt template and answer format;
- verifier/parser implementation;
- rollout count and sampling parameters;
- max-new-token limit and stop logic;
- reward scale: `0/1` vs `-1/+1`;
- trainable scope: full-parameter, state tuning, LoRA, or SFT-only;
- checkpoint step and whether the metric is moving average, small eval, or full eval.

For pass-k analysis, the most important group buckets are all0, mixed, and all1. Methods like hard-buffer re-screening or MaxRL-style advantages only have useful learning signal in mixed groups.

## What Is Excluded

The following files were not committed:

- RWKV/Qwen/Llama model weights and checkpoints;
- datasets and HF cache;
- generated response JSONL files and rollout dumps;
- large logs, temporary directories, and environment folders;
- conda/venv/cache directories.

Most launch scripts still contain absolute server paths such as `/root/autodl-tmp/...`. To rerun an experiment, restore the required external assets and either recreate those paths or edit the script paths.

## Recommended Navigation

1. Start with `autodl-tmp_code/README.md` for the experiment map.
2. Use `CODE_INDEX.md` to locate a specific training/eval/patch script.
3. For MATH500 evaluation, inspect `autodl-tmp_code/eval/` and the Albatross reference under `root_Albatross/` or `autodl-tmp_code/Albatross_ref_tmp/`.
4. For old GSM8K/RWKV training variants, search `autodl-tmp_code/` for `gsm8k`, `hardbuffer`, `pass8`, `purehard`, and `dynamic`.
5. For OPD/OPSD work, search `autodl-tmp_code/` for `opd`, `opsd`, `teacher`, and `distill`.
