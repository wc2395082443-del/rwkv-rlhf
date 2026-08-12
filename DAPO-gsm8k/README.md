# DAPO-gsm8k: Boxed Strict-CoT RL for RWKV7 g1i 1.5B

This project contains the cleaned code and lightweight result logs for a GSM8K RL run on RWKV7 g1i 1.5B. The experiment trains a boxed strict-CoT output format with a binary verifier reward.

## Prompt

```text
User: {problem}
Please reason step by step, and put the final answer within \boxed{}.
Assistant: <think>
```

Generation starts after the prefilled `<think>`. The model must close `</think>` and put the final answer after it, preferably as `\boxed{...}`.

## Reward / Eval Criterion

Strict reward is binary. A response receives `1` only if all conditions hold:

- final answer is correct under the verifier;
- exactly one valid `</think>` structure is present;
- final answer is extracted after `</think>`, preferably from `\boxed{}`;
- generation ends with EOS;
- response is not truncated;
- response does not trip the repeat / multilingual / degeneration gate.

Loose accuracy only checks final-answer correctness.

## Main Training Setup

- Base model: `rwkv7-g1i_preview5445-1.5b-20260729-ctx16384.pth`
- Dataset: GSM8K train / GSM8K test
- Full-parameter RL, bf16 forward with FP32 master AdamW
- No 8-bit optimizer, no CPU offload
- `max_new_tokens=2048`
- rollout: `temperature=1.0`, `top_p=1.0`, `top_k=0`
- eval: `temperature=0.3`, `top_p=0.4`, `top_k=500`
- dynamic sampling: `32` candidate questions, `16` rollouts/question
- `rollout_forward_batch=512`
- `micro_batch=3`
- `lr=2e-7`
- `kl_coef=0`
- `length_weight=0`
- train responses are not saved; eval responses are saved by `eval_gsm8k_boxed_strict_reward.py`.

## Key Result: Full GSM8K Eval

Full eval uses all 1319 GSM8K test questions and pass@1 generation.

| Model | Strict Acc | Loose Acc | Format | EOD | Trunc | Mean Tokens |
|---|---:|---:|---:|---:|---:|---:|
| Base g1i 1.5B | 53.15% | 53.22% | 68.16% | 68.46% | 1.67% | 792.3 |
| Step85 RL checkpoint | 67.25% | 67.32% | 96.13% | 96.21% | 0.08% | 466.7 |
| Gain | +14.10 pp | +14.10 pp | +27.98 pp | +27.75 pp | -1.59 pp | -325.6 |

The full-eval summary is in `results/base_vs_step85_full_eval_summary.json`.

## Continuation Runs

Continuation from Step85 for 20 steps stopped early at global step 105:

- sample pre-eval at step85 checkpoint: strict `66.29%` on 264 questions;
- eval at step100: strict `68.56%` on 264 questions;
- post-eval at step105: strict `68.94%` on 264 questions;
- final checkpoint: `final_step_105.pth` (not uploaded).

A further fixed 50-step continuation from step105 reached:

- pre-eval at step105 checkpoint: strict `69.32%` on 264 questions;
- eval at step125: strict `70.45%` on 264 questions;
- eval at step150: strict `70.83%` on 264 questions;
- post-eval at step155: strict `69.70%` on 264 questions;
- final checkpoint: `final_step_155.pth` (not uploaded).

These continuation numbers are sample evals, not full evals.

## Files

- `train_rl_thinking_cot_strict_boxedprompt_schedopt_earlystop.py`: main training script used for the successful run.
- `eval_gsm8k_boxed_strict_reward.py`: eval script; saves full `responses.jsonl` for eval runs.
- `reward.py`: format, answer extraction, degeneration gates.
- `infer_gpu_buffered_compact.py`: batched RWKV rollout/inference helper.
- `scripts/`: launch scripts for base-to-step85, continuations, and full eval.
- `results/`: lightweight metrics and validation summaries.
- `examples/step85_random10_responses.json`: 10 sampled Step85 eval responses for qualitative inspection.

## Notes

Checkpoints, base weights, data files, GPU monitor CSVs, logs, and full eval response files are intentionally not included. The full response files were kept on the training server during the experiment, but are too large/noisy for the cleaned code branch.

## Results

- [10 wrong-to-right qualitative examples](results/wrong_to_right_10.md): base failed strict eval, step85 passed strict eval.
