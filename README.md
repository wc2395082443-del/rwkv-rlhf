# Progress Advantage

This project compares policy trajectories against a reference policy using token-level log-probability progress, then evaluates several response reranking rules.

## Source Files

- `gsm8k_progress_advantage_eval.py`: rollout, reference scoring, progress scoring, reranking, and summary generation.
- `run_gsm8k_progress_advantage_paper.sh`: two-stage GSM8K GRPO run followed by Progress Advantage evaluation.

## Reported Results

- Original G1G: first-sample accuracy `53.071%`; best PA reranking `48.067%`.
- Dynamic-rescreen Stage 2 step 50: first-sample accuracy `65.353%`; PA positive-fraction reranking `67.248%`.
- Final dynamic-rescreen model: first-sample accuracy `65.959%`; PA positive-fraction reranking `64.973%`.

The isolated step-50 gain did not reproduce on the final checkpoint.
