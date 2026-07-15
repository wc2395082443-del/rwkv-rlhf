# Weekly Report - 2026-07-15

This report records the latest RWKV-7 G1F 1.5B reinforcement-learning experiments. Large model weights, checkpoints, generated responses, datasets, and raw logs remain outside the repository.

## 2. Progress Advantage

We tested Progress Advantage scoring by comparing the policy trajectory with the reference policy and reranking the eight sampled responses for each GSM8K question.

| Model / checkpoint | First-sample accuracy | Progress Advantage result |
| --- | ---: | ---: |
| Original G1G | 53.071% | Best PA reranking: 48.067% |
| GSM8K dynamic-rescreen Stage 2, step 50 | 65.353% | PA positive-fraction reranking: 67.248% |
| Final dynamic-rescreen model | 65.959% | PA positive-fraction reranking: 64.973% |

The isolated Stage 2 step-50 gain did not reproduce on the final checkpoint. Overall, Progress Advantage was not reliable enough to replace ordinary sampling or GRPO.

## 3. AntiDoom on MATH500

Because RWKV had a high truncation rate on MATH500, we tested an AntiDoom-style degeneration detector. Degenerate prefixes were used to construct `chosen` and `rejected` pairs, followed by 50 FTPO steps.

| Metric | Before FTPO | After FTPO |
| --- | ---: | ---: |
| Accuracy | 40.04% | 38.67% |
| Truncation rate | 50.00% | 43.46% |
| Average length | 586 tokens | 542 tokens |
| Pass-at-group | 41.41% | 41.80% |

AntiDoom reduced truncation and response length, but paid a small single-sample accuracy cost. It is useful as a degeneration-control mechanism, not as a direct accuracy improvement.

## 4. DeepMath Split and GRPO Ablations

Following the data-split recommendation, DeepMath was partitioned with seed 42:

| Split | Examples |
| --- | ---: |
| Train | 91,602 |
| Validation | 5,092 |
| Test | 5,145 |

The current `neg_adv_weight=1.0` GRPO branch used:

- 8 questions per rollout step and 8 samples per question;
- `max_new_tokens=1024`;
- rollout sampling `temperature=1`, `top_p=0.6`, `top_k=0`;
- learning rate `5e-7`, PPO1, K3 KL loss with coefficient `0.05`;
- `neg_adv_weight=1.0`;
- full-parameter BF16 ZeRO-3 offload;
- no hard buffer, length reward, zstd reward, or n-gram penalty.

The clean base-model pre-evaluation was `8.936%`. The selected `negw=1` continuation produced:

| Configuration | Full validation accuracy |
| --- | ---: |
| PPO4 continuation, `neg_adv_weight=1.0` | 28.123% |

The `28.123%` result is therefore an ablation result from the `negw=1` branch, not a direct base-to-post measurement.

### Standard-GRPO Ablation

We then removed the MaxRL-inspired group-correct-count scaling, used ordinary group reward standardization, and set `neg_adv_weight=1.0`:

| Step | Standard GRPO accuracy |
| ---: | ---: |
| 100 | 25.000% |
| 200 | 26.394% |
| 300 | 21.760% |

The standard-GRPO run learned faster early, but became unstable after step 200. By step 300, the average response length collapsed to about 19 tokens and the model frequently emitted short boxed guesses instead of a reasoning trace. The step-300 checkpoint save also hit a ZeRO-3 `param in flight` assertion; step-100 and step-200 checkpoints were saved successfully.

A follow-up run with the same standard-GRPO design and `lr=2e-7` is in progress. Its purpose is to test whether reducing the learning rate prevents the late-training collapse.

## Conclusions

1. Progress Advantage produced an isolated gain but no stable final-checkpoint improvement.
2. AntiDoom reduced truncation at a modest accuracy cost.
3. DeepMath gives a much larger in-domain RL signal than the earlier GSM8K and MATH500 runs.
4. The `negw=1` branch needs a lower learning rate after removing the MaxRL-inspired scaling; `5e-7` is too aggressive for the standard-GRPO formulation.
5. The next controlled comparison is standard GRPO at `2e-7`, followed by full validation at steps 100 and 200.
