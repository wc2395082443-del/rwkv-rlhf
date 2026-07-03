# RWKV GRPO vLLM-RWKV Validation

Date: 2026-06-26

## vLLM-RWKV Checkout

- Repository: `/home/caizus/Projects/Packages/vllm-rwkv`
- Remote: `origin https://github.com/rwkv-rs/vllm-rwkv`
- Branch: `feat/rwkv-faster3a`
- HEAD: `2660d700c53223dae3fd05a9312c632d80c805e6`
- HEAD date: `2026-06-26 07:20:14 -0500`
- Subject: `feat(rwkv): add RWKV7 Albatross support`
- Fetch check: `HEAD == origin/feat/rwkv-faster3a`

Runtime import used by smoke:

- `PYTHONPATH=/home/caizus/Projects/Packages/vllm-rwkv`
- `vllm.__version__ = 0.23.1rc1.dev223+ga346d589f.d20260626`
- `vllm.__file__ = /home/caizus/Projects/Packages/vllm-rwkv/vllm/__init__.py`
- `vllm.model_executor.models.rwkv7 = /home/caizus/Projects/Packages/vllm-rwkv/vllm/model_executor/models/rwkv7.py`
- Metadata source: `/home/caizus/Projects/Packages/vllm-rwkv/vllm.egg-info`
- `uv pip show --python .venv/bin/python vllm` reports no installed package; deployment for this smoke is source checkout via `PYTHONPATH`.

## Source Evidence

Current `vllm-rwkv` RWKV7 implements transactional online weight updates:

- `load_weights()` buffers incoming bucket weights while `_pending_weight_update` is active.
- `finish_weight_update()` validates the complete key set against the initial full checkpoint key set.
- It only commits preprocessed weights after the complete update is present, and rolls back on failure.

The verl IPC path calls `start_weight_update()` before bucket receive and `finish_weight_update()` after all buckets are received when the model exposes those methods.

This matches the old failure shape: the bad W&B run starts diverging immediately after the first online update, with rollout/actor probability max diff approaching 1 while `actor/ppo_kl` remains 0.

## Old W&B Failure Snapshot

Source: `analysis/wandb_v26y41n6/history.csv`

| step | corr | prob diff max | prob diff mean | rollout_corr/kl | training log ppl | response mean | reward mean | ppo_kl |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.9999125004 | 0.0868653953 | 0.0028218697 | 0.0001852905 | 1.0007269382 | 1918.03125 | 0.453125 | 0 |
| 2 | 0.9674799442 | 0.9988043904 | 0.0458493233 | 0.0738722757 | 0.9999666214 | 1933.609375 | 0.296875 | 0 |
| 3 | 0.9242096543 | 0.9992305040 | 0.0805032924 | 0.2140049040 | 1.3364921808 | 1827.640625 | 0.21875 | 0 |
| 28 | 0.3389478922 | 1.0 | 0.4892009795 | 11.4183673859 | 11.5837831497 | 2631.890625 | 0.015625 | 0 |
| 29 | 0.3744088709 | 1.0 | 0.4178475440 | 8.1386356354 | 9.4656190872 | 2745.734375 | 0.0 | 0 |
| 30 | 0.3149457276 | 1.0 | 0.5103431940 | 10.7500972748 | 10.8425769806 | 2653.0 | 0.015625 | 0 |

## Latest 2-Step Smoke

Log: `analysis/rwkv_grpo_smoke_2step_latest.log`

Command shape:

- `RWKV_MODEL_PATH=/home/caizus/Weights/RWKV/rwkv7/pth/rwkv7-g1f-1.5b-20260419-ctx8192.pth`
- `RWKV_LM_PATH=/home/caizus/Projects/MachineLearning/rwkv-lm`
- `VLLM_RWKV_PATH=/home/caizus/Projects/Packages/vllm-rwkv`
- `PYTHONPATH=/home/caizus/Projects/Packages/vllm-rwkv`
- `VLLM_USE_FLASHINFER_SAMPLER=0`
- `TRAIN_BATCH_SIZE=1`
- `PPO_MINI_BATCH_SIZE=1`
- `PPO_MICRO_BATCH_SIZE=1`
- `MAX_RESPONSE_LENGTH=64`
- `ROLLOUT_N=1`
- `ROLLOUT_GPU_MEM_UTIL=0.65`
- `trainer.total_training_steps=2`
- `trainer.logger=["console"]`
- `actor_rollout_ref.rollout.agent.num_workers=1`

Result: command exit code 0.

| step | corr | prob diff max | prob diff mean | rollout_corr/kl | training log ppl | response mean | reward mean | update_weights |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.0 | 0.1362084746 | 0.0021418817 | 1.3766150475 | 12.2960090637 | 64.0 | 0.0 | 7.7520685070 |
| 2 | 0.9999991655 | 0.0000695581 | 0.0000111609 | 1.5029438734 | 12.4200143814 | 64.0 | 0.0 | 3.8058343350 |

Interpretation:

- The old W&B run loses rollout/actor probability alignment immediately after step 1.
- With latest `vllm-rwkv`, step 2 remains aligned after the first online update.
- `actor/ppo_kl=0` is not used as a health signal here; the decisive metrics are the rollout/actor probability correlation and probability diff.

The smoke also logs a weakref/DataLoader worker cleanup traceback after training completion, but the process exit code is 0 and both step metrics are emitted.

## Verification

Commands run from `/home/caizus/Projects/MachineLearning/verl-rwkv`:

```bash
rtk git diff --check
rtk .venv/bin/python -m pytest tests/utils/test_attention_utils_on_cpu.py tests/utils/test_padding_on_cpu.py tests/workers/rollout/test_vllm_weight_update_utils_on_cpu.py tests/rwkv/test_weight_mapping_on_cpu.py tests/rwkv/test_native_templates_on_cpu.py -q
rtk env PYTHONPATH=/home/caizus/Projects/Packages/vllm-rwkv .venv/bin/python -m pytest /home/caizus/Projects/Packages/vllm-rwkv/tests/model_executor/models/test_rwkv7.py -q -k 'online_weight_update or abort_weight_update or direct_parameter_update'
```

Results:

- `git diff --check`: passed
- verl tests: `28 passed, 1 skipped, 4 warnings`
- vllm-rwkv tests: `7 passed, 85 deselected, 14 warnings`

## Environment Notes

Installed/available in `.venv` during validation:

- `watchfiles`
- `model_hosting_container_standards`
- `socksio`
- `torchvision`
- `prometheus_fastapi_instrumentator`

Not available:

- `flash_attn`
- `flash_attn_2_cuda`

`flash-attn==2.8.3.post1` did not have a matching wheel for this Linux aarch64 / CUDA 13 / Torch 2.11 environment. Source build attempts were not usable in this validation session. The local `verl.utils.attention_utils` compatibility path only replaces `flash_attn.bert_padding` padding helpers with equivalent torch indexing helpers; it does not replace attention kernels.
