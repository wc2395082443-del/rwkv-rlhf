#!/usr/bin/env bash
set -euo pipefail

export HF_HOME=/root/autodl-tmp/.hf_cache
export HF_DATASETS_CACHE=/root/autodl-tmp/.hf_cache/datasets
export TMPDIR=/root/autodl-tmp/tmp
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_V1=1
export RAY_LOGGING_LEVEL=ERROR
export HYDRA_FULL_ERROR=1

MODEL_PATH=${MODEL_PATH:-/root/autodl-tmp/models/DeepSeek-R1-Distill-Qwen-1.5B}
TRAIN_FILE=${TRAIN_FILE:-/root/autodl-tmp/official_repro_assets/DeepMath-103K-verl-deepmath_r1/train.parquet}
TEST_FILE=${TEST_FILE:-/root/autodl-tmp/official_repro_assets/DeepMath-103K-verl-deepmath_r1/test.parquet}
REWARD_FILE=${REWARD_FILE:-/root/autodl-tmp/trl-grpo-repro/deepmath_official_rewards.py}
OUT_DIR=${OUT_DIR:-/root/autodl-tmp/trl-grpo-repro/verl_outputs/deepmath_r1_1p5b_smoke}

mkdir -p "${OUT_DIR}"

/root/miniconda3/envs/verl-vllm/bin/python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.kl_ctrl.kl_coef=0.001 \
  algorithm.filter_groups.enable=True \
  algorithm.filter_groups.max_num_gen_batches=2 \
  algorithm.filter_groups.metric=acc \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${TEST_FILE}" \
  data.prompt_key=prompt \
  data.rm_system_prompt=True \
  data.return_raw_chat=True \
  data.gen_batch_size=4 \
  data.train_batch_size=4 \
  data.max_prompt_length=2048 \
  data.max_response_length=512 \
  data.filter_overlong_prompts=True \
  data.truncation='left' \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  +actor_rollout_ref.model.override_config.attention_dropout=0.0 \
  +actor_rollout_ref.model.override_config.embd_pdrop=0.0 \
  +actor_rollout_ref.model.override_config.resid_pdrop=0.0 \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=4 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.entropy_coeff=0.0 \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.actor.clip_ratio_low=0.2 \
  actor_rollout_ref.actor.clip_ratio_high=0.27 \
  actor_rollout_ref.actor.grad_clip=1.0 \
  actor_rollout_ref.actor.use_token_level_loss=True \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
  actor_rollout_ref.rollout.enable_chunked_prefill=True \
  actor_rollout_ref.rollout.max_num_batched_tokens=4096 \
  actor_rollout_ref.rollout.max_num_seqs=16 \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=4096 \
  actor_rollout_ref.rollout.temperature=0.6 \
  actor_rollout_ref.rollout.top_p=0.95 \
  actor_rollout_ref.rollout.n=4 \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
  actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.rollout.val_kwargs.n=1 \
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=4096 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  custom_reward_function.path="${REWARD_FILE}" \
  custom_reward_function.name=compute_score_deepmath_r1 \
  trainer.project_name='deepmath_official_like' \
  trainer.experiment_name='deepseek_r1_qwen_1p5b_smoke' \
  trainer.logger='["console"]' \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.test_freq=-1 \
  trainer.save_freq=-1 \
  trainer.total_training_steps=1 \
  trainer.default_local_dir="${OUT_DIR}"
