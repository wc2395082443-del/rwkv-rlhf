#!/bin/bash
set -x
source /root/miniconda3/etc/profile.d/conda.sh
conda activate verl-vllm
export CUDA_VISIBLE_DEVICES=0
export TMPDIR=/root/autodl-tmp/tmp
export PIP_CACHE_DIR=/root/autodl-tmp/pip-cache
export WANDB_MODE=disabled
ray stop --force || true
TS=$(date +%Y%m%d_%H%M%S)
OUT=/root/autodl-tmp/log/verl_llama_math500_full_vllm_tuned_${TS}
mkdir -p "$OUT"
export TENSORBOARD_DIR="$OUT/tensorboard"
export VERL_FILE_LOGGER_PATH="$OUT/metrics.jsonl"
cd /root/verl
python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    trainer.val_before_train=False \
    data.train_files=/root/autodl-tmp/data/verl_math500/train.parquet \
    data.val_files=/root/autodl-tmp/data/verl_math500/test.parquet \
    data.train_batch_size=8 \
    data.val_batch_size=8 \
    data.max_prompt_length=768 \
    data.max_response_length=256 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.shuffle=True \
    actor_rollout_ref.model.path=/root/autodl-tmp/models/Llama-3.2-3B-Instruct \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.lora_rank=32 \
    actor_rollout_ref.model.lora_alpha=32 \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.20 \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.rollout.max_num_seqs=8 \
    actor_rollout_ref.rollout.max_model_len=1024 \
    actor_rollout_ref.rollout.max_num_batched_tokens=1024 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
    algorithm.use_kl_in_reward=False \
    trainer.use_legacy_worker_impl=disable \
    trainer.critic_warmup=0 \
    trainer.logger='["console","file"]' \
    trainer.project_name=verl_llama_math500 \
    trainer.experiment_name=full_tuned_${TS} \
    trainer.default_local_dir="$OUT" \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=100 \
    trainer.total_epochs=1 \
    2>&1 | tee "$OUT/train.log"
