#!/usr/bin/env bash
set -xeuo pipefail

RWKV_MODEL_PATH=${RWKV_MODEL_PATH:-/workspace/Weights/RWKV/rwkv7-g1g-1.5b-20260526-ctx8192.pth}
RWKV_LM_PATH=${RWKV_LM_PATH:-/workspace/Projects/MachineLearning/rwkv-lm/RWKV-v7/train_temp}
VLLM_RWKV_PATH=${VLLM_RWKV_PATH:-}
PYTHON=${PYTHON:-.venv/bin/python}

DATA_ROOT=${DATA_ROOT:-/workspace/Datasets/gsm8k}
TRAIN_FILES=${TRAIN_FILES:-"['${DATA_ROOT}/train.parquet']"}
VAL_FILES=${VAL_FILES:-"['${DATA_ROOT}/test.parquet']"}

[[ -x "${PYTHON}" ]] || { echo "Python executable not found: ${PYTHON}. Run uv sync or set PYTHON."; exit 1; }
[[ -f "${RWKV_MODEL_PATH}" ]] || { echo "RWKV checkpoint not found: ${RWKV_MODEL_PATH}"; exit 1; }
[[ -d "${RWKV_LM_PATH}" ]] || { echo "rwkv-lm train_temp path not found: ${RWKV_LM_PATH}"; exit 1; }
[[ -f "${RWKV_LM_PATH}/train.py" ]] || { echo "rwkv-lm train.py not found under: ${RWKV_LM_PATH}"; exit 1; }

if [[ -n "${VLLM_RWKV_PATH}" ]]; then
    [[ -d "${VLLM_RWKV_PATH}" ]] || { echo "vLLM-RWKV repository not found: ${VLLM_RWKV_PATH}"; exit 1; }
    export PYTHONPATH="${VLLM_RWKV_PATH}${PYTHONPATH:+:${PYTHONPATH}}"
fi

export HYDRA_FULL_ERROR=${HYDRA_FULL_ERROR:-1}
export VLLM_RWKV7_WKV_MODE=${VLLM_RWKV7_WKV_MODE:-fp32io16}
export VLLM_RWKV7_EMB_DEVICE=${VLLM_RWKV7_EMB_DEVICE:-cpu}

NNODES=${NNODES:-1}
TRAIN_NGPUS_PER_NODE=${TRAIN_NGPUS_PER_NODE:-${NGPUS_PER_NODE:-7}}
ROLLOUT_NNODES=${ROLLOUT_NNODES:-${NNODES}}
ROLLOUT_NGPUS_PER_NODE=${ROLLOUT_NGPUS_PER_NODE:-1}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-56}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-56}
PPO_MICRO_BATCH_SIZE=${PPO_MICRO_BATCH_SIZE:-1}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-7168}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-32768}
RWKV_USE_DYNAMIC_BSZ=${RWKV_USE_DYNAMIC_BSZ:-True}

ADV_ESTIMATOR=${ADV_ESTIMATOR:-grpo}
REWARD_MANAGER=${REWARD_MANAGER:-naive}
REWARD_FUNCTION_PATH=${REWARD_FUNCTION_PATH:-${PWD}/examples/rwkv_trainer/math_verify_reward.py}
REWARD_FUNCTION_NAME=${REWARD_FUNCTION_NAME:-compute_score}

ACTOR_LR=${ACTOR_LR:-1e-6}
ACTOR_USE_KL_LOSS=${ACTOR_USE_KL_LOSS:-False}
ACTOR_KL_LOSS_COEF=${ACTOR_KL_LOSS_COEF:-0.0}
ACTOR_KL_LOSS_TYPE=${ACTOR_KL_LOSS_TYPE:-low_var_kl}
ACTOR_LR_WARMUP_STEPS=${ACTOR_LR_WARMUP_STEPS:-}
ACTOR_WEIGHT_DECAY=${ACTOR_WEIGHT_DECAY:-}
ACTOR_ENTROPY_COEFF=${ACTOR_ENTROPY_COEFF:-}
ACTOR_GRAD_CLIP=${ACTOR_GRAD_CLIP:-}
CLIP_RATIO_LOW=${CLIP_RATIO_LOW:-}
CLIP_RATIO_HIGH=${CLIP_RATIO_HIGH:-}
CLIP_RATIO_C=${CLIP_RATIO_C:-}

ROLLOUT_N=${ROLLOUT_N:-8}
ROLLOUT_TP=${ROLLOUT_TP:-1}
ROLLOUT_DP=${ROLLOUT_DP:-1}
ROLLOUT_PP=${ROLLOUT_PP:-1}
ROLLOUT_MODE=${ROLLOUT_MODE:-async}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.85}
ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-512}
ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-32768}
ROLLOUT_CHECKPOINT_ENGINE_BACKEND=${ROLLOUT_CHECKPOINT_ENGINE_BACKEND:-nccl}
ROLLOUT_CORRECTION_BYPASS_MODE=${ROLLOUT_CORRECTION_BYPASS_MODE:-True}

TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-}
SAVE_FREQ=${SAVE_FREQ:-50}
TEST_FREQ=${TEST_FREQ:-50}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-False}

PROJECT_NAME=${PROJECT_NAME:-verl_rwkv_grpo}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-rwkv7_grpo_vllm}

DATA=(
    algorithm.adv_estimator=${ADV_ESTIMATOR}
    algorithm.use_kl_in_reward=False
    data.train_files="${TRAIN_FILES}"
    data.val_files="${VAL_FILES}"
    data.train_batch_size=${TRAIN_BATCH_SIZE}
    data.max_prompt_length=${MAX_PROMPT_LENGTH}
    data.max_response_length=${MAX_RESPONSE_LENGTH}
    data.filter_overlong_prompts=True
    data.truncation=error
    reward.custom_reward_function.path="${REWARD_FUNCTION_PATH}"
    reward.custom_reward_function.name="${REWARD_FUNCTION_NAME}"
    reward.reward_manager.name=${REWARD_MANAGER}
)

MODEL=(
    model@actor_rollout_ref.model=rwkv_native
    actor_rollout_ref.model.path="${RWKV_MODEL_PATH}"
    actor_rollout_ref.model.rwkv_lm_path="${RWKV_LM_PATH}"
)

ACTOR=(
    actor@actor_rollout_ref.actor=rwkv_lm
    actor_rollout_ref.actor.engine.rwkv_lm_path="${RWKV_LM_PATH}"
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR}
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE}
    actor_rollout_ref.actor.use_dynamic_bsz=${RWKV_USE_DYNAMIC_BSZ}
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.actor.use_kl_loss=${ACTOR_USE_KL_LOSS}
    actor_rollout_ref.actor.kl_loss_coef=${ACTOR_KL_LOSS_COEF}
    actor_rollout_ref.actor.kl_loss_type=${ACTOR_KL_LOSS_TYPE}
)

[[ -n "${ACTOR_LR_WARMUP_STEPS}" ]] && ACTOR+=(actor_rollout_ref.actor.optim.lr_warmup_steps=${ACTOR_LR_WARMUP_STEPS})
[[ -n "${ACTOR_WEIGHT_DECAY}" ]] && ACTOR+=(actor_rollout_ref.actor.optim.weight_decay=${ACTOR_WEIGHT_DECAY})
[[ -n "${ACTOR_ENTROPY_COEFF}" ]] && ACTOR+=(actor_rollout_ref.actor.entropy_coeff=${ACTOR_ENTROPY_COEFF})
[[ -n "${ACTOR_GRAD_CLIP}" ]] && ACTOR+=(actor_rollout_ref.actor.optim.clip_grad=${ACTOR_GRAD_CLIP})
[[ -n "${CLIP_RATIO_LOW}" ]] && ACTOR+=(actor_rollout_ref.actor.clip_ratio_low=${CLIP_RATIO_LOW})
[[ -n "${CLIP_RATIO_HIGH}" ]] && ACTOR+=(actor_rollout_ref.actor.clip_ratio_high=${CLIP_RATIO_HIGH})
[[ -n "${CLIP_RATIO_C}" ]] && ACTOR+=(actor_rollout_ref.actor.clip_ratio_c=${CLIP_RATIO_C})

REF=(
    ref@actor_rollout_ref.ref=rwkv_lm
    actor_rollout_ref.ref.engine.rwkv_lm_path="${RWKV_LM_PATH}"
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE}
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${RWKV_USE_DYNAMIC_BSZ}
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
)

ROLLOUT=(
    actor_rollout_ref.hybrid_engine=False
    rollout.nnodes=${ROLLOUT_NNODES}
    rollout.n_gpus_per_node=${ROLLOUT_NGPUS_PER_NODE}
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.load_format=auto
    actor_rollout_ref.rollout.nnodes=${ROLLOUT_NNODES}
    actor_rollout_ref.rollout.n_gpus_per_node=${ROLLOUT_NGPUS_PER_NODE}
    actor_rollout_ref.rollout.mode=${ROLLOUT_MODE}
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}
    actor_rollout_ref.rollout.data_parallel_size=${ROLLOUT_DP}
    actor_rollout_ref.rollout.pipeline_model_parallel_size=${ROLLOUT_PP}
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL}
    actor_rollout_ref.rollout.n=${ROLLOUT_N}
    actor_rollout_ref.rollout.max_num_seqs=${ROLLOUT_MAX_NUM_SEQS}
    actor_rollout_ref.rollout.max_num_batched_tokens=${ROLLOUT_MAX_NUM_BATCHED_TOKENS}
    actor_rollout_ref.rollout.enable_prefix_caching=False
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE}
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${RWKV_USE_DYNAMIC_BSZ}
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.rollout.checkpoint_engine.backend=${ROLLOUT_CHECKPOINT_ENGINE_BACKEND}
    algorithm.rollout_correction.bypass_mode=${ROLLOUT_CORRECTION_BYPASS_MODE}
)

TOKENIZER_MODE_OVERRIDDEN=False
for arg in "$@"; do
    if [[ "${arg}" == *"actor_rollout_ref.rollout.engine_kwargs.vllm.tokenizer_mode="* ]]; then
        TOKENIZER_MODE_OVERRIDDEN=True
        break
    fi
done
if [[ "${TOKENIZER_MODE_OVERRIDDEN}" == "False" ]]; then
    ROLLOUT+=(+actor_rollout_ref.rollout.engine_kwargs.vllm.tokenizer_mode=rwkv)
fi

TRAINER=(
    critic.enable=False
    trainer.logger='["console"]'
    trainer.project_name="${PROJECT_NAME}"
    trainer.experiment_name="${EXPERIMENT_NAME}"
    trainer.nnodes=${NNODES}
    trainer.n_gpus_per_node=${TRAIN_NGPUS_PER_NODE}
    trainer.save_freq=${SAVE_FREQ}
    trainer.test_freq=${TEST_FREQ}
    trainer.val_before_train=${VAL_BEFORE_TRAIN}
    trainer.total_epochs=${TOTAL_EPOCHS}
)

[[ -n "${TOTAL_TRAINING_STEPS}" ]] && TRAINER+=(trainer.total_training_steps=${TOTAL_TRAINING_STEPS})

RAY_BIN="$(dirname "${PYTHON}")/ray"
if [[ -x "${RAY_BIN}" ]]; then
    "${RAY_BIN}" stop --force || true
fi

"${PYTHON}" -m verl.experimental.one_step_off_policy.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${REF[@]}" \
    "${ROLLOUT[@]}" \
    "${TRAINER[@]}" \
    "$@"
