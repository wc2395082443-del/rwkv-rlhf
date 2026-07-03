#!/bin/bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate verl-vllm
export CUDA_VISIBLE_DEVICES=0
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONWARNINGS=ignore
export TQDM_DISABLE=1
export NCCL_DEBUG=WARN
export PYTHONPATH=/root/OpenRLHF:$PYTHONPATH
TS=$(date +%Y%m%d_%H%M%S)
OUT=/root/autodl-tmp/log/openrlhf_math500_rb_${TS}
mkdir -p "$OUT"
cd /root/OpenRLHF
python -m openrlhf.cli.train_ppo_ray \
  --pretrain /root/autodl-tmp/models/Llama-3.2-3B-Instruct \
  --remote_rm_url /root/OpenRLHF/examples/python/math_reward_func.py \
  --save_path "$OUT" \
  --ckpt_path "$OUT/ckpt" \
  --save_steps -1 \
  --logging_steps 1 \
  --eval_steps 50 \
  --prompt_data /root/autodl-tmp/data/openrlhf_math500/train.jsonl \
  --eval_dataset /root/autodl-tmp/data/openrlhf_math500/test.jsonl \
  --input_key prompt \
  --label_key label \
  --apply_chat_template \
  --ref_num_nodes 1 \
  --ref_num_gpus_per_node 1 \
  --actor_num_nodes 1 \
  --actor_num_gpus_per_node 1 \
  --vllm_num_engines 1 \
  --vllm_tensor_parallel_size 1 \
  --colocate_all_models \
  --vllm_gpu_memory_utilization 0.35 \
  --vllm_enable_sleep \
  --deepspeed_enable_sleep \
  --enforce_eager \
  --advantage_estimator reinforce_baseline \
  --use_kl_loss \
  --kl_estimator k2 \
  --init_kl_coef 1e-4 \
  --actor_learning_rate 1e-6 \
  --n_samples_per_prompt 4 \
  --rollout_batch_size 32 \
  --micro_rollout_batch_size 4 \
  --train_batch_size 32 \
  --micro_train_batch_size 1 \
  --num_episodes 1 \
  --max_epochs 1 \
  --max_samples 2048 \
  --max_len 1024 \
  --max_new_tokens 256 \
  --zero_stage 3 \
  --param_dtype bf16 \
  --attn_implementation sdpa \
  --gradient_checkpointing \
  --packing_samples \
  --lora_rank 32 \
  --lora_alpha 32 \
  --target_modules all-linear \
  --use_tensorboard "$OUT/runs" \
  > "$OUT/train.log" 2>&1