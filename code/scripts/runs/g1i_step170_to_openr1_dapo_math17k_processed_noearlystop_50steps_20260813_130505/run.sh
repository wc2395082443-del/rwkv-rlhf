#!/usr/bin/env bash
set -euo pipefail
cd "/hy-tmp/RWKV7-g1i-strict-quality-gate-adamw-nooffload-20260810"
python3 train_rl_thinking_cot_strict_boxedprompt_schedopt_earlystop.py \
  --load_model "/hy-tmp/runs/g1i_step155_to_openr1_dapo_math17k_processed_maxnew6144_micro1_20260813_094132/final_step_170.pth" \
  --proj_dir "/hy-tmp/runs/g1i_step170_to_openr1_dapo_math17k_processed_noearlystop_50steps_20260813_130505" \
  --tokenizer /hy-tmp/RWKV-v7/rwkv_vocab_v20230424.txt \
  --train_jsonl /hy-tmp/data/openr1_dapo_math17k_processed/strict_split_seed42/train.jsonl \
  --eval_jsonl /hy-tmp/data/openr1_dapo_math17k_processed/strict_split_seed42/validation.jsonl \
  --full_eval_jsonl /hy-tmp/data/openr1_dapo_math17k_processed/strict_split_seed42/test.jsonl \
  --devices 1 --accelerator gpu --strategy auto --precision bf16 --grad_cp 1 \
  --total_steps 50 --step_offset 170 --random_seed 42 \
  --num_questions 8 --samples_per_question 16 \
  --max_new_tokens 6144 --micro_batch 1 --rollout_forward_batch 512 --use_stateful_rollout 1 \
  --temperature 1.0 --top_p 1.0 --top_k -1 \
  --ppo_epochs 1 \
  --eval_temperature 0.3 --eval_top_p 0.4 --eval_top_k 500 --eval_tail_max_tokens 0 \
  --lr 2e-7 --beta1 0.9 --beta2 0.99 --weight_decay 0.0 --grad_clip 1.0 \
  --neg_adv_weight 1.0 --kl_coef 0.0 --kl_mode k3_loss --diagnostic_ref_forward 0 \
  --length_weight 0.0 --zstd_penalty_weight 0.0 --ngram_penalty 0.0 \
  --hard_buffer_target_samples 0 --disable_extra_step 1 --extra_curriculum off \
  --eval_interval 50 --eval_sample_ratio 0.2 \
  --dapo_dynamic_sampling 1 --dapo_candidate_questions 32 --dapo_max_gen_batches 4 --dapo_min_valid_groups 8 \
  --dapo_dual_clip_c 10.0 --dapo_overlong_buffer_tokens 0 --dapo_overlong_penalty_factor 0.0 \
  --checkpoint_interval 0 --save_interval 0 --save_eval_checkpoint 0 --log_interval 1 \
  --preeval_sample_ratio 0.2 --skip_preeval 1 --skip_posteval 1 --final_full_eval 0 \
  --fp32_master_optimizer 1 --enable_progress_bar 0 --save_final_checkpoint 1 \
  --log_responses 0 --compute_grad_norm 0 --empty_cache_each_micro 0 \
  --train_early_stop_patience 999999 --train_early_stop_min_steps 999999
