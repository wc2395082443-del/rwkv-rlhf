#!/usr/bin/env bash
set -euo pipefail

EXP=/root/autodl-tmp/g1e_dynamic_rescreen_20260426_020719
PY=/root/miniconda3/bin/python
MODEL=$EXP/stage_4/run/checkpoint_step_100.pth
TOKENIZER=/root/RWKV-LM/RWKV-v7/rwkv_vocab_v20230424.txt
FULL_TRAIN=/root/RWKV-LM/RWKV7-statetuning/gsm8k_train_formatted.jsonl
SMALL_EVAL=/root/RWKV-LM/RWKV7-statetuning/gsm8k_test_formatted_1of8.jsonl
FULL_EVAL=/root/RWKV-LM/RWKV7-statetuning/gsm8k_test_formatted.jsonl
TRAIN_SCRIPT=/root/RWKV-LM/RWKV-v7/train_temp/train_rl_baseline.py
PASS8_SCRIPT=/root/autodl-tmp/dynamic_pass8_eval.py
BUILD_SUBSET=/root/autodl-tmp/build_core_sub_subset.py

mkdir -p $EXP/stage_4/pass8_full
echo [stage 4-resume] pass8 eval on full train model=$MODEL | tee -a $EXP/master.log
$PY $PASS8_SCRIPT \
  --model $MODEL \
  --eval_jsonl $FULL_TRAIN \
  --out_dir $EXP/stage_4/pass8_full \
  --rollout_forward_batch 192 \
  --chunk_size 24 \
  --group_size 8 \
  --temperature 1.0 --top_p 0.6 --top_k 0 \
  |& tee $EXP/stage_4/pass8_stdout_resume.log

NEXT_SUBSET=$EXP/subset_stage_5.jsonl
$PY $BUILD_SUBSET \
  --source_train_jsonl $FULL_TRAIN \
  --pass8_jsonl $EXP/stage_4/pass8_full/pass8_eval.jsonl \
  --out_jsonl $NEXT_SUBSET \
  |& tee $EXP/stage_4/build_subset_stdout_resume.log
printf '%s\n' $NEXT_SUBSET > $EXP/current_subset.txt

STAGE_DIR=$EXP/stage_5
mkdir -p $STAGE_DIR/run
echo [stage 5] train_jsonl=$NEXT_SUBSET model=$MODEL | tee -a $EXP/master.log
cd /root/RWKV-LM/RWKV-v7/train_temp
$PY $TRAIN_SCRIPT \
  --load_model $MODEL \
  --proj_dir $STAGE_DIR/run \
  --tokenizer $TOKENIZER \
  --train_jsonl $NEXT_SUBSET \
  --eval_jsonl $SMALL_EVAL \
  --full_eval_jsonl $FULL_EVAL \
  --devices 1 --accelerator gpu --strategy deepspeed_stage_3_offload --precision bf16 \
  --total_steps 100 --num_questions 24 --samples_per_question 8 --max_new_tokens 768 \
  --micro_batch 18 --rollout_forward_batch 192 --use_stateful_rollout 1 \
  --lr 5e-7 --neg_adv_weight 0.6 --kl_coef 0.05 --kl_mode k3_loss \
  --length_weight 0.0 --zstd_penalty_weight 0.0 --ngram_penalty 0.0 \
  --skip_preeval 1 --skip_posteval 1 --eval_interval 10 --save_interval 50 \
  --log_interval 1 --enable_progress_bar 0 \
  --hard_buffer_ttl 10 --hard_buffer_cooldown 5 --hard_buffer_target_samples 192 --hard_buffer_group_size 8 \
  --hard_buffer_extra_lr_scale 0.5 --hard_buffer_adv_clip 2.5 \
  --extra_curriculum pure_hard \
  --full_eval_early_stop_patience 0 --save_eval_checkpoint 1 --save_final_checkpoint 1 --disable_extra_step 0 \
  |& tee $STAGE_DIR/train_stdout.log
