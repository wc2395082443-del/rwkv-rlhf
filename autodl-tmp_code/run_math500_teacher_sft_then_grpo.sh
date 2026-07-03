#!/usr/bin/env bash
set -euo pipefail
CODE=/root/RWKV-LM/RWKV7-math500_teacher_trace_sft_20260613
BASE=/root/autodl-tmp/rwkv_models/ms_g1_1p5b/rwkv7-g1f-1.5b-20260419-ctx8192.pth
TOK=/root/RWKV-LM/rwkv_vocab_v20230424.txt
DATA=/root/Albatross/faster3a_2605/dataset/MATH500.jsonl
INIT=/root/autodl-tmp/logs/math500_teacher_trace_sft_20260613_004936/train/ckpt_step100.pth
OUT=/root/autodl-tmp/logs/math500_teacher_sft_then_grpo_$(date +%Y%m%d_%H%M%S)
export OUT
printf "%s\n" "$OUT" > /tmp/math500_teacher_sft_then_grpo_latest
mkdir -p "$OUT/train" "$OUT/eval"
{
  echo "[$(date '+%F %T')] OUT=$OUT"
  echo "[$(date '+%F %T')] INIT=$INIT"
  echo "[$(date '+%F %T')] START GRPO from teacher-SFT ckpt"
} | tee "$OUT/progress.log"
cd "$CODE"
/root/miniconda3/bin/python main.py \
  --train_jsonl "$DATA" \
  --eval_jsonl "$DATA" \
  --model "$BASE" \
  --full_init_ckpt "$INIT" \
  --tokenizer "$TOK" \
  --out_dir "$OUT/train" \
  --tune_mode full \
  --reward_mode trl_doc \
  --prompt_mode trl_doc \
  --num_questions 8 \
  --samples_per_question 8 \
  --total_steps 100 \
  --max_new_tokens 1024 \
  --temperature 0.8 \
  --top_p 1.0 \
  --top_k 0 \
  --eval_temperature 1.0 \
  --eval_top_p 0.28 \
  --eval_top_k 32 \
  --eval_max_new_tokens 1500 \
  --eval_chunk_size 64 \
  --min_tokens 1 \
  --length_weight 0.0 \
  --zstd_penalty_weight 0.0 \
  --ngram_penalty 0.0 \
  --neg_adv_weight 1.0 \
  --opd_coef 0.0 \
  --kl_coef 0.0 \
  --lr 1e-6 \
  --micro_batch 1 \
  --rollout_forward_batch 8 \
  --hard_buffer_target_samples 0 \
  --eval_interval 999999 \
  --skip_preeval 1 \
  --skip_posteval 1 \
  --save_interval 999999 \
  --save_last 1 \
  --save_responses 0 \
  --seed 42 2>&1 | tee "$OUT/train/run.log"
CKPT="$OUT/train/ckpt_step100.pth"
echo "[$(date '+%F %T')] GRPO DONE ckpt=$CKPT" | tee -a "$OUT/progress.log"
/root/miniconda3/bin/python main.py \
  --train_jsonl "$DATA" \
  --eval_jsonl "$DATA" \
  --model "$BASE" \
  --full_init_ckpt "$CKPT" \
  --tokenizer "$TOK" \
  --out_dir "$OUT/eval" \
  --tune_mode full \
  --reward_mode trl_doc \
  --prompt_mode trl_doc \
  --num_questions 8 \
  --samples_per_question 8 \
  --total_steps 0 \
  --max_new_tokens 1024 \
  --temperature 0.8 \
  --top_p 1.0 \
  --top_k 0 \
  --eval_temperature 1.0 \
  --eval_top_p 0.28 \
  --eval_top_k 32 \
  --eval_max_new_tokens 1500 \
  --eval_chunk_size 64 \
  --min_tokens 1 \
  --length_weight 0.0 \
  --zstd_penalty_weight 0.0 \
  --ngram_penalty 0.0 \
  --neg_adv_weight 1.0 \
  --opd_coef 0.0 \
  --kl_coef 0.0 \
  --lr 1e-6 \
  --micro_batch 1 \
  --rollout_forward_batch 8 \
  --hard_buffer_target_samples 0 \
  --eval_interval 999999 \
  --preeval_sample_ratio 1.0 \
  --skip_preeval 0 \
  --skip_posteval 1 \
  --save_interval 999999 \
  --save_last 0 \
  --save_responses 0 \
  --seed 42 2>&1 | tee "$OUT/eval/run.log"
/root/miniconda3/bin/python - <<'PY' | tee "$OUT/summary.tsv"
import json, os
out=os.environ['OUT']
train=[]
with open(os.path.join(out,'train','metrics.jsonl'), encoding='utf-8') as f:
    for line in f:
        if line.strip():
            d=json.loads(line)
            if d.get('split')=='train': train.append(d)
last_eval={}
with open(os.path.join(out,'eval','metrics.jsonl'), encoding='utf-8') as f:
    for line in f:
        if line.strip(): last_eval=json.loads(line)
last50=train[-50:]
print('post_acc\tavg_train_acc_last50\tavg_trunc_last50\tavg_step_time\tckpt')
print(f"{last_eval.get('accuracy'):.6f}\t{sum(x.get('accuracy',0) for x in last50)/max(1,len(last50)):.6f}\t{sum(x.get('trunc_rate',0) for x in last50)/max(1,len(last50)):.6f}\t{sum(x.get('time',0) for x in train)/max(1,len(train)):.3f}\t{os.path.join(out,'train','ckpt_step100.pth')}")
PY
echo "[$(date '+%F %T')] ALL DONE" | tee -a "$OUT/progress.log"

