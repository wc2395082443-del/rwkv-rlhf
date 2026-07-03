#!/usr/bin/env bash
set -euo pipefail
CODE=/root/RWKV-LM/RWKV7-math500_teacher_trace_sft_20260613
BASE=/root/autodl-tmp/rwkv_models/ms_g1_1p5b/rwkv7-g1f-1.5b-20260419-ctx8192.pth
TOK=/root/RWKV-LM/rwkv_vocab_v20230424.txt
TRAIN=/root/autodl-tmp/data/gsm8k/gsm8k_train_formatted.jsonl
TEST=/root/autodl-tmp/data/gsm8k/gsm8k_test_formatted.jsonl
SFTDATA=/root/autodl-tmp/data/gsm8k/gsm8k_train_cot_teacher_sft_20260613.jsonl
OUT=/root/autodl-tmp/logs/gsm8k_teacher_sft_grpo_$(date +%Y%m%d_%H%M%S)
export OUT
printf "%s\n" "$OUT" > /tmp/gsm8k_teacher_sft_grpo_latest
mkdir -p "$OUT/base_eval" "$OUT/sft" "$OUT/sft_eval" "$OUT/grpo" "$OUT/grpo_eval"
{
  echo "[$(date '+%F %T')] OUT=$OUT"
  echo "[$(date '+%F %T')] CODE=$CODE"
  echo "[$(date '+%F %T')] TRAIN=$TRAIN TEST=$TEST SFTDATA=$SFTDATA"
  sha256sum "$CODE"/sft_distill.py "$CODE"/main.py "$CODE"/train.py "$CODE"/infer.py "$CODE"/reward.py || true
} | tee "$OUT/progress.log"
cd "$CODE"
run_eval() {
  local outdir="$1"
  local init_ckpt="$2"
  local tag="$3"
  echo "[$(date '+%F %T')] START EVAL $tag init=$init_ckpt" | tee -a "$OUT/progress.log"
  local init_args=()
  if [ -n "$init_ckpt" ]; then init_args=(--full_init_ckpt "$init_ckpt"); fi
  /root/miniconda3/bin/python main.py \
    --train_jsonl "$TRAIN" \
    --eval_jsonl "$TEST" \
    --model "$BASE" \
    "${init_args[@]}" \
    --tokenizer "$TOK" \
    --out_dir "$outdir" \
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
    --eval_max_new_tokens 1024 \
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
    --seed 42 2>&1 | tee "$outdir/run.log"
  echo "[$(date '+%F %T')] DONE EVAL $tag" | tee -a "$OUT/progress.log"
}

run_eval "$OUT/base_eval" "" "base"

echo "[$(date '+%F %T')] START GSM8K COT SFT" | tee -a "$OUT/progress.log"
/root/miniconda3/bin/python sft_distill.py \
  --train_jsonl "$SFTDATA" \
  --model "$BASE" \
  --tokenizer "$TOK" \
  --out_dir "$OUT/sft" \
  --steps 200 \
  --batch_size 8 \
  --micro_batch 1 \
  --lr 5e-7 \
  --model_dtype bf16 \
  --prompt_mode trl_doc \
  --logit_chunk_tokens 128 \
  --save_interval 200 \
  --save_last 1 \
  --seed 42 2>&1 | tee "$OUT/sft/run.log"
SFT_CKPT="$OUT/sft/ckpt_step200.pth"
echo "[$(date '+%F %T')] DONE SFT ckpt=$SFT_CKPT" | tee -a "$OUT/progress.log"

run_eval "$OUT/sft_eval" "$SFT_CKPT" "sft200"

echo "[$(date '+%F %T')] START GRPO from SFT" | tee -a "$OUT/progress.log"
/root/miniconda3/bin/python main.py \
  --train_jsonl "$TRAIN" \
  --eval_jsonl "$TEST" \
  --model "$BASE" \
  --full_init_ckpt "$SFT_CKPT" \
  --tokenizer "$TOK" \
  --out_dir "$OUT/grpo" \
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
  --eval_max_new_tokens 1024 \
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
  --seed 42 2>&1 | tee "$OUT/grpo/run.log"
GRPO_CKPT="$OUT/grpo/ckpt_step100.pth"
echo "[$(date '+%F %T')] DONE GRPO ckpt=$GRPO_CKPT" | tee -a "$OUT/progress.log"

run_eval "$OUT/grpo_eval" "$GRPO_CKPT" "sft200_grpo100"

/root/miniconda3/bin/python - <<'PY' | tee "$OUT/summary.tsv"
import json, os
out=os.environ['OUT']
def last_metric(path):
    last={}
    if os.path.exists(path):
        for line in open(path, encoding='utf-8'):
            if line.strip(): last=json.loads(line)
    return last
base=last_metric(os.path.join(out,'base_eval','metrics.jsonl'))
sft=last_metric(os.path.join(out,'sft_eval','metrics.jsonl'))
grpo=last_metric(os.path.join(out,'grpo_eval','metrics.jsonl'))
train=[]
mp=os.path.join(out,'grpo','metrics.jsonl')
if os.path.exists(mp):
    for line in open(mp, encoding='utf-8'):
        if line.strip():
            d=json.loads(line)
            if d.get('split')=='train': train.append(d)
last50=train[-50:]
print('stage\tacc\tavg_len\ttrunc\trepeat\teval_time')
for name,d in [('base',base),('sft200',sft),('sft200_grpo100',grpo)]:
    print(f"{name}\t{d.get('accuracy',float('nan')):.6f}\t{d.get('avg_length',float('nan')):.3f}\t{d.get('trunc_rate',float('nan')):.6f}\t{d.get('repeat_rate',float('nan')):.6f}\t{d.get('eval_time',float('nan')):.1f}")
print('grpo_last50_train_acc\tgrpo_last50_trunc\tgrpo_avg_step_time\tckpt')
print(f"{sum(x.get('accuracy',0) for x in last50)/max(1,len(last50)):.6f}\t{sum(x.get('trunc_rate',0) for x in last50)/max(1,len(last50)):.6f}\t{sum(x.get('time',0) for x in train)/max(1,len(train)):.3f}\t{os.path.join(out,'grpo','ckpt_step100.pth')}")
PY
echo "[$(date '+%F %T')] ALL DONE" | tee -a "$OUT/progress.log"

