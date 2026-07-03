#!/usr/bin/env bash
set -euo pipefail
CODE=/root/RWKV-LM/RWKV7-math500_teacher_trace_sft_20260613
BASE=/root/autodl-tmp/rwkv_models/ms_g1_1p5b/rwkv7-g1f-1.5b-20260419-ctx8192.pth
TOK=/root/RWKV-LM/rwkv_vocab_v20230424.txt
DATA=/root/autodl-tmp/data/math500_teacher7b_correct_sft_20260613.jsonl
EVAL=/root/Albatross/faster3a_2605/dataset/MATH500.jsonl
OUT=/root/autodl-tmp/logs/math500_teacher_trace_sft_$(date +%Y%m%d_%H%M%S)
export OUT
printf "%s\n" "$OUT" > /tmp/math500_teacher_trace_sft_latest
mkdir -p "$OUT/train" "$OUT/eval"
{
  echo "[$(date '+%F %T')] OUT=$OUT"
  echo "[$(date '+%F %T')] CODE=$CODE"
  sha256sum "$CODE"/sft_distill.py "$CODE"/main.py "$CODE"/train.py "$CODE"/infer.py "$CODE"/rwkv7_trainable.py || true
  echo "[$(date '+%F %T')] START SFT"
} | tee "$OUT/progress.log"
cd "$CODE"
/root/miniconda3/bin/python sft_distill.py \
  --train_jsonl "$DATA" \
  --model "$BASE" \
  --tokenizer "$TOK" \
  --out_dir "$OUT/train" \
  --steps 100 \
  --batch_size 4 \
  --micro_batch 1 \
  --lr 5e-7 \
  --model_dtype bf16 \
  --prompt_mode trl_doc \
  --logit_chunk_tokens 128 \
  --save_interval 100 \
  --save_last 1 \
  --seed 42 2>&1 | tee "$OUT/train/run.log"
CKPT="$OUT/train/ckpt_step100.pth"
echo "[$(date '+%F %T')] SFT DONE ckpt=$CKPT" | tee -a "$OUT/progress.log"
/root/miniconda3/bin/python main.py \
  --train_jsonl "$EVAL" \
  --eval_jsonl "$EVAL" \
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
last_train={}
with open(os.path.join(out,'train','metrics.jsonl'), encoding='utf-8') as f:
    for line in f:
        if line.strip(): last_train=json.loads(line)
last_eval={}
with open(os.path.join(out,'eval','metrics.jsonl'), encoding='utf-8') as f:
    for line in f:
        if line.strip(): last_eval=json.loads(line)
print('sft_steps\tlr\ttrain_loss\ttrain_ppl\ttrain_tok_s\tpost_acc\tavg_len\ttrunc\trepeat')
print(f"{last_train.get('step')}\t{last_train.get('lr')}\t{last_train.get('loss'):.6f}\t{last_train.get('ppl'):.4f}\t{last_train.get('tokens_per_sec'):.2f}\t{last_eval.get('accuracy'):.6f}\t{last_eval.get('avg_length'):.3f}\t{last_eval.get('trunc_rate'):.6f}\t{last_eval.get('repeat_rate'):.6f}")
PY
echo "[$(date '+%F %T')] ALL DONE" | tee -a "$OUT/progress.log"
