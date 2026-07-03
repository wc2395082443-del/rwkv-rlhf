#!/usr/bin/env bash
set -u
CODE=$(cat /tmp/rwkv_opsd_code_latest)
PY=/root/miniconda3/bin/python
TRAIN_JSON=/root/Albatross/faster3a_2605/dataset/MATH500.jsonl
EVAL_JSON=/root/Albatross/faster3a_2605/dataset/MATH500.jsonl
MODEL=/root/autodl-tmp/rwkv_models/ms_g1_1p5b/rwkv7-g1f-1.5b-20260419-ctx8192.pth
TOK=/root/RWKV-LM/rwkv_vocab_v20230424.txt
TS=$(date +%Y%m%d_%H%M%S)
OUT=/root/autodl-tmp/logs/math500_opsd_sweep_${TS}
mkdir -p "$OUT"
echo "$OUT" > /tmp/math500_opsd_sweep_latest
cd "$CODE" || exit 1
{
  echo "[$(date '+%F %T')] OUT=$OUT"
  echo "[$(date '+%F %T')] code_dir=$CODE"
  sha256sum main.py train.py infer.py stateful_rollout.py rwkv7_trainable.py reward.py 2>/dev/null || true
  echo "[$(date '+%F %T')] sweep opsd_coef: 0.005 0.02 0.05 0.1"
  echo "[$(date '+%F %T')] OPSD: answer_hint self branch, top_k=64, correct_only=1, mixed_only=1, agreement_gate=1"
} | tee "$OUT/progress.log"
printf "opsd_coef\tpost_acc\tdelta_vs_base0422\tgroups_used_rate\tgroups_used\tgroups_total\tall0_rate\tall1_rate\tavg_train_acc\tavg_trunc\tavg_len\tavg_step_time\tavg_opsd_loss\topsd_gate_rate\tckpt\ttrain_dir\teval_dir\n" > "$OUT/summary.tsv"

run_one() {
  local COEF="$1"
  local TAG="opsd_${COEF//./p}"
  local RDIR="$OUT/$TAG"
  local TDIR="$RDIR/train"
  local EDIR="$RDIR/eval"
  mkdir -p "$TDIR" "$EDIR"
  "$PY" main.py \
    --train_jsonl "$TRAIN_JSON" \
    --eval_jsonl "$EVAL_JSON" \
    --model "$MODEL" \
    --tokenizer "$TOK" \
    --out_dir "$TDIR" \
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
    --opsd_coef "$COEF" \
    --opsd_temp 1.0 \
    --opsd_top_k 64 \
    --opsd_correct_only 1 \
    --opsd_mixed_only 1 \
    --opsd_agreement_gate 1 \
    --opsd_margin 0.0 \
    --logit_chunk_tokens 128 \
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
    --seed 42 > "$TDIR/run.log" 2>&1
  local RC=$?
  echo "[$(date '+%F %T')] TRAIN DONE $TAG rc=$RC" | tee -a "$OUT/progress.log"
  local CKPT="$TDIR/ckpt_step100.pth"
  if [[ $RC -ne 0 || ! -s "$CKPT" ]]; then
    echo "[$(date '+%F %T')] SKIP EVAL $TAG missing_ckpt=$CKPT" | tee -a "$OUT/progress.log"
    printf "%s\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\t%s\t%s\t%s\n" "$COEF" "$CKPT" "$TDIR" "$EDIR" >> "$OUT/summary.tsv"
    return 0
  fi
  echo "[$(date '+%F %T')] START EVAL $TAG ckpt=$CKPT" | tee -a "$OUT/progress.log"
  "$PY" main.py \
    --train_jsonl "$TRAIN_JSON" \
    --eval_jsonl "$EVAL_JSON" \
    --model "$MODEL" \
    --full_init_ckpt "$CKPT" \
    --tokenizer "$TOK" \
    --out_dir "$EDIR" \
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
    --opsd_coef 0.0 \
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
    --seed 42 > "$EDIR/run.log" 2>&1
  echo "[$(date '+%F %T')] EVAL DONE $TAG" | tee -a "$OUT/progress.log"
  "$PY" - "$COEF" "$TDIR" "$EDIR" "$CKPT" "$OUT/summary.tsv" <<'PY'
import json, sys, os, statistics
coef, tdir, edir, ckpt, summary = sys.argv[1:]
rows=[]
try:
    with open(os.path.join(tdir,'metrics.jsonl')) as f:
        rows=[json.loads(x) for x in f if x.strip()]
except FileNotFoundError:
    pass
post_acc='NA'
try:
    with open(os.path.join(edir,'metrics.jsonl')) as f:
        for line in f:
            if line.strip():
                obj=json.loads(line)
                if 'accuracy' in obj: post_acc=float(obj['accuracy'])
except FileNotFoundError:
    pass
if rows:
    gtot=sum(float(r.get('groups_total',0)) for r in rows)
    guse=sum(float(r.get('groups_used',0)) for r in rows)
    all0=sum(float(r.get('groups_all_wrong',0)) for r in rows)
    all1=sum(float(r.get('groups_all_correct',0)) for r in rows)
    avg=lambda k: statistics.mean(float(r.get(k,0.0)) for r in rows)
    vals=[guse/gtot if gtot else 0.0,guse,gtot,all0/gtot if gtot else 0.0,all1/gtot if gtot else 0.0,avg('accuracy'),avg('trunc_rate'),avg('avg_length'),avg('time'),avg('avg_opsd_loss'),avg('opsd_gate_rate')]
else:
    vals=['NA']*11
if isinstance(post_acc,float): post_s=f'{post_acc:.6f}'; delta_s=f'{post_acc-0.422:.6f}'
else: post_s=delta_s='NA'
def fmt(x): return x if isinstance(x,str) else f'{x:.6f}'
with open(summary,'a') as out:
    out.write('\t'.join([coef,post_s,delta_s]+[fmt(v) for v in vals]+[ckpt,tdir,edir])+'\n')
PY
  tail -1 "$OUT/summary.tsv" | tee -a "$OUT/progress.log"
}

for c in 0.005 0.02 0.05 0.1; do
  echo "[$(date '+%F %T')] START TRAIN opsd_${c//./p} coef=$c" | tee -a "$OUT/progress.log"
  run_one "$c"
done

echo "[$(date '+%F %T')] ALL DONE" | tee -a "$OUT/progress.log"

