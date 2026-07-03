set -euo pipefail
ROOT=/root/RWKV-LM/RWKV7-deepmath_trl_doc_g1f1p5b_fp32kernel
DATA_DIR=$ROOT/data_deepmath_trl_doc
MODEL=/dev/shm/rwkv_models/rwkv7-g1f-1.5b-20260419-ctx8192.pth
TOKENIZER=/root/RWKV-LM/rwkv_vocab_v20230424.txt
BASE_OUT=/root/autodl-tmp/lr_sweep_g1f1p5b_deepmath_100_$(date +%Y%m%d_%H%M%S)
mkdir -p "$BASE_OUT"
LRS=(1e-7 2e-7 3e-7 5e-7 7e-7 1e-6)
echo "base_out=$BASE_OUT"
printf "%s\n" "${LRS[@]}" > "$BASE_OUT/lr_list.txt"
cd "$ROOT"
for LR in "${LRS[@]}"; do
  TAG=$(echo "$LR" | sed 's/-/m/g; s/\./p/g')
  OUT_DIR="$BASE_OUT/lr_${TAG}"
  mkdir -p "$OUT_DIR"
  echo "[$(date)] START lr=$LR out=$OUT_DIR"
  /root/miniconda3/bin/python main.py \
    --train_jsonl "$DATA_DIR/deepmath_train_rwkv.jsonl" \
    --eval_jsonl "$DATA_DIR/deepmath_test_rwkv.jsonl" \
    --model "$MODEL" \
    --tokenizer "$TOKENIZER" \
    --out_dir "$OUT_DIR" \
    --tune_mode full \
    --model_dtype bf16 \
    --reward_mode trl_doc \
    --prompt_mode trl_doc \
    --num_questions 8 \
    --samples_per_question 8 \
    --total_steps 100 \
    --max_new_tokens 1024 \
    --temperature 1.0 \
    --top_p 1.0 \
    --top_k 0 \
    --min_tokens 1 \
    --length_weight 0.0 \
    --zstd_penalty_weight 0.0 \
    --ngram_penalty 0.0 \
    --neg_adv_weight 1.0 \
    --kl_coef 0.0 \
    --lr "$LR" \
    --micro_batch 1 \
    --rollout_forward_batch 8 \
    --hard_buffer_target_samples 0 \
    --eval_interval 999999 \
    --skip_preeval 1 \
    --skip_posteval 1 \
    --save_interval 999999 \
    --save_last 1 \
    --final_full_eval 1 \
    --seed 42 | tee "$OUT_DIR/train_stdout.log"
  echo "[$(date)] DONE lr=$LR out=$OUT_DIR"
done
