#!/bin/bash
set -euo pipefail

ROOT=/root/RWKV-LM/RWKV7-deepmath_trl_doc_g1f1p5b
DATA_DIR=${DATA_DIR:-$ROOT/data_deepmath_trl_doc}
MODEL=${MODEL:-/dev/shm/rwkv_models/rwkv7-g1f-1.5b-20260419-ctx8192.pth}
TOKENIZER=${TOKENIZER:-/root/RWKV-LM/rwkv_vocab_v20230424.txt}
STAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR=${OUT_DIR:-$ROOT/log/trl_doc_g1f1p5b_${STAMP}}

mkdir -p "$DATA_DIR" "$ROOT/log" "$OUT_DIR"

if [ ! -f "$DATA_DIR/deepmath_train_rwkv.jsonl" ]; then
  /root/miniconda3/bin/python "$ROOT/prepare_deepmath_rwkv_jsonl.py"     --dataset_path /dev/shm/official_repro_assets/DeepMath-103K-trl-hf-official     --out_dir "$DATA_DIR"
fi

cd "$ROOT"
/root/miniconda3/bin/python main.py   --train_jsonl "$DATA_DIR/deepmath_train_rwkv.jsonl"   --eval_jsonl "$DATA_DIR/deepmath_test_rwkv.jsonl"   --model "$MODEL"   --tokenizer "$TOKENIZER"   --out_dir "$OUT_DIR"   --tune_mode state   --reward_mode trl_doc   --prompt_mode trl_doc   --num_questions 8   --samples_per_question 8   --total_steps 300   --max_new_tokens 1024   --temperature 1.0   --top_p 1.0   --top_k 0   --min_tokens 1   --length_weight 0.0   --zstd_penalty_weight 0.0   --ngram_penalty 0.0   --neg_adv_weight 1.0   --kl_coef 0.0   --lr 1e-5   --micro_batch 2   --rollout_forward_batch 8   --hard_buffer_target_samples 0   --eval_interval 999999   --skip_preeval 1   --skip_posteval 1   --save_interval 999999   --save_last 1   --seed 42 | tee "$OUT_DIR/train_stdout.log"
