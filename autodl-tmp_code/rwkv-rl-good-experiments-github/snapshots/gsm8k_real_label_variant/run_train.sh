#!/bin/bash
# 示例训练脚本

# 设置路径
TRAIN_DATA="data/train.jsonl"
EVAL_DATA="data/eval.jsonl"
MODEL_PATH="models/rwkv-7b.pth"
TOKENIZER_PATH="tokenizer/rwkv_vocab_v20230424.txt"
OUTPUT_DIR="output/grpo_$(date +%Y%m%d_%H%M%S)"

# 基本配置
NUM_QUESTIONS=24         # 每步采样的题目数
SAMPLES_PER_QUESTION=8   # 每道题的采样次数
TOTAL_STEPS=100          # 总训练步数

# 生成配置
MAX_NEW_TOKENS=2048
TEMPERATURE=1.0
TOP_P=0.9
TOP_K=0

# 奖励配置
MIN_TOKENS=50
LENGTH_WEIGHT=0.5

# 训练配置
LEARNING_RATE=1e-5
KL_COEF=0.01
GRAD_CLIP=1.0
MICRO_BATCH=4

# 运行训练
python main.py \
    --train_jsonl "$TRAIN_DATA" \
    --eval_jsonl "$EVAL_DATA" \
    --model "$MODEL_PATH" \
    --tokenizer "$TOKENIZER_PATH" \
    --out_dir "$OUTPUT_DIR" \
    --num_questions $NUM_QUESTIONS \
    --samples_per_question $SAMPLES_PER_QUESTION \
    --total_steps $TOTAL_STEPS \
    --max_new_tokens $MAX_NEW_TOKENS \
    --temperature $TEMPERATURE \
    --top_p $TOP_P \
    --top_k $TOP_K \
    --min_tokens $MIN_TOKENS \
    --length_weight $LENGTH_WEIGHT \
    --lr $LEARNING_RATE \
    --kl_coef $KL_COEF \
    --grad_clip $GRAD_CLIP \
    --micro_batch $MICRO_BATCH \
    --log_interval 1 \
    --save_interval 10 \
    --seed 42

echo "训练完成! 输出目录: $OUTPUT_DIR"
