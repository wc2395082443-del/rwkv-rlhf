#!/bin/bash

# ========== 路径配置 ==========
load_model="C:\RWKV-LM\rwkv7-g1b-1.5b-20251202-ctx8192.pth"  # 基础模型路径
proj_dir="./output_neko"                                    # 输出目录
data_file="C:\RWKV-LM\RWKV_State-tuning-test-main\RWKV_State-tuning-test-main\NekoQA-10K.jsonl"                     # 数据集路径

# ========== 模型参数 ==========
n_layer=24          # RWKV-7 1.5B 的层数
n_embd=2048         # RWKV-7 1.5B 的嵌入维度
vocab_size=65536    # 词表大小
ctx_len=1024        # 上下文长度

# ========== 训练参数 ==========
micro_bsz=8         # 每个 GPU 的 batch size（根据显存调整）
epoch_steps=1000    # 每个 epoch 的步数
epoch_count=10      # 总训练轮数（State Tuning 不需要太多）
epoch_save=1        # 每几个 epoch 保存一次

# ========== 学习率 ==========
lr_init=1e-3        # 初始学习率（State Tuning 可以用较高学习率）
lr_final=1e-5       # 最终学习率

# ========== 硬件配置 ==========
devices=1           # GPU 数量
strategy="deepspeed_stage_1"  # 训练策略
grad_cp=1           # 梯度检查点（1=节省显存，0=更快但需更多显存）

# ========== RWKV 特定参数 ==========
my_testing="x070"   # RWKV-7 用 x070，RWKV-6 用 x060
peft_type="state"   # 微调类型：state
op="fla"            # 算子：State Tuning 只支持 fla
data_type="jsonl"   # 数据格式

# ========== 执行训练 ==========
python train.py \
  --load_model $load_model \
  --proj_dir $proj_dir \
  --data_file $data_file \
  --vocab_size $vocab_size \
  --data_type $data_type \
  --n_layer $n_layer \
  --n_embd $n_embd \
  --ctx_len $ctx_len \
  --micro_bsz $micro_bsz \
  --epoch_steps $epoch_steps \
  --epoch_count $epoch_count \
  --epoch_save $epoch_save \
  --lr_init $lr_init \
  --lr_final $lr_final \
  --accelerator gpu \
  --precision bf16 \
  --devices $devices \
  --strategy $strategy \
  --grad_cp $grad_cp \
  --my_testing $my_testing \
  --peft $peft_type \
  --op $op
