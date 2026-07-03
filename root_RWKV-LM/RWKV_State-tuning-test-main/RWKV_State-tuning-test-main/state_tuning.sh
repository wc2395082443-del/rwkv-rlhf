load_model="/your/path/rwkv7-g1b-1.5b-20251202-ctx8192.pth" # 填写下载好的RWKV-7模型的路径
proj_dir='/your/path/' # 训练输出和保存state的路径
data_file='NekoQA-10K.jsonl'

# n_layer和n_embd根据基底RWKV模型的参数设置
n_layer=24
n_embd=2048

micro_bsz=8 # 微批次大小，根据数据量和显存大小调整
epoch_save=1 # 保存state的频率
epoch_steps=1000 # 每个训练轮次的步数，增加会拉长单个epoch的训练时间
ctx_len=1024 # 微调模型的上下文长度

# train.py文件位于RWKV-PEFT路径下
python /RWKV-PEFT/train.py --load_model $load_model \
--proj_dir $proj_dir \
--data_file $data_file \
# 词表大小 
--vocab_size 65536 \
# 训练语料的文件格式
--data_type jsonl \
--n_layer $n_layer \
--n_embd $n_embd \
--ctx_len $ctx_len \
--micro_bsz $micro_bsz \
--epoch_steps $epoch_steps \
# 总训练轮次，state tuning不需要过多反复训练
--epoch_count 10 \
--epoch_save $epoch_save \
--lr_init 1e-3 \
--lr_final 1e-5 \
--accelerator gpu \
--precision bf16 \
# 使用的GPU数量
--devices 2 \
# lightning 训练策略参数
--strategy deepspeed_stage_1 \
# 梯度累积步数，0训练更快但需更多显存，1训练较慢但节省显存
--grad_cp 1 \
# 训练的RWKV模型版本，v7选x070，v6选x060
--my_testing "x070" \
# 微调训练类型，state tuning微调填state
--peft state \
# 选择算子，state tuning仅支持fla算子
--op fla
