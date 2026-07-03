# GRPO训练框架重构版

## 概述

这是一个完全重构的GRPO (Group Relative Policy Optimization) 训练框架，用于RWKV模型的强化学习微调。

## 文件结构

```
.
├── utils.py          # 工具函数 (日志、数据读取、提示构建等)
├── reward.py         # 答案提取和奖励函数
├── infer.py          # 推理逻辑 (包含正确的采样流程)
├── train.py          # GRPO训练逻辑
└── main.py           # 主程序入口
```

## 主要改进

### 1. Next Token Prediction 采样流程

**正确的顺序**: 重复惩罚 → 温度 → Top-K → 计算概率 → Top-P → 采样

```python
# 在infer.py中实现
def sample_next_token(logits, token_counts, temperature, top_p, top_k, ...):
    # 1. 保存原始log概率 (用于RL)
    original_logp = F.log_softmax(logits, dim=-1)
    
    # 2. 应用重复惩罚
    logits = apply_repetition_penalty(logits, token_counts, ...)
    
    # 3. 应用温度
    logits = apply_temperature(logits, temperature)
    
    # 4. 应用Top-K
    logits = apply_top_k(logits, top_k)
    
    # 5. 计算概率
    probs = F.softmax(logits, dim=-1)
    
    # 6. 应用Top-P
    probs = apply_top_p(probs, top_p)
    
    # 7. 采样
    token_ids = torch.multinomial(probs, num_samples=1)
    
    # 返回采样token和原始log概率
    return token_ids, original_logp[token_ids]
```

### 2. GRPO训练方式

完全对齐GRPO论文的实现：

**Reference Model**: 使用初态的无梯度time_state的infer model，不是old logp
- 在每次采样后，使用reference model重新计算log概率
- Reference model = infer_model + 初始time_state (无梯度)

**损失函数**: 
- 逐token计算
- 除以token长度和group size进行归一化
- 公式: `loss = -advantage * log_ratio / (token_length * group_size)`

**KL散度**: 使用无偏KL散度
```
D_KL[ref || policy] = exp(log_ratio) - log_ratio - 1
其中 log_ratio = ref_logp - policy_logp
```

**Advantage**: 使用均值-方差归一化
```python
mean_reward = sum(rewards) / len(rewards)
std_reward = sqrt(variance)
advantage = (reward - mean_reward) / std_reward
```

### 3. 采样方式

- 每步从训练集中随机抽取 `num_questions` 道题 (默认24，可修改)
- 每道题做 `samples_per_question` 次采样 (默认8，可修改)
- 每道题的所有样本作为一个group
- 在group内计算advantage

### 4. 奖励函数

```python
def calculate_reward(text, ground_truth, token_length, min_tokens, max_tokens, length_weight):
    # 1. 答案正确: +1分
    if is_correct:
        reward += 1.0
        
        # 2. 格式正确: +1分 (在答案正确基础上)
        if is_format_correct:
            reward += 1.0
    
    # 3. 长度奖励
    lambda_val = 0.5 - (token_length - min_tokens) / (max_tokens - min_tokens)
    if is_correct:
        length_reward = length_weight * lambda_val
    else:
        length_reward = length_weight * min(0, lambda_val)
    
    reward += length_reward
    return reward
```

### 5. 答案提取

支持多种格式，使用正则匹配：

1. `\boxed{answer}`
2. `answer is answer`
3. `answer: answer`
4. 最后一行作为答案

**冲突处理**: 如果命中多个答案，使用**最后一个**

**正则匹配**: 允许中间有任意空格和标点符号

## 使用方法

### 基本用法

```bash
python main.py \
    --train_jsonl data/train.jsonl \
    --eval_jsonl data/eval.jsonl \
    --model models/rwkv-7b.pth \
    --tokenizer tokenizer/rwkv_vocab_v20230424.txt \
    --out_dir output \
    --total_steps 100 \
    --num_questions 24 \
    --samples_per_question 8
```

### 主要参数

**数据集**:
- `--train_jsonl`: 训练数据路径
- `--eval_jsonl`: 评估数据路径 (可选)
- `--max_data_samples`: 最大数据样本数 (可选)

**模型**:
- `--model`: 模型路径
- `--tokenizer`: 分词器路径
- `--state_init`: 初始time_state路径 (可选)
- `--ctx_len`: 上下文长度 (默认4096)

**采样配置**:
- `--num_questions`: 每步采样的题目数 (默认24)
- `--samples_per_question`: 每道题的采样次数 (默认8)

**生成配置**:
- `--max_new_tokens`: 最大生成token数 (默认2048)
- `--temperature`: 温度 (默认1.0)
- `--top_p`: top-p参数 (默认0.9)
- `--top_k`: top-k参数 (默认0)

**奖励配置**:
- `--min_tokens`: 最小token数 (默认50)
- `--length_weight`: 长度奖励权重 (默认0.5)

**训练配置**:
- `--total_steps`: 总训练步数 (默认100)
- `--lr`: 学习率 (默认1e-5)
- `--kl_coef`: KL系数 (默认0.01)
- `--grad_clip`: 梯度裁剪 (默认1.0)

**正则化**:
- `--time_state_l2`: time_state L2正则化 (默认0.0)
- `--time_state_clamp`: time_state裁剪 (默认0.0)

## 数据格式

训练数据应为JSONL格式，每行一个JSON对象：

```json
{"problem": "问题描述", "answer": "答案"}
```

## 输出

训练过程会在输出目录生成以下文件：

- `train.log`: 训练日志
- `metrics.jsonl`: 训练指标 (accuracy, reward, loss, kl等)
- `responses.jsonl`: 所有生成的响应
- `ckpt_stepX.pth`: 检查点文件

## 关键特性

1. ✅ **正确的采样顺序**: 重复惩罚 → 温度 → Top-K → Top-P
2. ✅ **GRPO算法**: 使用reference model而非old logp
3. ✅ **无偏KL散度**: `exp(log_ratio) - log_ratio - 1`
4. ✅ **逐token损失**: 除以token长度和group size
5. ✅ **灵活的采样**: 可配置题目数和每题采样次数
6. ✅ **完整的奖励函数**: 正确性 + 格式 + 长度
7. ✅ **鲁棒的答案提取**: 支持多种格式，正则匹配

## 注意事项

1. 确保 `rwkv7_trainable.py` 和 `reference/` 目录在Python路径中
2. 需要CUDA支持以获得最佳性能
3. 根据显存调整 `micro_batch` 大小
4. 建议从小的 `num_questions` 开始测试

## 许可证

MIT License
