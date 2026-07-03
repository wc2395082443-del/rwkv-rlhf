# GRPO训练配置说明

## 配置参数详解

### 1. 采样配置

```bash
--num_questions 24              # 每步从训练集随机抽取的题目数
                                # - 增大可以增加训练样本多样性
                                # - 减小可以加快训练速度
                                # - 推荐: 16-32

--samples_per_question 8        # 每道题采样的响应数 (group size)
                                # - 这些样本会在group内计算advantage
                                # - 增大可以提高优势估计的准确性
                                # - 减小可以加快训练速度
                                # - 推荐: 4-16
```

### 2. 生成配置

```bash
--max_new_tokens 2048           # 最大生成token数
                                # - 根据问题复杂度调整
                                # - 数学问题推荐: 1024-2048

--temperature 1.0               # 温度参数
                                # - 越高越随机，越低越确定性
                                # - 训练时推荐: 0.8-1.2
                                # - 推理时推荐: 0.0-0.3

--top_p 0.9                     # Nucleus采样参数
                                # - 保留累积概率达到top_p的token
                                # - 推荐: 0.8-0.95

--top_k 0                       # Top-K采样参数
                                # - 0表示不使用top-k
                                # - 如果使用，推荐: 40-100
```

### 3. 奖励配置

```bash
--min_tokens 50                 # 长度奖励的最小token数
                                # - 低于此长度会有负奖励(如果答错)
                                # - 根据问题类型调整

--length_weight 0.5             # 长度奖励的权重
                                # - 控制长度奖励的重要性
                                # - 推荐: 0.3-0.7
                                # - 公式: lambda = 0.5 - (len-min)/(max-min)
                                #   正确: reward += weight * lambda
                                #   错误: reward += weight * min(0, lambda)
```

### 4. 训练配置

```bash
--total_steps 100               # 总训练步数
                                # - 根据数据集大小和收敛情况调整
                                # - 小数据集: 50-200
                                # - 大数据集: 200-1000

--lr 1e-5                       # 学习率
                                # - 推荐范围: 1e-6 到 1e-4
                                # - time_state通常需要较小的学习率

--kl_coef 0.01                  # KL散度系数
                                # - 控制与reference model的偏离程度
                                # - 越大越保守，越小越激进
                                # - 推荐: 0.005-0.05

--grad_clip 1.0                 # 梯度裁剪
                                # - 防止梯度爆炸
                                # - 推荐: 0.5-2.0

--micro_batch 4                 # Micro batch大小
                                # - 根据显存调整
                                # - 显存受限时减小
                                # - 推荐: 2-8

--ppo_epochs 1                  # PPO epoch数
                                # - 每批数据重复训练的次数
                                # - 推荐: 1-3
```

### 5. 正则化配置

```bash
--time_state_l2 0.0             # time_state L2正则化
                                # - 防止time_state偏离初始值太远
                                # - 如果出现不稳定，可以尝试: 0.001-0.01
                                # - 默认不使用

--time_state_clamp 0.0          # time_state值裁剪
                                # - 限制time_state的绝对值
                                # - 如果出现数值问题，可以尝试: 1.0-10.0
                                # - 默认不使用
```

### 6. 日志和保存配置

```bash
--log_interval 1                # 日志打印间隔 (步数)
--save_interval 10              # 检查点保存间隔 (步数)
--eval_interval 5               # 评估间隔 (步数，如果实现了评估)
```

## 推荐配置组合

### 快速测试配置
```bash
--num_questions 8 \
--samples_per_question 4 \
--total_steps 20 \
--micro_batch 2
```

### 标准训练配置
```bash
--num_questions 24 \
--samples_per_question 8 \
--total_steps 100 \
--lr 1e-5 \
--kl_coef 0.01 \
--micro_batch 4
```

### 高质量训练配置
```bash
--num_questions 32 \
--samples_per_question 16 \
--total_steps 200 \
--lr 5e-6 \
--kl_coef 0.02 \
--micro_batch 4 \
--time_state_l2 0.001
```

### 显存受限配置
```bash
--num_questions 16 \
--samples_per_question 4 \
--micro_batch 2 \
--max_new_tokens 1024
```

## 调优建议

1. **收敛太慢**:
   - 增大学习率 (lr)
   - 减小KL系数 (kl_coef)
   - 增大samples_per_question

2. **训练不稳定**:
   - 减小学习率 (lr)
   - 增大KL系数 (kl_coef)
   - 添加L2正则化 (time_state_l2)
   - 添加值裁剪 (time_state_clamp)

3. **显存不足**:
   - 减小micro_batch
   - 减小num_questions
   - 减小samples_per_question
   - 减小max_new_tokens

4. **生成质量差**:
   - 调整temperature (推理时用低温度)
   - 调整top_p
   - 检查奖励函数是否合理
   - 增加训练步数

5. **过拟合**:
   - 增大KL系数
   - 添加L2正则化
   - 减少训练步数
   - 增加数据多样性
