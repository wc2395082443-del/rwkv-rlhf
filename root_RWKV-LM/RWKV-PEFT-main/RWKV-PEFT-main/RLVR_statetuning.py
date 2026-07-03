import os, sys, re, json, torch, random
import numpy as np
from argparse import ArgumentParser
from lightning import Trainer, LightningModule, seed_everything
from torch.nn import functional as F
os.environ["RWKV_JIT_ON"] = "0"
# 强制指定算子实现为纯 pytorch 模式
os.environ["WKV"] = "pytorch"
# 关闭融合内核
os.environ["FUSED_KERNEL"] = "0"
os.environ["RWKV_JIT_ON"] = "0"       # 彻底关闭 JIT，防止它去跑 nvcc
os.environ["FUSED_KERNEL"] = "0"     # 彻底关闭融合算子，不再需要 rwkvfla
os.environ["RWKV_MY_RTT"] = "7"      # 强制指定版本 7
os.environ["WKV"] = "pytorch"        # 强制算子路径走 pytorch 模式
from lightning_utilities.core.rank_zero import rank_zero_info
def set_rwkv_env(args):
    os.environ["RWKV_MY_TESTING"] = args.my_testing
    os.environ["RWKV_CTXLEN"] = str(args.ctx_len)
    os.environ["RWKV_HEAD_SIZE_A"] = str(args.head_size_a)
    os.environ["RWKV_FLOAT_MODE"] = args.precision
    os.environ["RWKV_JIT_ON"] = "0"
    os.environ["WKV"] = args.op
    os.environ["FUSED_KERNEL"] = '1' if args.fused_kernel else '0'
    # 核心：State Tuning 必须开启这个
    if args.peft == 'state':
        os.environ["RWKV_TRAIN_TYPE"] = 'state'
    else:
        os.environ["RWKV_TRAIN_TYPE"] = args.train_type
# 解决 Windows 环境变量找不到的问题
if os.name == 'nt':
    os.environ["PATH"] = os.environ["PATH"] + ";" + "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.9\\bin"
# 假设的基础组件
from rwkvt.peft_loading import load_peft_model
from rwkvt.lightning_train.trainer import train_callback

########################################################################################################
# 1. 题目难度追踪器 (用于筛选 50% 正确率的题目)
########################################################################################################
import re
def load_gsm8k_1percent(file_path):
    """
    只加载 JSONL 文件中前 1% 的数据。
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到数据集文件: {file_path}")

    all_data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                all_data.append(json.loads(line))
    
    # 计算 1% 的数量
    num_to_keep = max(1, len(all_data) // 100)
    selected_data = all_data[:num_to_keep]
    
    print(f"📊 数据集总数: {len(all_data)}")
    print(f"🚀 已截取前 1% 数据进行训练，共计: {len(selected_data)} 道题")
    
    # 格式化数据
    formatted_dataset = []
    for idx, item in enumerate(selected_data):
        formatted_dataset.append({
            'id': idx,
            'problem': item.get('problem') or item.get('question'),
            'solution': item.get('solution') or item.get('answer')
        })
    return formatted_dataset
def get_reward_gsm8k(completion, ground_truth):
    """
    可验证奖励函数：
    从模型生成的 completion 和标准答案 ground_truth 中提取数字并比对。
    """
    def extract_answer(text):
        if not text:
            return None
        
        # 1. 尝试提取 LaTeX \boxed{} 里的内容 (常见于标准答案)
        boxed_match = re.findall(r'\\boxed\{([^}]+)\}', text)
        if boxed_match:
            res = boxed_match[-1].replace(',', '').strip()
            # 提取里面的数字
            nums = re.findall(r'-?\d+\.?\d*', res)
            if nums: return nums[0]

        # 2. 尝试提取 GSM8K 格式的 '####' 后的数字
        hash_match = re.findall(r'####\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)', text)
        if hash_match:
            return hash_match[-1].replace(',', '')

        # 3. 尝试提取 "answer is" 后的数字
        alt_match = re.findall(r'(?:answer is|答案是)[:\s]*\$?\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)', text.lower())
        if alt_match:
            return alt_match[-1].replace(',', '')

        # 4. 最后兜底：提取文本中出现的最后一个数字
        all_nums = re.findall(r'-?\d+(?:,\d{3})*(?:\.\d+)?', text)
        if all_nums:
            return all_nums[-1].replace(',', '')
        
        return None

    # 提取预测值和真实值
    pred_str = extract_answer(completion)
    gt_str = extract_answer(ground_truth)

    if pred_str is None or gt_str is None:
        return 0.0

    try:
        # 转换为浮点数比对，允许极小的数值误差
        pred_val = float(pred_str)
        gt_val = float(gt_str)
        return 1.0 if abs(pred_val - gt_val) < 1e-4 else 0.0
    except Exception:
        # 如果转换失败（例如提取到的是奇怪的字符），返回 0 分
        return 0.0
import json
import os

def load_gsm8k_jsonl(file_path):
    """
    从 JSONL 文件加载 GSM8K 数据集。
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"数据集文件未找到: {file_path}")

    dataset = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                # 兼容不同的字段命名习惯
                problem = data.get('problem') or data.get('question')
                solution = data.get('solution') or data.get('answer')
                
                if problem and solution:
                    dataset.append({
                        'id': line_idx, # 给每个题一个唯一 ID 用于 Tracker 记录历史正确率
                        'problem': problem,
                        'solution': solution
                    })
            except json.JSONDecodeError:
                print(f"警告：跳过无法解析的行 {line_idx}")
                
    print(f"成功加载数据集，共 {len(dataset)} 条题目。")
    return dataset
class QuestionTracker:
    def __init__(self, data_list):
        # data_list 每个元素需有唯一 ID
        self.questions = data_list
        self.stats = {i: {"correct": 0, "total": 0} for i in range(len(data_list))}

    def get_accuracy(self, q_idx):
        stat = self.stats[q_idx]
        if stat["total"] == 0: return 0.5  # 没做过的题默认为 0.5，增加采样机会
        return stat["correct"] / stat["total"]

    def sample_indices(self, num_to_sample):
        """
        采样逻辑：优先采样正确率在 50% 左右的题目
        权重计算：1 / (|accuracy - 0.5| + 0.01)
        """
        indices = list(range(len(self.questions)))
        weights = []
        for i in indices:
            acc = self.get_accuracy(i)
            # 距离 0.5 越近，权重越高
            weight = 1.0 / (abs(acc - 0.5) + 0.01)
            weights.append(weight)
        
        # 归一化权重
        weights = np.array(weights)
        weights /= weights.sum()
        
        return np.random.choice(indices, size=num_to_sample, p=weights, replace=False)

    def update(self, q_idx, is_correct):
        self.stats[q_idx]["total"] += 1
        if is_correct:
            self.stats[q_idx]["correct"] += 1

########################################################################################################
# 2. RLVR 训练模块 (核心要求实现)
########################################################################################################

class RWKV7_RLVR_v2(LightningModule):
    def __init__(self, model, tokenizer, tracker, args):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.tracker = tracker
        self.args = args
        self.target_correct_count = 64  # 截图要求：直到得到 64 条正确答案

    def training_step(self, batch, batch_idx):
        # 注意：这里的 batch 只是一个占位，我们手动从 tracker 中采样
        correct_trajectories = []
        attempts = 0

        # --- 截图要求：直到得到 64 条正确答案停止 ---
        self.model.eval()
        while len(correct_trajectories) < self.target_correct_count:
            # 1. 按照“正确率 50% 优先”原则采样题目
            q_idx = self.tracker.sample_indices(1)[0]
            item = self.tracker.questions[q_idx]
            
            prompt = f"Question: {item['problem']}\n\nAnswer: Let's solve this step by step.\n"
            input_ids = self.tokenizer.encode(prompt)
            
            # 2. 采样 1 份答案 (每次 rollout 1 份答案)
            with torch.no_grad():
                out_ids = self.model.generate(
                    torch.tensor([input_ids], device=self.device),
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.8
                )[0]
            
            completion = self.tokenizer.decode(out_ids[len(input_ids):])
            
            # 3. 验证正确性 (Verifiable Reward)
            is_correct = get_reward_gsm8k(completion, item['solution'])
            
            # 更新历史正确率
            self.tracker.update(q_idx, is_correct > 0)
            
            if is_correct > 0:
                correct_trajectories.append({
                    "ids": out_ids,
                    "q_idx": q_idx
                })
            
            attempts += 1
            if attempts > 500: # 安全阈值，防止死循环
                break

        # --- 截图要求：训练一个 step 取这 64 条答案 ---
        self.model.train()
        if len(correct_trajectories) == 0: return None

        # 将 64 条正确轨迹打包
        all_ids = torch.stack([t["ids"] for t in correct_trajectories])
        
        # 计算 Loss (State Tuning)
        # 因为全是正确答案，这里演变为最大化这些正确路径的似然概率
        logits = self.model(all_ids)
        # 传统的 RLVR/GRPO 会减去基线，但截图要求只取正确答案训练
        # 此时类似于带有难度筛选的 Rejection Sampling Fine-Tuning
        loss = self.calculate_loss(logits, all_ids) 
        
        self.log("rollout_eff", len(correct_trajectories) / attempts) # 记录采样效率
        self.log("loss", loss, prog_bar=True)
        return loss

    def calculate_loss(self, logits, ids):
        # 标准的交叉熵 Loss，仅计算生成部分的梯度
        # 梯度会流回 RWKV-7 的 State 参数
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = ids[..., 1:].contiguous()
        loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return loss

    def configure_optimizers(self):
        # 只训练 State 相关参数
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        return torch.optim.Adam(
        trainable_params, 
        lr=self.args.lr_init, 
        betas=(0.9, 0.999), 
        eps=1e-8
    )
########################################################################################################
# 3. 启动逻辑
########################################################################################################

def p_args():
    parser = ArgumentParser()

    # --- 模型与路径配置 ---
    parser.add_argument("--load_model", default="C:\\RWKV-LM\\rwkv7-g1b-1.5b-20251202-ctx8192.pth", type=str, help="预训练模型路径")
    parser.add_argument("--proj_dir", default="out/rwkv7_rlvr_gsm8k", type=str, help="训练输出保存目录")
    parser.add_argument("--data_file", default="C:\\RWKV-LM\\RWKV7-statetuning\\gsm8k_test_formatted.jsonl", type=str, help="GSM8K 训练集路径")
    parser.add_argument("--vocab_size", default=65536, type=int)
    parser.add_argument("--head_size", default=64, type=int)

    # --- 训练超参数 (State Tuning) ---
    parser.add_argument("--ctx_len", default=1024, type=int, help="推理与训练的最大上下文长度")
    parser.add_argument("--epoch_count", default=50, type=int, help="训练轮数")
    parser.add_argument("--micro_bsz", default=1, type=int, help="RL采样时的并行Batch（建议1，采样循环在内部实现）")
    parser.add_argument("--lr_init", default=1e-4, type=float, help="State 参数的学习率（建议比SFT小）")
    parser.add_argument("--betas", default=(0.9, 0.99), type=tuple)
    parser.add_argument("--precision", default="bf16", type=str, choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--devices", default=1, type=int, help="GPU 数量")
    
    # --- PEFT 设置 ---
    parser.add_argument("--peft", default="state", type=str, help="必须设置为 state 以启用 State Tuning")

    # --- RLVR 核心要求设置 (对应截图内容) ---
    parser.add_argument("--target_correct_count", default=64, type=int, help="每步训练必须凑齐的正确答案数量")
    parser.add_argument("--max_attempts", default=512, type=int, help="单步采样最大尝试次数（防止模型太弱死循环）")
    parser.add_argument("--temp", default=0.8, type=float, help="采样温度")
    parser.add_argument("--top_p", default=0.85, type=float, help="采样 Top-P")
    parser.add_argument("--max_new_tokens", default=512, type=int, help="生成回答的最大长度")
    parser.add_argument("--load_model", default="", type=str)  # full path, with .pth
    parser.add_argument("--wandb", default="", type=str)  # wandb project name. if "" then don't use wandb
    parser.add_argument("--proj_dir", default="out", type=str)
    parser.add_argument("--random_seed", default="-1", type=int)

    parser.add_argument("--data_file", default="", type=str)
    parser.add_argument("--data_type", default="utf-8", type=str) #binidx / sft
    parser.add_argument("--vocab_size", default=0, type=int)  # vocab_size = 0 means auto (for char-level LM and .txt data)

    parser.add_argument("--ctx_len", default=1024, type=int)
    parser.add_argument("--epoch_steps", default=1000, type=int)  # a mini "epoch" has [epoch_steps] steps
    parser.add_argument("--epoch_count", default=500, type=int)  # train for this many "epochs". will continue afterwards with lr = lr_final
    parser.add_argument("--epoch_begin", default=0, type=int)  # if you load a model trained for x "epochs", set epoch_begin = x
    parser.add_argument("--epoch_save", default=5, type=int)  # save the model every [epoch_save] "epochs"

    parser.add_argument("--micro_bsz", default=12, type=int)  # micro batch size (batch size per GPU)
    parser.add_argument("--n_layer", default=6, type=int)
    parser.add_argument("--n_embd", default=512, type=int)
    parser.add_argument("--dim_att", default=0, type=int)
    parser.add_argument("--dim_ffn", default=0, type=int)
    parser.add_argument("--pre_ffn", default=0, type=int)  # replace first att layer by ffn (sometimes better)
    parser.add_argument("--head_qk", default=0, type=int)  # my headQK trick
    parser.add_argument("--tiny_att_dim", default=0, type=int)  # tiny attention dim
    parser.add_argument("--tiny_att_layer", default=-999, type=int)  # tiny attention @ which layer

    parser.add_argument("--lr_init", default=6e-4, type=float)  # 6e-4 for L12-D768, 4e-4 for L24-D1024, 3e-4 for L24-D2048
    parser.add_argument("--lr_final", default=1e-5, type=float)
    parser.add_argument("--warmup_steps", default=-1, type=int)  # try 50 if you load a model
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.99, type=float)  # use 0.999 when your model is close to convergence
    parser.add_argument("--adam_eps", default=1e-8, type=float)
    parser.add_argument("--grad_cp", default=0, type=int)  # gradient checkpt: saves VRAM, but slower
    parser.add_argument("--dropout", default=0, type=float) # try 0.01 / 0.02 / 0.05 / 0.1
    parser.add_argument("--weight_decay", default=0, type=float) # try 0.1 / 0.01 / 0.001
    parser.add_argument("--weight_decay_final", default=-1, type=float)

    parser.add_argument("--my_pile_version", default=1, type=int)  # my special pile version
    parser.add_argument("--my_pile_stage", default=0, type=int)  # my special pile mode
    parser.add_argument("--my_pile_shift", default=-1, type=int)  # my special pile mode - text shift
    parser.add_argument("--my_pile_edecay", default=0, type=int)
    parser.add_argument("--layerwise_lr", default=1, type=int)  # layerwise lr for faster convergence (but slower it/s)
    parser.add_argument("--ds_bucket_mb", default=200, type=int)  # deepspeed bucket size in MB. 200 seems enough
    # parser.add_argument("--cuda_cleanup", default=0, type=int)  # extra cuda cleanup (sometimes helpful)

    parser.add_argument("--my_sample_len", default=0, type=int)
    parser.add_argument("--my_ffn_shift", default=1, type=int)
    parser.add_argument("--my_att_shift", default=1, type=int)
    parser.add_argument("--head_size_a", default=64, type=int) # can try larger values for larger models
    parser.add_argument("--head_size_divisor", default=8, type=int)
    parser.add_argument("--my_pos_emb", default=0, type=int)
    parser.add_argument("--load_partial", default=0, type=int)
    parser.add_argument("--magic_prime", default=0, type=int)
    parser.add_argument("--my_qa_mask", default=0, type=int)
    parser.add_argument("--my_random_steps", default=0, type=int)
    parser.add_argument("--my_testing", default='7', type=str)
    parser.add_argument("--my_exit", default=99999999, type=int)
    parser.add_argument("--my_exit_tokens", default=0, type=int)

    parser.add_argument("--peft", default="none", type=str)# lora pissa DiSHA
    #parser.add_argument("--train_parts", default=["time", "ln"], type=list)##emb , head
    parser.add_argument("--train_parts", default=["time", "ln"], nargs='*', help="List of parts to train emb head time ln")

    #LORA
    parser.add_argument("--lora_config", default='{"lora_load":"", "lora_r":8, "lora_alpha":32, "lora_dropout":0.01}', type=json.loads)

    parser.add_argument(
        "--peft_config",
        type=str,
        default="{}",
        help="PEFT config JSON string, e.g. '{\"r\":8, \"alpha\":32, \"dropout\":0.05, \"target_modules\":[\"receptance\",\"key\",\"value\",\"output\"]}'"
    )

    # #LISA
    # parser.add_argument("--lisa_config", default='{"lisa_r":2, "lisa_k":100}', type=json.loads)

    #PISSA
    parser.add_argument("--pissa_config", default='{"pissa_load":"", "pissa_init":"", "pissa_r":8, "svd_niter":4}', type=json.loads)

    #Bone
    parser.add_argument("--miss_config", default='{"mode":"mode", "load":"", "r":64}', type=json.loads)
    parser.add_argument("--merge", type=int, default=1, help="1=merge PEFT weights, 0=save PEFT-only")

    #quant
    parser.add_argument("--quant", default="none", type=str)

    #dataset
    parser.add_argument("--dataload", default="pad", type=str)

    parser.add_argument("--chunk_ctx", default=512, type=int)
    #fla
    parser.add_argument("--fla", action="store_true")
    parser.add_argument("--train_type", default="none", type=str)

    #loss_mask
    parser.add_argument("--loss_mask", default="none", type=str)### pad qa se
    parser.add_argument("--mask_id", default='{"mask0":"0", "mask1":"1"}', type=json.loads)
    parser.add_argument("--data_shuffle", default=1, type=int)


    #new optim
    parser.add_argument("--optimizer", default="none", type=str)

    #acc_grad_batchs
    parser.add_argument("--avg_loss", default=0, type=int)


    parser.add_argument("--sft_field", default=None, type=str, nargs='+', help='List of fields for SFT')
    parser.add_argument("--sft_split", default="train", type=str)


    parser.add_argument("--op", default="pytorch", type=str)
    parser.add_argument("--fused_kernel", action='store_true', help="Enable rwkv-fla fused kernel")

    parser.add_argument("--lr_schedule", default="cos", type=str)        #['cos', 'wsd']


    parser.add_argument("--accelerator", default="gpu", type=str)
    parser.add_argument("--strategy", default="auto", type=str)
    parser.add_argument("--devices", default=1, type=int)
    parser.add_argument("--num_nodes", default=1, type=int)
    parser.add_argument("--precision", default="fp16", type=str)
    parser.add_argument("--accumulate_grad_batches", default=1, type=int)

    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--persistent_workers", action="store_true")
    parser.add_argument("--prefetch_factor", default=None, type=int)
    args = parser.parse_args()
    return args

########################################################################################################
# 启动主逻辑
########################################################################################################

def main():
    rank_zero_info("########## work in progress ##########")
    args = p_args()
    seed_everything(42)
    os.environ["RWKV_MY_TESTING"] = '7'
    os.environ["RWKV_CTXLEN"] = str(args.ctx_len)
    os.environ["RWKV_HEAD_SIZE_A"] = str(args.head_size_a)
    ######state tuning
    if args.peft=='state':
        os.environ["RWKV_TRAIN_TYPE"] = 'state'
    else:
        os.environ["RWKV_TRAIN_TYPE"] = args.train_type


    print(f"########## WKV OP           {args.op}               ##########\n" * 1)
    print(f"########## FUSED OP    {args.fused_kernel}          ##########\n" * 1)
    os.environ["WKV"]= args.op
    os.environ["FUSED_KERNEL"] = '0'
    # 1. 加载数据与难度追踪器
    # 使用之前补充的 load_gsm8k_1percent 函数
    raw_data = load_gsm8k_1percent(args.data_file)
    tracker = QuestionTracker(raw_data)

    # 2. 加载模型 (核心：RWKV-7 + State Tuning)
    # 假设 load_peft_model 会处理 requires_grad=False 并识别 peft='state'
    args, model = load_peft_model(args)

    # 3. 初始化自定义 RLVR 模块
    # 传入之前定义的 RWKV7_RLVR_v2 类
    rl_model = RWKV7_RLVR_v2(model, model.tokenizer, tracker, args)

    # 4. 配置训练器
    trainer = Trainer(
        accelerator="gpu",
        devices=args.devices,
        precision=args.precision,
        max_epochs=args.epoch_count,
        # 截图要求“没有 buffer”，意味着我们不使用长期的回放池，随采随练
        callbacks=[train_callback(args)],
        enable_checkpointing=True,
        logger=True,
        accumulate_grad_batches=1
    )

    # 5. 启动训练
    # 因为采样逻辑在 training_step 内部实现，这里传入一个简单的 Range 数据加载器即可
    # 每一轮调用都会执行一次“采样64条 -> 训练”的循环
    from torch.utils.data import DataLoader, Dataset
    class DummyDataset(Dataset):
        def __len__(self): return 100000 
        def __getitem__(self, idx): return {}

    dummy_loader = DataLoader(DummyDataset(), batch_size=1)
    
    print(f"🚀 开始 RLVR State-Tuning...")
    print(f"目标：每步凑齐 {args.target_correct_count} 条正确答案")
    print(f"策略：优先采样历史正确率 50% 左右的题目")
    
    trainer.fit(rl_model, dummy_loader)

if __name__ == "__main__":
    main()