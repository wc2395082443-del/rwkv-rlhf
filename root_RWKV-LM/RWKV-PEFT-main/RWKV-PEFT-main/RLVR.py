########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################
# ===== RLVR imports =====
import sys
import os

EVAL_DIR = r"C:\RWKV-LM\Albatross\faster_251101"
os.chdir(EVAL_DIR)
if EVAL_DIR not in sys.path:
    sys.path.append(EVAL_DIR)
from reference.utils import TRIE_TOKENIZER
from reference.rwkv7 import RWKV_x070
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import os, sys

from eval import (
    build_prompt,
    verify_gsm8k_answer,
)

import os
import sys
import logging
# logging.basicConfig(level=logging.INFO)
from typing import Optional, Dict, Sequence, List, Literal
import subprocess
import torch
import jsonlines
import lightning as L
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, args, processor=None):
        self.args = args
        self.processor = processor
        self.data_type = args.data_type
        self.data = []

        # 针对 RLVR 模式下的 jsonl 加载优化
        if args.data_type == "jsonl" or args.train_type == "rlvr":
            print(f"载入 RLVR 专用数据集: {args.data_file}")
            with jsonlines.open(args.data_file) as file:
                for item in file:
                    # 适配 gsm8k_test_formatted.jsonl 的 Key 名
                    # 优先取 problem/solution，兼容 question/answer
                    question = item.get("problem") or item.get("question")
                    answer = item.get("solution") or item.get("answer") or item.get("original_answer")
                    
                    if question and answer:
                        self.data.append({
                            "question": question,
                            "answer": answer
                        })
            
            # 根据 epoch_steps 截断数据
            if args.epoch_steps > 0 and args.epoch_steps < len(self.data):
                self.data = self.data[:args.epoch_steps]
        
        # 保留原有的 binidx 等其他加载逻辑 (如果需要兼容)
        elif args.data_type == "binidx":
            # ... 原有的 MMapIndexedDataset 逻辑 ...
            pass

        print(f"✅ 数据集加载完成: 共 {len(self.data)} 条样本 (用于 {args.train_type})")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # RLVR 模式：直接返回 Dict 包含原始文本
        if self.args.train_type == "rlvr":
            return {
                "question": self.data[idx]["question"],
                "answer": self.data[idx]["answer"]
            }
        
        # 如果是传统的 SFT 或预训练模式，保留你原来的逻辑
        # (此处省略你原有的 tokenization 和 padding 逻辑)
        return self.data[idx]
def rlvr_collate_fn(batch):
    # 如果 micro_bsz 是 1，batch 就是 [item]，我们取第一个
    return batch[0]
class MyDataModule(L.LightningDataModule):
    def __init__(self, args, processor=None):
        super().__init__()
        self.args = args
        self.processor = processor
        self.train_dataset = None

    def setup(self, stage=None):
        # 实例化适配后的 Dataset
        self.train_dataset = MyDataset(self.args, self.processor)

    def train_dataloader(self):
        # RLVR 通常 micro_bsz 较小 (例如 1 或 2)，因为要进行自回归生成
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.micro_bsz,
            shuffle=True, 
            num_workers=self.args.num_workers,
            pin_memory=True,
            collate_fn=rlvr_collate_fn if self.args.micro_bsz == 1 else None
            # RLVR 建议不使用复杂的 collate_fn，直接返回 Dict 列表即可
        )
# 1. 强制定位 Ninja 绝对路径
# 既然提示 Ninja 运行正常，我们需要确保这个路径被传递给 subprocess
python_scripts = os.path.join(os.path.dirname(sys.executable), "Scripts")
ninja_exe = os.path.join(python_scripts, "ninja.exe")

if os.path.exists(ninja_exe):
    os.environ["PATH"] = python_scripts + os.pathsep + os.environ["PATH"]
    # 这一行是关键：让 torch 寻找 ninja 时直接能找到
    sys.path.insert(0, python_scripts) 
    print(f"✅ 强行锁定 Ninja 位置: {ninja_exe}")

# 2. 解决 LNK1181: 无法打开输入文件“aio.lib”和“cufile.lib”
# 这是 CUDA 12.8 在 Windows 上的 Bug，创建空文件欺骗链接器
def fix_cuda_12_libs():
    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path:
        lib_dir = os.path.join(cuda_path, "lib", "x64")
        for lib_name in ["aio.lib", "cufile.lib"]:
            lib_file = os.path.join(lib_dir, lib_name)
            if not os.path.exists(lib_file):
                try:
                    with open(lib_file, 'w') as f: pass
                    print(f"🛠️ 已自动创建缺失库补丁: {lib_name}")
                except Exception as e:
                    print(f"⚠️ 补丁创建失败（请以管理员权限运行）: {e}")

fix_cuda_12_libs()

# 3. 彻底禁用 Torch 的自动 Ninja 检查（如果上述仍失败）
# 这是一个黑科技：直接修改 torch 内部变量
import torch.utils.cpp_extension
def dummy_verify_ninja(): return True
torch.utils.cpp_extension.verify_ninja_availability = dummy_verify_ninja
print("🚀 已绕过 Torch 内部 Ninja 检查")
os.environ['PATH'] = "C:\\Program Files (x86)\\Microsoft Visual Studio\\18\\BuildTools\\VC\\Tools\\MSVC\\14.50.35717\\bin\\Hostx64\\x64\\"
if __name__ == "__main__":
    from argparse import ArgumentParser
    from lightning import Trainer
    from lightning.pytorch import seed_everything
    from lightning_utilities.core.rank_zero import rank_zero_info
    import lightning as pl
    import json
    from rwkvt.args_type import TrainingArgs
    rank_zero_info("########## work in progress ##########")

    parser = ArgumentParser()

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
    parser.add_argument("--my_testing", default='x070', type=str)
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


    parser.add_argument("--op", default="cuda", type=str)
    parser.add_argument("--fused_kernel", action='store_true', help="Enable rwkv-fla fused kernel")

    parser.add_argument("--lr_schedule", default="cos", type=str)        #['cos', 'wsd']


    parser.add_argument("--accelerator", default="gpu", type=str)
    parser.add_argument("--strategy", default="auto", type=str)
    parser.add_argument("--devices", default=1, type=int)
    parser.add_argument("--num_nodes", default=1, type=int)
    parser.add_argument("--precision", default="fp32", type=str)
    parser.add_argument("--accumulate_grad_batches", default=1, type=int)

    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--persistent_workers", action="store_true")
    parser.add_argument("--prefetch_factor", default=None, type=int)
    args = parser.parse_args()

    ########################################################################################################

    import os, warnings, math, datetime, sys, time
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    if "deepspeed" in args.strategy:
        import deepspeed
    # from pytorch_lightning import seed_everything

    if args.random_seed >= 0:
        print(f"########## WARNING: GLOBAL SEED {args.random_seed} THIS WILL AFFECT MULTIGPU SAMPLING ##########\n" * 3)
        seed_everything(args.random_seed)

    np.set_printoptions(precision=4, suppress=True, linewidth=200)
    warnings.filterwarnings("ignore", ".*Consider increasing the value of the `num_workers` argument*")
    warnings.filterwarnings("ignore", ".*The progress bar already tracks a metric with the*")
    # os.environ["WDS_SHOW_SEED"] = "1"

    #args.vocab_size = get_vocab_size(args)
    args.my_timestamp = datetime.datetime.today().strftime("%Y-%m-%d-%H-%M-%S")
    args.enable_checkpointing = False
    args.replace_sampler_ddp = False
    args.logger = False
    args.gradient_clip_val = 1.0
    args.num_sanity_val_steps = 0
    args.check_val_every_n_epoch = int(1e20)
    args.log_every_n_steps = int(1e20)
      # continue forever
    args.max_epochs = args.epoch_count
    if args.dataload =='get':
        args.max_epochs = -1
    args.betas = (args.beta1, args.beta2)
    args.real_bsz = int(args.num_nodes) * int(args.devices) * args.micro_bsz
    os.environ["RWKV_MY_TESTING"] = args.my_testing
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
    os.environ["FUSED_KERNEL"] = '1' if args.fused_kernel else '0'

    if args.dim_att <= 0:
        args.dim_att = args.n_embd
    if args.dim_ffn <= 0:
        args.dim_ffn = int((args.n_embd * 3.5) // 32 * 32) # default = 3.5x emb size


    args.run_name = f"{args.vocab_size} ctx{args.ctx_len} L{args.n_layer} D{args.n_embd}"
    if not os.path.exists(args.proj_dir):
        os.makedirs(args.proj_dir)

    
    try:
        deepspeed_version = deepspeed.__version__
    except:
        deepspeed_version = None
        pass


    assert args.data_type in ["utf-8", "utf-16le", "numpy", "binidx", "dummy", "uint16", "sft", 'jsonl']

    if args.lr_final == 0 or args.lr_init == 0:
        rank_zero_info("\n\nNote: lr_final = 0 or lr_init = 0. Using linear LR schedule instead.\n\n")

    assert args.precision in ["fp32", "tf32", "fp16", "bf16"]
    os.environ["RWKV_FLOAT_MODE"] = args.precision
    if args.precision == "fp32":
        for i in range(10):
            rank_zero_info("\n\nNote: you are using fp32 (very slow). Try bf16 / tf32 for faster training.\n\n")
    if args.precision == "fp16":
        rank_zero_info("\n\nNote: you are using fp16 (might overflow). Try bf16 / tf32 for stable training.\n\n")

    os.environ["RWKV_JIT_ON"] = "0"
    if "deepspeed_stage_3" in args.strategy:
        os.environ["RWKV_JIT_ON"] = "0"

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    if args.precision == "fp32":
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    else:
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True

    if "32" in args.precision:
        args.precision = 32
    elif args.precision == "fp16":
        args.precision = 16
    else:
        args.precision = "bf16"

    ########################################################################################################

    from rwkvt.lightning_train.trainer import train_callback
    from rwkvt.peft_loading import load_peft_model
    #from rwkvt.dataset.dataset import MyDataModule
    args, model = load_peft_model(args)
    # ================= RLVR MODULE =================
    # ================= RLVR switch =================

    import lightning as pl
    import torch
    import torch.nn.functional as F

    class RLVRModule(L.LightningModule):
        def __init__(self, model, tokenizer, args):
            super().__init__()
            self.model = model
            self.tokenizer = tokenizer
            self.args = args

        def generate_with_logprob(self, prompt):
        # 1. 针对 RWKV7 的初始化状态
            state = self.model.model.generate_init_state() if hasattr(self.model.model, 'generate_init_state') else None
            
            # 2. 编码 Prompt 并转换为 Tensor
            input_ids = self.tokenizer.encode(prompt)
            # 关键修改：转换为 Tensor 并增加 Batch 维度，移动到模型所在的设备
            input_tensor = torch.tensor([input_ids], device=self.device) 

            # 3. 预填充 Prompt (Prefilling)
            # 注意：RWKV7 的 forward 正常会返回 logits 和 state
            out, state = self.model.model.forward(input_tensor, state)
            
            tokens = []
            logprobs = []
            
            # 取最后一个 token 的输出作为生成起点
            # input_ids[-1:] 得到的是一个只有一个元素的 list，同样需要转 Tensor
            next_token_tensor = torch.tensor([[input_ids[-1]]], device=self.device)

            for _ in range(self.args.my_sample_len or 256):
                # 4. 前向传播获取下一个 token 的 logits
                logits, state = self.model.model.forward(next_token_tensor, state)
                
                # logits 形状通常是 [B, T, V]，因为我们输入是单 token，取 [0, -1, :]
                probs = F.softmax(logits[0, -1, :], dim=-1)
                
                # 采样
                token = torch.multinomial(probs, 1).item()
                logprob = torch.log(probs[token] + 1e-8)
                
                tokens.append(token)
                logprobs.append(logprob)
                
                # 更新下一个输入
                next_token_tensor = torch.tensor([[token]], device=self.device)
                
                if token == 0: # 假设 0 是终止符
                    break

            return tokens, torch.stack(logprobs)

        def training_step(self, batch, batch_idx):
            from eval import build_prompt, verify_gsm8k_answer # 确保导入
            question, answer = batch["question"], batch["answer"]
            prompt = build_prompt(question)
            
            tokens, logprobs = self.generate_with_logprob(prompt)
            response = self.tokenizer.decode(tokens)
            
            # 计算奖励
            result = verify_gsm8k_answer(response, answer)
            reward = torch.tensor(float(result["is_correct"]), device=self.device)
            
            # RL Loss (Policy Gradient 简化版)
            loss = - reward * logprobs.sum()
            
            self.log("reward", reward, prog_bar=True, logger=True)
            self.log("loss", loss, prog_bar=True)
            return loss

        def configure_optimizers(self):
            # 1. 识别并分组参数
            param_dict = {}
            for n, p in self.model.named_parameters():
                if p.requires_grad:
                    # 尝试获取参数自带的 scale，如果没有则默认为 1.0
                    scale = getattr(p, 'my_lr_scale', 1.0)
                    if scale not in param_dict:
                        param_dict[scale] = []
                    param_dict[scale].append(p)
            
            # 2. 构造符合 RWKV 训练器要求的参数组
            optim_groups = []
            for scale, params in param_dict.items():
                optim_groups.append({
                    "params": params, 
                    "my_lr_scale": scale  # 必须显式包含这个 Key
                })

            # 如果没有任何可训练参数（例如 State Tuning 没搜到参数），添加一个空组防止报错
            if not optim_groups:
                optim_groups = [{"params": self.model.parameters(), "my_lr_scale": 1.0}]

            return torch.optim.AdamW(
                optim_groups,
                lr=self.args.lr_init,
                betas=self.args.betas,
                eps=self.args.adam_eps,
                weight_decay=self.args.weight_decay
            )
    if args.train_type == "rlvr":
                    print("🚀 Using RLVR training mode")

                    tokenizer = TRIE_TOKENIZER(
                        "C:\\RWKV-LM\\rwkv_vocab_v20230424.txt"
                    )

                    model = RLVRModule(model, tokenizer, args)

    # 强制使用 GPU
    import torch
    if torch.cuda.is_available():
        torch.cuda.set_device(0)  # 使用第一个 GPU
        print(f"✓ 强制使用 GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("❌ 错误：未检测到 CUDA GPU")
        exit(1)
    #trainer = Trainer(accelerator=args.accelerator,strategy=args.strategy,devices=args.devices,num_nodes=args.num_nodes,precision=args.precision,
    #logger=args.logger,callbacks=[train_callback(args)],max_epochs=args.max_epochs,check_val_every_n_epoch=args.check_val_every_n_epoch,num_sanity_val_steps=args.num_sanity_val_steps,
    #log_every_n_steps=args.log_every_n_steps,enable_checkpointing=args.enable_checkpointing,accumulate_grad_batches=args.accumulate_grad_batches,gradient_clip_val=args.gradient_clip_val)
    # 修复 Windows GPU 设备问题
    import torch
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        print(f"✓ 强制使用 GPU: {torch.cuda.get_device_name(0)}")

    # 修改 Trainer 配置
    trainer = Trainer(
        accelerator="gpu",
        devices=[0],  # 明确指定 GPU 0（列表形式）
        strategy="auto",  # 使用 auto 而不是 single_device
        num_nodes=args.num_nodes,
        precision="bf16",  # 使用推荐的 bf16-mixed
        logger=args.logger,
        callbacks=[train_callback(args)],
        max_epochs=args.epoch_count,
        check_val_every_n_epoch=args.epoch_count,
        num_sanity_val_steps=0,
        log_every_n_steps=1,
        enable_checkpointing=False,
        enable_progress_bar=True,
        enable_model_summary=False,
        gradient_clip_val=1.0,
        accumulate_grad_batches=args.accumulate_grad_batches,
    )


 
    train_data = MyDataModule(
    args, processor=None
  )
    train_data.setup()
    trainer.fit(model, train_data)
