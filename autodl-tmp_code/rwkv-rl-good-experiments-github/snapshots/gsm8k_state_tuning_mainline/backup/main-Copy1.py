#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import torch
import copy

from utils import read_jsonl, set_seed, ProgressTracker, now_str
from reward import calculate_reward
from infer import AlbatrossBatchInference
from train import GRPOTrainer, GRPOConfig


# 模型相关配置
HEAD_SIZE = 64


def normalize_model_arg(model_arg: str):
    """规范化模型参数"""
    model_arg = model_arg.strip()
    if model_arg.endswith(".pth"):
        base = model_arg[:-4]
        pth = model_arg
    else:
        base = model_arg
        pth = model_arg + ".pth"
    if not os.path.isfile(pth) and os.path.isfile(base):
        pth = base
    return base, pth


def _torch_load_weights(path: str):
    """加载torch权重"""
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_train_model_rwkv7_cuda(pth_path: str, device: str, ctx_len: int):
    """加载训练模型"""
    from types import SimpleNamespace
    from rwkv7_trainable import RWKV7
    
    sd = _torch_load_weights(pth_path)
    
    n_embd = sd["emb.weight"].shape[1]
    vocab_size = sd["emb.weight"].shape[0]
    n_layer = max(int(k.split(".")[1]) for k in sd if k.startswith("blocks.")) + 1
    dim_ffn = sd.get("blocks.0.ffn.key.weight", torch.zeros(n_embd * 4, n_embd)).shape[0]
    
    args = SimpleNamespace(
        n_embd=n_embd,
        vocab_size=vocab_size,
        n_layer=n_layer,
        dim_att=n_embd,
        dim_ffn=dim_ffn,
        head_size_a=HEAD_SIZE,
        head_size_divisor=8,
        ctx_len=ctx_len,
        chunk_ctx=ctx_len,
        grad_cp=0,
        train_type="state",
        peft="none",
        my_testing="x070",
    )
    
    model = RWKV7(args)
    model.load_state_dict(sd, strict=False)
    model.args = args
    model = model.to(device).to(torch.bfloat16)
    return model, args


def load_infer_model_albatross(base_name_no_pth: str):
    """加载推理模型"""
    import types
    from reference.rwkv7 import RWKV_x070
    
    args = types.SimpleNamespace()
    args.vocab_size = 65536
    args.MODEL_NAME = base_name_no_pth
    model = RWKV_x070(args)
    return model, args


def freeze_except_time_state(model: torch.nn.Module) -> int:
    """冻结除time_state外的所有参数"""
    cnt = 0
    for n, p in model.named_parameters():
        if "time_state" in n:
            p.requires_grad = True
            cnt += p.numel()
        else:
            p.requires_grad = False
    return cnt


def load_time_state_only(model: torch.nn.Module, path: str) -> bool:
    """仅加载time_state参数"""
    if not path or not os.path.exists(path):
        return False
    sd = _torch_load_weights(path)
    if 'time_state' in sd and isinstance(sd['time_state'], dict):
        sd = sd['time_state']
    hit = 0
    for n, p in model.named_parameters():
        if n in sd:
            p.data.copy_(sd[n].to(p.device).to(p.dtype))
            hit += 1
    return hit > 0


def parse_args():
    """解析命令行参数"""
    ap = argparse.ArgumentParser()
    
    # 数据集
    ap.add_argument("--train_jsonl", type=str, required=True, help="训练数据路径")
    ap.add_argument("--eval_jsonl", type=str, default=None, help="评估数据路径")
    ap.add_argument("--max_data_samples", type=int, default=None, help="最大数据样本数")
    
    # 模型
    ap.add_argument("--model", type=str, required=True, help="模型路径")
    ap.add_argument("--tokenizer", type=str, required=True, help="分词器路径")
    ap.add_argument("--state_init", type=str, default=None, help="初始time_state路径")
    ap.add_argument("--ctx_len", type=int, default=8192, help="上下文长度")
    
    # 采样配置
    ap.add_argument("--num_questions", type=int, default=24, help="每步采样的题目数")
    ap.add_argument("--samples_per_question", type=int, default=8, help="每道题的采样次数")
    
    # 生成配置
    ap.add_argument("--max_new_tokens", type=int, default=768, help="最大生成token数")
    ap.add_argument("--temperature", type=float, default=1.0, help="温度")
    ap.add_argument("--top_p", type=float, default=0.6, help="top-p参数")
    ap.add_argument("--top_k", type=int, default=0, help="top-k参数")
    ap.add_argument("--eval_temperature", type=float, default=0.3, help="评估温度")
    ap.add_argument("--eval_top_p", type=float, default=0.4, help="评估top-p参数")
    ap.add_argument("--eval_top_k", type=int, default=0, help="评估top-k参数")
    
    # 奖励配置
    ap.add_argument("--min_tokens", type=int, default=200, help="最小token数")
    ap.add_argument("--length_weight", type=float, default=0.5, help="长度奖励权重")
    
    # 训练配置
    ap.add_argument("--total_steps", type=int, default=200, help="总训练步数")
    ap.add_argument("--ppo_epochs", type=int, default=1, help="PPO epoch数")
    ap.add_argument("--micro_batch", type=int, default=4, help="micro batch大小")
    ap.add_argument("--lr", type=float, default=1e-5, help="学习率")
    ap.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪")
    ap.add_argument("--kl_coef", type=float, default=0.001, help="KL系数")
    
    # 正则化
    ap.add_argument("--time_state_l2", type=float, default=1e-7, help="time_state L2正则化")
    ap.add_argument("--time_state_clamp", type=float, default=10.0, help="time_state裁剪")
    
    # 日志和保存
    ap.add_argument("--out_dir", type=str, default="./output", help="输出目录")
    ap.add_argument("--log_interval", type=int, default=1, help="日志间隔")
    ap.add_argument("--save_interval", type=int, default=50, help="保存间隔")
    ap.add_argument("--eval_interval", type=int, default=5, help="评估间隔")
    ap.add_argument("--eval_sample_ratio", type=float, default=1.0, help="评估采样比例 (1.0=全量)")
    
    # 其他
    ap.add_argument("--seed", type=int, default=42, help="随机种子")
    
    return ap.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 设置环境变量
    os.environ["RWKV_HEAD_SIZE_A"] = str(HEAD_SIZE)
    os.environ["RWKV_MY_TESTING"] = "x070"
    os.environ["RWKV_TRAIN_TYPE"] = "state"
    os.environ["RWKV_CTXLEN"] = str(int(args.ctx_len))
    os.environ["FUSED_KERNEL"] = "0"
    os.environ["WKV"] = "cuda"
    
    # 创建输出目录
    os.makedirs(args.out_dir, exist_ok=True)
    
    # 加载数据
    print(f"加载训练数据: {args.train_jsonl}")
    train_data = read_jsonl(args.train_jsonl, max_samples=args.max_data_samples)
    if not train_data:
        raise RuntimeError("训练数据为空")
    print(f"训练数据: {len(train_data)} 样本")
    
    if args.eval_jsonl:
        print(f"加载评估数据: {args.eval_jsonl}")
        test_data = read_jsonl(args.eval_jsonl)
        print(f"评估数据: {len(test_data)} 样本")
    else:
        # 使用训练数据的一部分作为测试数据
        test_size = min(128, len(train_data) // 5)
        test_data = train_data[:test_size]
        train_data = train_data[test_size:]
        print(f"划分数据: 训练={len(train_data)}, 测试={len(test_data)}")
    
    # 加载分词器
    from reference.utils import TRIE_TOKENIZER
    tok = TRIE_TOKENIZER(args.tokenizer)
    encode = lambda s: tok.encode(s)
    
    def safe_decode(ids):
        try:
            return tok.decode(ids, utf8_errors="replace")
        except:
            try:
                return tok.decode(ids)
            except:
                try:
                    b = tok.decodeBytes(ids)
                    return b.decode("utf-8", errors="replace")
                except:
                    return "".join(chr(int(x) % 256) for x in ids)
    
    decode = safe_decode
    
    # 加载模型
    print(f"加载模型: {args.model}")
    base_name, pth_path = normalize_model_arg(args.model)
    if not os.path.isfile(pth_path):
        raise FileNotFoundError(f"找不到模型文件: {pth_path}")
    
    print("加载训练模型...")
    train_model, _ = load_train_model_rwkv7_cuda(pth_path, device=device, ctx_len=int(args.ctx_len))
    
    # 加载初始time_state
    if args.state_init:
        ok = load_time_state_only(train_model, args.state_init)
        print(f"加载初始time_state: {ok} (from {args.state_init})")

    # reference model: 初始权重冻结副本
    ref_model = copy.deepcopy(train_model)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    print("加载推理模型...")
    infer_model, _ = load_infer_model_albatross(base_name)
    
    # 冻结参数
    trainable = freeze_except_time_state(train_model)
    if trainable <= 0:
        raise RuntimeError("没有可训练的time_state参数")
    print(f"可训练参数: {trainable}")

    # 创建reference model (冻结副本)
    # ref_model 已在上面创建
    
    # 创建配置
    cfg = GRPOConfig(
        num_questions=int(args.num_questions),
        samples_per_question=int(args.samples_per_question),
        max_new_tokens=int(args.max_new_tokens),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        top_k=int(args.top_k),
        eval_temperature=float(args.eval_temperature),
        eval_top_p=float(args.eval_top_p),
        eval_top_k=int(args.eval_top_k),
        ppo_epochs=int(args.ppo_epochs),
        micro_batch=int(args.micro_batch),
        lr=float(args.lr),
        grad_clip=float(args.grad_clip),
        min_tokens=int(args.min_tokens),
        max_tokens=int(args.max_new_tokens),
        length_weight=float(args.length_weight),
        kl_coef=float(args.kl_coef),
        time_state_l2=float(args.time_state_l2),
        time_state_clamp=float(args.time_state_clamp),
        log_interval=int(args.log_interval),
        save_interval=int(args.save_interval),
        eval_interval=int(args.eval_interval),
        eval_sample_ratio=float(args.eval_sample_ratio),
    )
    
    # 创建推理引擎
    infer_engine = AlbatrossBatchInference(
        infer_model=infer_model,
        train_model=train_model,
        encode_fn=encode,
        decode_fn=decode,
        device=device,
        cfg=cfg,
    )
    
    # 创建训练器
    trainer = GRPOTrainer(
        train_model=train_model,
        ref_model=ref_model,
        infer_engine=infer_engine,
        encode_fn=encode,
        decode_fn=decode,
        train_data=train_data,
        test_data=test_data,
        out_dir=args.out_dir,
        device=device,
        cfg=cfg,
        seed=int(args.seed),
    )
    
    # 开始训练
    print("\n" + "="*50)
    print(f"开始GRPO训练 | 总步数: {args.total_steps}")
    print("="*50 + "\n")
    
    try:
        pre_acc = trainer.evaluate(step=0, tag="pre_eval", sample_ratio=1.0)
        if pre_acc is not None:
            print(f"[pre_eval] acc={pre_acc:.4f}")
        trainer.train(total_steps=int(args.total_steps))
        post_acc = trainer.evaluate(step=int(args.total_steps), tag="post_eval")
        if post_acc is not None:
            print(f"[post_eval] acc={post_acc:.4f}")
    except KeyboardInterrupt:
        print("\n训练被用户中断")
    except Exception as e:
        print(f"\n训练出错: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n训练结束!")


if __name__ == "__main__":
    main()