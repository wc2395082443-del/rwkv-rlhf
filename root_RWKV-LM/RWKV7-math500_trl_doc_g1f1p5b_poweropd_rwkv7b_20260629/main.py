#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import torch
import sys
from pathlib import Path
from typing import Optional

# 确保能导入当前目录的模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import read_jsonl, set_seed, now_str
from infer import AlbatrossBatchInference
from stateful_rollout_albatross import AlbatrossTrainRollout
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
    """??torch???????state_dict?????checkpoint"""
    try:
        sd = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        sd = torch.load(path, map_location="cpu")

    # Training checkpoints may be wrapped as {"step": ..., "model": state_dict}.
    if isinstance(sd, dict) and "emb.weight" not in sd and isinstance(sd.get("model"), dict):
        model_sd = sd["model"]
        if "emb.weight" in model_sd:
            return model_sd

    return sd


def load_train_model_rwkv7_cuda(pth_path: str, device: str, ctx_len: int, train_type: str = "state", load_dtype: str = "bf16"):
    """加载训练模型"""
    from types import SimpleNamespace
    from rwkv7_trainable import RWKV7
    
    print(f"[load_train_model] load weights: {pth_path}", flush=True)
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
        grad_cp=1,
        train_type=train_type,
        peft="none",
        my_testing="x070",
    )
    
    print(f"[load_train_model] build model train_type={train_type} layers={n_layer} emb={n_embd}", flush=True)
    model = RWKV7(args)
    print("[load_train_model] load_state_dict", flush=True)
    model.load_state_dict(sd, strict=False)
    model.args = args
    dtype_map = {"bf16": torch.bfloat16, "fp32": torch.float32}
    target_dtype = dtype_map.get(str(load_dtype).lower(), torch.bfloat16)
    print(f"[load_train_model] move to {device} {target_dtype}", flush=True)
    model = model.to(device=device, dtype=target_dtype)
    print("[load_train_model] done", flush=True)
    return model, args


def load_hf_teacher(model_path: str, device: str, dtype: str = "bf16"):
    import json
    from transformers import AutoTokenizer, AutoModelForCausalLM
    mp = Path(model_path)
    index_path = mp / 'model.safetensors.index.json'
    if index_path.is_file():
        obj = json.loads(index_path.read_text(encoding='utf-8'))
        missing = []
        for fname in sorted(set(obj.get('weight_map', {}).values())):
            if not (mp / fname).is_file():
                missing.append(fname)
        if missing:
            raise FileNotFoundError(f'Teacher model is incomplete under {model_path}. Missing shards: {missing}')
    torch_dtype = torch.bfloat16 if str(dtype).lower() == 'bf16' else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch_dtype, device_map=None)
    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, tokenizer


def load_rwkv_teacher(pth_path: str, device: str, ctx_len: int, dtype: str = "bf16"):
    model, _ = load_train_model_rwkv7_cuda(
        pth_path,
        device=device,
        ctx_len=ctx_len,
        train_type="state",
        load_dtype=dtype,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


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


def unfreeze_all_parameters(model: torch.nn.Module) -> int:
    """解冻全部参数"""
    cnt = 0
    for _, p in model.named_parameters():
        p.requires_grad = True
        cnt += p.numel()
    return cnt


def cast_trainable_time_state_fp32(model: torch.nn.Module) -> int:
    cnt = 0
    for n, p in model.named_parameters():
        if p.requires_grad and 'time_state' in n:
            if p.dtype != torch.float32:
                p.data = p.data.to(dtype=torch.float32)
            cnt += p.numel()
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
    ap.add_argument("--tune_mode", type=str, default="full", choices=["state", "full"], help="训练模式: state=仅time_state, full=全参微调")
    ap.add_argument("--ctx_len", type=int, default=8192, help="上下文长度")
    ap.add_argument("--model_dtype", type=str, default="bf16", choices=["bf16", "fp32"], help="??/????????")
    ap.add_argument("--reward_mode", type=str, default="rwkv", choices=["rwkv", "trl_doc"], help="????")
    ap.add_argument("--prompt_mode", type=str, default="rwkv_boxed", choices=["rwkv_boxed", "trl_doc", "question_only"], help="?????")
    ap.add_argument("--save_responses", type=int, default=1, help="???????????")
    
    # 采样配置
    ap.add_argument("--num_questions", type=int, default=24, help="每步采样的题目数")
    ap.add_argument("--samples_per_question", type=int, default=8, help="每道题的采样次数")
    ap.add_argument("--hard_buffer_ttl", type=int, default=10, help="hard buffer TTL (steps)")
    ap.add_argument("--hard_buffer_cooldown", type=int, default=5, help="hard buffer cooldown (steps)")
    ap.add_argument("--hard_buffer_target_samples", type=int, default=192, help="extra-stage target sample count")
    ap.add_argument("--hard_buffer_group_size", type=int, default=8, help="extra-stage samples per question")
    ap.add_argument("--hard_buffer_extra_lr_scale", type=float, default=0.5, help="lr scale for hard-buffer extra step")
    ap.add_argument("--hard_buffer_adv_clip", type=float, default=2.5, help="adv clip for hard-buffer extra step")
    
    # 生成配置
    ap.add_argument("--max_new_tokens", type=int, default=1024, help="最大生成token数")
    ap.add_argument("--temperature", type=float, default=1.0, help="温度")
    ap.add_argument("--top_p", type=float, default=0.6, help="top-p参数")
    ap.add_argument("--top_k", type=int, default=0, help="top-k参数")
    ap.add_argument("--eval_temperature", type=float, default=0.3, help="评估温度")
    ap.add_argument("--eval_top_p", type=float, default=0.4, help="评估top-p参数")
    ap.add_argument("--eval_top_k", type=int, default=500, help="评估top-k参数")
    
    # 奖励配置
    ap.add_argument("--min_tokens", type=int, default=200, help="最小token数")
    ap.add_argument("--length_weight", type=float, default=0.0, help="长度奖励权重")
    
    # --- [新增] 高级奖励配置 ---
    ap.add_argument("--zstd_threshold", type=float, default=2.5, help="Zstd压缩比阈值")
    ap.add_argument("--zstd_penalty_weight", type=float, default=0.0, help="Zstd惩罚权重")
    ap.add_argument("--ngram_penalty", type=float, default=0.0, help="N-gram重复惩罚")
    
    # 训练配置
    ap.add_argument("--total_steps", type=int, default=200, help="总训练步数")
    ap.add_argument("--ppo_epochs", type=int, default=1, help="PPO epoch数")
    ap.add_argument("--micro_batch", type=int, default=4, help="micro batch大小")
    ap.add_argument("--rollout_forward_batch", type=int, default=8, help="full模式下rollout/eval前向micro batch")
    ap.add_argument("--lr", type=float, default=6e-5, help="学习率")
    ap.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪")
    ap.add_argument('--kl_coef', type=float, default=0.05, help='KL??(?kl_mode??)')
    ap.add_argument('--kl_mode', type=str, default='k1_reward', choices=['k1_reward','k3_loss'], help='KL??')
    
    # --- [新增] 负样本降权 ---
    ap.add_argument("--neg_adv_weight", type=float, default=0.0, help="负Advantage权重 (0.0-1.0)")
    
    # 正则化
    ap.add_argument("--time_state_l2", type=float, default=0, help="time_state L2正则化")
    ap.add_argument("--time_state_clamp", type=float, default=10.0, help="time_state裁剪")
    
    # 日志和保存
    ap.add_argument("--out_dir", type=str, default="./output", help="输出目录")
    ap.add_argument("--log_interval", type=int, default=1, help="日志间隔")
    ap.add_argument("--save_interval", type=int, default=50, help="保存间隔")
    ap.add_argument("--eval_interval", type=int, default=5, help="评估间隔")
    ap.add_argument("--eval_sample_ratio", type=float, default=1.0, help="eval sample ratio during training (1.0=full)")
    ap.add_argument("--preeval_sample_ratio", type=float, default=1.0, help="pre_eval sample ratio (1.0=full)")
    ap.add_argument("--posteval_sample_ratio", type=float, default=1.0, help="post_eval sample ratio (1.0=full)")
    ap.add_argument("--skip_preeval", type=int, default=0, help="skip pre_eval (1=yes)")
    ap.add_argument("--skip_posteval", type=int, default=0, help="skip post_eval (1=yes)")
    ap.add_argument("--save_last", type=int, default=0, help="force save ckpt at last step (1=yes)")
    ap.add_argument("--final_full_eval", type=int, default=0, help="force full_eval at last step (1=yes)")
    
    # 其他
    ap.add_argument("--seed", type=int, default=42, help="随机种子")
    ap.add_argument("--teacher_model", type=str, default="", help="Teacher model path for OPD (HF dir or RWKV .pth)")
    ap.add_argument("--teacher_kind", type=str, default="auto", choices=["auto", "hf", "rwkv"], help="Teacher backend for OPD")
    ap.add_argument("--opd_weight", type=float, default=0.0, help="teacher distillation loss weight")
    ap.add_argument("--opd_max_new_tokens", type=int, default=512, help="teacher generation max_new_tokens")
    ap.add_argument("--opd_temperature", type=float, default=0.7, help="teacher generation temperature")
    ap.add_argument("--opd_top_p", type=float, default=0.9, help="teacher generation top_p")
    ap.add_argument("--opd_micro_batch", type=int, default=2, help="student OPD micro batch")
    ap.add_argument("--opd_mode", type=str, default="tokenkl", choices=["tokenkl", "power"], help="OPD objective variant")
    ap.add_argument("--poweropd_alpha", type=float, default=0.3, help="PowerOPD alpha; 0 recovers log-ratio reward")
    ap.add_argument("--logit_chunk_tokens", type=int, default=128, help="chunk size for teacher logits projection")
    
    return ap.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    
    # 设置环境变量
    os.environ["RWKV_HEAD_SIZE_A"] = str(HEAD_SIZE)
    os.environ["RWKV_MY_TESTING"] = "x070"
    rwkv_train_type = "fullstate" if args.tune_mode == "full" else "state"
    os.environ["RWKV_TRAIN_TYPE"] = rwkv_train_type
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
    train_model, _ = load_train_model_rwkv7_cuda(
        pth_path,
        device=device,
        ctx_len=int(args.ctx_len),
        train_type=rwkv_train_type,
        load_dtype=str(args.model_dtype),
    )
    
    # 加载初始time_state
    if args.state_init:
        ok = load_time_state_only(train_model, args.state_init)
        print(f"加载初始time_state: {ok} (from {args.state_init})")

    print("加载参考模型...")
    ref_model, _ = load_train_model_rwkv7_cuda(
        pth_path,
        device=device,
        ctx_len=int(args.ctx_len),
        train_type=rwkv_train_type,
        load_dtype=str(args.model_dtype),
    )
    if args.state_init:
        ok = load_time_state_only(ref_model, args.state_init)
        print(f"加载参考time_state: {ok} (from {args.state_init})")
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    if args.tune_mode == "full":
        if str(args.model_dtype).lower() == "fp32":
            print("full + fp32: disable stateful rollout, use train_model full rollout")
            infer_model = None
        else:
            print("full + bf16: use stateful rollout cache")
            infer_model = AlbatrossTrainRollout(train_model=train_model, device=device)
    else:
        print("state tune: use reference rollout model")
        infer_model, _ = load_infer_model_albatross(base_name)
    
    if args.tune_mode == "full":
        trainable = unfreeze_all_parameters(train_model)
        fp32_trainable = 0
        print(f"训练模式: full | 可训练参数: {trainable}")
    else:
        trainable = freeze_except_time_state(train_model)
        if trainable <= 0:
            raise RuntimeError("没有可训练的time_state参数")
        fp32_trainable = cast_trainable_time_state_fp32(train_model)
        print(f"训练模式: state | 可训练参数: {trainable} (fp32={fp32_trainable})")

    # 创建配置
    # --- [修改] 传入新增的参数 ---
    teacher_model = None
    teacher_tokenizer = None
    teacher_kind = "hf"
    teacher_path = str(getattr(args, "teacher_model", "")).strip()
    if teacher_path and float(args.opd_weight) > 0:
        teacher_kind = str(getattr(args, "teacher_kind", "auto")).lower()
        if teacher_kind == "auto":
            teacher_kind = "rwkv" if teacher_path.endswith(".pth") else "hf"
        print(f"load OPD teacher ({teacher_kind}): {teacher_path}")
        if teacher_kind == "rwkv":
            teacher_model = load_rwkv_teacher(teacher_path, device=device, ctx_len=int(args.ctx_len), dtype=str(args.model_dtype))
            teacher_tokenizer = "__rwkv_shared__"
        else:
            teacher_model, teacher_tokenizer = load_hf_teacher(teacher_path, device=device, dtype=str(args.model_dtype))

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
        rollout_forward_batch=int(args.rollout_forward_batch),
        lr=float(args.lr),
        grad_clip=float(args.grad_clip),
        min_tokens=int(args.min_tokens),
        max_tokens=int(args.max_new_tokens),
        length_weight=float(args.length_weight),
        
        # 新增参数
        zstd_threshold=float(args.zstd_threshold),
        zstd_penalty_weight=float(args.zstd_penalty_weight),
        ngram_penalty=float(args.ngram_penalty),
        neg_adv_weight=float(args.neg_adv_weight),
        
        kl_coef=float(args.kl_coef),
        kl_mode=str(args.kl_mode),
        time_state_l2=float(args.time_state_l2),
        time_state_clamp=float(args.time_state_clamp),
        log_interval=int(args.log_interval),
        save_interval=int(args.save_interval),
        eval_interval=int(args.eval_interval),
        eval_sample_ratio=float(args.eval_sample_ratio),
        save_last=bool(int(args.save_last)),
        final_full_eval=bool(int(args.final_full_eval)),
        hard_buffer_ttl=int(args.hard_buffer_ttl),
        hard_buffer_cooldown=int(args.hard_buffer_cooldown),
        hard_buffer_target_samples=int(args.hard_buffer_target_samples),
        hard_buffer_group_size=int(args.hard_buffer_group_size),
        hard_buffer_extra_lr_scale=float(args.hard_buffer_extra_lr_scale),
        hard_buffer_adv_clip=float(args.hard_buffer_adv_clip),
        tune_mode=str(args.tune_mode),
        reward_mode=str(args.reward_mode),
        prompt_mode=str(args.prompt_mode),
        save_responses=bool(int(args.save_responses)),
        opd_enabled=bool(str(args.teacher_model).strip() and float(args.opd_weight) > 0),
        opd_weight=float(args.opd_weight),
        opd_max_new_tokens=int(args.opd_max_new_tokens),
        opd_temperature=float(args.opd_temperature),
        opd_top_p=float(args.opd_top_p),
        opd_micro_batch=int(args.opd_micro_batch),
        opd_mode=str(args.opd_mode),
        poweropd_alpha=float(args.poweropd_alpha),
        logit_chunk_tokens=int(args.logit_chunk_tokens),
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
        teacher_model=teacher_model,
        teacher_tokenizer=teacher_tokenizer,
        teacher_kind=teacher_kind,
    )
    
    # 开始训练
    print("\n" + "="*50)
    print(f"开始GRPO训练 | 总步数: {args.total_steps}")
    print(f"策略: NegAdvWeight={args.neg_adv_weight}, Zstd={args.zstd_threshold}")
    print("="*50 + "\n")
    
    try:
        if int(args.skip_preeval) != 1:
            print(f"start pre_eval | sample_ratio={float(args.preeval_sample_ratio):.3f}")
            pre_acc = trainer.evaluate(step=0, tag="pre_eval", sample_ratio=float(args.preeval_sample_ratio))
            if pre_acc is not None:
                print(f"[pre_eval] acc={pre_acc:.4f}")
        else:
            print("skip pre_eval")
        trainer.train(total_steps=int(args.total_steps))
        if int(args.skip_posteval) != 1:
            print(f"start post_eval | sample_ratio={float(args.posteval_sample_ratio):.3f}")
            post_acc = trainer.evaluate(step=int(args.total_steps), tag="post_eval", sample_ratio=float(args.posteval_sample_ratio))
            if post_acc is not None:
                print(f"[post_eval] acc={post_acc:.4f}")
        else:
            print("skip post_eval")
    except KeyboardInterrupt:
        print("\n训练被用户中断")
    except Exception as e:
        print(f"\n训练出错: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n训练结束!")


if __name__ == "__main__":
    main()
