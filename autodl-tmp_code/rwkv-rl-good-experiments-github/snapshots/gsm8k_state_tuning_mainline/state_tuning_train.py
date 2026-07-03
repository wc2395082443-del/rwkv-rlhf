#!/usr/bin/env python3
"""
RWKV7 State-Tuning with Albatross Inference

- 训练：RWKV7 (CUDA kernel, full-seq)
- 推理：RWKV_x070 (albatross)
"""
import sys
import os
os.environ["RWKV_HEAD_SIZE_A"] = "64"
os.environ["RWKV_MY_TESTING"] = "x070"
os.environ["RWKV_TRAIN_TYPE"] = "state"
os.environ["RWKV_CTXLEN"] = "4096"
sys.path.append("C:\\RWKV-LM\\Albatross\\")
sys.path.append("/mnt/program/_RWKV_/_ref_/RWKV-CUDA/rwkv7_fast_fused")
import types
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import time
import argparse
from types import SimpleNamespace
from typing import List, Tuple, Optional
import re
from tqdm import tqdm
# 训练模型
from rwkv7_trainable import RWKV7

# 推理模型 (albatross)
from reference.rwkv7 import RWKV_x070
from reference.utils import TRIE_TOKENIZER, sampler_simple_batch

HEAD_SIZE = 64


########################################################################################################
# Tokenizer
########################################################################################################

def get_tokenizer(tokenizer_path=None):
    """使用albatross的TRIE_TOKENIZER"""
    if tokenizer_path is None:
        tokenizer_path = "reference/rwkv_vocab_v20230424.txt"
    tokenizer = TRIE_TOKENIZER(tokenizer_path)
    
    encode = lambda s: tokenizer.encode(s)
    def safe_decode(ids):
        try:
            # 尝试正常解码
            return tokenizer.decode(ids)
        except UnicodeDecodeError:
            # 如果失败，说明末尾有残缺字节。
            # TRIE_TOKENIZER 的 decode 方法通常不接受 errors 参数（看你的报错栈似乎是硬编码的）
            # 所以我们只能手动处理：去掉最后一个 token 再试一次，或者直接忽略错误
            
            # 方法 A: 如果 tokenizer.decode 支持 errors 参数（看报错代码似乎支持，但没传进去）
            # 我们可以尝试 hack 一下，或者用更通用的方法：
            
            # 方法 B (通用): 将 token ids 转为 bytes，然后手动 decode 并忽略错误
            # 注意：这需要 tokenizer 提供 ids -> bytes 的能力。
            # 如果 TRIE_TOKENIZER 没有这个接口，我们用最笨的办法：
            
            # 方法 C (最稳妥): 递归删除最后一个 token 直到能解码（通常只需要删1个）
            temp_ids = ids[:]
            while len(temp_ids) > 0:
                try:
                    return tokenizer.decode(temp_ids)
                except UnicodeDecodeError:
                    temp_ids.pop() # 扔掉最后一个可能残缺的 token
            return "" # 实在解不出来就返回空
            
    decode = safe_decode
    return encode, decode, tokenizer
def extract_answer(text):
    """从模型输出中提取 \\boxed{...} 或最后的数字"""
    if not text: return ""
    # 1. 优先找 boxed
    matches = re.findall(r'\\boxed\{(.*?)\}', text)
    if matches:
        return matches[-1]
    # 2. 备选：找 "The answer is ..." 后面的内容 (视情况而定)
    return ""

def normalize_answer(text):
    """标准化答案（去空格，转小写）"""
    if not text: return ""
    return text.strip().replace(" ", "")

########################################################################################################
# 模型加载
########################################################################################################

def load_train_model(model_path, device='cuda', ctx_len=4096):
    """加载训练用的RWKV7模型（带time_state）"""
    print(f"Loading train model: {model_path}...")
    sd = torch.load(model_path + ".pth", map_location='cpu', weights_only=True)
    
    n_embd = sd['emb.weight'].shape[1]
    vocab_size = sd['emb.weight'].shape[0]
    n_layer = max(int(k.split('.')[1]) for k in sd if k.startswith('blocks.')) + 1
    dim_ffn = sd.get('blocks.0.ffn.key.weight', torch.zeros(n_embd*4, n_embd)).shape[0]
    
    args = SimpleNamespace(
        n_embd=n_embd, vocab_size=vocab_size, n_layer=n_layer,
        dim_att=n_embd, dim_ffn=dim_ffn, head_size_a=64, head_size_divisor=8,
        ctx_len=ctx_len, chunk_ctx=ctx_len, grad_cp=0, train_type='state', peft='none', my_testing='x070',
    )
    
    model = RWKV7(args)
    model.load_state_dict(sd, strict=False)
    print(f"Train model: {n_layer} layers, {n_embd} dim")
    return model.to(device).to(torch.bfloat16), args
def evaluate_math(train_model, infer_model, encode, decode, data_path, device):
    """MATH500 评测专用函数"""
    print("\n" + "=" * 60)
    print(f"开始 MATH500 评测 (Data: {data_path})")
    print("=" * 60)

    # 初始化高速推理引擎
    inference = AlbatrossInference(infer_model, train_model, device)
    
    # 1. 加载数据
    data = []
    if not os.path.exists(data_path):
        print(f"Error: 数据文件 {data_path} 不存在")
        # 尝试自动下载
        try:
            from datasets import load_dataset
            print("正在从 HuggingFace 下载 MATH-500...")
            ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
            data = [{"problem": x['problem'], "solution": x['solution']} for x in ds]
        except:
            return
    else:
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))

    print(f"共加载 {len(data)} 道题")
    
    correct = 0
    total = 0
    
    # 2. 循环评测
    # 你的模型训练格式是 User: ... Assistant: ...
    PROMPT_TEMPLATE = "User: {problem}\n\nAssistant:"
    
    pbar = tqdm(data)
    for item in pbar:
        problem = item['problem']
        ground_truth = extract_answer(item['solution']) # 从标准答案提取 boxed
        
        prompt = PROMPT_TEMPLATE.format(problem=problem)
        tokens = encode(prompt)
        
        # 使用 Albatross 高速生成
        # max_tokens 可以设大一点，数学题步骤多
        with torch.no_grad():
            # temperature=0 做题效果最好 (Greedy)
            gen_tokens, gen_text = inference.generate(
                tokens, max_tokens=512, temperature=0, decode_fn=decode
            )
            
        # 提取模型答案
        model_ans = extract_answer(gen_text)
        
        # 判分
        is_right = False
        if model_ans and ground_truth:
            if normalize_answer(model_ans) == normalize_answer(ground_truth):
                is_right = True
                
        if is_right:
            correct += 1
        total += 1
        
        pbar.set_description(f"Acc: {correct/total:.2%} ({correct}/{total})")

    print(f"\nFinal Accuracy: {correct/total:.2%}")

def load_inference_model(model_path, device='cuda'):
    """加载推理用的RWKV_x070模型（albatross）"""
    print(f"Loading inference model: {model_path}...")
    
    args = types.SimpleNamespace()
    args.vocab_size = 65536
    args.MODEL_NAME = model_path  # 不带.pth
    
    model = RWKV_x070(args)
    print(f"Inference model loaded")
    return model, args


########################################################################################################
# State管理
########################################################################################################

def freeze_except_state(model):
    """冻结除time_state外的所有参数"""
    cnt = 0
    for n, p in model.named_parameters():
        if 'time_state' in n:
            p.requires_grad = True
            cnt += p.numel()
        else:
            p.requires_grad = False
    print(f"Trainable: {cnt:,} params (time_state only)")
    return cnt


def save_state(model, path):
    """保存time_state参数"""
    sd = {n: p.data.cpu() for n, p in model.named_parameters() if 'time_state' in n}
    torch.save(sd, path)
    print(f"Saved: {path}")


def load_state(model, path):
    """加载time_state参数"""
    if not os.path.exists(path):
        print(f"State file not found: {path}")
        return False
    sd = torch.load(path, map_location='cpu', weights_only=True)
    for n, p in model.named_parameters():
        if n in sd:
            p.data.copy_(sd[n].to(p.device).to(p.dtype))
    print(f"Loaded: {path}")
    return True


########################################################################################################
# State-Cache推理 (使用albatross)
########################################################################################################

class AlbatrossInference:
    """使用albatross RWKV_x070进行推理"""
    
    def __init__(self, infer_model, train_model, device='cuda'):
        self.infer_model = infer_model
        self.train_model = train_model
        self.device = device
        
        # 从训练模型获取参数
        self.n_layer = len(train_model.blocks)
        self.n_embd = train_model.emb.weight.shape[1]
        self.n_head = self.n_embd // HEAD_SIZE
        
        print(f"AlbatrossInference: {self.n_layer} layers, {self.n_embd} dim, {self.n_head} heads")
    
    def init_state_with_time_state(self, B=1):
        """
        用训练模型的time_state初始化推理state
        
        albatross state结构:
        - state[0]: (L, 2, B, C) - shift states (att和ffn各一个)
        - state[1]: (L, B, H, 64, 64) - wkv states
        
        time_state: (H, 64, 64) per layer
        """
        # 先生成零初始化的state
        state = self.infer_model.generate_zero_state(B)
        
        # 用time_state覆盖wkv部分
        for i, block in enumerate(self.train_model.blocks):
            ts = block.att.time_state  # (H, 64, 64)
            # state[1]: (L, B, H, 64, 64)
            # 扩展到batch维度并复制
            state[1][i] = ts.unsqueeze(0).expand(B, -1, -1, -1).clone()
        
        return state
    
    def generate(self, prompt_tokens: List[int], max_tokens: int = 50, 
                 temperature: float = 0.0, top_p: float = 0.9,
                 stop_on_think_close: bool = True,
                 decode_fn=None) -> Tuple[List[int], str]:
        """
        使用albatross生成
        
        Args:
            prompt_tokens: prompt的token列表
            max_tokens: 最大生成长度
            temperature: 采样温度，0表示greedy
            top_p: nucleus sampling参数
            stop_on_think_close: 遇到</think>停止
            decode_fn: 解码函数，用于检测停止条件
        
        Returns:
            (generated_tokens, generated_text)
        """
        B = 1
        state = self.init_state_with_time_state(B)
        
        # Prime: 处理prompt
        tokens_batch = [prompt_tokens]  # list of list
        out = self.infer_model.forward_batch(tokens_batch, state)  # state就地修改
        
        generated = []
        for _ in range(max_tokens):
            # 采样
            if temperature <= 0:
                token = int(out[0].argmax().item())
            else:
                # 使用albatross的sampler
                s = sampler_simple_batch(out, noise=temperature)
                token = int(s.reshape(-1)[0].item())
            
            if token == 19797:  # EOS
                break
            
            generated.append(token)
            
            # 检查停止条件
            if stop_on_think_close and decode_fn is not None:
                text = decode_fn(generated)
                if "</think>" in text:
                    break
            
            # 单token forward
            out = self.infer_model.forward_batch([[token]], state)
        
        text = decode_fn(generated) if decode_fn else ""
        return generated, text
    
    def generate_with_logprobs(self, prompt_tokens: List[int], max_tokens: int = 150,
                               temperature: float = 1.0, top_p: float = 0.9,
                               stop_on_think_close: bool = True,
                               decode_fn=None) -> Tuple[List[int], List[float], str]:
        """
        生成并返回log probabilities（用于RL训练）
        """
        B = 1
        state = self.init_state_with_time_state(B)
        
        tokens_batch = [prompt_tokens]
        out = self.infer_model.forward_batch(tokens_batch, state)
        
        generated = []
        logprobs = []
        
        for _ in range(max_tokens):
            # 计算log prob
            logits = out[0]  # (vocab,)
            log_probs_all = F.log_softmax(logits.float(), dim=-1)
            
            # 采样
            if temperature <= 0:
                token = int(logits.argmax().item())
            else:
                scaled = logits.float() / temperature
                probs = F.softmax(scaled, dim=-1)
                
                # top-p
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cumsum = torch.cumsum(sorted_probs, dim=-1)
                mask = cumsum - sorted_probs > top_p
                mask[0] = False
                sorted_probs[mask] = 0
                sorted_probs = sorted_probs / sorted_probs.sum()
                
                idx = torch.multinomial(sorted_probs, 1)
                token = int(sorted_idx[idx].item())
            
            if token == 0:
                break
            
            logp = float(log_probs_all[token].item())
            
            generated.append(token)
            logprobs.append(logp)
            
            # 检查停止
            if stop_on_think_close and decode_fn is not None:
                text = decode_fn(generated)
                if "</think>" in text:
                    break
            
            out = self.infer_model.forward_batch([[token]], state)
        
        text = decode_fn(generated) if decode_fn else ""
        return generated, logprobs, text


########################################################################################################
# 训练
########################################################################################################

def train(train_model, encode, data_path, device, lr=1e-4, epochs=1, ctx_len=4096, log_every=10, plot_dir='loss_plots'):
    """训练time_state参数"""
    print("\n" + "=" * 60)
    print("训练 (使用CUDA kernel)")
    print("=" * 60)
    
    # 训练中作图：不在控制台显示，直接保存为文件（50步、500步、训练结束）
    def is_main_process():
        if not torch.distributed.is_available():
            return True
        if not torch.distributed.is_initialized():
            return True
        return torch.distributed.get_rank() == 0
    
    can_plot = False
    plt = None
    if is_main_process():
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as _plt
            plt = _plt
            can_plot = True
        except Exception as e:
            print(f"Warning: matplotlib unavailable, plotting disabled. ({e})")
            can_plot = False
    
    os.makedirs(plot_dir, exist_ok=True)
    loss_steps = []
    loss_values = []
    global_step = 0
    
    plot_targets = {
        50: "step050",
        500: "step500",
    }
    
    def save_loss_curve(tag):
        if (not can_plot) or (plt is None):
            return
        if not is_main_process():
            return
        if len(loss_steps) == 0:
            return
        
        out_path = os.path.join(plot_dir, f"loss_curve_{tag}.png")
        try:
            plt.figure()
            plt.plot(loss_steps, loss_values)
            plt.xlabel("Step")
            plt.ylabel("Loss")
            plt.title(f"Training Loss ({tag})")
            plt.tight_layout()
            plt.savefig(out_path, dpi=200)
            plt.close()
            print(f"Saved loss curve: {out_path}")
        except Exception as e:
            print(f"Warning: failed to save loss curve ({out_path}): {e}")
    
    freeze_except_state(train_model)
    opt = torch.optim.AdamW([p for p in train_model.parameters() if p.requires_grad], lr=lr)
    
    # 加载数据
    samples = []
    #with open(data_path) as f:
    #    for line in f:
    #        d = json.loads(line)
    #        text = d.get('text') or f"{d.get('instruction','')}\n{d.get('input','')}\n{d.get('output','')}"
    #        if text:
    #            toks = encode(text)[:ctx_len]
    #            if len(toks) > 16:
    #                samples.append(toks)
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                d = json.loads(line)
            except:
                continue # 跳过坏行

            text = ""
            
            # 1. 优先适配你的数学数据格式 (problem -> solution)
            if 'problem' in d and 'solution' in d:
                # 构造成 User/Assistant 对话格式，这样模型能学会“回答问题”
                # 如果你的模型是 Base 模型而非 Chat 模型，也可以用 "Question: ...\nAnswer: ..."
                text = f"User: {d['problem']}\n\nAssistant: {d['solution']}"
            
            # 2. 兼容 math500 测试集格式 (如果有 answer 字段)
            elif 'problem' in d and 'answer' in d:
                text = f"User: {d['problem']}\n\nAssistant: {d['answer']}"

            # 3. 兼容标准格式
            elif 'text' in d:
                text = d['text']
            else:
                text = f"{d.get('instruction','')}\n{d.get('input','')}\n{d.get('output','')}"

            # 只有当文本够长且有效时才加入训练
            if text and len(text) > 5:
                # 加上 <|endoftext|> (通常是 0) 这是一个好习惯，让模型知道何时结束
                # 假设 tokenizer.encode 没加，我们手动处理一下，或者依赖 ctx_len 截断
                toks = encode(text)
                # 加上 EOS token (根据你的 vocab，通常是 0)
                toks = toks + [0] 
                
                toks = toks[:ctx_len]
                if len(toks) > 16:
                    samples.append(toks)
    print(f"Loaded {len(samples)} samples")
    
    if not samples:
        print("Error: No samples!")
        return
    
    train_model.train()
    import random
    
    for ep in range(epochs):
        random.shuffle(samples)
        loss_sum = 0
        
        for i, toks in enumerate(samples):
            x = torch.tensor([toks[:-1]], dtype=torch.long, device=device)
            y = torch.tensor([toks[1:]], dtype=torch.long, device=device)
            
            logits = train_model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in train_model.parameters() if p.requires_grad], 1.0)
            opt.step()
            
            global_step += 1
            loss_steps.append(global_step)
            loss_values.append(float(loss.item()))
            
            if global_step in plot_targets:
                save_loss_curve(plot_targets[global_step])
            
            loss_sum += loss.item()
            if (i+1) % log_every == 0:
                print(f"Ep{ep+1} Step{i+1}/{len(samples)} Loss:{loss_sum/(i+1):.4f}")
        
        print(f"Epoch {ep+1} done, Loss: {loss_sum/len(samples):.4f}")
    
    save_loss_curve("final")


########################################################################################################
# 测试
########################################################################################################

def test(train_model, infer_model, encode, decode, device):
    """测试生成"""
    print("\n" + "=" * 60)
    print("测试 (使用albatross推理)")
    print("=" * 60)
    
    inference = AlbatrossInference(infer_model, train_model, device)
    
    prompts = [
        "User: 天上有几个太阳\n\nAssistant:",
        "User: 啦啦啦\n\nAssistant:",
        "User: 宝宝，如果我走了，你会怎么做？\n\nAssistant:",
    ]
    
    train_model.eval()
    
    for prompt in prompts:
        tokens = encode(prompt)
        print(f"\n--- Prompt ({len(tokens)} tokens) ---")
        print(f"{prompt}")
        
        # Greedy
        start = time.time()
        with torch.no_grad():
            gen, text = inference.generate(tokens, max_tokens=150, temperature=0, decode_fn=decode)
        elapsed = time.time() - start
        print(f"[Greedy] {len(gen)} tok, {elapsed:.2f}s, {len(gen)/elapsed:.1f} t/s")
        print(f"  {decode(tokens + gen)}")
        
        # Sample
        start = time.time()
        with torch.no_grad():
            gen, text = inference.generate(tokens, max_tokens=150, temperature=0.7, top_p=0.9, decode_fn=decode)
        elapsed = time.time() - start
        print(f"[Sample] {len(gen)} tok, {elapsed:.2f}s, {len(gen)/elapsed:.1f} t/s")
        print(f"  {decode(tokens + gen)}")


def speed_comparison(train_model, infer_model, device):
    """速度对比"""
    print("\n" + "=" * 60)
    print("速度对比: Albatross vs Full-Seq")
    print("=" * 60)
    
    inference = AlbatrossInference(infer_model, train_model, device)
    prompt_tokens = list(range(1, 21))
    
    for gen_len in [10, 30, 50, 100]:
        # Albatross (state-cache)
        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            state = inference.init_state_with_time_state(1)
            out = inference.infer_model.forward_batch([prompt_tokens], state)
            for _ in range(gen_len):
                token = int(out[0].argmax().item())
                out = inference.infer_model.forward_batch([[token]], state)
        torch.cuda.synchronize()
        alba_time = time.time() - start
        
        # Full-seq
        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            tokens = list(prompt_tokens)
            for _ in range(gen_len):
                idx = torch.tensor([tokens], dtype=torch.long, device=device)
                logits = train_model(idx)
                token = int(logits[0, -1, :].argmax().item())
                tokens.append(token)
        torch.cuda.synchronize()
        fs_time = time.time() - start
        
        speedup = fs_time / alba_time if alba_time > 0 else 0
        print(f"Gen {gen_len:3d}: Alba={alba_time*1000:.0f}ms, FS={fs_time*1000:.0f}ms, Speedup={speedup:.1f}x")


########################################################################################################
# Main
########################################################################################################

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help="模型路径（不带.pth）")
    parser.add_argument('--action', choices=['test', 'train', 'speed','eval'], default='test')
    parser.add_argument('--data', default='nekoqa_10k_formatted.jsonl')
    parser.add_argument('--state', default='state.pth')
    parser.add_argument('--tokenizer', default='reference/rwkv_vocab_v20230424.txt')
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--ctx_len', type=int, default=1024)
    parser.add_argument('--log_every', type=int, default=10)
    parser.add_argument('--plot_dir', default='loss_plots')
    args = parser.parse_args()
    
    device = 'cuda'
    encode, decode, tokenizer = get_tokenizer(args.tokenizer)
    
    # 加载两个模型
    train_model, train_args = load_train_model(args.model, device, args.ctx_len)
    infer_model, infer_args = load_inference_model(args.model, device)
    
    # 加载已有的state（如果有）
    if os.path.exists(args.state):
        load_state(train_model, args.state)
    
    if args.action == 'test':
        test(train_model, infer_model, encode, decode, device)
        speed_comparison(train_model, infer_model, device)
        
    elif args.action == 'speed':
        speed_comparison(train_model, infer_model, device)
        
    elif args.action == 'train':
        train(train_model, encode, args.data, device, args.lr, args.epochs, args.ctx_len, args.log_every, args.plot_dir)
        save_state(train_model, args.state)
        print("\n--- 训练后测试 ---")
        test(train_model, infer_model, encode, decode, device)
    elif args.action == 'eval':
        # 新增的评测入口
        # 注意：这里复用 --data 参数作为测试集路径
        evaluate_math(train_model, infer_model, encode, decode, args.data, device)

if __name__ == "__main__":
    main()
