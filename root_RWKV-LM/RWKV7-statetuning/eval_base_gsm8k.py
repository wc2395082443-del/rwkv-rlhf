import os
import sys
import json
import re
import argparse
import gc
from types import SimpleNamespace
from tqdm import tqdm

# ==========================================
# 1. 环境变量配置 (以此为准)
# ==========================================
os.environ["RWKV_CTXLEN"] = "8192"
os.environ["RWKV_HEAD_SIZE_A"] = "64"
os.environ["RWKV_HEAD_SIZE"] = "64"
os.environ["RWKV_CUDA_ON"] = "1" 

# ==========================================
# 2. 路径配置
# ==========================================
sys.path.append("C:\\RWKV-LM\\Albatross\\")
sys.path.append("/mnt/program/_RWKV_/_ref_/RWKV-CUDA/rwkv7_fast_fused")

import torch
from torch.nn import functional as F

# ==========================================
# 3. 导入并暴力修正 RWKV 模块
# ==========================================
import reference.rwkv7 as rwkv7_module
from reference.rwkv7 import RWKV_x070
from reference.utils import TRIE_TOKENIZER

# 🚨🚨🚨 暴力修正：强制覆盖模块内的全局变量 🚨🚨🚨
print(f"DEBUG: Original HEAD_SIZE in module: {getattr(rwkv7_module, 'HEAD_SIZE', 'Not Found')}")
rwkv7_module.HEAD_SIZE = 64
print(f"DEBUG: Patched HEAD_SIZE in module to: {rwkv7_module.HEAD_SIZE}")

torch.set_float32_matmul_precision('high')

# ==========================================
# 4. 默认路径
# ==========================================
DEFAULT_MODEL_PATH = r"C:\RWKV-LM\rwkv7-g1b-1.5b-20251202-ctx8192.pth"
DEFAULT_TOKENIZER_PATH = r"C:\RWKV-LM\rwkv_vocab_v20230424.txt"

# ==========================================
# 5. 采样逻辑
# ==========================================
def sample_logits(logits, occurrence, params):
    if params['alpha_presence'] > 0 or params['alpha_frequency'] > 0:
        presence_penalty = (occurrence > 0).float() * params['alpha_presence']
        frequency_penalty = occurrence * params['alpha_frequency']
        logits -= (presence_penalty + frequency_penalty)

    if params['temperature'] != 1.0:
        logits /= params['temperature']

    probs = F.softmax(logits, dim=-1)

    if params['top_k'] > 0:
        top_k_val, _ = torch.topk(probs, params['top_k'])
        probs[probs < top_k_val[-1]] = 0
        probs = probs / probs.sum()

    if params['top_p'] < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_indices_to_remove = cumulative_probs > params['top_p']
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
        probs[indices_to_remove] = 0
        probs = probs / probs.sum()

    try:
        token = torch.multinomial(probs, 1).item()
    except:
        token = torch.argmax(logits).item()
    return token

# ==========================================
# 6. CUDAGraph 引擎
# ==========================================
class CUDAGraphEngine:
    def __init__(self, model_path, decode_params):
        # 1. 路径处理：确保本地检查时有 .pth
        real_path = model_path
        if not real_path.endswith(".pth"):
            real_path += ".pth"
            
        print(f"Loading Model Config from: {real_path}")
        
        # 2. 自动探测参数
        try:
            tmp = torch.load(real_path, map_location='cpu')
            n_embd = tmp['emb.weight'].shape[1]
            n_layer = 0
            for k in tmp.keys():
                if 'ln1.weight' in k: n_layer += 1
            del tmp
            gc.collect()
            print(f"Detected Config: n_embd={n_embd}, n_layer={n_layer}")
        except Exception as e:
            print(f"Error loading config: {e}")
            n_embd = 2048 # 1.5B default
            n_layer = 24

        # 3. 传给 RWKV_x070 时去掉 .pth
        model_name_for_args = real_path.replace(".pth", "")

        args = SimpleNamespace(
            vocab_size=65536,
            head_size=64,
            n_embd=n_embd,
            n_layer=n_layer,
            ctx_len=8192,
            MODEL_NAME=model_name_for_args
        )
        
        print(f"Model Args: {args}")
        
        self.model = RWKV_x070(args)
        self.model.eval()
        self.decode_params = decode_params
        self.tokenizer = TRIE_TOKENIZER(DEFAULT_TOKENIZER_PATH)
        
        print("Initializing CUDAGraph...")
        self.n_embd = self.model.args.n_embd
        self.vocab_size = self.model.args.vocab_size
        
        # 静态变量：用于 graph capture
        self.static_input = torch.empty((self.n_embd), device="cuda", dtype=torch.half)
        tmp_state = self.model.generate_zero_state(1)
        self.static_state = [
            torch.empty_like(tmp_state[0], device="cuda"), 
            torch.empty_like(tmp_state[1], device="cuda")
        ]
        self.static_output = torch.empty((self.vocab_size), device="cuda", dtype=torch.half)
        
        # === Warmup ===
        print("Warming up CUDA & JIT...")
        
        # 基础 CUDA 初始化
        torch.matmul(torch.ones(1, 1, device='cuda'), torch.ones(1, 1, device='cuda'))
        
        # 🔧 修复：使用正确的方式预热
        # 方法1：使用 forward() 预热（推荐）
        warmup_state = self.model.generate_zero_state(1)
        warmup_tokens = [1, 2, 3]  # 随便几个 token
        try:
            _ = self.model.forward(warmup_tokens, warmup_state)
            torch.cuda.synchronize()
            print("Warmup with forward() successful")
        except Exception as e:
            print(f"Warmup warning: {e}")
        
        # 方法2：如果需要预热 forward_one_alt，使用正确的输入格式
        try:
            # forward_one_alt 需要 embedding 向量作为输入
            warmup_emb = self.model.z['emb.weight'][1].to(dtype=torch.half, device='cuda')
            warmup_state2 = self.model.generate_zero_state(1)
            _ = self.model.forward_one_alt(warmup_emb, warmup_state2)
            torch.cuda.synchronize()
            print("Warmup with forward_one_alt() successful")
        except Exception as e:
            print(f"Warmup warning (forward_one_alt): {e}")
        
        # === 初始化静态 state ===
        init_state = self.model.generate_zero_state(1)
        self.static_state[0].copy_(init_state[0])
        self.static_state[1].copy_(init_state[1])
        
        # === Capture Graph ===
        print("Capturing Graph...")
        self.g = torch.cuda.CUDAGraph()
        
        # 填充一个有效的 embedding
        test_emb = self.model.z['emb.weight'][1].to(dtype=torch.half, device='cuda')
        self.static_input.copy_(test_emb)
        
        with torch.cuda.graph(self.g):
            self.static_output = self.model.forward_one_alt(self.static_input, self.static_state)
            
        print("Engine Ready!")

    def generate(self, prompt, max_new_tokens=512):
        tokens = self.tokenizer.encode(prompt)
        
        occurrence = torch.zeros(self.vocab_size, device="cuda", dtype=torch.float32)
        for t in tokens: 
            occurrence[t] += 1

        # 初始化 state 并处理 prompt
        state = self.model.generate_zero_state(1)
        out = self.model.forward(tokens, state)
        token = sample_logits(out, occurrence, self.decode_params)
        
        generated = []
        
        # 复制 state 到静态变量
        self.static_state[0].copy_(state[0])
        self.static_state[1].copy_(state[1])
        
        for _ in range(max_new_tokens):
            if token == 0: 
                break
            generated.append(token)
            
            # Decay occurrence
            if self.decode_params['alpha_decay'] < 1.0:
                occurrence *= self.decode_params['alpha_decay']
            occurrence[token] += 1
            
            # 早停检测
            if len(generated) % 10 == 0:
                curr_text = self.tokenizer.decode(generated)
                if "\n\nUser:" in curr_text or "Q:" in curr_text:
                    break

            # 获取 embedding 并复制到静态输入
            emb = self.model.z['emb.weight'][token].to(dtype=torch.half, device='cuda')
            self.static_input.copy_(emb)
            
            # Replay graph
            self.g.replay()
            
            # 采样下一个 token
            logits = self.static_output.float()
            token = sample_logits(logits, occurrence, self.decode_params)
            
        return self.tokenizer.decode(generated)

# ==========================================
# 7. 主程序
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=DEFAULT_MODEL_PATH)
    parser.add_argument('--data', default="gsm8k_test_formatted.jsonl") 
    parser.add_argument('--output', default="gsm8k_sampler_result.jsonl")
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    
    DECODE_PARAMS = {
        'temperature': 0.3,
        'top_k': 500,
        'top_p': 0.4,
        'alpha_presence': 0.5,
        'alpha_frequency': 0.1,
        'alpha_decay': 0.99
    }
    
    print(f"Using Decode Params: {DECODE_PARAMS}")
    print(f"Model Args: {args}")
    
    # 数据文件检查
    if not os.path.exists(args.data):
        if os.path.exists(os.path.join("C:\\RWKV-LM", args.data)):
            args.data = os.path.join("C:\\RWKV-LM", args.data)
        else:
            print(f"Warning: Data file '{args.data}' not found. Creating dummy test.")
            dummy_data = [{"question": "What is 1+1?", "answer": "2"}]
            with open("dummy_test.jsonl", "w", encoding='utf-8') as f:
                f.write(json.dumps(dummy_data[0]))
            args.data = "dummy_test.jsonl"

    # 加载数据
    data = []
    with open(args.data, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    if args.limit > 0: 
        data = data[:args.limit]
    
    print(f"Loaded {len(data)} test cases")
    
    # 初始化引擎
    engine = CUDAGraphEngine(args.model, DECODE_PARAMS)
    
    prompt_template = "Question: {q}\nLet's think step by step.\nAnswer:"
    
    print(f"Start generating for {len(data)} items...")
    correct = 0
    total = 0
    
    with open(args.output, 'w', encoding='utf-8') as f_out:
        for item in tqdm(data):
            q = item.get('problem', '') or item.get('question', '')
            gt = item.get('solution', '') or item.get('answer', '')
            
            response = engine.generate(prompt_template.format(q=q))
            
            # 提取数字答案
            pred_nums = re.findall(r'-?\d+\.?\d*', response.replace(',', ''))
            gt_nums = re.findall(r'-?\d+\.?\d*', gt.replace(',', ''))
            
            is_right = False
            if pred_nums and gt_nums:
                try:
                    is_right = float(pred_nums[-1]) == float(gt_nums[-1])
                except: 
                    pass
            
            if is_right:
                correct += 1
            total += 1
            
            record = {
                "q": q, 
                "gt": gt, 
                "pred": response, 
                "ok": is_right,
                "accuracy": correct / total
            }
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            f_out.flush()
    
    print(f"\nDone! Results saved to {args.output}")
    print(f"Final Accuracy: {correct}/{total} = {correct/total*100:.2f}%")

if __name__ == "__main__":
    main()




