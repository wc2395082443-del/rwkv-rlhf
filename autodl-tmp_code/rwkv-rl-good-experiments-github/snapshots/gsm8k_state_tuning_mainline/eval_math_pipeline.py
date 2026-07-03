import os
import sys
import json
import torch
import re
import argparse
import time
from tqdm import tqdm
from types import SimpleNamespace
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
from sympy import simplify

# ================= 环境配置 =================
# 确保能引用到 albatross 和 rwkv7 代码
sys.path.append("C:\\RWKV-LM\\Albatross\\")
sys.path.append("/mnt/program/_RWKV_/_ref_/RWKV-CUDA/rwkv7_fast_fused")

# 引入模型定义
from rwkv7_trainable import RWKV7
from reference.rwkv7 import RWKV_x070
from reference.utils import TRIE_TOKENIZER, sampler_simple_batch

# 设置环境变量 (与训练脚本保持一致)
os.environ["RWKV_HEAD_SIZE_A"] = "64"
os.environ["RWKV_CTXLEN"] = "4096"

HEAD_SIZE = 64

# ==========================================
# 核心组件 1: 数学判分器 (SymPy)
# ==========================================
class MathGrader:
    def __init__(self):
        # 允许隐式乘法 (如 2x)
        self.transformations = (standard_transformations + (implicit_multiplication_application,))

    def extract_boxed(self, text):
        """提取 \\boxed{...} 内容，支持嵌套括号"""
        if not text: return None
        start_idx = text.find("\\boxed{")
        if start_idx == -1: return None
        
        idx = start_idx + 7
        brace_count = 1
        content_start = idx
        while idx < len(text):
            if text[idx] == '{': brace_count += 1
            elif text[idx] == '}': brace_count -= 1
            if brace_count == 0: return text[content_start:idx]
            idx += 1
        return None

    def clean_latex(self, text):
        """字符串清洗"""
        if not text: return ""
        text = str(text).strip()
        text = text.replace(" ", "").replace("\n", "")
        text = text.replace(r"\mathrm", "").replace(r"\text", "")
        text = text.replace(r"\left", "").replace(r"\right", "")
        text = text.replace(r"\,", "").replace(r"\!", "")
        text = text.replace(r"\dfrac", r"\frac")
        # 去掉末尾的句号
        if text.endswith("."): text = text[:-1]
        return text

    def parse_sympy(self, latex_str):
        """转为 SymPy 表达式"""
        s = latex_str
        # 简单的分数转换
        s = re.sub(r'\\frac\{([^{}]+)\}\{([^{}]+)\}', r'(\1)/(\2)', s)
        s = s.replace("^", "**").replace(r"\%", "/100").replace("\\", "")
        try:
            return parse_expr(s, transformations=self.transformations)
        except:
            return None

    def is_equivalent(self, model_output, ground_truth):
        # 1. 提取
        pred = self.extract_boxed(model_output)
        truth = self.extract_boxed(ground_truth)
        
        # 兜底：如果模型没输出 boxed，尝试取最后一行
        if pred is None:
            lines = model_output.strip().split('\n')
            if lines: pred = lines[-1]
        
        if truth is None: truth = ground_truth # GT 有时没 boxed

        if not pred or not truth: return False

        # 2. 字符串清洗对比
        clean_pred = self.clean_latex(pred)
        clean_truth = self.clean_latex(truth)
        if clean_pred == clean_truth: return True

        # 3. SymPy 数学对比
        expr_pred = self.parse_sympy(clean_pred)
        expr_truth = self.parse_sympy(clean_truth)
        
        if expr_pred is not None and expr_truth is not None:
            try:
                if simplify(expr_pred - expr_truth) == 0:
                    return True
            except:
                pass
        return False

# ==========================================
# 核心组件 2: 推理引擎 (Albatross + State)
# ==========================================
class InferenceEngine:
    def __init__(self, model_path, state_path, device='cuda'):
        print(f"Loading Base Model: {model_path}")
        # 1. 加载推理模型 (Albatross)
        # 注意：Albatross 内部会自动把模型加载到 GPU (如果可用)
        args = SimpleNamespace(vocab_size=65536, MODEL_NAME=model_path)
        self.model = RWKV_x070(args) 
        
        # 2. 初始化 Albatross State
        print("Initializing State...")
        B = 1
        self.init_state = self.model.generate_zero_state(B)
        
        # 3. 直接加载 State 权重文件并注入
        print(f"Loading State weights: {state_path}")
        state_dict = torch.load(state_path, map_location='cpu', weights_only=True)
        
        # 遍历每一层，寻找对应的 time_state 并注入
        n_layer = self.model.args.n_layer
        injected_count = 0
        
        for i in range(n_layer):
            key = f'blocks.{i}.att.time_state'
            
            if key in state_dict:
                # 获取权重
                ts = state_dict[key].float()
                
                # === 关键修改：动态获取目标设备 ===
                # 我们直接看 init_state 在哪个设备上，就把 ts 搬过去
                # self.init_state[1] 是 list，取第 i 层
                target_device = self.init_state[1][i].device
                
                if ts.device != target_device:
                    ts = ts.to(target_device)
                
                # 注入 (扩展 Batch 维度)
                self.init_state[1][i] = ts.unsqueeze(0).expand(B, -1, -1, -1).clone()
                injected_count += 1
            else:
                # 某些层可能没有被训练到，或者命名不同
                pass

        print(f"Successfully injected state for {injected_count}/{n_layer} layers.")
        
        del state_dict
        torch.cuda.empty_cache()
        
        self.tokenizer = TRIE_TOKENIZER("C:\\RWKV-LM\\rwkv_vocab_v20230424.txt")

    def generate(self, prompt, max_tokens=1024):
        tokens = self.tokenizer.encode(prompt)
        
        # 复制一份初始 State
        state = [s.clone() for s in self.init_state]
        
        # Prefill
        out = self.model.forward_batch([tokens], state)
        
        generated = []
        for _ in range(max_tokens):
            token = int(out[0].argmax().item())
            if token == 0: break
            
            generated.append(token)
            out = self.model.forward_batch([[token]], state)
            
        return self.tokenizer.decode(generated)


# ==========================================
# 主流程
# ==========================================

def run_generation(args):
    """阶段 1: 生成回复"""
    if os.path.exists(args.output) and not args.overwrite:
        print(f"Output file {args.output} exists. Skipping generation.")
        return

    # 加载数据
    data = []
    with open(args.data, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except: pass
            
    print(f"Loaded {len(data)} problems.")
    
    # 初始化引擎
    engine = InferenceEngine(args.model, args.state)
    
    # 准备写入
    with open(args.output, 'w', encoding='utf-8') as f_out:
        pbar = tqdm(data)
        for item in pbar:
            problem = item.get('problem', '')
            solution = item.get('solution', '') or item.get('answer', '')
            
            # Prompt 模板
            prompt = f"User: {problem}\n\nAssistant:"
            
            # 生成
            try:
                response = engine.generate(prompt)
            except Exception as e:
                print(f"Error generating: {e}")
                response = ""
            
            # 保存
            record = {
                "problem": problem,
                "ground_truth": solution,
                "model_output": response
            }
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            f_out.flush()

def run_evaluation(args):
    """阶段 2: 评测"""
    if not os.path.exists(args.output):
        print(f"File {args.output} not found. Run generation first.")
        return

    print(f"Evaluating results from {args.output}...")
    grader = MathGrader()
    
    correct = 0
    total = 0
    
    results = []
    with open(args.output, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line)
            except: continue
            
            is_correct = grader.is_equivalent(item['model_output'], item['ground_truth'])
            
            if is_correct:
                correct += 1
            total += 1
            
            # 可选：打印错题
            # if not is_correct:
            #     print(f"GT: {grader.extract_boxed(item['ground_truth'])} | Pred: {grader.extract_boxed(item['model_output'])}")

    print("\n" + "="*40)
    print(f"Final Accuracy: {correct/total:.2%} ({correct}/{total})")
    print("="*40)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help="Base model name (no .pth)")
    parser.add_argument('--state', required=True, help="Path to trained state .pth")
    parser.add_argument('--data', required=True, help="Path to math500.jsonl")
    parser.add_argument('--output', default="math_eval_result.jsonl", help="Output file")
    parser.add_argument('--tokenizer', default="C:\RWKV-LM\rwkv_vocab_v20230424.txt")
    parser.add_argument('--overwrite', action='store_true', help="Overwrite existing output")
    parser.add_argument('--only_eval', action='store_true', help="Skip generation, only eval")
    
    args = parser.parse_args()
    
    # 修正 tokenizer 路径给全局使用
    # (InferenceEngine 里硬编码了路径，如果需要灵活可以传参修改类)
    
    if not args.only_eval:
        run_generation(args)    
        
    run_evaluation(args)

if __name__ == "__main__":
    main()
