#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import time
import math
import random
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional

import torch
import torch.nn.functional as F


# =========================================================
# Utils
# =========================================================
def now_str() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())

def read_jsonl(path: str, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    """Read JSONL file with optional sample limit"""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_samples is not None and i >= max_samples:
                break
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data

def append_jsonl(path: str, obj: Dict[str, Any]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# =========================================================
# Progress Visualization
# =========================================================

class ProgressTracker:
    """Real-time progress tracking and visualization"""
    
    def __init__(self, total_steps: int, log_path: str):
        self.total_steps = total_steps
        self.log_path = log_path
        self.start_time = time.time()
        self.step_times = []
        
    def update(self, step: int, metrics: Dict[str, Any]):
        """Update progress with current metrics"""
        elapsed = time.time() - self.start_time
        self.step_times.append(elapsed)
        
        # Calculate ETA
        if len(self.step_times) > 1:
            avg_step_time = (self.step_times[-1] - self.step_times[0]) / len(self.step_times)
            eta_seconds = avg_step_time * (self.total_steps - step)
            eta_str = self._format_time(eta_seconds)
        else:
            eta_str = "calculating..."
        
        # Progress bar
        progress = step / self.total_steps
        bar_length = 40
        filled = int(bar_length * progress)
        bar = "#" * filled + "-" * (bar_length - filled)
        
        # Format metrics display
        metric_str = " | ".join([f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" 
                                  for k, v in metrics.items()])
        
        # Print progress
        print(f"\r[{bar}] {progress*100:.1f}% | Step {step}/{self.total_steps} | "
              f"Elapsed: {self._format_time(elapsed)} | ETA: {eta_str} | {metric_str}", 
              end="", flush=True)
        
        # Save to log
        log_entry = {
            "step": step,
            "progress": progress,
            "elapsed_seconds": elapsed,
            "eta_seconds": eta_seconds if len(self.step_times) > 1 else None,
            **metrics
        }
        append_jsonl(self.log_path, log_entry)
    
    def _format_time(self, seconds: float) -> str:
        """Format seconds to readable time string"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        if hours > 0:
            return f"{hours}h{minutes}m{secs}s"
        elif minutes > 0:
            return f"{minutes}m{secs}s"
        else:
            return f"{secs}s"
    
    def finish(self):
        """Print completion message"""
        total_time = time.time() - self.start_time
        print(f"\n✓ Training completed in {self._format_time(total_time)}")


# =========================================================
# Prompt (keep 'think'!)
# =========================================================

def build_prompt(problem: str) -> str:
    p = (problem or "").strip()
    return (
        f"User: {p}\n"
        f"请将最终答案放在\\boxed{{...}}里，并且最终只给出\\boxed{{...}}这一行，不要输出多余内容。 think\n"
        f"Assistant: <think>\n"
    )


# =========================================================
# Answer extraction & judging
# =========================================================

def _strip_math_delims(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    s = s.replace(r"\left", "").replace(r"\right", "")
    s = re.sub(r"\\[,\;\!\:]\s*", "", s)
    return s.strip()

def _find_balanced_brace(text: str, brace_start: int) -> Optional[Tuple[str, int]]:
    if brace_start < 0 or brace_start >= len(text) or text[brace_start] != "{":
        return None
    depth = 0
    i = brace_start
    while i < len(text):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[brace_start + 1:i], i
        i += 1
    return None

def extract_last_boxed(text: str) -> Optional[str]:
    if not text:
        return None
    key = r"\boxed{"
    idx = text.rfind(key)
    if idx < 0:
        return None
    brace = idx + len(key) - 1
    got = _find_balanced_brace(text, brace)
    if got is None:
        return None
    inner, _ = got
    return _strip_math_delims(inner)

def extract_final_answer(text: str) -> Optional[str]:
    a = extract_last_boxed(text)
    if a:
        return a
    # fallback: last non-empty line
    lines = [x.strip() for x in (text or "").splitlines() if x.strip()]
    if not lines:
        return None
    last = lines[-1].replace("</think>", "").strip()
    return _strip_math_delims(last) if last else None

def boxed_complete(text: str) -> bool:
    k = text.find(r"\boxed{")
    if k < 0:
        return False
    i = k + len(r"\boxed{")
    depth = 1
    while i < len(text):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return True
        i += 1
    return False


# =========================================================
# Answer extraction and verification logic
# =========================================================

def extract_answer(text):
    """Extract answer from model output - optimized for \\boxed{} format"""
    if not text or not isinstance(text, str):
        return None
    
    text = text.strip()
    
    # 1. Priority match \boxed{number}
    boxed_patterns = [
        r'\\boxed\{([^}]+)\}',
        r'\\boxed\s*\{([^}]+)\}',
        r'boxed\{([^}]+)\}',
        r'\{([0-9,\.\-\s]+)\}',
    ]
    
    for pattern in boxed_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            answer = matches[-1].strip()
            numbers = re.findall(r'-?\d+\.?\d*', answer.replace(',', ''))
            if numbers:
                return numbers[-1]
    
    # 2. Try matching answer after ####
    if '####' in text:
        after_hash = text.split('####')[-1].strip()
        numbers = re.findall(r'-?\d+\.?\d*', after_hash.replace(',', ''))
        if numbers:
            return numbers[0]
    
    # 3. Try extracting number from last line
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    if lines:
        last_line = lines[-1]
        numbers = re.findall(r'-?\d+\.?\d*', last_line.replace(',', ''))
        if numbers:
            return numbers[-1]
    
    # 4. Fallback: extract last number in full text
    all_numbers = re.findall(r'-?\d+\.?\d*', text.replace(',', ''))
    if all_numbers:
        return all_numbers[-1]
    
    return None


def normalize_answer(answer):
    """Normalize answer format"""
    if answer is None:
        return None
    
    answer_str = str(answer).strip()
    answer_str = answer_str.replace(',', '').replace('$', '').replace('%', '')
    answer_str = answer_str.replace('\\', '').replace('{', '').replace('}', '')
    answer_str = re.sub(r'[a-zA-Z\s]+$', '', answer_str).strip()
    
    try:
        num = float(answer_str)
        if abs(num - round(num)) < 1e-9:
            return str(int(round(num)))
        formatted = f"{num:.10f}".rstrip('0').rstrip('.')
        return formatted
    except (ValueError, TypeError):
        return answer_str.strip()


def compare_answers(pred, gold, tolerance=1e-6):
    """Compare two answers for equality"""
    if pred is None or gold is None:
        return False
    
    pred_norm = normalize_answer(pred)
    gold_norm = normalize_answer(gold)
    
    if pred_norm is None or gold_norm is None:
        return False
    
    # 1. Exact string match
    if pred_norm == gold_norm:
        return True
    
    # 2. Numerical comparison
    try:
        pred_num = float(pred_norm)
        gold_num = float(gold_norm)
        
        abs_diff = abs(pred_num - gold_num)
        if abs_diff < tolerance:
            return True
        
        if abs(gold_num) > tolerance:
            rel_diff = abs_diff / abs(gold_num)
            if rel_diff < tolerance:
                return True
        
        return False
    except (ValueError, TypeError):
        pass
    
    # 3. Case-insensitive string comparison
    return pred_norm.lower() == gold_norm.lower()


def verify_gsm8k_answer(model_response, correct_answer, verbose=False):
    """Verify GSM8K answer"""
    extracted = extract_answer(model_response)
    gold_extracted = extract_answer(correct_answer) or correct_answer
    pred_norm = normalize_answer(extracted)
    gold_norm = normalize_answer(gold_extracted)
    is_correct = compare_answers(extracted, gold_extracted)
    
    result = {
        'is_correct': is_correct,
        'extracted_answer': extracted,
        'normalized_pred': pred_norm,
        'normalized_gold': gold_norm,
        'raw_response': model_response[:200] if model_response else None
    }
    
    if verbose:
        print(f"Raw response: {model_response[:100]}...")
        print(f"Extracted: {extracted}")
        print(f"Pred (normalized): {pred_norm}")
        print(f"Gold (normalized): {gold_norm}")
        print(f"Match: {is_correct}")
    
    return result


# =========================================================
# Zhipu API judge (via OpenAI-compatible API)
# =========================================================

import requests

ZHIPU_API_URL = "https://www.packyapi.com/v1/chat/completions"
ZHIPU_API_KEY = ""


def judge_with_zhipu(pred_full_output: str, gt: str) -> Tuple[bool, str]:
    """Use Zhipu API to judge if the prediction is correct"""
    pred = (pred_full_output or "").strip()
    gt = (gt or "").strip()

    prompt = (
        "你是一个严格的答案判定器。给定标准答案(GT)与模型的完整输出(OUTPUT)，判断模型的回答是否正确。\n"
        "你需要从模型输出中找到最终答案（通常在\\boxed{}中），然后判断其是否与标准答案等价。\n"
        "等价包括：数学等价、同义表述、可化简的表达式、等值分数/小数等。\n"
        "若无法确定或信息不足，一律判为不等价并输出 0。\n"
        "请只输出一个字符：1 或 0。\n\n"
        f"GT: {gt}\n"
        f"OUTPUT: {pred}\n"
        "Output: "
    )

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {ZHIPU_API_KEY}"
    }

    data = {
        "model": "glm-4.7",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 16,
        "temperature": 0.0,
    }

    response = requests.post(ZHIPU_API_URL, headers=headers, json=data, timeout=60)
    response.raise_for_status()
    result = response.json()
    raw = result.get("choices", [{}])[0].get("message", {}).get("content", "")

    m = re.search(r"(?<!\d)([01])(?!\d)", raw.strip())
    ok = (m is not None and m.group(1) == "1")
    return ok, raw


def judge_answer_llm(pred_text: str, gt_text: str) -> Tuple[float, Dict[str, Any]]:
    """Judge correctness using Zhipu API (LLM-based)"""
    gt_ans = _strip_math_delims(gt_text)

    dbg: Dict[str, Any] = {
        "pred_full_output": pred_text[:500] if pred_text else None,
        "gt": gt_ans,
        "method": "zhipu-api",
        "error": None,
        "judge_llm_raw": None,
        "judge_llm_decision": None,
        "correct": False,
    }

    if not pred_text or not pred_text.strip():
        dbg["method"] = "no_output"
        return 0.0, dbg

    try:
        ok, raw = judge_with_zhipu(pred_full_output=pred_text, gt=gt_ans)
        dbg["judge_llm_raw"] = raw
        dbg["judge_llm_decision"] = int(bool(ok))
        dbg["correct"] = bool(ok)
        return (1.0 if ok else 0.0), dbg
    except Exception as e:
        # Fallback to strict normalized string compare
        dbg["method"] = "fallback_string_exact"
        dbg["error"] = repr(e)
        pred_ans = extract_final_answer(pred_text)
        if pred_ans:
            p_norm = _strip_math_delims(pred_ans).replace(" ", "")
            g_norm = _strip_math_delims(gt_ans).replace(" ", "")
            ok = (p_norm == g_norm)
            dbg["correct"] = ok
            return (1.0 if ok else 0.0), dbg
        return 0.0, dbg


def judge_answer_dispatch(
    judge_type: str,
    model_response: str,
    gold_answer: str,
    llm_judge_fn=None,
) -> Tuple[float, Dict[str, Any]]:
    """
    Dispatch to appropriate judge based on judge_type
    Returns: (reward, debug_info)
    """
    if judge_type == "llm":
        if llm_judge_fn is None:
            raise ValueError("llm_judge_fn must be provided when judge_type=llm")
        reward, dbg = llm_judge_fn(model_response, gold_answer)
        return reward, dbg

    elif judge_type == "rule":
        result = verify_gsm8k_answer(
            model_response=model_response,
            correct_answer=gold_answer,
            verbose=False
        )
        dbg = {
            "pred_extracted": result["extracted_answer"],
            "gt": normalize_answer(extract_answer(gold_answer) or gold_answer),
            "method": "rule",
            "error": None,
            "pred_parsed": result["normalized_pred"],
            "gt_parsed": result["normalized_gold"],
            "correct": result["is_correct"],
            "truncated_forced_zero": False,
        }
        return (1.0 if result["is_correct"] else 0.0), dbg

    else:
        raise ValueError(f"Unknown judge_type: {judge_type}")


# =========================================================
# Config
# =========================================================

STOP_TOKENS = ("\n\nUser:", "\n\nQuestion:", "Q:", "<|endoftext|>", "\nUser:", "\nQuestion:")

@dataclass
class DAPOConfig:
    # sampling / batch
    batch_prompts: int = 1
    group_size: int = 16
    rollout_n: int = 32
    train_bsz: int = 32
    max_new_tokens: int = 512
    temperature: float = 1.0
    top_p: float = 0.7
    rollout_temperature: float = 1.0
    rollout_top_p: float = 0.6
    top_k: int = 0
    buffer_length_weight: float = 0.001
    buffer_save_path: Optional[str] = None
    buffer_load_path: Optional[str] = None
    buffer_save_interval: int = 5
    buffer_cold_path: Optional[str] = None
    buffer_min_init: int = 0
    buffer_warmup_rounds: int = 1
    mask_token0: bool = True
    presence_penalty:float= 0.5 
    frequency_penalty:float=0.1
    alpha_decay:float=0.99

    # stop checks
    stop_on_think_close: bool = False
    stop_on_user: bool = True
    stop_on_boxed: bool = True
    stop_check_every: int = 1
    stop_check_window: int = 96

    # dynamic sampling
    dynamic_sampling_max_tries: int = 200
    collect_chunk: int = 4

    # PPO/DAPO
    ppo_epochs: int = 1
    micro_batch: int = 4
    lr: float = 1e-4
    eps_low: float = 0.2
    eps_high: float = 0.5
    grad_clip: float = 0.2

    # stability
    kl_coef: float = 0.08
    target_kl: float = 0.01
    adaptive_kl: bool = True
    time_state_l2: float = 5e-6
    time_state_clamp: float = 3.0

    # logging / save
    log_interval: int = 1
    save_interval: int = 50
    infer_check_interval: int = 50

    # eval
    eval_interval: int = 5
    eval_n: int = 16
    eval_temperature: float = 0.0
    eval_top_p: float = 1.0
    eval_top_k: int = 0
    eval_max_new_tokens: int = 256

    # judge type
    judge_type: str = "llm"  # "llm" or "rule"

    # faulthandler
    enable_faulthandler: bool = False
    hang_dump_s: float = 0.0


# =========================================================
# Model helpers
# =========================================================

HEAD_SIZE = 64

def normalize_model_arg(model_arg: str) -> Tuple[str, str]:
    """Return (base_name_no_pth_for_albatross, pth_path_for_torch_load)."""
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
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")

def load_train_model_rwkv7_cuda(pth_path: str, device: str, ctx_len: int):
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
    import types
    from reference.rwkv7 import RWKV_x070

    args = types.SimpleNamespace()
    args.vocab_size = 65536
    args.MODEL_NAME = base_name_no_pth
    model = RWKV_x070(args)
    return model, args

def freeze_except_time_state(model: torch.nn.Module) -> int:
    cnt = 0
    for n, p in model.named_parameters():
        if "time_state" in n:
            p.requires_grad = True
            cnt += p.numel()
        else:
            p.requires_grad = False
    return cnt

def save_time_state_only(model: torch.nn.Module, path: str):
    sd = {n: p.detach().cpu() for n, p in model.named_parameters() if "time_state" in n}
    torch.save(sd, path)

def load_time_state_only(model: torch.nn.Module, path: str) -> bool:
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


# =========================================================
# Albatross batched inference
# =========================================================

class AlbatrossBatchInference:
    def __init__(self, infer_model, train_model, encode_fn, decode_fn, device: str, cfg: DAPOConfig):
        self.infer_model = infer_model
        self.train_model = train_model
        self.encode = encode_fn
        self.decode = decode_fn
        self.device = device
        self.cfg = cfg

    def init_state_with_time_state(self, B: int):
        state = self.infer_model.generate_zero_state(B)
        for i, block in enumerate(self.train_model.blocks):
            ts = block.att.time_state
            state[1][i] = ts.unsqueeze(0).expand(B, -1, -1, -1).clone()
        return state

    @torch.no_grad()
    def prime_prompts(self, prompt_tokens_list: List[List[int]]):
        B = len(prompt_tokens_list)
        state = self.init_state_with_time_state(B)
        out = self.infer_model.forward_batch(prompt_tokens_list, state)
        if torch.is_tensor(out) and out.dim() == 3:
            out = out[:, -1, :]
        return out, state

    @torch.no_grad()
    def generate_group_parallel(
        self,
        prompt_tokens_list: List[List[int]],
        group_size: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        stop_on_think_close: bool,
        stop_on_user: bool,
        stop_on_boxed: bool,
        stop_check_every: int,
        stop_check_window: int,
    ) -> Tuple[List[List[int]], List[List[float]], List[str], List[bool]]:

        Bp = len(prompt_tokens_list)
        if Bp == 0:
            return [], [], [], []

        # 1. 初始 Prompt 预处理
        last_logits, state = self.prime_prompts(prompt_tokens_list)

        B = Bp * group_size
        last_logits = last_logits.repeat_interleave(group_size, dim=0).contiguous()

        # RWKV7 State 复制
        state0 = state[0].repeat_interleave(group_size, dim=2).contiguous()
        state1 = state[1].repeat_interleave(group_size, dim=1).contiguous()
        state = [state0, state1]

        comp_tokens: List[List[int]] = [[] for _ in range(B)]
        old_logps: List[List[float]] = [[] for _ in range(B)]
        active = torch.ones((B,), device=last_logits.device, dtype=torch.bool)
        truncated = [False for _ in range(B)]

        # 2. 对齐 Benchmark 的惩罚参数
        alpha_presence = getattr(self.cfg, 'presence_penalty', 0.5)
        alpha_frequency = getattr(self.cfg, 'frequency_penalty', 0.1)
        alpha_decay = getattr(self.cfg, 'alpha_decay', 0.99)
        occurences = torch.zeros((B, last_logits.size(-1)), device=last_logits.device)

        # 3. 主生成循环
        for t in range(max_new_tokens):
            if not active.any():
                break

            # --- A. 基础 Logits 处理 ---
            logits = last_logits.float()
            
            # 重要：在应用任何惩罚前记录原始 Log Softmax，用于 RL 训练
            raw_logp_all = F.log_softmax(logits, dim=-1)

            # --- B. 应用 Presence & Frequency Penalty (对齐 Benchmark) ---
            if alpha_presence > 0 or alpha_frequency > 0:
                mask = (occurences > 0).float()
                penalty = (mask * alpha_presence) + (occurences * alpha_frequency)
                logits = logits - penalty

            # --- C. Top-K 过滤 ---
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -1e30

            # --- D. 计算概率分布 ---
            probs = F.softmax(logits, dim=-1)

            # --- E. Top-P (Nucleus) 过滤 (对齐 Benchmark 健壮逻辑) ---
            if 0.0 < top_p < 1.0:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0 # 确保至少保留一个候选
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                probs[indices_to_remove] = 0
                probs = probs / probs.sum(dim=-1, keepdim=True)

            # --- F. Temperature 缩放 (Benchmark 风格: 在 Probs 上缩放) ---
            if temperature != 1.0 and temperature > 0:
                probs = torch.pow(probs, 1.0 / temperature)
                probs = probs / probs.sum(dim=-1, keepdim=True)

            # --- G. 采样 ---
            token_ids = torch.multinomial(probs, num_samples=1).squeeze(-1)
            
            # 提取被选中 Token 的原始对数概率 (用于计算策略梯度)
            picked_logp = raw_logp_all.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)

            # --- H. 更新惩罚计数器并衰减 ---
            occurences.scatter_add_(1, token_ids.view(-1, 1), torch.ones_like(token_ids.view(-1, 1), dtype=torch.float32))
            occurences *= alpha_decay

            # --- I. 状态掩码处理 ---
            token_ids = torch.where(active, token_ids, torch.zeros_like(token_ids))
            picked_logp = torch.where(active, picked_logp, torch.zeros_like(picked_logp))

            tok_cpu = token_ids.detach().cpu().tolist()
            lp_cpu = picked_logp.detach().cpu().tolist()

            for i in range(B):
                if active[i]:
                    comp_tokens[i].append(int(tok_cpu[i]))
                    old_logps[i].append(float(lp_cpu[i]))

            # --- J. 停止准则检查 ---
            if (stop_on_think_close or stop_on_user or stop_on_boxed) and (t % max(1, stop_check_every) == 0):
                for i in range(B):
                    if not active[i]: continue
                    
                    # decode full output to avoid missing earlier stop markers
                    w = comp_tokens[i]
                    s = self.decode(w)
                    
                    stop_hit = False
                    if stop_on_boxed and ("\\boxed{" in s and "}" in s.split("\\boxed{")[-1]):
                        stop_hit = True
                    elif stop_on_think_close and ("</think>" in s):
                        stop_hit = True
                    elif stop_on_user and any(tok in s for tok in STOP_TOKENS):
                        stop_hit = True
                    if stop_hit:
                        active[i] = False

            # --- K. 下一步模型推理 ---
            step_tokens_batch = [[int(x)] for x in tok_cpu]
            last_logits = self.infer_model.forward_batch(step_tokens_batch, state)
            if torch.is_tensor(last_logits) and last_logits.dim() == 3:
                last_logits = last_logits[:, -1, :]

        # 4. 结果包装
        for i in range(B):
            if active[i]:
                truncated[i] = True

        comp_text = [self.decode(x) for x in comp_tokens]
        return comp_tokens, old_logps, comp_text, truncated


# =========================================================
# Trainer
# =========================================================

class DAPOStateTuningTrainer:
    def __init__(
        self,
        train_model,
        infer_engine: AlbatrossBatchInference,
        encode_fn,
        decode_fn,
        train_data: List[Dict[str, Any]],
        test_data: List[Dict[str, Any]],
        out_dir: str,
        device: str,
        cfg: DAPOConfig,
        seed: int = 42,
        train_idx_map: Optional[Dict[int, int]] = None,
        train_orig_indices: Optional[List[int]] = None,
    ):
        self.model = train_model
        self.infer = infer_engine
        self.encode = encode_fn
        self.decode = decode_fn
        self.train_data = train_data
        self.test_data = test_data
        self.train_idx_map = train_idx_map or {}
        self.train_orig_indices = train_orig_indices or []
        self.out_dir = out_dir
        self.device = device
        self.cfg = cfg
        self.rng = random.Random(seed)
        self._correct_cnt = [0 for _ in range(len(self.train_data))]
        self._attempt_cnt = [0 for _ in range(len(self.train_data))]
        self._global_correct = 0
        self._global_attempts = 0
        self._p_sigma = 0.5 / math.sqrt(2.0 * math.log(4.0))
        self.stats_path = os.path.join(out_dir, "buffer_stats.json")
        self.train_ids_path = os.path.join(out_dir, "train_ids.jsonl")

        os.makedirs(out_dir, exist_ok=True)
        self.log_path = os.path.join(out_dir, "train.log")
        self.gen_dump_path = os.path.join(out_dir, "gen_judgements.jsonl")
        self.infer_check_path = os.path.join(out_dir, "infer_check.jsonl")
        self.eval_path = os.path.join(out_dir, "eval.jsonl")
        self.metrics_path = os.path.join(out_dir, "metrics.jsonl")
        
        # Progress tracker
        self.progress_log_path = os.path.join(out_dir, "progress.jsonl")


        self._hang_f = None
        if cfg.enable_faulthandler:
            try:
                import faulthandler
                hang_path = os.path.join(out_dir, "hang_tracebacks.log")
                self._hang_f = open(hang_path, "a", encoding="utf-8", buffering=1)
                faulthandler.enable(file=self._hang_f, all_threads=True)
                if float(cfg.hang_dump_s) > 0:
                    faulthandler.dump_traceback_later(float(cfg.hang_dump_s), repeat=True, file=self._hang_f)
            except Exception:
                self._hang_f = None

        params = [p for p in self.model.parameters() if p.requires_grad]
        if not params:
            raise RuntimeError("No trainable params (expected time_state only).")
        self.opt = torch.optim.Adam(params, lr=self.cfg.lr, betas=(0.9, 0.99), eps=1e-18)

        self._ts_init: Dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for n, p in self.model.named_parameters():
                if "time_state" in n:
                    self._ts_init[n] = p.detach().clone()

    def _log(self, msg: str):
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass

    def _load_answer_buffer(self, path: str):
        if not path or not os.path.isfile(path):
            return
        loaded = 0
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    idx = obj.get("problem_idx")
                    tokens = obj.get("tokens")
                    old_logps = obj.get("old_logps")
                    text = obj.get("text", "")
                    if idx is None or not isinstance(tokens, list) or not isinstance(old_logps, list):
                        continue
                    if len(tokens) != len(old_logps):
                        continue
                    idx = int(idx)
                    if self.train_idx_map:
                        if idx not in self.train_idx_map:
                            continue
                        idx = self.train_idx_map[idx]
                    self._update_answer_buffer(idx, tokens, old_logps, text)
                    loaded += 1
        except Exception:
            return
        self._log(f"[buffer] loaded {loaded} answers from {path}")

    def _save_answer_buffer(self, path: str):
        if not path:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for idx, entries in self._answer_buffer.items():
                for e in entries:
                    orig_idx = idx
                    if self.train_orig_indices and 0 <= idx < len(self.train_orig_indices):
                        orig_idx = int(self.train_orig_indices[idx])
                    f.write(json.dumps({
                        "problem_idx": int(orig_idx),
                        "tokens": e.get("tokens", []),
                        "old_logps": e.get("old_logps", []),
                        "text": e.get("text", ""),
                        "len": int(e.get("len", 0)),
                    }, ensure_ascii=False) + "\n")

    def _update_answer_buffer(self, problem_idx: int, comp_tokens: List[int], old_logps: List[float], text: str):
        if not comp_tokens:
            return
        entry = {
            "tokens": comp_tokens,
            "old_logps": old_logps,
            "text": text,
            "len": len(comp_tokens),
        }
        buf = self._answer_buffer.get(problem_idx)
        if not buf:
            self._answer_buffer[problem_idx] = [entry]
            return
        if not any(e["tokens"] == entry["tokens"] for e in buf):
            buf.append(entry)

    def _sample_from_buffer(self, buf: List[Dict[str, Any]]):
        if len(buf) == 1:
            return buf[0]
        weights = [math.exp(-self.cfg.buffer_length_weight * float(e["len"])) for e in buf]
        return self.rng.choices(buf, weights=weights, k=1)[0]

    def _posterior_rate(self, idx: int) -> float:
        if self._global_attempts <= 0:
            return 0.5
        num = float(self._correct_cnt[idx] + self._global_correct)
        den = float(self._attempt_cnt[idx] + self._global_attempts)
        if den <= 0:
            return 0.5
        p = num / den
        if p < 0.0:
            return 0.0
        if p > 1.0:
            return 1.0
        return p

    def _problem_weight(self, p: float) -> float:
        d = p - 0.5
        return math.exp(-(d * d) / (2.0 * self._p_sigma * self._p_sigma))

    def _sample_problem_idx(self, exclude: set) -> int:
        candidates = [i for i in range(len(self.train_data)) if i not in exclude]
        if not candidates:
            return self.rng.randrange(len(self.train_data))
        weights = [self._problem_weight(self._posterior_rate(i)) for i in candidates]
        return self.rng.choices(candidates, weights=weights, k=1)[0]

    def _update_stats(self, idx: int, correct: bool):
        self._attempt_cnt[idx] += 1
        self._global_attempts += 1
        if correct:
            self._correct_cnt[idx] += 1
            self._global_correct += 1

    def _write_stats(self, step: int):
        items = []
        for i in range(len(self.train_data)):
            orig_idx = i
            if self.train_orig_indices and 0 <= i < len(self.train_orig_indices):
                orig_idx = int(self.train_orig_indices[i])
            items.append({
                "problem_idx": int(orig_idx),
                "correct": int(self._correct_cnt[i]),
                "attempts": int(self._attempt_cnt[i]),
                "posterior": float(self._posterior_rate(i)),
            })
        payload = {
            "time": now_str(),
            "step": int(step),
            "global_correct": int(self._global_correct),
            "global_attempts": int(self._global_attempts),
            "items": items,
        }
        with open(self.stats_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)

    def _write_train_ids(self, step: int, indices: list, attempts: int):
        orig_indices = []
        for i in indices:
            oi = i
            if self.train_orig_indices and 0 <= i < len(self.train_orig_indices):
                oi = int(self.train_orig_indices[i])
            orig_indices.append(int(oi))
        append_jsonl(self.train_ids_path, {
            "time": now_str(),
            "step": int(step),
            "train_bsz": int(len(indices)),
            "attempts": int(attempts),
            "problem_indices": orig_indices,
        })

    def _time_state_stats(self):(self):
        mx = 0.0
        rms_sum = 0.0
        cnt = 0
        bad = False
        with torch.no_grad():
            for n, p in self.model.named_parameters():
                if "time_state" not in n:
                    continue
                if torch.isnan(p).any() or torch.isinf(p).any():
                    bad = True
                v = p.detach().float()
                mx = max(mx, float(v.abs().max().item()))
                rms_sum += float((v * v).mean().sqrt().item())
                cnt += 1
        return {"absmax": mx, "rms_avg": (rms_sum / max(1, cnt)), "bad": bad}

    def _pad_batch(self, seqs: List[List[int]], pad_id: int = 0) -> Tuple[torch.Tensor, List[int]]:
        lens = [len(s) for s in seqs]
        T = max(lens)
        B = len(seqs)
        x = torch.full((B, T), pad_id, dtype=torch.long, device=self.device)
        for i, s in enumerate(seqs):
            if s:
                x[i, :len(s)] = torch.tensor(s, dtype=torch.long, device=self.device)
        return x, lens

    def _compute_advantages(self, rewards: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        mean = rewards.mean()
        std = rewards.std(unbiased=False)
        return (rewards - mean) / (std + eps)

    def _ppo_clipped_objective(self, ratio: torch.Tensor, adv: torch.Tensor) -> torch.Tensor:
        unclipped = ratio * adv
        clipped = torch.clamp(ratio, 1.0 - self.cfg.eps_low, 1.0 + self.cfg.eps_high) * adv
        return torch.where(adv >= 0, torch.minimum(unclipped, clipped), torch.maximum(unclipped, clipped))

    def _maybe_adapt_lr_by_kl(self, approx_kl: float):
        if not self.cfg.adaptive_kl:
            return
        if approx_kl is None:
            return
        lr_now = float(self.opt.param_groups[0]["lr"])
        if approx_kl > self.cfg.target_kl * 2.0:
            new_lr = max(lr_now * 0.5, 1e-6)
            if new_lr < lr_now:
                self.opt.param_groups[0]["lr"] = new_lr
                self._log(f"[KL-ADAPT] kl={approx_kl:.6f} high -> lr {lr_now:.2e} -> {new_lr:.2e}")
        elif approx_kl < self.cfg.target_kl * 0.25:
            new_lr = min(lr_now * 1.1, self.cfg.lr)
            if new_lr > lr_now:
                self.opt.param_groups[0]["lr"] = new_lr
                self._log(f"[KL-ADAPT] kl={approx_kl:.6f} low  -> lr {lr_now:.2e} -> {new_lr:.2e}")

    @torch.no_grad()
    def _infer_once(self, problem: str, gt: str, max_new: int, temperature: float, top_p: float, top_k: int) -> Dict[str, Any]:
        prompt = build_prompt(problem)
        ids = self.encode(prompt)
        max_prompt_len = int(self.model.args.ctx_len) - int(max_new) - 4
        max_prompt_len = max(64, max_prompt_len)
        if len(ids) > max_prompt_len:
            ids = ids[-max_prompt_len:]

        comp_tokens, _, comp_texts, truncs = self.infer.generate_group_parallel(
            prompt_tokens_list=[ids],
            group_size=1,
            max_new_tokens=max_new,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop_on_think_close=self.cfg.stop_on_think_close,
            stop_on_user=self.cfg.stop_on_user,
            stop_on_boxed=self.cfg.stop_on_boxed,
            stop_check_every=max(1, self.cfg.stop_check_every // 2),
            stop_check_window=max(64, self.cfg.stop_check_window),
        )

        txt = comp_texts[0]
        trunc = bool(truncs[0])
        if trunc:
            r = 0.0
            jdbg = {
                "pred_extracted": None,
                "gt": normalize_answer(extract_answer(gt) or gt),
                "method": "truncated_skip_judge",
                "error": None,
                "pred_parsed": None,
                "gt_parsed": None,
                "correct": False,
                "truncated_forced_zero": True,
            }
        else:
            r, jdbg = judge_answer_dispatch(
                judge_type=self.cfg.judge_type,
                model_response=txt,
                gold_answer=gt,
                llm_judge_fn=judge_answer_llm if self.cfg.judge_type == "llm" else None,
            )

        return {
            "prompt": prompt,
            "completion": txt,
            "truncated": trunc,
            "reward": float(r),
            "judge": jdbg,
            "gen_len": len(comp_tokens[0]),
        }

    @torch.no_grad()
    def sanity_infer_check(self, step: int, n: int = 3):
        for _ in range(n):
            if not self.train_data:
                return
            ex = self.train_data[self.rng.randrange(len(self.train_data))]
            rec = self._infer_once(
                problem=ex.get("problem", ""),
                gt=str(ex.get("solution", "")),
                max_new=self.cfg.eval_max_new_tokens,
                temperature=self.cfg.temperature,
                top_p=self.cfg.top_p,
                top_k=self.cfg.top_k,
            )
            append_jsonl(self.infer_check_path, {
                "time": now_str(),
                "step": step,
                "problem": ex.get("problem", ""),
                "gt": str(ex.get("solution", "")),
                **rec
            })

    @torch.no_grad()
    def evaluate(self, step: int, dataset=None, tag: str = "eval"):
        eval_data = dataset if dataset is not None else (self.test_data if self.test_data else self.train_data)
        if not eval_data:
            self._log("WARN: empty eval set.")
            return

        self.model.eval()
        collect_details = (tag == "eval")
        total_correct = 0
        total_trunc = 0
        total_len = 0
        details = [] if collect_details else None

        chunk_size = int(self.cfg.eval_n) if int(self.cfg.eval_n) > 0 else len(eval_data)
        chunk_size = max(1, min(chunk_size, len(eval_data)))

        for start in range(0, len(eval_data), chunk_size):
            ex_list = eval_data[start:start + chunk_size]
            probs = [ex.get("problem", "") for ex in ex_list]
            gts = [str(ex.get("solution", "")) for ex in ex_list]
            prompt_strs = [build_prompt(p) for p in probs]

            prompt_tokens_list = []
            for ps in prompt_strs:
                ids = self.encode(ps)
                max_prompt_len = int(self.model.args.ctx_len) - int(self.cfg.eval_max_new_tokens) - 4
                max_prompt_len = max(64, max_prompt_len)
                if len(ids) > max_prompt_len:
                    ids = ids[-max_prompt_len:]
                prompt_tokens_list.append(ids)

            comp_tokens_flat, _, comp_text_flat, truncated_flat = self.infer.generate_group_parallel(
                prompt_tokens_list=prompt_tokens_list,
                group_size=1,
                max_new_tokens=self.cfg.eval_max_new_tokens,
                temperature=self.cfg.eval_temperature,
                top_p=self.cfg.eval_top_p,
                top_k=self.cfg.eval_top_k,
                stop_on_think_close=self.cfg.stop_on_think_close,
                stop_on_user=self.cfg.stop_on_user,
                stop_on_boxed=self.cfg.stop_on_boxed,
                stop_check_every=max(1, self.cfg.stop_check_every // 2),
                stop_check_window=max(64, self.cfg.stop_check_window),
            )

            for i in range(len(prompt_tokens_list)):
                txt = comp_text_flat[i]
                trunc = bool(truncated_flat[i])
                if trunc:
                    r = 0.0
                    jdbg = {
                        "pred_extracted": None,
                        "gt": normalize_answer(extract_answer(gts[i]) or gts[i]),
                        "method": "truncated_skip_judge",
                        "error": None,
                        "pred_parsed": None,
                        "gt_parsed": None,
                        "correct": False,
                        "truncated_forced_zero": True,
                    }
                else:
                    r, jdbg = judge_answer_dispatch(
                        judge_type=self.cfg.judge_type,
                        model_response=txt,
                        gold_answer=gts[i],
                        llm_judge_fn=judge_answer_llm if self.cfg.judge_type == "llm" else None,
                    )

                if r >= 0.5:
                    total_correct += 1
                if trunc:
                    total_trunc += 1

                gen_len = len(comp_tokens_flat[i])
                total_len += gen_len
                if collect_details:
                    details.append({
                        "problem": probs[i],
                        "gt": gts[i],
                        "prompt": prompt_strs[i],
                        "completion": txt,
                        "truncated": trunc,
                        "reward": float(r),
                        "judge": jdbg,
                        "gen_len": gen_len,
                    })

        eval_n = len(eval_data)
        acc = total_correct / max(1, eval_n)
        trunc_rate = total_trunc / max(1, eval_n)
        avg_len = total_len / max(1, eval_n)

        append_jsonl(self.eval_path, {
            "time": now_str(),
            "step": step,
            "tag": tag,
            "eval_n": eval_n,
            "acc": acc,
            "trunc_rate": trunc_rate,
            "avg_len": avg_len,
            "eval_temperature": self.cfg.eval_temperature,
            "eval_top_p": self.cfg.eval_top_p,
            "eval_top_k": self.cfg.eval_top_k,
            "eval_max_new_tokens": self.cfg.eval_max_new_tokens,
            "details": details,
        })

        append_jsonl(self.metrics_path, {
            "time": now_str(),
            "step": step,
            "tag": tag,
            "acc": acc,
            "trunc_rate": trunc_rate,
            "avg_len": avg_len,
        })
        self._log(f"[EVAL step {step}][{tag}] acc={acc:.3f} trunc={trunc_rate:.3f} avg_len={avg_len:.1f} "
                  f"(temp={self.cfg.eval_temperature}, top_p={self.cfg.eval_top_p}, max_new={self.cfg.eval_max_new_tokens})")

    def train(self, total_steps: int):
        self._log(f"train begin: steps={total_steps} rollout_n={self.cfg.rollout_n} train_bsz={self.cfg.train_bsz} "
                  f"lr={self.cfg.lr} kl_coef={self.cfg.kl_coef} l2={self.cfg.time_state_l2} clamp={self.cfg.time_state_clamp} "
                  f"judge_type={self.cfg.judge_type}")
        st0 = self._time_state_stats()
        self._log(f"time_state init: absmax={st0['absmax']:.6f} rms={st0['rms_avg']:.6f} bad={st0['bad']}")

        # Initialize progress tracker
        progress = ProgressTracker(total_steps, self.progress_log_path)

        for step in range(1, total_steps + 1):
            t0 = time.time()
            train_bsz = int(self.cfg.train_bsz)
            sample_cnt = 0
            sample_correct = 0
            sample_trunc = 0
            sample_len_total = 0

            selected = set()
            trajs = []
            attempts = 0

            while len(selected) < train_bsz:
                data_idx = self._sample_problem_idx(selected)
                ex = self.train_data[data_idx]
                prompt = build_prompt(ex.get("problem", ""))
                prompt_tokens = self.encode(prompt)
                max_prompt_len = int(self.model.args.ctx_len) - int(self.cfg.max_new_tokens) - 4
                max_prompt_len = max(64, max_prompt_len)
                if len(prompt_tokens) > max_prompt_len:
                    prompt_tokens = prompt_tokens[-max_prompt_len:]

                comp_tokens_flat, old_logps_flat, comp_text_flat, truncated_flat = self.infer.generate_group_parallel(
                    prompt_tokens_list=[prompt_tokens],
                    group_size=1,
                    max_new_tokens=self.cfg.max_new_tokens,
                    temperature=self.cfg.rollout_temperature,
                    top_p=self.cfg.rollout_top_p,
                    top_k=self.cfg.top_k,
                    stop_on_think_close=self.cfg.stop_on_think_close,
                    stop_on_user=self.cfg.stop_on_user,
                    stop_on_boxed=self.cfg.stop_on_boxed,
                    stop_check_every=self.cfg.stop_check_every,
                    stop_check_window=self.cfg.stop_check_window,
                )

                ctoks = comp_tokens_flat[0]
                ologp = old_logps_flat[0]
                ctext = comp_text_flat[0]
                trunc = bool(truncated_flat[0])

                attempts += 1
                sample_cnt += 1
                sample_len_total += len(ctoks)

                if trunc:
                    r = 0.0
                    jdbg = {
                        "pred_extracted": None,
                        "gt": normalize_answer(extract_answer(str(ex.get("solution", ""))) or str(ex.get("solution", ""))),
                        "method": "truncated_skip_judge",
                        "error": None,
                        "pred_parsed": None,
                        "gt_parsed": None,
                        "correct": False,
                        "truncated_forced_zero": True,
                    }
                else:
                    r, jdbg = judge_answer_dispatch(
                        judge_type=self.cfg.judge_type,
                        model_response=ctext,
                        gold_answer=str(ex.get("solution", "")),
                        llm_judge_fn=judge_answer_llm if self.cfg.judge_type == "llm" else None,
                    )

                if r >= 0.5:
                    sample_correct += 1
                if trunc:
                    sample_trunc += 1

                self._update_stats(data_idx, r >= 0.5)

                individual_response = {
                    "time": now_str(),
                    "step": step,
                    "prompt_idx": 0,
                    "group_idx": 0,
                    "problem": ex.get("problem", ""),
                    "response": ctext,
                    "reward": float(r),
                    "is_truncated": trunc,
                    "judge_detail": jdbg,
                }
                res_path = os.path.join(self.out_dir, "responses.jsonl")
                with open(res_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(individual_response, ensure_ascii=False) + "
")

                append_jsonl(self.gen_dump_path, {
                    "time": now_str(),
                    "step": step,
                    "problem_idx": int(data_idx),
                    "problem": ex.get("problem", ""),
                    "solution": str(ex.get("solution", "")),
                    "prompt": prompt,
                    "group_size": 1,
                    "max_new_tokens": self.cfg.max_new_tokens,
                    "sample": {
                        "text": ctext,
                        "truncated": trunc,
                        "reward": float(r),
                        "judge": jdbg,
                    },
                })

                if r >= 0.5 and (not trunc) and (data_idx not in selected):
                    selected.add(data_idx)
                    full = prompt_tokens + ctoks
                    trajs.append({
                        "full_tokens": full,
                        "prompt_len": len(prompt_tokens),
                        "comp_len": len(ctoks),
                        "old_logps": ologp,
                        "adv": 1.0,
                        "reward": 1.0,
                    })

            self._write_train_ids(step, list(selected), attempts)

            train_tries = attempts

            total_comp_tokens = sum(int(tr["comp_len"]) for tr in trajs)
            if total_comp_tokens <= 0:
                self._log("WARN: total_comp_tokens=0.")
                continue

            last_loss = None
            last_kl = None
            last_clipfrac = None
            last_grad = None

            trajs_sorted = sorted(trajs, key=lambda x: len(x["full_tokens"]), reverse=True)

            for _ep in range(self.cfg.ppo_epochs):
                self.model.train()
                self.opt.zero_grad(set_to_none=True)

                approx_kl_sum = 0.0
                approx_kl_cnt = 0
                clip_hits = 0.0
                clip_cnt = 0

                mb = max(1, int(self.cfg.micro_batch))
                for s in range(0, len(trajs_sorted), mb):
                    batch = trajs_sorted[s:s + mb]
                    seqs = [b["full_tokens"] for b in batch]
                    padded, _ = self._pad_batch(seqs, pad_id=0)

                    inp = padded[:, :-1].contiguous()
                    tgt = padded[:, 1:].contiguous()

                    logits = self.model(inp)
                    if torch.is_tensor(logits) and logits.dim() == 2:
                        logits = logits.unsqueeze(0)

                    logp = F.log_softmax(logits.float(), dim=-1)
                    picked = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).contiguous()

                    obj_sum = 0.0
                    kl_sum_tok = 0.0
                    ts_reg = 0.0

                    for bi, tr in enumerate(batch):
                        prompt_len = int(tr["prompt_len"])
                        comp_len = int(tr["comp_len"])
                        old_lp = torch.tensor(tr["old_logps"], device=self.device, dtype=torch.float32)
                        adv = torch.tensor(tr["adv"], device=self.device, dtype=torch.float32)

                        start = prompt_len - 1
                        end = start + comp_len

                        new_lp = picked[bi, start:end].to(torch.float32)

                        if new_lp.numel() != old_lp.numel():
                            m = min(new_lp.numel(), old_lp.numel())
                            new_lp = new_lp[:m]
                            old_lp = old_lp[:m]
                            if m <= 0:
                                continue

                        log_ratio = new_lp - old_lp
                        ratio = torch.exp(log_ratio)

                        obj = self._ppo_clipped_objective(ratio, adv)
                        obj_sum = obj_sum + obj.sum()

                        kl_tok = 0.5 * (log_ratio ** 2)
                        kl_sum_tok = kl_sum_tok + kl_tok.sum()

                        with torch.no_grad():
                            approx_kl_sum += float(kl_tok.mean().item())
                            approx_kl_cnt += 1
                            clipped = (ratio < (1.0 - self.cfg.eps_low)) | (ratio > (1.0 + self.cfg.eps_high))
                            clip_hits += float(clipped.float().mean().item())
                            clip_cnt += 1

                    if self.cfg.time_state_l2 > 0:
                        for n, p in self.model.named_parameters():
                            if "time_state" in n:
                                ts_reg = ts_reg + (p.float() - self._ts_init[n].float()).pow(2).mean()

                    loss = -(obj_sum / float(total_comp_tokens))
                    loss = loss + float(self.cfg.kl_coef) * (kl_sum_tok / float(total_comp_tokens))
                    if self.cfg.time_state_l2 > 0:
                        loss = loss + float(self.cfg.time_state_l2) * ts_reg

                    loss.backward()
                    last_loss = float(loss.detach().item())

                if self.cfg.grad_clip and self.cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad],
                        self.cfg.grad_clip
                    )

                with torch.no_grad():
                    g2 = 0.0
                    for p in self.model.parameters():
                        if p.requires_grad and p.grad is not None:
                            g = p.grad.detach().float()
                            g2 += float((g.norm(2) ** 2).item())
                    last_grad = math.sqrt(g2)

                self.opt.step()

                if self.cfg.time_state_clamp and self.cfg.time_state_clamp > 0:
                    cv = float(self.cfg.time_state_clamp)
                    with torch.no_grad():
                        for n, p in self.model.named_parameters():
                            if "time_state" in n:
                                p.data.clamp_(-cv, cv)

                last_kl = float(approx_kl_sum / max(1, approx_kl_cnt))
                last_clipfrac = float(clip_hits / max(1, clip_cnt))
                self._maybe_adapt_lr_by_kl(last_kl)

            avg_r = sample_correct / max(1, sample_cnt)
            dt = time.time() - t0
            st = self._time_state_stats()
            lr_now = float(self.opt.param_groups[0]["lr"])
            
            # Update progress tracker
            trunc_rate = sample_trunc / max(1, sample_cnt)
            avg_len = sample_len_total / max(1, sample_cnt)
            progress_metrics = {
                "acc": sample_correct / max(1, sample_cnt),
                "trunc_rate": trunc_rate,
                "avg_len": avg_len,
                "avg_reward": avg_r,
                "loss": last_loss if last_loss else 0.0,
                "grad": last_grad if last_grad else 0.0,
                "approx_kl": last_kl if last_kl else 0.0,
                "clip_frac": last_clipfrac if last_clipfrac else 0.0,
                "absmax": st["absmax"],
                "rms": st["rms_avg"],
            }
            progress.update(step, progress_metrics)
            append_jsonl(self.metrics_path, {
                "time": now_str(),
                "step": step,
                "tag": "train",
                "acc": progress_metrics["acc"],
                "trunc_rate": trunc_rate,
                "avg_len": avg_len,
            })

            if step % self.cfg.log_interval == 0:
                global_rate = self._global_correct / max(1, self._global_attempts)
                self._log(
                    f"[step {step}/{total_steps}] "
                    f"train_bsz={train_bsz} attempts={attempts} unique_correct={len(trajs)} | "
                    f"samples={sample_cnt} acc={sample_correct/max(1,sample_cnt):.3f} trunc={sample_trunc/max(1,sample_cnt):.3f} | "
                    f"global={self._global_correct}/{self._global_attempts} p={global_rate:.3f} | "
                    f"avg_reward={avg_r:.4f} loss={last_loss} grad={last_grad:.3f} "
                    f"approx_kl={last_kl:.6f} clip_frac={last_clipfrac:.3f} | "
                    f"ts(absmax={st['absmax']:.4f}, rms={st['rms_avg']:.4f}, bad={st['bad']}) | "
                    f"hp(lr={lr_now:.2e}, kl_coef={self.cfg.kl_coef}, l2={self.cfg.time_state_l2}, clamp={self.cfg.time_state_clamp}, "
                    f"temp={self.cfg.rollout_temperature}, top_p={self.cfg.rollout_top_p}, max_new={self.cfg.max_new_tokens}) "
                    f"step_time={dt:.1f}s"
                )

            if step % self.cfg.infer_check_interval == 0:
 == 0:
                self.model.eval()
                self.sanity_infer_check(step=step, n=3)

            if step % self.cfg.eval_interval == 0:
                self.model.eval()
                self.evaluate(step=step)

            if step % self.cfg.save_interval == 0 or step == total_steps:
                ckpt_path = os.path.join(self.out_dir, f"ckpt_step{step}.pth")
                torch.save({
                    "time": now_str(),
                    "step": step,
                    "cfg": self.cfg.__dict__,
                    "time_state": {n: p.detach().cpu() for n, p in self.model.named_parameters() if "time_state" in n},
                }, ckpt_path)

                latest_ts_path = os.path.join(self.out_dir, "latest_time_state.pth")
                save_time_state_only(self.model, latest_ts_path)

                self._log(f"saved: {ckpt_path}")
                self._log(f"saved: {latest_ts_path}")

            self._write_stats(step)

        progress.finish()

        if self.cfg.enable_faulthandler and float(self.cfg.hang_dump_s) > 0:
            try:
                import faulthandler
                faulthandler.cancel_dump_traceback_later()
            except Exception:
                pass

        self._log("train end.")


# =========================================================
# Sanity check
# =========================================================

@torch.no_grad()
def sanity_check_train_vs_albatross(train_model, infer_model, infer_engine, encode, decode, device="cuda"):
    prompt = "User: What is 2+2? think\nAssistant: <think>\n"
    ids = encode(prompt)
    if not ids:
        raise RuntimeError("encode(prompt) returned empty")

    t = torch.tensor([ids], device=device, dtype=torch.long)
    logits_train = train_model(t)[0, -1].float()

    state = infer_engine.init_state_with_time_state(B=1)
    out2 = infer_model.forward_batch([ids], state)
    if torch.is_tensor(out2) and out2.dim() == 3:
        out2 = out2[:, -1, :]
    logits_infer = out2[0].float()

    top_train = int(torch.argmax(logits_train).item())
    top_infer = int(torch.argmax(logits_infer).item())

    print("[sanity] top1_train =", top_train, "->", repr(decode([top_train])), flush=True)
    print("[sanity] top1_infer =", top_infer, "->", repr(decode([top_infer])), flush=True)

    if top_train != top_infer:
        raise RuntimeError("SANITY FAIL: train vs infer top1 mismatch (check model name / tokenizer / env vars).")


# =========================================================
# Main
# =========================================================

def read_jsonl(path: str, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    """读取 JSONL 文件，支持可选的样本数量限制"""
    data = []
    if not os.path.exists(path):
        return data
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_samples is not None and i >= max_samples:
                break
            line = line.strip()
            if not line: continue
            data.append(json.loads(line))
    return data

# =========================================================
# 2. 进度条选取与可视化功能
# =========================================================

class ProgressTracker:
    """实时进度跟踪与可视化：显示进度条、ETA、已用时间和自定义指标"""
    
    def __init__(self, total_steps: int, log_path: str):
        self.total_steps = total_steps
        self.log_path = log_path
        self.start_time = time.time()
        self.step_times = []
        
    def _format_time(self, seconds: float) -> str:
        hours, rem = divmod(int(seconds), 3600)
        minutes, secs = divmod(rem, 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}" if hours > 0 else f"{minutes:02d}:{secs:02d}"

    def update(self, step: int, metrics: Dict[str, Any]):
        elapsed = time.time() - self.start_time
        self.step_times.append(elapsed)
        
        # 计算 ETA (预计剩余时间)
        if len(self.step_times) > 1:
            avg_time_per_step = (self.step_times[-1] - self.step_times[0]) / (len(self.step_times) - 1)
            eta = avg_time_per_step * (self.total_steps - step)
            eta_str = self._format_time(eta)
        else:
            eta_str = "--:--"
        
        # 动态进度条绘制
        progress = step / self.total_steps
        bar_len = 30
        filled = int(bar_len * progress)
        bar = "#" * filled + "-" * (bar_len - filled)
        
        # 指标字符串
        metric_str = " | ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" 
                                 for k, v in metrics.items()])
        
        # 打印至终端 (\r 实现覆盖当前行)
        print(f"\r进度: [{bar}] {progress*100:5.1f}% | 步数: {step}/{self.total_steps} | "
              f"耗时: {self._format_time(elapsed)} | ETA: {eta_str} | {metric_str}", end="", flush=True)
        
        # 写入日志文件
        log_entry = {"step": step, "time": self._format_time(elapsed), **metrics}
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")

    def finish(self):
        print(f"\n训练完成！总耗时: {self._format_time(time.time() - self.start_time)}")

# =========================================================
# 3. 完整的 Main 函数
# =========================================================

def main():
    ap = argparse.ArgumentParser(description="RLVR v2 - Enhanced Training Script")
    # --- 基础与路径参数 ---
    ap.add_argument("--model", type=str, required=True, help="model path, with or without .pth")
    ap.add_argument("--train_jsonl", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)

    # --- 数据控制 (新增功能) ---
    ap.add_argument("--max_data_samples", type=int, default=None, 
                    help="Dataset excerpt: only use the first n samples (None = all)")

    # --- 训练核心参数 ---
    ap.add_argument("--total_steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ctx_len", type=int, default=8192)

    # --- 采样与推理参数 ---
    ap.add_argument("--batch_prompts", type=int, default=1)
    ap.add_argument("--group_size", type=int, default=16)
    ap.add_argument("--rollout_n", type=int, default=32)
    ap.add_argument("--train_bsz", type=int, default=32)
    ap.add_argument("--max_new_tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--top_p", type=float, default=0.4)
    ap.add_argument("--rollout_temperature", type=float, default=1.0)
    ap.add_argument("--rollout_top_p", type=float, default=0.6)
    ap.add_argument("--top_k", type=int, default=500)
    ap.add_argument("--mask_token0", action="store_true")
    ap.add_argument("--dynamic_sampling_max_tries", type=int, default=200)
    ap.add_argument("--collect_chunk", type=int, default=4)
    ap.add_argument("--presence_penalty", type=float, default=0.5)
    ap.add_argument("--frequency_penalty", type=float, default=0.1)
    ap.add_argument("--alpha_decay", type=float, default=0.99)
    ap.add_argument("--buffer_length_weight", type=float, default=0.001)
    ap.add_argument("--buffer_save_path", type=str, default=None)
    ap.add_argument("--buffer_load_path", type=str, default=None)
    ap.add_argument("--buffer_save_interval", type=int, default=5)
    ap.add_argument("--buffer_cold_path", type=str, default=None)
    ap.add_argument("--buffer_min_init", type=int, default=None, help="min buffer items before training (default=train_bsz)")
    ap.add_argument("--buffer_warmup_rounds", type=int, default=1, help="max warmup rollout rounds before training")

    # --- 优化器与 PPO 参数 ---
    ap.add_argument("--ppo_epochs", type=int, default=1)
    ap.add_argument("--micro_batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--eps_low", type=float, default=0.2)
    ap.add_argument("--eps_high", type=float, default=0.5)
    ap.add_argument("--grad_clip", type=float, default=0.2)

    # --- KL 散度与状态正则化 ---
    ap.add_argument("--kl_coef", type=float, default=0.08)
    ap.add_argument("--target_kl", type=float, default=0.01)
    ap.add_argument("--adaptive_kl", action="store_true")
    ap.add_argument("--time_state_l2", type=float, default=1e-6)
    ap.add_argument("--time_state_clamp", type=float, default=5.0)

    # --- 记录与评估周期 ---
    ap.add_argument("--log_interval", type=int, default=1)
    ap.add_argument("--save_interval", type=int, default=50)
    ap.add_argument("--infer_check_interval", type=int, default=50)
    ap.add_argument("--eval_interval", type=int, default=5)
    ap.add_argument("--eval_n", type=int, default=16)
    ap.add_argument("--eval_temperature", type=float, default=None)
    ap.add_argument("--eval_top_p", type=float, default=None)
    ap.add_argument("--eval_top_k", type=int, default=None)
    ap.add_argument("--eval_max_new_tokens", type=int, default=None)
    ap.add_argument("--run_full_pre", action="store_true", help="run full dataset eval before training")

    # --- 其他 ---
    ap.add_argument("--state_init", type=str, default=None, help="optional: load time_state-only checkpoint")
    ap.add_argument("--tokenizer", type=str, default="reference/rwkv_vocab_v20230424.txt")
    ap.add_argument("--enable_faulthandler", action="store_true")
    ap.add_argument("--hang_dump_s", type=float, default=0.0)
    ap.add_argument("--judge_type", type=str, default="rule", choices=["llm", "rule"],
                    help="Reward judge type: llm (LLM-as-judge) or rule (extract+compare)")
    ap.add_argument("--stop_on_think_close", action="store_true")
    ap.add_argument("--stop_on_user", action="store_true", default=True)
    ap.add_argument("--stop_on_boxed", action="store_true", default=True)
    ap.add_argument("--no_stop_on_think_close", action="store_false", dest="stop_on_think_close")
    ap.add_argument("--no_stop_on_user", action="store_false", dest="stop_on_user")
    ap.add_argument("--no_stop_on_boxed", action="store_false", dest="stop_on_boxed")
    ap.add_argument("--stop_check_every", type=int, default=1)
    ap.add_argument("--stop_check_window", type=int, default=128)
    args = ap.parse_args()

    if args.buffer_min_init is None:
        args.buffer_min_init = args.train_bsz

    # --- 环境初始化 ---
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    os.environ["RWKV_HEAD_SIZE_A"] = str(HEAD_SIZE)
    os.environ["RWKV_MY_TESTING"] = "x070"
    os.environ["RWKV_TRAIN_TYPE"] = "state"
    os.environ["RWKV_CTXLEN"] = str(int(args.ctx_len))
    os.environ["FUSED_KERNEL"] = "0"
    os.environ["WKV"] = "cuda"

    os.makedirs(args.out_dir, exist_ok=True)

    # --- 功能实现 1: 选取数据集前 n 条 ---
    print(f"Loading data from {args.train_jsonl}...")
    data = read_jsonl(args.train_jsonl, max_samples=args.max_data_samples)
    if not data:
        raise RuntimeError("empty train_jsonl or max_samples resulted in no data")
    print(f"Data loaded: {len(data)} samples selected.")
    if len(data) < 2:
        raise RuntimeError("need at least 2 samples to split train/test")

    split_seed = int(args.seed)
    split_rng = random.Random(split_seed)
    idxs = list(range(len(data)))
    split_rng.shuffle(idxs)
    test_size = min(128, len(data) - 1)
    test_idxs = idxs[:test_size]
    train_idxs = idxs[test_size:]
    train_data = [data[i] for i in train_idxs]
    test_data = [data[i] for i in test_idxs]
    print(f"Split done: train={len(train_data)} test={len(test_data)} (seed={split_seed})")

    # --- 模型与分词器加载 ---
    from reference.utils import TRIE_TOKENIZER
    tok = TRIE_TOKENIZER(args.tokenizer)
    encode = lambda s: tok.encode(s)

    def safe_decode(ids):
        try:
            return tok.decode(ids, utf8_errors="replace")
        except:
            try: return tok.decode(ids)
            except:
                try:
                    b = tok.decodeBytes(ids)
                    return b.decode("utf-8", errors="replace")
                except:
                    return "".join(chr(int(x) % 256) for x in ids)
    decode = safe_decode

    base_name, pth_path = normalize_model_arg(args.model)
    if not os.path.isfile(pth_path):
        raise FileNotFoundError(f"Cannot find model pth: {pth_path}")

    train_model, _ = load_train_model_rwkv7_cuda(pth_path, device=device, ctx_len=int(args.ctx_len))
    infer_model, _ = load_infer_model_albatross(base_name)

    trainable = freeze_except_time_state(train_model)
    if trainable <= 0:
        raise RuntimeError("No trainable time_state found.")

    if args.state_init:
        ok = load_time_state_only(train_model, args.state_init)
        print(f"[state_init] loaded={ok} from {args.state_init}")

    # --- 配置构建 ---
    buffer_save_path = args.buffer_save_path
    if buffer_save_path is None:
        buffer_save_path = os.path.join(args.out_dir, "answer_buffer.jsonl")
    buffer_cold_path = args.buffer_cold_path
    if buffer_cold_path is None:
        buffer_cold_path = os.path.join(args.out_dir, "answer_buffer_cold.jsonl")

    cfg = DAPOConfig(
        batch_prompts=int(args.batch_prompts),
        group_size=int(args.group_size),
        rollout_n=int(args.rollout_n),
        train_bsz=int(args.train_bsz),
        max_new_tokens=int(args.max_new_tokens),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        rollout_temperature=float(args.rollout_temperature),
        rollout_top_p=float(args.rollout_top_p),
        top_k=int(args.top_k),
        buffer_length_weight=float(args.buffer_length_weight),
        buffer_save_path=str(buffer_save_path) if buffer_save_path else None,
        buffer_load_path=str(args.buffer_load_path) if args.buffer_load_path else None,
        buffer_save_interval=int(args.buffer_save_interval),
        buffer_cold_path=str(buffer_cold_path) if buffer_cold_path else None,
        buffer_min_init=int(args.buffer_min_init),
        buffer_warmup_rounds=int(args.buffer_warmup_rounds),
        mask_token0=bool(args.mask_token0),
        dynamic_sampling_max_tries=int(args.dynamic_sampling_max_tries),
        collect_chunk=int(args.collect_chunk),
        ppo_epochs=int(args.ppo_epochs),
        micro_batch=int(args.micro_batch),
        lr=float(args.lr),
        eps_low=float(args.eps_low),
        eps_high=float(args.eps_high),
        grad_clip=float(args.grad_clip),
        kl_coef=float(args.kl_coef),
        target_kl=float(args.target_kl),
        adaptive_kl=bool(args.adaptive_kl),
        time_state_l2=float(args.time_state_l2),
        time_state_clamp=float(args.time_state_clamp),
        log_interval=int(args.log_interval),
        save_interval=int(args.save_interval),
        infer_check_interval=int(args.infer_check_interval),
        eval_interval=int(args.eval_interval),
        eval_n=int(args.eval_n),
        eval_temperature=float(args.eval_temperature) if args.eval_temperature is not None else float(args.temperature),
        eval_top_p=float(args.eval_top_p) if args.eval_top_p is not None else float(args.top_p),
        eval_top_k=int(args.eval_top_k) if args.eval_top_k is not None else int(args.top_k),
        eval_max_new_tokens=int(args.eval_max_new_tokens) if args.eval_max_new_tokens is not None else int(args.max_new_tokens),
        enable_faulthandler=bool(args.enable_faulthandler),
        hang_dump_s=float(args.hang_dump_s),
        presence_penalty=float(args.presence_penalty),
        frequency_penalty=float(args.frequency_penalty),
        alpha_decay=float(args.alpha_decay),
        judge_type=str(args.judge_type),
        stop_on_think_close=bool(args.stop_on_think_close),
        stop_on_user=bool(args.stop_on_user),
        stop_on_boxed=bool(args.stop_on_boxed),
        stop_check_every=int(args.stop_check_every),
        stop_check_window=int(args.stop_check_window),
    )

    # --- 引擎与 Trainer 初始化 ---
    infer_engine = AlbatrossBatchInference(infer_model, train_model, encode, decode, device=device, cfg=cfg)
    
    # Sanity Check
    sanity_check_train_vs_albatross(train_model, infer_model, infer_engine, encode, decode, device=device)

    train_idx_map = {orig_idx: i for i, orig_idx in enumerate(train_idxs)}
    trainer = DAPOStateTuningTrainer(
        train_model=train_model,
        infer_engine=infer_engine,
        encode_fn=encode,
        decode_fn=decode,
        train_data=train_data,
        test_data=test_data,
        train_idx_map=train_idx_map,
        train_orig_indices=train_idxs,
        out_dir=args.out_dir,
        device=device,
        cfg=cfg,
        seed=int(args.seed),
    )

    # --- 功能实现 2: 进度条功能 ---
    # 我们为 trainer 注入一个 ProgressTracker 
    log_file = os.path.join(args.out_dir, f"train_log_{now_str()}.jsonl")
    tracker = ProgressTracker(total_steps=int(args.total_steps), log_path=log_file)
    
    # 覆盖 trainer 的 log 逻辑或在训练循环外围观察
    # 这里建议修改 trainer.train 方法的实现，或者如果 trainer 支持 callback，则在此调用。
    # 假设 trainer.train 内部有打印逻辑，我们可以通过 tracker 让它更美观。
    
    print("\n" + "="*50)
    print(f"STARTING RLVR TRAINING | Steps: {args.total_steps}")
    print("="*50 + "\n")

    try:
        # 如果 trainer.train 内部已经写死，你可以通过修改 trainer 类来配合 tracker
        # 或者在这里直接启动，并在终端显示最终耗时
        print(f"\n[RLVR_v2] Initialized. Training on {len(train_data)} samples (test={len(test_data)}).\n")
        trainer.model.eval()
        if args.run_full_pre:
            trainer.evaluate(step=0, dataset=data, tag="full_pre")
        else:
            print("[eval] skip full_pre (use --run_full_pre to enable)")
        trainer.train(total_steps=int(args.total_steps))
        trainer.model.eval()
        trainer.evaluate(step=int(args.total_steps), dataset=data, tag="full_post")
    except KeyboardInterrupt:
        print("\n[Terminated] Training interrupted by user.")
    finally:
        tracker.finish()

if __name__ == "__main__":
    main()
