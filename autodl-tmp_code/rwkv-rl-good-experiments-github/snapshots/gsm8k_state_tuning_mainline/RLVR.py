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

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
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
# Prompt (keep 'think'!)
# =========================================================

def build_prompt(problem: str) -> str:
    p = (problem or "").strip()
    # 你说换行不关键，这里用最简单的 1 个换行，且保留 "think"
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

def _latex_to_sympyish(s: str) -> str:
    if s is None:
        return ""
    s = _strip_math_delims(s)
    s = s.replace(r"\cdot", "*").replace(r"\times", "*")
    s = s.replace("^", "**")
    s = s.replace(r"\pi", "pi")
    s = s.replace(r"\infty", "oo").replace("∞", "oo")
    s = s.replace("−", "-")
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)

    # \frac{a}{b}
    while True:
        idx = s.find(r"\frac{")
        if idx < 0:
            break
        brace1 = idx + len(r"\frac")
        got1 = _find_balanced_brace(s, brace1)
        if got1 is None:
            break
        a, end1 = got1
        if end1 + 1 >= len(s) or s[end1 + 1] != "{":
            break
        got2 = _find_balanced_brace(s, end1 + 1)
        if got2 is None:
            break
        b, end2 = got2
        s = s[:idx] + f"(({_latex_to_sympyish(a)})/({_latex_to_sympyish(b)}))" + s[end2 + 1:]

    # \sqrt{a}
    while True:
        idx = s.find(r"\sqrt{")
        if idx < 0:
            break
        brace = idx + len(r"\sqrt")
        got = _find_balanced_brace(s, brace)
        if got is None:
            break
        inner, end = got
        s = s[:idx] + f"sqrt({_latex_to_sympyish(inner)})" + s[end + 1:]

    s = s.replace("\\", "")
    return s.strip()

# =========================================================
# Zhipu API judge (via OpenAI-compatible API)
#   - API URL: https://www.packyapi.com/v1/chat/completions
#   - Pass full output to judge instead of extracting answer
# =========================================================

import requests

ZHIPU_API_URL = "https://www.packyapi.com/v1/chat/completions"
ZHIPU_API_KEY = ""


def judge_with_zhipu(pred_full_output: str, gt: str) -> Tuple[bool, str]:
    """
    Use Zhipu API to judge if the prediction is correct.
    Pass full output to the model for judging.
    """
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


def judge_answer(pred_text: str, gt_text: str) -> Tuple[float, Dict[str, Any]]:
    """
    Judge correctness using Zhipu API.
    Pass full output to judge instead of extracting answer first.

    Returns:
        reward (float): 1.0 if correct else 0.0
        dbg (dict): debug info
    """
    gt_ans = _strip_math_delims(gt_text)

    dbg: Dict[str, Any] = {
        "pred_full_output": pred_text[:500] if pred_text else None,  # truncate for logging
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

    # -------- LLM-based judging (Zhipu API) --------
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
########################################################################################################
# 答案提取和验证逻辑
########################################################################################################

def extract_answer(text):
    """
    从模型输出中提取答案 - 针对 \\boxed{} 格式优化
    
    Args:
        text: 模型输出的文本
    
    Returns:
        str: 提取的答案，如果提取失败返回 None
    """
    if not text or not isinstance(text, str):
        return None
    
    text = text.strip()
    
    # 1. 优先匹配 \boxed{数字}
    boxed_patterns = [
        r'\\boxed\{([^}]+)\}',           # 标准格式
        r'\\boxed\s*\{([^}]+)\}',        # 允许空格
        r'boxed\{([^}]+)\}',             # 缺少反斜杠
        r'\{([0-9,\.\-\s]+)\}',          # 只有花括号
    ]
    
    for pattern in boxed_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            # 取最后一个匹配（如果有多个）
            answer = matches[-1].strip()
            # 从答案中提取数字
            numbers = re.findall(r'-?\d+\.?\d*', answer.replace(',', ''))
            if numbers:
                return numbers[-1]  # 返回最后一个数字
    
    # 2. 尝试匹配 #### 后面的答案（GSM8K标准格式）
    if '####' in text:
        after_hash = text.split('####')[-1].strip()
        numbers = re.findall(r'-?\d+\.?\d*', after_hash.replace(',', ''))
        if numbers:
            return numbers[0]
    
    # 3. 如果没有找到 boxed，尝试提取最后一行的数字
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    if lines:
        last_line = lines[-1]
        numbers = re.findall(r'-?\d+\.?\d*', last_line.replace(',', ''))
        if numbers:
            return numbers[-1]
    
    # 4. 兜底：提取全文最后一个数字
    all_numbers = re.findall(r'-?\d+\.?\d*', text.replace(',', ''))
    if all_numbers:
        return all_numbers[-1]
    
    return None


def normalize_answer(answer):
    """
    标准化答案格式
    
    Args:
        answer: 原始答案字符串或数字
    
    Returns:
        str: 标准化后的答案字符串
    """
    if answer is None:
        return None
    
    # 清理常见格式字符
    answer_str = str(answer).strip()
    answer_str = answer_str.replace(',', '').replace('$', '').replace('%', '')
    answer_str = answer_str.replace('\\', '').replace('{', '').replace('}', '')
    
    # 移除可能的文本后缀（如 "dollars", "meters" 等）
    answer_str = re.sub(r'[a-zA-Z\s]+$', '', answer_str).strip()
    
    # 尝试转换为数字
    try:
        num = float(answer_str)
        
        # 处理整数（460.0 -> 460）
        if abs(num - round(num)) < 1e-9:
            return str(int(round(num)))
        
        # 处理小数（保留必要精度）
        formatted = f"{num:.10f}".rstrip('0').rstrip('.')
        return formatted
        
    except (ValueError, TypeError):
        # 无法转换为数字，返回清理后的字符串
        return answer_str.strip()


def compare_answers(pred, gold, tolerance=1e-6):
    """
    比较两个答案是否相等
    
    Args:
        pred: 预测答案
        gold: 正确答案
        tolerance: 数值比较的容差
    
    Returns:
        bool: 是否相等
    """
    if pred is None or gold is None:
        return False
    
    # 标准化
    pred_norm = normalize_answer(pred)
    gold_norm = normalize_answer(gold)
    
    if pred_norm is None or gold_norm is None:
        return False
    
    # 1. 字符串完全匹配
    if pred_norm == gold_norm:
        return True
    
    # 2. 数值比较
    try:
        pred_num = float(pred_norm)
        gold_num = float(gold_norm)
        
        # 绝对误差
        abs_diff = abs(pred_num - gold_num)
        if abs_diff < tolerance:
            return True
        
        # 相对误差（避免除零）
        if abs(gold_num) > tolerance:
            rel_diff = abs_diff / abs(gold_num)
            if rel_diff < tolerance:
                return True
        
        return False
        
    except (ValueError, TypeError):
        pass
    
    # 3. 大小写不敏感的字符串比较
    return pred_norm.lower() == gold_norm.lower()


def verify_gsm8k_answer(model_response, correct_answer, verbose=False):
    """
    验证 GSM8K 答案
    
    Args:
        model_response: 模型的完整回复
        correct_answer: 正确答案
        verbose: 是否输出详细信息
    
    Returns:
        dict: 验证结果
    """
    # 提取答案
    extracted = extract_answer(model_response)
    
    # 标准化
    pred_norm = normalize_answer(extracted)
    gold_norm = normalize_answer(correct_answer)
    
    # 比较
    is_correct = compare_answers(pred_norm, gold_norm)
    
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

def judge_answer_dispatch(
    judge_type: str,
    model_response: str,
    gold_answer: str,
    llm_judge_fn=judge_answer,
):
    """
    根据 judge_type 分发判题逻辑
    返回 reward ∈ {0.0, 1.0}
    """
    if judge_type == "llm":
        if llm_judge_fn is None:
            raise ValueError("llm_judge_fn must be provided when judge_type=llm")
        reward, _ = llm_judge_fn(model_response, gold_answer)
        return reward

    elif judge_type == "rule":
        result = verify_gsm8k_answer(
            model_response=model_response,
            correct_answer=gold_answer,
            verbose=False
        )
        return 1.0 if result["is_correct"] else 0.0

    else:
        raise ValueError(f"Unknown judge_type: {judge_type}")


# =========================================================
# Config
# =========================================================

@dataclass
class DAPOConfig:
    # sampling / batch
    batch_prompts: int = 8
    group_size: int = 8
    max_new_tokens: int = 512
    temperature: float = 1.0
    top_p: float = 0.7
    top_k: int = 0
    mask_token0: bool = True

    # stop checks
    stop_on_think_close: bool = True
    stop_on_user: bool = True
    stop_on_boxed: bool = True
    stop_check_every: int = 8
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
    kl_coef: float = 0.05
    target_kl: float = 0.01
    adaptive_kl: bool = True
    time_state_l2: float = 5e-6
    time_state_clamp: float = 3.0

    # logging / save
    log_interval: int = 10
    save_interval: int = 50
    infer_check_interval: int = 50  # cheap sanity

    # eval
    eval_interval: int = 20
    eval_n: int = 10
    eval_temperature: float = 0.0  # greedy by default
    eval_top_p: float = 1.0
    eval_top_k: int = 0
    eval_max_new_tokens: int = 256

    # faulthandler (OFF by default to avoid segfault)
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
        # user gave exact file path without .pth assumption
        pth = model_arg
        if pth.endswith(".pth"):
            base = pth[:-4]
    return base, pth

def _torch_load_weights(path: str):
    # be compatible with torch versions that don't support weights_only
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
    # Handle nested checkpoint format: {'time_state': {...}, ...}
    if 'time_state' in sd and isinstance(sd['time_state'], dict):
        sd = sd['time_state']
    hit = 0
    for n, p in model.named_parameters():
        if n in sd:
            p.data.copy_(sd[n].to(p.device).to(p.dtype))
            hit += 1
    return hit > 0


# =========================================================
# Albatross batched inference (parallel group sampling)
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
            ts = block.att.time_state  # (H,64,64)
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

        last_logits, state = self.prime_prompts(prompt_tokens_list)

        B = Bp * group_size
        last_logits = last_logits.repeat_interleave(group_size, dim=0).contiguous()

        # repeat state for group
        state0 = state[0].repeat_interleave(group_size, dim=2).contiguous()
        state1 = state[1].repeat_interleave(group_size, dim=1).contiguous()
        state = [state0, state1]

        comp_tokens: List[List[int]] = [[] for _ in range(B)]
        old_logps: List[List[float]] = [[] for _ in range(B)]
        active = torch.ones((B,), device=last_logits.device, dtype=torch.bool)
        truncated = [False for _ in range(B)]

        def sample_next(logits_2d: torch.Tensor) -> torch.Tensor:
            if temperature <= 0:
                return torch.argmax(logits_2d, dim=-1)

            x = logits_2d.float() / float(temperature)
            V = x.size(-1)

            # optional: mask token 0
            if self.cfg.mask_token0:
                x[:, 0] = -1e30

            k_cap = 0
            if top_k and top_k > 0:
                k_cap = int(min(top_k, V))
            elif top_p and 0.0 < top_p < 1.0:
                k_cap = int(min(2048, V))

            if k_cap > 0:
                topv, topi = torch.topk(x, k=k_cap, dim=-1)
                if top_p and 0.0 < top_p < 1.0:
                    probs = F.softmax(topv, dim=-1)
                    cdf = torch.cumsum(probs, dim=-1)
                    keep = cdf <= float(top_p)
                    keep[:, 0] = True
                    topv = topv.masked_fill(~keep, -1e30)
                probs = F.softmax(topv, dim=-1)
                pick = torch.multinomial(probs, 1).squeeze(-1)
                return topi.gather(-1, pick.unsqueeze(-1)).squeeze(-1)

            probs = F.softmax(x, dim=-1)
            return torch.multinomial(probs, 1).squeeze(-1)

        for t in range(max_new_tokens):
            if not bool(active.any().item()):
                break

            logits = last_logits
            # sample token
            token_ids = sample_next(logits)

            # behavior-policy logp
            logp_all = F.log_softmax(logits.float(), dim=-1)
            picked_logp = logp_all.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)

            # inactive: feed 0 as dummy (NOT EOS), but don't append
            token_ids = torch.where(active, token_ids, torch.zeros_like(token_ids))
            picked_logp = torch.where(active, picked_logp, torch.zeros_like(picked_logp))

            tok_cpu = token_ids.detach().cpu().tolist()
            lp_cpu = picked_logp.detach().cpu().tolist()

            for i in range(B):
                if not active[i]:
                    continue
                comp_tokens[i].append(int(tok_cpu[i]))
                old_logps[i].append(float(lp_cpu[i]))

            # stop check
            if (stop_on_think_close or stop_on_user or stop_on_boxed) and (t % max(1, stop_check_every) == 0):
                for i in range(B):
                    if not active[i]:
                        continue
                    w = comp_tokens[i][-stop_check_window:] if stop_check_window > 0 else comp_tokens[i]
                    s = self.decode(w)
                    if stop_on_boxed and boxed_complete(s):
                        active[i] = False
                        continue
                    if stop_on_think_close and ("</think>" in s):
                        active[i] = False
                        continue
                    if stop_on_user and (("\nUser:" in s) or ("\n\nUser:" in s)):
                        active[i] = False
                        continue

            # forward 1 token per element
            step_tokens_batch = [[int(x)] for x in tok_cpu]
            last_logits = self.infer_model.forward_batch(step_tokens_batch, state)
            if torch.is_tensor(last_logits) and last_logits.dim() == 3:
                last_logits = last_logits[:, -1, :]

        # mark truncated
        for i in range(B):
            if bool(active[i].item()):
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
        data: List[Dict[str, Any]],
        out_dir: str,
        device: str,
        cfg: DAPOConfig,
        seed: int = 42,
    ):
        self.model = train_model
        self.infer = infer_engine
        self.encode = encode_fn
        self.decode = decode_fn
        self.data = data
        self.out_dir = out_dir
        self.device = device
        self.cfg = cfg
        self.rng = random.Random(seed)

        os.makedirs(out_dir, exist_ok=True)
        self.log_path = os.path.join(out_dir, "train.log")
        self.gen_dump_path = os.path.join(out_dir, "gen_judgements.jsonl")
        self.infer_check_path = os.path.join(out_dir, "infer_check.jsonl")
        self.eval_path = os.path.join(out_dir, "eval.jsonl")

        # faulthandler is OFF by default (avoid segfault)
        self._hang_f = None
        if cfg.enable_faulthandler:
            try:
                import faulthandler
                hang_path = os.path.join(out_dir, "hang_tracebacks.log")
                self._hang_f = open(hang_path, "a", encoding="utf-8", buffering=1)
                faulthandler.enable(file=self._hang_f, all_threads=True)
                # IMPORTANT: only enable timer if hang_dump_s > 0
                if float(cfg.hang_dump_s) > 0:
                    faulthandler.dump_traceback_later(float(cfg.hang_dump_s), repeat=True, file=self._hang_f)
            except Exception:
                self._hang_f = None

        params = [p for p in self.model.parameters() if p.requires_grad]
        if not params:
            raise RuntimeError("No trainable params (expected time_state only).")
        self.opt = torch.optim.AdamW(params, lr=self.cfg.lr)

        # save init time_state for L2 pull-to-init
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

    def _time_state_stats(self):
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
        # KL too high -> reduce lr
        if approx_kl > self.cfg.target_kl * 2.0:
            new_lr = max(lr_now * 0.5, 1e-6)
            if new_lr < lr_now:
                self.opt.param_groups[0]["lr"] = new_lr
                self._log(f"[KL-ADAPT] kl={approx_kl:.6f} high -> lr {lr_now:.2e} -> {new_lr:.2e}")
        # KL too low -> slightly increase (cap at initial cfg.lr)
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
                "gt": _strip_math_delims(gt),
                "method": "truncated_skip_judge",
                "error": None,
                "pred_parsed": None,
                "gt_parsed": None,
                "correct": False,
                "truncated_forced_zero": True,
            }
        else:
            r, jdbg0 = judge_answer(txt, gt)
            jdbg = dict(jdbg0)
            jdbg["truncated_forced_zero"] = False

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
            ex = self.data[self.rng.randrange(len(self.data))]
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
    def evaluate(self, step: int):
        idxs = [self.rng.randrange(len(self.data)) for _ in range(self.cfg.eval_n)]
        correct = 0
        trunc_cnt = 0
        lens = []
        details = []
        for i in idxs:
            ex = self.data[i]
            gt = str(ex.get("solution", ""))
            rec = self._infer_once(
                problem=ex.get("problem", ""),
                gt=gt,
                max_new=self.cfg.eval_max_new_tokens,
                temperature=self.cfg.eval_temperature,
                top_p=self.cfg.eval_top_p,
                top_k=self.cfg.eval_top_k,
            )
            if rec["reward"] >= 0.5:
                correct += 1
            if rec["truncated"]:
                trunc_cnt += 1
            lens.append(rec["gen_len"])
            details.append({
                "problem": ex.get("problem", ""),
                "gt": gt,
                **rec
            })

        acc = correct / max(1, self.cfg.eval_n)
        trunc_rate = trunc_cnt / max(1, self.cfg.eval_n)
        avg_len = sum(lens) / max(1, len(lens))

        append_jsonl(self.eval_path, {
            "time": now_str(),
            "step": step,
            "eval_n": self.cfg.eval_n,
            "acc": acc,
            "trunc_rate": trunc_rate,
            "avg_len": avg_len,
            "eval_temperature": self.cfg.eval_temperature,
            "eval_top_p": self.cfg.eval_top_p,
            "eval_top_k": self.cfg.eval_top_k,
            "eval_max_new_tokens": self.cfg.eval_max_new_tokens,
            "details": details,
        })

        self._log(f"[EVAL step {step}] acc={acc:.3f} trunc={trunc_rate:.3f} avg_len={avg_len:.1f} "
                  f"(temp={self.cfg.eval_temperature}, top_p={self.cfg.eval_top_p}, max_new={self.cfg.eval_max_new_tokens})")

    def train(self, total_steps: int):
        self._log(f"train begin: steps={total_steps} batch_prompts={self.cfg.batch_prompts} group={self.cfg.group_size} "
                  f"lr={self.cfg.lr} kl_coef={self.cfg.kl_coef} l2={self.cfg.time_state_l2} clamp={self.cfg.time_state_clamp}")
        st0 = self._time_state_stats()
        self._log(f"time_state init: absmax={st0['absmax']:.6f} rms={st0['rms_avg']:.6f} bad={st0['bad']}")

        for step in range(1, total_steps + 1):
            t0 = time.time()

            # step stats
            cand_prompts = 0
            kept_prompts = 0
            drop_all0 = 0
            drop_all1 = 0
            sample_cnt = 0
            sample_correct = 0
            sample_trunc = 0

            # ---------------- collect batch (dynamic sampling) ----------------
            buffer = []
            tries = 0

            while len(buffer) < self.cfg.batch_prompts and tries < self.cfg.dynamic_sampling_max_tries:
                tries += 1
                need = self.cfg.batch_prompts - len(buffer)
                chunk = min(self.cfg.collect_chunk, need)

                ex_list = [self.data[self.rng.randrange(len(self.data))] for _ in range(chunk)]
                prompt_strs = [build_prompt(ex.get("problem", "")) for ex in ex_list]
                gts = [str(ex.get("solution", "")) for ex in ex_list]
                probs = [ex.get("problem", "") for ex in ex_list]

                prompt_tokens_list = []
                for ps in prompt_strs:
                    ids = self.encode(ps)
                    max_prompt_len = int(self.model.args.ctx_len) - int(self.cfg.max_new_tokens) - 4
                    max_prompt_len = max(64, max_prompt_len)
                    if len(ids) > max_prompt_len:
                        ids = ids[-max_prompt_len:]
                    prompt_tokens_list.append(ids)

                comp_tokens_flat, old_logps_flat, comp_text_flat, truncated_flat = self.infer.generate_group_parallel(
                    prompt_tokens_list=prompt_tokens_list,
                    group_size=self.cfg.group_size,
                    max_new_tokens=self.cfg.max_new_tokens,
                    temperature=self.cfg.temperature,
                    top_p=self.cfg.top_p,
                    top_k=self.cfg.top_k,
                    stop_on_think_close=self.cfg.stop_on_think_close,
                    stop_on_user=self.cfg.stop_on_user,
                    stop_on_boxed=self.cfg.stop_on_boxed,
                    stop_check_every=self.cfg.stop_check_every,
                    stop_check_window=self.cfg.stop_check_window,
                )

                for pi in range(chunk):
                    group = []
                    rewards = []
                    judges = []
                    drop_reason = None

                    for gi in range(self.cfg.group_size):
                        idx = pi * self.cfg.group_size + gi
                        ctoks = comp_tokens_flat[idx]
                        ologp = old_logps_flat[idx]
                        ctext = comp_text_flat[idx]
                        trunc = bool(truncated_flat[idx])

                        if trunc:
                            r = 0.0
                            jdbg = {
                                "pred_extracted": None,
                                "gt": _strip_math_delims(gts[pi]),
                                "method": "truncated_skip_judge",
                                "error": None,
                                "pred_parsed": None,
                                "gt_parsed": None,
                                "correct": False,
                                "truncated_forced_zero": True,
                            }
                        else:
                            #r, jdbg0 = judge_answer(ctext, gts[pi])
                            r = judge_answer_dispatch(
                                judge_type='rule',
                                model_response=ctext,
                                gold_answer=gts[pi],
                            )
                            #jdbg = dict(jdbg0)
                            #jdbg["truncated_forced_zero"] = False
                            jdbg = {
                                "pred_extracted": None,
                                "gt": _strip_math_delims(gts[pi]),
                                "method": "truncated_skip_judge",
                                "error": None,
                                "pred_parsed": None,
                                "gt_parsed": None,
                                "correct": True if r >= 0.5 else False,
                                "truncated_forced_zero": True,
                            }

                        group.append((ctoks, ologp, ctext, trunc))
                        rewards.append(float(r))
                        judges.append(jdbg)

                        sample_cnt += 1
                        if r >= 0.5:
                            sample_correct += 1
                        if trunc:
                            sample_trunc += 1

                    cand_prompts += 1
                    rsum = sum(rewards)
                    if rsum == 0.0:
                        drop_reason = "all0"
                        drop_all0 += 1
                    elif rsum == float(self.cfg.group_size):
                        drop_reason = "all1"
                        drop_all1 += 1
                    else:
                        kept_prompts += 1

                    append_jsonl(self.gen_dump_path, {
                        "time": now_str(),
                        "step": step,
                        "try": tries,
                        "kept": (drop_reason is None),
                        "drop_reason": drop_reason,
                        "problem": probs[pi],
                        "solution": gts[pi],
                        "prompt": prompt_strs[pi],
                        "group_size": self.cfg.group_size,
                        "max_new_tokens": self.cfg.max_new_tokens,
                        "samples": [
                            {"i": gi, "text": group[gi][2], "truncated": bool(group[gi][3]),
                             "reward": float(rewards[gi]), "judge": judges[gi]}
                            for gi in range(self.cfg.group_size)
                        ]
                    })

                    if drop_reason is not None:
                        continue

                    buffer.append((prompt_tokens_list[pi], gts[pi], group, rewards))

            if not buffer:
                self._log(f"WARN: empty batch after dynamic sampling. tries={tries}/{self.cfg.dynamic_sampling_max_tries}. "
                          f"Consider increasing temperature/top_p or reduce group_size.")
                continue

            # ---------------- build trajectories ----------------
            trajs = []
            for (prompt_tokens, gt, group, rewards_list) in buffer:
                rewards_t = torch.tensor(rewards_list, device=self.device, dtype=torch.float32)
                adv = self._compute_advantages(rewards_t)

                for i, (comp_tokens, old_logps, _, trunc) in enumerate(group):
                    if not comp_tokens:
                        continue
                    full = prompt_tokens + comp_tokens
                    trajs.append({
                        "full_tokens": full,
                        "prompt_len": len(prompt_tokens),
                        "comp_len": len(comp_tokens),
                        "old_logps": old_logps,
                        "adv": float(adv[i].item()),
                        "reward": float(rewards_list[i]),
                    })

            if not trajs:
                self._log("WARN: no trajs after expansion.")
                continue

            total_comp_tokens = sum(int(tr["comp_len"]) for tr in trajs)
            if total_comp_tokens <= 0:
                self._log("WARN: total_comp_tokens=0.")
                continue

            # ---------------- PPO/DAPO update ----------------
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

                    logits = self.model(inp)  # (B, T-1, V)
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

                        # token-level KL penalty
                        kl_tok = 0.5 * (log_ratio ** 2)
                        kl_sum_tok = kl_sum_tok + kl_tok.sum()

                        with torch.no_grad():
                            approx_kl_sum += float(kl_tok.mean().item())
                            approx_kl_cnt += 1
                            clipped = (ratio < (1.0 - self.cfg.eps_low)) | (ratio > (1.0 + self.cfg.eps_high))
                            clip_hits += float(clipped.float().mean().item())
                            clip_cnt += 1

                    # time_state pull-to-init
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

                # clamp time_state
                if self.cfg.time_state_clamp and self.cfg.time_state_clamp > 0:
                    cv = float(self.cfg.time_state_clamp)
                    with torch.no_grad():
                        for n, p in self.model.named_parameters():
                            if "time_state" in n:
                                p.data.clamp_(-cv, cv)

                last_kl = float(approx_kl_sum / max(1, approx_kl_cnt))
                last_clipfrac = float(clip_hits / max(1, clip_cnt))
                self._maybe_adapt_lr_by_kl(last_kl)

            # ---------------- logging / save / eval ----------------
            avg_r = sum(tr["reward"] for tr in trajs) / max(1, len(trajs))
            dt = time.time() - t0
            st = self._time_state_stats()
            lr_now = float(self.opt.param_groups[0]["lr"])

            if step % self.cfg.log_interval == 0:
                self._log(
                    f"[step {step}/{total_steps}] "
                    f"collect kept={len(buffer)}/{self.cfg.batch_prompts} cand={cand_prompts} "
                    f"drop(all0={drop_all0}, all1={drop_all1}) tries={tries} | "
                    f"samples={sample_cnt} acc={sample_correct/max(1,sample_cnt):.3f} trunc={sample_trunc/max(1,sample_cnt):.3f} | "
                    f"avg_reward={avg_r:.4f} loss={last_loss} grad={last_grad:.3f} "
                    f"approx_kl={last_kl:.6f} clip_frac={last_clipfrac:.3f} | "
                    f"ts(absmax={st['absmax']:.4f}, rms={st['rms_avg']:.4f}, bad={st['bad']}) | "
                    f"hp(lr={lr_now:.2e}, kl_coef={self.cfg.kl_coef}, l2={self.cfg.time_state_l2}, clamp={self.cfg.time_state_clamp}, "
                    f"temp={self.cfg.temperature}, top_p={self.cfg.top_p}, max_new={self.cfg.max_new_tokens}) "
                    f"step_time={dt:.1f}s"
                )

            if step % self.cfg.infer_check_interval == 0:
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

        # cancel timer if enabled
        if self.cfg.enable_faulthandler and float(self.cfg.hang_dump_s) > 0:
            try:
                import faulthandler
                faulthandler.cancel_dump_traceback_later()
            except Exception:
                pass

        self._log("train end.")


# =========================================================
# Sanity check: train vs infer last-logits top1
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True, help="model path, with or without .pth")
    ap.add_argument("--train_jsonl", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--total_steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ctx_len", type=int, default=8192)

    ap.add_argument("--batch_prompts", type=int, default=8)
    ap.add_argument("--group_size", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.7)
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--mask_token0", action="store_true")

    ap.add_argument("--dynamic_sampling_max_tries", type=int, default=200)
    ap.add_argument("--collect_chunk", type=int, default=4)

    ap.add_argument("--ppo_epochs", type=int, default=1)
    ap.add_argument("--micro_batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--eps_low", type=float, default=0.2)
    ap.add_argument("--eps_high", type=float, default=0.5)
    ap.add_argument("--grad_clip", type=float, default=0.2)

    ap.add_argument("--kl_coef", type=float, default=0.08)
    ap.add_argument("--target_kl", type=float, default=0.01)
    ap.add_argument("--adaptive_kl", action="store_true")

    ap.add_argument("--time_state_l2", type=float, default=1e-6)
    ap.add_argument("--time_state_clamp", type=float, default=5.0)

    ap.add_argument("--log_interval", type=int, default=10)
    ap.add_argument("--save_interval", type=int, default=50)
    ap.add_argument("--infer_check_interval", type=int, default=50)

    ap.add_argument("--eval_interval", type=int, default=20)
    ap.add_argument("--eval_n", type=int, default=10)
    ap.add_argument("--eval_temperature", type=float, default=0.0)
    ap.add_argument("--eval_top_p", type=float, default=1.0)
    ap.add_argument("--eval_top_k", type=int, default=0)
    ap.add_argument("--eval_max_new_tokens", type=int, default=1024)

    ap.add_argument("--state_init", type=str, default=None, help="optional: load time_state-only checkpoint")
    ap.add_argument("--tokenizer", type=str, default="reference/rwkv_vocab_v20230424.txt")

    # faulthandler (off by default!)
    ap.add_argument("--enable_faulthandler", action="store_true")
    ap.add_argument("--hang_dump_s", type=float, default=0.0)
    ap.add_argument(
    "--judge_type",
    type=str,
    default="llm",
    choices=["llm", "rule"],
    help="Reward judge type: llm (LLM-as-judge) or rule (extract+compare)"
)

    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # IMPORTANT: set env vars BEFORE importing rwkvt kernels
    os.environ["RWKV_HEAD_SIZE_A"] = str(HEAD_SIZE)
    os.environ["RWKV_MY_TESTING"] = "x070"
    os.environ["RWKV_TRAIN_TYPE"] = "state"
    os.environ["RWKV_CTXLEN"] = str(int(args.ctx_len))
    os.environ["FUSED_KERNEL"] = "0"
    os.environ["WKV"] = "cuda"  # must be CUDA (no fla)

    os.makedirs(args.out_dir, exist_ok=True)

    data = read_jsonl(args.train_jsonl)
    if not data:
        raise RuntimeError("empty train_jsonl")

    # tokenizer (TRIE_TOKENIZER)
    from reference.utils import TRIE_TOKENIZER
    tok = TRIE_TOKENIZER(args.tokenizer)

    encode = lambda s: tok.encode(s)

    def safe_decode(ids):
        # prefer replace to avoid UnicodeDecodeError
        try:
            return tok.decode(ids, utf8_errors="replace")
        except TypeError:
            pass
        try:
            return tok.decode(ids)
        except UnicodeDecodeError:
            try:
                b = tok.decodeBytes(ids)
                return b.decode("utf-8", errors="replace")
            except Exception:
                return "".join(chr(int(x) % 256) for x in ids)

    decode = safe_decode

    base_name, pth_path = normalize_model_arg(args.model)
    if not os.path.isfile(pth_path):
        raise FileNotFoundError(f"Cannot find model pth: {pth_path}")

    train_model, _ = load_train_model_rwkv7_cuda(pth_path, device=device, ctx_len=int(args.ctx_len))
    infer_model, _ = load_infer_model_albatross(base_name)

    trainable = freeze_except_time_state(train_model)
    if trainable <= 0:
        raise RuntimeError("No trainable time_state found. Check model weights / naming.")

    if args.state_init:
        ok = load_time_state_only(train_model, args.state_init)
        print(f"[state_init] loaded={ok} from {args.state_init}", flush=True)

    cfg = DAPOConfig(
        batch_prompts=int(args.batch_prompts),
        group_size=int(args.group_size),
        max_new_tokens=int(args.max_new_tokens),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        top_k=int(args.top_k),
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
        eval_temperature=float(args.eval_temperature),
        eval_top_p=float(args.eval_top_p),
        eval_top_k=int(args.eval_top_k),
        eval_max_new_tokens=int(args.eval_max_new_tokens),

        enable_faulthandler=bool(args.enable_faulthandler),
        hang_dump_s=float(args.hang_dump_s),
    )

    infer_engine = AlbatrossBatchInference(infer_model, train_model, encode, decode, device=device, cfg=cfg)

    # sanity: train vs albatross last logits alignment
    sanity_check_train_vs_albatross(train_model, infer_model, infer_engine, encode, decode, device=device)

    trainer = DAPOStateTuningTrainer(
        train_model=train_model,
        infer_engine=infer_engine,
        encode_fn=encode,
        decode_fn=decode,
        data=data,
        out_dir=args.out_dir,
        device=device,
        cfg=cfg,
        seed=int(args.seed),
    )

    trainer.train(total_steps=int(args.total_steps))


if __name__ == "__main__":
    main()
