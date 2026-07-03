#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import time
import math
import random
import sys
import subprocess
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
        f"Please put the final answer in \\boxed{{...}} and output only that line. think\n"
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
    """Extract answer from model output"""
    if not text or not isinstance(text, str):
        return None

    text = text.strip()

    # 1. Balanced \boxed{...}
    boxed = extract_last_boxed(text)
    if boxed:
        boxed = boxed.strip()
        frac = re.findall(r'-?\d+\s*/\s*-?\d+', boxed.replace(',', ''))
        if frac:
            return frac[-1].replace(' ', '')
        numbers = re.findall(r'-?\d+\.?\d*', boxed.replace(',', ''))
        if numbers:
            return numbers[-1]
        return boxed if boxed else None

    # 2. Plain boxed{...} without backslash
    boxed_plain = re.findall(r'(?<!\\)boxed\{([^}]+)\}', text, re.IGNORECASE)
    if boxed_plain:
        answer = boxed_plain[-1].strip()
        frac = re.findall(r'-?\d+\s*/\s*-?\d+', answer.replace(',', ''))
        if frac:
            return frac[-1].replace(' ', '')
        numbers = re.findall(r'-?\d+\.?\d*', answer.replace(',', ''))
        if numbers:
            return numbers[-1]
        return answer if answer else None

    # 3. Try matching answer after ####
    if '####' in text:
        after_hash = text.split('####')[-1].strip()
        frac = re.findall(r'-?\d+\s*/\s*-?\d+', after_hash.replace(',', ''))
        if frac:
            return frac[-1].replace(' ', '')
        numbers = re.findall(r'-?\d+\.?\d*', after_hash.replace(',', ''))
        if numbers:
            return numbers[0]

    # 4. Fallback: extract last fraction in full text
    frac = re.findall(r'-?\d+\s*/\s*-?\d+', text.replace(',', ''))
    if frac:
        return frac[-1].replace(' ', '')

    # 5. Fallback: extract last number in full text
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
        if not math.isfinite(num):
            return answer_str.strip()
        if abs(num - round(num)) < 1e-9:
            return str(int(round(num)))
        formatted = f"{num:.10f}".rstrip('0').rstrip('.')
        return formatted
    except (ValueError, TypeError, OverflowError):
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


def _compute_reward_components(correct_any: bool, correct_boxed: bool, has_boxed: bool, truncated: bool) -> Tuple[float, Dict[str, float]]:
    reward = 0.0
    if correct_any:
        reward += 0.6

    format_reward = 0.0
    if not truncated:
        if has_boxed:
            format_reward += 0.15
        if correct_boxed:
            format_reward += 0.25

    reward += format_reward

    if truncated:
        reward -= 0.1

    parts = {
        'correct_any': 0.6 if correct_any else 0.0,
        'boxed_present': 0.15 if (has_boxed and not truncated) else 0.0,
        'boxed_correct': 0.25 if (correct_boxed and not truncated) else 0.0,
        'truncated_penalty': -0.1 if truncated else 0.0,
        'format_reward': format_reward,
    }
    return reward, parts

def judge_answer_dispatch(
    judge_type: str,
    model_response: str,
    gold_answer: str,
    truncated: bool,
    llm_judge_fn=None,
) -> Tuple[float, Dict[str, Any]]:
    """
    Dispatch to appropriate judge based on judge_type
    Returns: (reward, debug_info)
    """
    pred_text = model_response or ""
    gold_extracted = extract_answer(gold_answer) or gold_answer

    boxed_ans = extract_last_boxed(pred_text)
    correct_boxed = False
    if boxed_ans is not None:
        correct_boxed = compare_answers(boxed_ans, gold_extracted)

    has_boxed = "\\boxed{" in pred_text

    if judge_type == "llm":
        if llm_judge_fn is None:
            raise ValueError("llm_judge_fn must be provided when judge_type=llm")
        ok, raw = judge_with_zhipu(pred_full_output=pred_text, gt=gold_answer)
        correct_any = bool(ok)
        pred_extracted = extract_answer(pred_text)
        pred_norm = normalize_answer(pred_extracted)
        gold_norm = normalize_answer(gold_extracted)
        reward, parts = _compute_reward_components(correct_any, correct_boxed, has_boxed, truncated)
        dbg = {
            "pred_extracted": pred_extracted,
            "gt": normalize_answer(gold_extracted),
            "method": "llm",
            "error": None,
            "pred_parsed": pred_norm,
            "gt_parsed": gold_norm,
            "correct": correct_any,
            "correct_boxed": correct_boxed,
            "has_boxed": has_boxed,
            "truncated": truncated,
            "reward_parts": parts,
            "judge_llm_raw": raw,
            "judge_llm_decision": int(correct_any),
        }
        return reward, dbg

    elif judge_type == "rule":
        result = verify_gsm8k_answer(
            model_response=pred_text,
            correct_answer=gold_answer,
            verbose=False
        )
        correct_any = bool(result["is_correct"])
        reward, parts = _compute_reward_components(correct_any, correct_boxed, has_boxed, truncated)
        dbg = {
            "pred_extracted": result["extracted_answer"],
            "gt": normalize_answer(gold_extracted),
            "method": "rule",
            "error": None,
            "pred_parsed": result["normalized_pred"],
            "gt_parsed": result["normalized_gold"],
            "correct": correct_any,
            "correct_boxed": correct_boxed,
            "has_boxed": has_boxed,
            "truncated": truncated,
            "reward_parts": parts,
        }
        return reward, dbg

    else:
        raise ValueError(f"Unknown judge_type: {judge_type}")


# =========================================================
# Config
# =========================================================

STOP_TOKENS = ("\n\nUser:", "\n\nQuestion:", "Q:", "<|endoftext|>", "\nUser:", "\nQuestion:")

def _has_repetition(tokens, min_ngram=12, max_ngram=64, window=256):
    if len(tokens) < min_ngram * 2:
        return False
    if window and len(tokens) > window:
        w = tokens[-window:]
    else:
        w = tokens
    for n in (64, 48, 32, 24, 16, 12):
        if len(w) >= 2 * n and w[-n:] == w[-2 * n:-n]:
            return True
    n = 8
    if len(w) >= 3 * n and w[-n:] == w[-2 * n:-n] == w[-3 * n:-2 * n]:
        return True
    if len(w) >= 32 and len(set(w[-32:])) == 1:
        return True
    return False

@dataclass
class DAPOConfig:
    # sampling / batch
    batch_prompts: int = 1
    group_size: int = 16
    rollout_n: int = 32
    train_bsz: int = 32
    max_new_tokens: int = 768
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
    prior_mean_override: float = -1.0
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
    neg_w: float = 0.2

    # stability
    kl_coef: float = 0.001
    target_kl: float = 0.01
    adaptive_kl: bool = True
    time_state_l2: float = 1e-7
    time_state_clamp: float = 10.0

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

            # --- C. Temperature 缩放 (Logits 上) ---
            if temperature != 1.0 and temperature > 0:
                logits = logits / temperature

            # --- D. Top-K 过滤 ---
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -1e30

            # --- E. 计算概率分布 ---
            probs = F.softmax(logits, dim=-1)

            # --- F. Top-P (Nucleus) 过滤 (对齐 Benchmark 健壮逻辑) ---
            if 0.0 < top_p < 1.0:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                probs[indices_to_remove] = 0
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
            if t % max(1, stop_check_every) == 0:
                for i in range(B):
                    if not active[i]:
                        continue

                    w = comp_tokens[i]
                    stop_hit = False
                    if stop_on_think_close or stop_on_user:
                        # decode full output to avoid missing earlier stop markers
                        s = self.decode(w)
                        if stop_on_think_close and ("</think>" in s):
                            stop_hit = True
                        elif stop_on_user and any(tok in s for tok in STOP_TOKENS):
                            stop_hit = True

                    if not stop_hit and _has_repetition(w, window=stop_check_window):
                        stop_hit = True

                    if stop_hit:
                        active[i] = False
