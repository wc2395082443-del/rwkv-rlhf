#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
from typing import Optional, Tuple
import zstandard as zstd
global_compressor = zstd.ZstdCompressor(level=9)
def _strip_math_delims(s: str) -> str:
    """去除数学分隔符"""
    if s is None:
        return ""
    s = str(s).strip()
    s = s.replace(r"\left", "").replace(r"\right", "")
    s = re.sub(r"\\[,\;\!\:]\s*", "", s)
    return s.strip()

def _extract_number_or_frac(s: str) -> Optional[str]:
    if not s:
        return None

    # 1. 基础清洗
    # 假设 _strip_math_delims 去掉了 $ 等包裹符
    s = _strip_math_delims(s) 
    # 统一负号：将 Unicode 减号替换为标准 ASCII 减号
    s = s.replace("−", "-").replace("–", "-")
    # 移除千分位逗号、百分号等干扰
    s = s.replace(",", "").replace("$", "").replace("%", "")

    # 2. 关键修复：将 LaTeX 分数 \frac{a}{b} 转换为 a/b
    # 这一步必须在正则匹配之前做
    s = re.sub(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"\1/\2", s)

    # 3. 提取分数 (支持 1/2, -3/4 等)
    # 现在的 s 里的 \frac{1}{2} 已经变成了 1/2，可以被匹配了
    frac_pattern = r"[-+]?\d+\s*/\s*[-+]?\d+"
    fracs = re.findall(frac_pattern, s)
    if fracs:
        # 找到最后一个分数，去掉可能的空格
        return fracs[-1].replace(" ", "")

    # 4. 提取数字 (整数或小数)
    # 改进正则：支持 .5 这种写法
    # 逻辑：
    #   [-+]?  : 可选正负号
    #   ( ... )
    #     \d+\.?\d* : 123, 123., 123.45
    #     |
    #     \.\d+     : .45
    num_pattern = r"[-+]?(?:\d+\.?\d*|\.\d+)"
    nums = re.findall(num_pattern, s)
    
    if nums:
        # 过滤掉只有符号没有数字的情况 (如 "-", "+")
        valid_nums = [n for n in nums if re.search(r"\d", n)]
        if valid_nums:
            return valid_nums[-1]

    return None



def _find_balanced_brace(text: str, brace_start: int) -> Optional[Tuple[str, int]]:
    """查找平衡的大括号"""
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
    """提取最后一个\\boxed{}内容"""
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


def _iter_boxed(text: str):
    for m in re.finditer(r"\\boxed\{", text):
        brace_start = m.end() - 1
        got = _find_balanced_brace(text, brace_start)
        if got is None:
            continue
        inner, _ = got
        yield inner, m.start()


def extract_answer(text: str, ground_truth: str = None) -> Optional[str]:
    """
    从模型输出中提取答案，支持多种格式：
    1. \\boxed{${answer}}
    2. answer is ${answer}
    3. answer: ${answer}
    4. 行内纯数字/分数

    冲突处理：如果命中多个答案，使用最后一个答案
    """
    if not text or not isinstance(text, str):
        return None

    candidates = []
    # 格式1: \boxed{answer}
    for inner, pos in _iter_boxed(text):
        boxed_num = _extract_number_or_frac(inner)
        if boxed_num:
            candidates.append((pos, boxed_num))

    # 格式2和3: answer is / answer:
    # 允许关键字和答案之间有任意空格与标点（不允许字母/数字夹在中间）
    patterns = [
        r"answer\s*is[^A-Za-z0-9]*([$+-]?\d[\d,]*(?:\.\d+)?(?:\s*/\s*[$+-]?\d[\d,]*(?:\.\d+)?)?)",
        r"answer\s*:[^A-Za-z0-9]*([$+-]?\d[\d,]*(?:\.\d+)?(?:\s*/\s*[$+-]?\d[\d,]*(?:\.\d+)?)?)",
    ]

    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            answer = match.group(1).strip()
            answer = _extract_number_or_frac(answer)
            if answer:
                candidates.append((match.start(), answer))

    # 格式4: 行内纯数字/分数
    line_only_pattern = re.compile(
        r"^\s*[$+-]?\d[\d,]*(?:\.\d+)?(?:\s*/\s*[$+-]?\d[\d,]*(?:\.\d+)?)?\s*%?\s*$"
    )
    offset = 0
    for line in text.splitlines(True):
        raw = line.strip()
        if raw and line_only_pattern.match(raw):
            ans = _extract_number_or_frac(raw)
            if ans:
                candidates.append((offset, ans))
        offset += len(line)

    if candidates:
        candidates.sort(key=lambda x: x[0])
        return candidates[-1][1]

    return None


def _extract_ground_truth(ground_truth: str) -> Optional[str]:
    if not ground_truth:
        return None
    gt = extract_answer(ground_truth)
    if gt:
        return gt
    return _extract_number_or_frac(ground_truth)


def check_answer_correct(extracted: str, ground_truth: str) -> bool:
    """
    改进版：支持数值等价性比较的答案检查
    """
    if not extracted or not ground_truth:
        return False

    # 假设 _extract_ground_truth 逻辑正确，提取出 "####" 后的部分
    gt_extracted = _extract_ground_truth(ground_truth)
    if not gt_extracted:
        return False

    def clean_number_str(s: str) -> str:
        """移除所有非数字字符（除了小数点和负号），并处理千分位"""
        s = str(s).strip().lower()
        s = _strip_math_delims(s) # 假设你已有的去Latex函数
        # 移除逗号 (关键！)
        s = s.replace(',', '') 
        return s

    ext_clean = clean_number_str(extracted)
    gt_clean = clean_number_str(gt_extracted)

    # 1. 尝试数值比较 (最稳健的方法)
    try:
        # 转换为 float 进行比较，解决 5 == 5.0 的问题
        # 使用 round 防止浮点数精度问题，通常保留 6 位小数足够
        is_float_equal = abs(float(ext_clean) - float(gt_clean)) < 1e-6
        if is_float_equal:
            return True
    except ValueError:
        # 如果无法转为数字（例如答案是纯文本），则回退到字符串比较
        pass

    # 2. 回退到字符串比较 (处理非数字答案)
    # 移除所有空白再次比较
    return ext_clean.replace(" ", "") == gt_clean.replace(" ", "")


def check_format_correct(text: str, ground_truth: str) -> bool:
    """
    严格格式检查：
    返回 True 当且仅当：
    1. 模型输出了 \\boxed{...} 格式
    2. 且 \\boxed{...} 中提取出的答案与 ground_truth 数值一致
    """
    if not text or not ground_truth:
        return False

    # 1. 专门提取 \boxed 中的内容
    # 我们只关心 boxed，忽略 answer is 或行内数字
    boxed_candidates = []
    
    # 注意：这里假设你已经有了 _iter_boxed 函数
    # 如果没有，我在下面补上了它的实现
    for inner, pos in _iter_boxed(text):
        # 提取并清洗 boxed 内部的数字/分数
        val = _extract_number_or_frac(inner)
        if val:
            boxed_candidates.append((pos, val))

    # 如果连 boxed 都没有，直接 False
    if not boxed_candidates:
        return False

    # 2. 锁定最终答案
    # 按照惯例，如果输出了多个 boxed，以最后一个为准
    boxed_candidates.sort(key=lambda x: x[0])
    final_boxed_answer = boxed_candidates[-1][1]

    # 3. 验证答案正确性
    # 直接复用你之前写好的 check_answer_correct 进行数值对比
    # 注意：这里我们将 final_boxed_answer 作为 extracted 传入
    return check_answer_correct(final_boxed_answer, ground_truth)



def calculate_reward(
    text: str,
    ground_truth: str,
    token_length: int,
    min_tokens: int = 50,
    max_tokens: int = 2048,
    length_weight: float = 0.5,
    repeat_ngram: bool = False,
    repeat_penalty: float = 0,
    zstd_threshold: float = 3,  # 阈值：超过这个压缩率视为复读
    zstd_penalty_weight: float = 0.5
) -> Tuple[float, bool, bool]:
    reward, is_correct, is_format_correct, _ = calculate_reward_details(
        text=text,
        ground_truth=ground_truth,
        token_length=token_length,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
        length_weight=length_weight,
        repeat_ngram=repeat_ngram,
        repeat_penalty=repeat_penalty,
        zstd_threshold= zstd_threshold,  # 阈值：超过这个压缩率视为复读
        zstd_penalty_weight=zstd_penalty_weight
    )
    return reward, is_correct, is_format_correct


def calculate_reward_details(
    text: str,
    ground_truth: str,
    token_length: int,
    min_tokens: int = 50,
    max_tokens: int = 2048,
    length_weight: float = 0.5,
    repeat_ngram: bool = False,
    repeat_penalty: float = 0,
    zstd_threshold: float = 3,  # 阈值：超过这个压缩率视为复读
    zstd_penalty_weight: float = 0.5
) -> Tuple[float, bool, bool, dict]:
    extracted = extract_answer(text, ground_truth)
    gt_extracted = _extract_ground_truth(ground_truth)
    is_correct = check_answer_correct(extracted, ground_truth)
    is_format_correct = check_format_correct(text,ground_truth)

    reward = 0.0
    correct_reward = 0.0
    format_reward = 0.0
    if is_correct:
        correct_reward = 1.0
        reward += correct_reward
        if is_format_correct:
            format_reward = 1.0
            reward += format_reward

    token_length = max(min_tokens, min(token_length, max_tokens))
    if max_tokens > min_tokens:
        lambda_val = 0.5 - (token_length - min_tokens) / (max_tokens - min_tokens)
    else:
        lambda_val = 0.0

    if is_correct:
        length_reward = length_weight * lambda_val
    else:
        length_reward = length_weight * min(0.0, lambda_val)

    reward += length_reward

    repeat_reward = 0.0
    if repeat_ngram:
        repeat_reward = repeat_penalty
        reward += repeat_reward
    comp_reward = 0.0
    zstd_ratio = 0.0
    
    if text:
        raw = text.encode("utf-8", errors="ignore")
        # 只有文本足够长才有压缩检测的意义，太短的文本压缩率波动大
        if len(raw) > 100: 
            comp = global_compressor.compress(raw)
            # 避免除以零
            if len(comp) > 0:
                zstd_ratio = len(raw) / len(comp)
            
            # 逻辑：只有当压缩率超过阈值时才开始扣分
            # 例如：阈值 4.5，实际 6.5，超出了 2.0
            # 扣分 = 2.0 * 0.5 = -1.0
                excess = zstd_ratio - zstd_threshold
                comp_reward = -(max(0,excess) * zstd_penalty_weight)
                
                # 可选：设置一个最大惩罚上限，防止 reward 变成负无穷
                # repeat_reward = max(repeat_reward, -5.0)
    reward += comp_reward
    details = {
        "extracted_answer": extracted,
        "ground_truth_answer": gt_extracted,
        "correct_reward": correct_reward,
        "format_reward": format_reward,
        "length_lambda": lambda_val,
        "length_reward": length_reward,
        "repeat_ngram": repeat_ngram,
        "repeat_penalty": repeat_reward,
        "min_tokens": min_tokens,
        "max_tokens": max_tokens,
        "zstd_ratio": zstd_ratio,
        "zstd_penalty": comp_reward,
        "token_length": token_length,
    }
    return reward, is_correct, is_format_correct, details