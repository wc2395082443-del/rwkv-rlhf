#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
from typing import Optional, Tuple


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
    s = _strip_math_delims(s)
    s = s.replace(",", "").replace("$", "").replace("%", "")
    frac = re.findall(r"[-+]?\d+\s*/\s*[-+]?\d+", s)
    if frac:
        return frac[-1].replace(" ", "")
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", s)
    if nums:
        return nums[-1]
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
    """检查提取的答案是否正确"""
    if not extracted or not ground_truth:
        return False

    gt_extracted = _extract_ground_truth(ground_truth)
    if not gt_extracted:
        return False

    def normalize(s: str) -> str:
        s = str(s).strip().lower()
        s = _strip_math_delims(s)
        s = re.sub(r"\s+", " ", s)
        return s

    ext_norm = normalize(extracted)
    gt_norm = normalize(gt_extracted)
    return ext_norm == gt_norm


def check_format_correct(text: str) -> bool:
    """
    检查格式是否正确
    要求至少满足一种正确格式：
    1. 包含 \\boxed{...}
    2. 包含 answer is ...
    3. 包含 answer: ...
    """
    if not text:
        return False

    if r"\boxed{" in text:
        return True

    if re.search(r"answer\s*is[^A-Za-z0-9]*\S", text, re.IGNORECASE):
        return True

    if re.search(r"answer\s*:[^A-Za-z0-9]*\S", text, re.IGNORECASE):
        return True

    return False


def calculate_reward(
    text: str,
    ground_truth: str,
    token_length: int,
    min_tokens: int = 50,
    max_tokens: int = 2048,
    length_weight: float = 0.5,
    repeat_ngram: bool = False,
    repeat_penalty: float = -0.5
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
    repeat_penalty: float = -0.5
) -> Tuple[float, bool, bool, dict]:
    extracted = extract_answer(text, ground_truth)
    gt_extracted = _extract_ground_truth(ground_truth)
    is_correct = check_answer_correct(extracted, ground_truth)
    is_format_correct = check_format_correct(text)

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
        "token_length": token_length,
    }
    return reward, is_correct, is_format_correct, details