#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import math
from typing import Optional, Tuple
import zstandard as zstd

global_compressor = zstd.ZstdCompressor(level=9)
BOOL_ANSWERS = {"yes", "no", "true", "false"}


def _strip_math_delims(s: str) -> str:
    """???????"""
    if s is None:
        return ""
    s = str(s).strip()
    s = s.replace(r"\left", "").replace(r"\right", "")
    s = re.sub(r"\\[,\;\!\:]\s*", "", s)
    return s.strip()


def _extract_bool_answer(text: str) -> Optional[str]:
    if not text:
        return None
    boxed = re.findall(r"\\boxed\s*\{\s*(yes|no|true|false)\s*\}", text, flags=re.IGNORECASE)
    if boxed:
        return boxed[-1].lower()
    matches = re.findall(r"\b(yes|no|true|false)\b", text, flags=re.IGNORECASE)
    if matches:
        return matches[-1].lower()
    return None


def _extract_number_or_frac(s: str) -> Optional[str]:
    if not s:
        return None

    s = _strip_math_delims(s)
    s = s.replace("?", "-").replace("?", "-")
    s = s.replace(",", "").replace("$", "").replace("%", "")
    s = re.sub(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"\1/\2", s)

    frac_pattern = r"[-+]?\d+\s*/\s*[-+]?\d+"
    fracs = re.findall(frac_pattern, s)
    if fracs:
        return fracs[-1].replace(" ", "")

    num_pattern = r"[-+]?(?:\d+\.?\d*|\.\d+)"
    nums = re.findall(num_pattern, s)
    if nums:
        valid_nums = [n for n in nums if re.search(r"\d", n)]
        if valid_nums:
            return valid_nums[-1]

    return None


def _find_balanced_brace(text: str, brace_start: int) -> Optional[Tuple[str, int]]:
    """????????"""
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
    """??????\\boxed{}??"""
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
    ?????????????????? bool ???
    """
    if not text or not isinstance(text, str):
        return None

    gt_norm = str(ground_truth).strip().lower() if ground_truth is not None else ""
    if gt_norm in BOOL_ANSWERS:
        return _extract_bool_answer(text)

    candidates = []
    for inner, pos in _iter_boxed(text):
        boxed_num = _extract_number_or_frac(inner)
        if boxed_num:
            candidates.append((pos, boxed_num))

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
    if ground_truth is None:
        return None
    gt_text = str(ground_truth).strip()
    if not gt_text:
        return None
    gt_norm = gt_text.lower()
    if gt_norm in BOOL_ANSWERS:
        return gt_norm
    gt = extract_answer(gt_text)
    if gt:
        return gt
    return _extract_number_or_frac(gt_text)


def check_answer_correct(extracted: str, ground_truth: str) -> bool:
    """
    ???????? bool ?????
    """
    if extracted is None or ground_truth is None:
        return False

    gt_extracted = _extract_ground_truth(ground_truth)
    if gt_extracted is None:
        return False

    ext_text = str(extracted).strip().lower()
    if gt_extracted in BOOL_ANSWERS:
        return ext_text == gt_extracted

    def clean_number_str(s: str) -> str:
        s = str(s).strip().lower()
        s = _strip_math_delims(s)
        s = s.replace(',', '')
        return s

    ext_clean = clean_number_str(extracted)
    gt_clean = clean_number_str(gt_extracted)

    try:
        if abs(float(ext_clean) - float(gt_clean)) < 1e-6:
            return True
    except ValueError:
        pass

    return ext_clean.replace(" ", "") == gt_clean.replace(" ", "")


def check_format_correct(text: str, ground_truth: str) -> bool:
    """
    ???????boxed ????????? ground_truth ???
    """
    if not text or not ground_truth:
        return False

    gt_extracted = _extract_ground_truth(ground_truth)
    if gt_extracted in BOOL_ANSWERS:
        boxed = extract_last_boxed(text)
        return boxed is not None and str(boxed).strip().lower() == gt_extracted

    boxed_candidates = []
    for inner, pos in _iter_boxed(text):
        val = _extract_number_or_frac(inner)
        if val:
            boxed_candidates.append((pos, val))

    if not boxed_candidates:
        return False

    boxed_candidates.sort(key=lambda x: x[0])
    final_boxed_answer = boxed_candidates[-1][1]
    return check_answer_correct(final_boxed_answer, ground_truth)


def calculate_reward(
    text: str,
    ground_truth: str,
    token_length: int,
    min_tokens: int = 50,
    max_tokens: int = 2048,
    length_weight: float = 0.0,
    repeat_ngram: bool = False,
    repeat_penalty: float = 0,
    zstd_threshold: float = 2.5,
    zstd_penalty_weight: float = 0.5,
    reward_mode: str = "rwkv"
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
        zstd_threshold=zstd_threshold,
        zstd_penalty_weight=zstd_penalty_weight,
        reward_mode=reward_mode,
    )
    return reward, is_correct, is_format_correct


def calculate_reward_details(
    text: str,
    ground_truth: str,
    token_length: int,
    min_tokens: int = 50,
    max_tokens: int = 2048,
    length_weight: float = 0.0,
    repeat_ngram: bool = False,
    repeat_penalty: float = 0,
    zstd_threshold: float = 2.5,
    zstd_penalty_weight: float = 0.5,
    reward_mode: str = "rwkv"
) -> Tuple[float, bool, bool, dict]:
    extracted = extract_answer(text, ground_truth)
    gt_extracted = _extract_ground_truth(ground_truth)
    is_correct = check_answer_correct(extracted, ground_truth)
    is_format_correct = check_format_correct(text, ground_truth)

    reward = 0.0
    correct_reward = 0.0
    format_reward = 0.0
    token_length = max(min_tokens, min(token_length, max_tokens))
    lambda_val = 0.0
    length_reward = 0.0
    repeat_reward = 0.0
    comp_reward = 0.0
    zstd_ratio = 0.0

    if reward_mode == "trl_doc":
        if is_correct:
            correct_reward = 1.0
            reward = 1.0
    else:
        if is_correct:
            correct_reward = 1.0
            reward += correct_reward
            if is_format_correct:
                format_reward = 1.0
                reward += format_reward

        if max_tokens > min_tokens:
            lambda_val = 0.5 - (token_length - min_tokens) / (max_tokens - min_tokens)
        else:
            lambda_val = 0.0

        if is_correct:
            length_reward = length_weight * lambda_val
        else:
            length_reward = length_weight * min(0.0, lambda_val)

        if length_reward > 0.25:
            length_reward = 0.25
        elif length_reward < -0.25:
            length_reward = -0.25

        reward += length_reward

        if text:
            raw = text.encode("utf-8", errors="ignore")
            if len(raw) > 100:
                comp = global_compressor.compress(raw)
                if len(comp) > 0:
                    zstd_ratio = len(raw) / len(comp)
                    x = zstd_ratio - zstd_threshold
                    if x > 0:
                        b = 1.0
                        x1, y1 = 0.5, 0.1
                        x2, y2 = 1.0, 0.25
                        e1 = math.exp(b * x1) - 1.0
                        e2 = math.exp(b * x2) - 1.0
                        a = (y1 - (x1 / x2) * y2) / (e1 - (x1 / x2) * e2)
                        c = (y2 - a * e2) / x2
                        penalty = a * (math.exp(b * x) - 1.0) + c * x
                        if penalty > 1.0:
                            penalty = 1.0
                        comp_reward = -float(zstd_penalty_weight) * penalty
        reward += comp_reward

        if is_correct:
            neg = 0.0
            for v in (length_reward, repeat_reward, comp_reward):
                if v < 0:
                    neg += v
            if neg < -0.5:
                reward += (-0.5 - neg)

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
        "reward_mode": reward_mode,
    }
    return reward, is_correct, is_format_correct, details
