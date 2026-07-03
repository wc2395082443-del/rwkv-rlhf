#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import math
from typing import Optional, Tuple

import zstandard as zstd

try:
    from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse as math_verify_parse, verify as math_verify_verify
    MATH_VERIFY_AVAILABLE = True
    _MATH_VERIFY_PRED_CONFIG = (LatexExtractionConfig(), ExprExtractionConfig())
    _MATH_VERIFY_GOLD_CONFIG = (LatexExtractionConfig(boxed_match_priority=0), ExprExtractionConfig())
except Exception:
    ExprExtractionConfig = None
    LatexExtractionConfig = None
    math_verify_parse = None
    math_verify_verify = None
    MATH_VERIFY_AVAILABLE = False
    _MATH_VERIFY_PRED_CONFIG = ()
    _MATH_VERIFY_GOLD_CONFIG = ()

global_compressor = zstd.ZstdCompressor(level=9)


def _strip_math_delims(s: str) -> str:
    """???????"""
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
    ??????????????????
    1. \\boxed{${answer}}
    2. answer is ${answer}
    3. answer: ${answer}
    4. ?????/??

    ??????????????????????
    """
    if not text or not isinstance(text, str):
        return None

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


def _normalize_answer_judge(answer_judge: str) -> str:
    mode = str(answer_judge or "legacy").strip().lower()
    return mode if mode in {"legacy", "math_verify", "auto"} else "legacy"


def _stringify_parsed_answer(parsed) -> Optional[str]:
    if not parsed:
        return None
    if isinstance(parsed, (list, tuple)):
        return " || ".join(str(x) for x in parsed)
    return str(parsed)


def _parse_with_math_verify(text: str, extraction_config) -> list:
    if not MATH_VERIFY_AVAILABLE or not text:
        return []
    try:
        parsed = math_verify_parse(
            str(text),
            extraction_config=extraction_config,
            fallback_mode="first_match",
            extraction_mode="any_match",
            parsing_timeout=5,
            raise_on_error=False,
        )
    except Exception:
        return []
    return parsed or []


def _parse_math_verify_gold(ground_truth: str) -> list:
    gt = str(ground_truth or "").strip()
    if not gt:
        return []
    for cand in (f"${gt}$", rf"\boxed{{{gt}}}", gt):
        for cfg in (_MATH_VERIFY_PRED_CONFIG, _MATH_VERIFY_GOLD_CONFIG):
            parsed = _parse_with_math_verify(cand, cfg)
            if parsed:
                return parsed
    return []


def _parse_math_verify_pred(text: str) -> list:
    parsed = _parse_with_math_verify(text, _MATH_VERIFY_PRED_CONFIG)
    if parsed:
        return parsed

    boxed = extract_last_boxed(text)
    if boxed:
        for cand in (boxed, f"${boxed}$", rf"\boxed{{{boxed}}}"):
            parsed = _parse_with_math_verify(cand, _MATH_VERIFY_PRED_CONFIG)
            if parsed:
                return parsed

    extracted = extract_answer(text)
    if extracted:
        parsed = _parse_with_math_verify(extracted, _MATH_VERIFY_PRED_CONFIG)
        if parsed:
            return parsed

    return []


def _extract_ground_truth(ground_truth: str) -> Optional[str]:
    if not ground_truth:
        return None
    gt = extract_answer(ground_truth)
    if gt:
        return gt
    return _extract_number_or_frac(ground_truth)


def check_answer_correct(extracted: str, ground_truth: str) -> bool:
    """?????????????????"""
    if not extracted or not ground_truth:
        return False

    gt_extracted = _extract_ground_truth(ground_truth)
    if not gt_extracted:
        return False

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


def judge_answer(text: str, ground_truth: str, answer_judge: str = "legacy") -> Tuple[bool, Optional[str], Optional[str]]:
    mode = _normalize_answer_judge(answer_judge)
    if mode in {"math_verify", "auto"} and MATH_VERIFY_AVAILABLE:
        pred_parsed = _parse_math_verify_pred(text)
        gold_parsed = _parse_math_verify_gold(ground_truth)
        pred_display = _stringify_parsed_answer(pred_parsed)
        gt_display = _stringify_parsed_answer(gold_parsed)
        if pred_parsed and gold_parsed:
            try:
                ok = bool(
                    math_verify_verify(
                        gold_parsed,
                        pred_parsed,
                        strict=True,
                        timeout_seconds=5,
                        raise_on_error=False,
                    )
                )
            except Exception:
                ok = False
            if ok:
                return True, pred_display, gt_display

    extracted = extract_answer(text, ground_truth)
    gt_extracted = _extract_ground_truth(ground_truth)
    return check_answer_correct(extracted, ground_truth), (pred_display if mode in {"math_verify", "auto"} and MATH_VERIFY_AVAILABLE and pred_display else extracted), (gt_display if mode in {"math_verify", "auto"} and MATH_VERIFY_AVAILABLE and gt_display else gt_extracted)


def check_format_correct(text: str, ground_truth: str, answer_judge: str = "legacy") -> bool:
    """
    ???????
    ?? True ?????
    1. ????? \\boxed{...} ??
    2. ? \\boxed{...} ??????? ground_truth ??
    """
    if not text or not ground_truth:
        return False

    final_boxed = extract_last_boxed(text)
    if not final_boxed:
        return False

    mode = _normalize_answer_judge(answer_judge)
    if mode in {"math_verify", "auto"} and MATH_VERIFY_AVAILABLE:
        gold_parsed = _parse_math_verify_gold(ground_truth)
        if gold_parsed:
            for cand in (rf"\boxed{{{final_boxed}}}", f"${final_boxed}$", final_boxed):
                pred_parsed = _parse_with_math_verify(cand, _MATH_VERIFY_PRED_CONFIG)
                if not pred_parsed:
                    continue
                try:
                    if bool(
                        math_verify_verify(
                            gold_parsed,
                            pred_parsed,
                            strict=True,
                            timeout_seconds=5,
                            raise_on_error=False,
                        )
                    ):
                        return True
                except Exception:
                    pass

    final_boxed_answer = _extract_number_or_frac(final_boxed)
    if not final_boxed_answer:
        return False
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
    zstd_penalty_weight: float = 0.2,
    answer_judge: str = "legacy",
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
        answer_judge=answer_judge,
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
    zstd_penalty_weight: float = 0.2,
    answer_judge: str = "legacy",
) -> Tuple[float, bool, bool, dict]:
    is_correct, extracted, gt_extracted = judge_answer(text, ground_truth, answer_judge=answer_judge)
    is_format_correct = check_format_correct(text, ground_truth, answer_judge=answer_judge)

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

    if length_reward > 0.25:
        length_reward = 0.25
    elif length_reward < -0.25:
        length_reward = -0.25

    reward += length_reward

    repeat_reward = 0.0
    repeat_penalty = 0.0
    if repeat_ngram:
        repeat_reward = 0.0
        reward += repeat_reward
    comp_reward = 0.0
    zstd_ratio = 0.0

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
        "answer_judge": _normalize_answer_judge(answer_judge),
        "math_verify_available": MATH_VERIFY_AVAILABLE,
    }
    return reward, is_correct, is_format_correct, details
