#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import math
from typing import Optional, Tuple
import zstandard as zstd

global_compressor = zstd.ZstdCompressor(level=9)
BOOL_ANSWERS = {"yes", "no", "true", "false"}


def _strip_math_delims(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    s = s.replace(r"\left", "").replace(r"\right", "")
    s = re.sub(r"\\[,;\!\:]\s*", "", s)
    return s.strip()


def _strip_outer_braces(s: str) -> str:
    s = str(s).strip()
    while len(s) >= 2 and s[0] == "{" and s[-1] == "}":
        depth = 0
        ok = True
        for i, ch in enumerate(s):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and i != len(s) - 1:
                    ok = False
                    break
        if not ok or depth != 0:
            break
        s = s[1:-1].strip()
    return s


def _fix_fracs(string: str) -> str:
    while "\\frac " in string:
        string = string.replace("\\frac ", "\\frac")
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if len(substr) > 0 and substr[0] == "{":
                new_str += substr
            else:
                if len(substr) < 2:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    post_substr = substr[2:] if len(substr) > 2 else ""
                    new_str += "{" + a + "}{" + b + "}" + post_substr
                else:
                    post_substr = substr[2:] if len(substr) > 2 else ""
                    new_str += "{" + a + "}" + b + post_substr
    return new_str


def _process_and_or_inside_text(string: str) -> str:
    string = re.sub(r"\s*\\text\{\s*(or|and)\s*\}\s*", ",", string)
    string = re.sub(r",\s*,", ",", string)
    return string


def _remove_right_units(expr: str) -> str:
    if "\\text" in expr:
        try:
            splits = re.split(r"\\text\s*\{\s*", expr)
            if len(splits) == 2 and splits[0] not in ("", "("):
                return splits[0]
        except Exception:
            pass

    if "\\text{" in expr:
        return re.sub(r"\\text\{([^}]+)\}", r"\1", expr)
    if "\\mbox{" in expr:
        splits = expr.split("\\mbox{")
        if len(splits) == 2:
            return splits[0]
    return expr


def _fix_sqrt(string: str) -> str:
    return re.sub(r"\\sqrt(\s*\w+)", r"\\sqrt{\1}", string)


def _inject_implicit_mixed_number(step: str) -> str:
    return re.compile(r"([0-9]) +([0-9])").sub(r"\1+\2", step)


def _inject_implicit_mixed_fraction(step: str) -> str:
    pattern = re.compile(r"(\d+) *\\frac\{(\d+)\}\{(\d+)\}")

    def replacer(match):
        whole_part = match.group(1)
        numerator = match.group(2)
        denominator = match.group(3)
        if whole_part:
            return f"{whole_part}+{numerator}/{denominator}"
        return f"{numerator}/{denominator}"

    return pattern.sub(replacer, step)


def normalize_answer_string(expr: str) -> Optional[str]:
    if expr is None:
        return None

    expr = _strip_math_delims(expr)
    expr = _strip_outer_braces(expr)
    expr = _process_and_or_inside_text(expr)
    expr = _remove_right_units(expr)

    for surround_str in ["\\text", "\\mathrm", "\\mathcal", "\\textbf", "\\textit"]:
        expr = expr.replace(surround_str, "")

    expr = expr.replace(r"\!", "")
    expr = expr.replace("\\%", "%")
    expr = expr.replace("\\$", "$")
    expr = expr.replace("$", "")
    expr = expr.replace("%", "")
    expr = expr.replace("^{\\circ}", "")
    expr = expr.replace(" or ", ",")
    expr = expr.replace(" and ", ",")
    expr = expr.replace("?", "-").replace("?", "-")
    expr = re.sub(r"\^ *\\circ", "", expr)
    expr = _fix_sqrt(expr)
    expr = _fix_fracs(expr)
    expr = re.sub(r"- *", "-", expr)
    expr = _inject_implicit_mixed_number(expr)
    expr = _inject_implicit_mixed_fraction(expr)
    expr = expr.strip()

    if re.fullmatch(r"-?\d{1,3}(,\d{3})+(\.\d+)?", expr):
        expr = expr.replace(",", "")

    expr = re.sub(r"\s+", "", expr)
    return expr or None


def _extract_number_or_frac(s: str) -> Optional[str]:
    if not s:
        return None

    s = _strip_math_delims(s)
    s = s.replace("?", "-").replace("?", "-")
    s = s.replace(",", "").replace("$", "").replace("%", "")
    s = re.sub(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"\1/\2", s)

    fracs = re.findall(r"[-+]?\d+\s*/\s*[-+]?\d+", s)
    if fracs:
        return fracs[-1].replace(" ", "")

    nums = re.findall(r"[-+]?(?:\d+\.?\d*|\.\d+)", s)
    valid_nums = [n for n in nums if re.search(r"\d", n)]
    if valid_nums:
        return valid_nums[-1]
    return None


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


def _iter_boxed(text: str):
    for m in re.finditer(r"\\boxed\{", text):
        brace_start = m.end() - 1
        got = _find_balanced_brace(text, brace_start)
        if got is None:
            continue
        inner, _ = got
        yield inner, m.start()


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


def _normalize_choice_answer(expr: str) -> str:
    expr = expr.strip()
    m = re.fullmatch(r"\(?\s*([A-Za-z])\s*\)?", expr)
    if m:
        return m.group(1).upper()
    return expr


def _normalize_set_prefix(expr: str) -> str:
    expr = expr.strip()
    expr = re.sub(r"^[A-Za-z]+\\in", "", expr)
    expr = re.sub(r"^[A-Za-z]+?", "", expr)
    return expr.strip()


def _normalize_latex_constants(expr: str) -> str:
    replacements = {
        r"\pi/2": r"\frac{\pi}{2}",
        r"\pi/3": r"\frac{\pi}{3}",
        r"\pi/4": r"\frac{\pi}{4}",
        r"\pi/6": r"\frac{\pi}{6}",
    }
    for src, dst in replacements.items():
        expr = expr.replace(src, dst)
    return expr


def _split_top_level(expr: str, sep: str):
    parts = []
    buf = []
    depth_paren = 0
    depth_brace = 0
    i = 0
    while i < len(expr):
        ch = expr[i]
        if ch in '([':
            depth_paren += 1
        elif ch in ')]':
            depth_paren = max(0, depth_paren - 1)
        elif ch == '{':
            depth_brace += 1
        elif ch == '}':
            depth_brace = max(0, depth_brace - 1)
        if depth_paren == 0 and depth_brace == 0 and expr.startswith(sep, i):
            parts.append(''.join(buf).strip())
            buf = []
            i += len(sep)
            continue
        buf.append(ch)
        i += 1
    parts.append(''.join(buf).strip())
    return parts


def _extract_expr_candidate(s: str) -> Optional[str]:
    if s is None:
        return None
    raw = _strip_math_delims(str(s))
    raw_choice = _normalize_choice_answer(raw)
    if raw_choice in {"A", "B", "C", "D", "E"}:
        return raw_choice
    s = normalize_answer_string(s)
    if not s:
        return None
    s = _normalize_set_prefix(s)
    s = _normalize_latex_constants(s)
    choice = _normalize_choice_answer(s)
    if choice in {"A", "B", "C", "D", "E"}:
        return choice
    return s if s else None


def extract_answer(text: str, ground_truth: str = None) -> Optional[str]:
    if not text or not isinstance(text, str):
        return None

    gt_candidate = _extract_expr_candidate(ground_truth) if ground_truth is not None else None
    if gt_candidate in {"A", "B", "C", "D", "E"}:
        for inner, _ in _iter_boxed(text):
            choice = _normalize_choice_answer(_strip_math_delims(inner))
            if choice in {"A", "B", "C", "D", "E"}:
                return choice
        boxed = extract_last_boxed(text)
        if boxed:
            choice = _normalize_choice_answer(_strip_math_delims(boxed))
            if choice in {"A", "B", "C", "D", "E"}:
                return choice

    candidates = []
    for inner, pos in _iter_boxed(text):
        expr = _extract_expr_candidate(inner)
        if expr:
            candidates.append((pos, expr))
        else:
            boxed_num = _extract_number_or_frac(inner)
            if boxed_num:
                candidates.append((pos, boxed_num))

    expr_patterns = [
        r"answer\s*is\s*(.+)$",
        r"answer\s*:\s*(.+)$",
        r"final\s*answer\s*is\s*(.+)$",
    ]
    for pattern in expr_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE):
            tail = match.group(1).strip().splitlines()[0].strip()
            expr = _extract_expr_candidate(tail)
            if expr:
                candidates.append((match.start(), expr))
            else:
                answer = _extract_number_or_frac(tail)
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
    if not ground_truth:
        return None
    expr = _extract_expr_candidate(ground_truth)
    if expr:
        return expr
    gt = extract_answer(ground_truth)
    if gt:
        return gt
    return _extract_number_or_frac(ground_truth)


def is_digit(s) -> tuple[bool, Optional[float]]:
    try:
        num = float(str(s).replace(",", ""))
        return True, num
    except Exception:
        return False, None


def format_intervals(prediction: str) -> str:
    patterns = {
        "Interval(": r"^Interval\((.*)\)$",
        "Interval.Ropen(": r"^Interval\.Ropen\((.*)\)$",
        "Interval.Lopen(": r"^Interval\.Lopen\((.*)\)$",
        "Interval.open(": r"^Interval\.open\((.*)\)$",
    }
    for key, pattern in patterns.items():
        match = re.match(pattern, prediction)
        if match:
            inner_content = match.group(1)
            if key == "Interval(":
                return f"[{inner_content}]"
            if key == "Interval.Ropen(":
                return f"[{inner_content})"
            if key == "Interval.Lopen(":
                return f"({inner_content}]"
            if key == "Interval.open(":
                return f"({inner_content})"
    return prediction


def symbolic_equal(a: str, b: str, tolerance: float = 1e-4) -> bool:
    try:
        import sympy
        from sympy.parsing.latex import parse_latex
        from sympy.parsing.sympy_parser import parse_expr
    except Exception:
        return False

    def _parse(s: str):
        for fn in (parse_expr, parse_latex):
            try:
                return fn(s)
            except Exception:
                pass
        return s

    a = _parse(a)
    b = _parse(b)
    try:
        if sympy.simplify(a - b) == 0:
            return True
    except Exception:
        pass
    try:
        return math.isclose(float(sympy.N(a)), float(sympy.N(b)), rel_tol=tolerance, abs_tol=tolerance)
    except Exception:
        return False


def math_equal(prediction, reference, include_percentage: bool = True, tolerance: float = 1e-4) -> bool:
    prediction = _extract_expr_candidate(prediction)
    reference = _extract_expr_candidate(reference)

    if prediction is None or reference is None:
        return False
    if prediction == reference:
        return True
    if prediction.strip().lower() == reference.strip().lower():
        return True
    if prediction.replace(" ", "") == reference.replace(" ", ""):
        return True

    pred_is_digit, pred_val = is_digit(prediction)
    ref_is_digit, ref_val = is_digit(reference)
    if pred_is_digit and ref_is_digit:
        candidates = [ref_val / 100.0, ref_val, ref_val * 100.0] if include_percentage else [ref_val]
        return any(math.isclose(pred_val, x, rel_tol=tolerance, abs_tol=tolerance) for x in candidates)

    prediction = _normalize_set_prefix(format_intervals(str(prediction).strip()))
    reference = _normalize_set_prefix(format_intervals(str(reference).strip()))

    pred_choice = _normalize_choice_answer(prediction)
    ref_choice = _normalize_choice_answer(reference)
    if pred_choice == ref_choice:
        return True

    if (
        prediction and reference and prediction[0] in "([" and prediction[-1] in ")]"
        and reference[0] == prediction[0] and reference[-1] == prediction[-1]
    ):
        pred_parts = _split_top_level(prediction[1:-1], ',')
        ref_parts = _split_top_level(reference[1:-1], ',')
        if len(pred_parts) == len(ref_parts) and len(pred_parts) > 1:
            return all(math_equal(p, r, include_percentage, tolerance) for p, r in zip(pred_parts, ref_parts))

    if r"\cup" in prediction and r"\cup" in reference:
        pred_parts = _split_top_level(prediction, r'\cup')
        ref_parts = _split_top_level(reference, r'\cup')
        if len(pred_parts) == len(ref_parts) and len(pred_parts) > 1:
            return all(math_equal(p, r, include_percentage, tolerance) for p, r in zip(pred_parts, ref_parts))

    if ',' in prediction and ',' in reference:
        pred_parts = _split_top_level(prediction, ',')
        ref_parts = _split_top_level(reference, ',')
        if len(pred_parts) == len(ref_parts) and len(pred_parts) > 1:
            return all(math_equal(p, r, include_percentage, tolerance) for p, r in zip(pred_parts, ref_parts))

    return symbolic_equal(prediction, reference, tolerance)


def check_answer_correct(extracted: str, ground_truth: str) -> bool:
    if extracted is None or not ground_truth:
        return False
    gt_extracted = _extract_ground_truth(ground_truth)
    if gt_extracted is None:
        return False
    return math_equal(extracted, gt_extracted, include_percentage=True, tolerance=1e-4)


def check_format_correct(text: str, ground_truth: str) -> bool:
    if not text or not ground_truth:
        return False

    boxed_candidates = []
    for inner, pos in _iter_boxed(text):
        val = _extract_expr_candidate(inner)
        if val:
            boxed_candidates.append((pos, val))
        else:
            num = _extract_number_or_frac(inner)
            if num:
                boxed_candidates.append((pos, num))

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
