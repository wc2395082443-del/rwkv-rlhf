import contextlib
import re
import signal
from math import isclose
from typing import Union

from math_verify import parse, verify


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
                try:
                    assert len(substr) >= 2
                except Exception:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    return new_str


def _strip_properly_formatted_commas(expr: str) -> str:
    pattern = re.compile(r"(\d)(,)(\d\d\d)($|\D)")
    while True:
        next_expr = pattern.sub("\\1\\3\\4", expr)
        if next_expr == expr:
            break
        expr = next_expr
    return next_expr


def _str_is_int(x: str) -> bool:
    try:
        x = _strip_properly_formatted_commas(x)
        x = float(x)
        return abs(x - int(round(x))) <= 1e-7
    except Exception:
        return False


def _str_to_int(x: str) -> int:
    x = x.replace(",", "")
    if "_" in x:
        x = x.split("_")[0]
    x = float(x)
    return int(x)


def _inject_implicit_mixed_number(step: str) -> str:
    return re.compile(r"([0-9]) +([0-9])").sub("\\1+\\2", step)


def _remove_right_units(expr: str) -> str:
    if "\\text" in expr:
        try:
            splits = re.split(r"\\text\s*{\s*", expr)
            assert len(splits) == 2 and splits[0] not in ("", "(")
            return splits[0]
        except AssertionError:
            pass

    if "\\text{" in expr:
        return re.sub(r"\\text{([^}]+)}", r"\1", expr)
    if "\\mbox{" in expr:
        splits = expr.split("\\mbox{")
        if len(splits) == 2:
            return splits[0]
    return expr


def _process_and_or_inside_text(string: str) -> str:
    string = re.sub(r"\s*\\text{\s*(or|and)\s*}\s*", ",", string)
    string = re.sub(r",\s*,", ",", string)
    return string


def _remove_left_and_right(expr: str) -> str:
    expr = re.sub(r"\\left", "", expr)
    expr = re.sub(r"\\right", "", expr)
    return expr


def _fix_sqrt(string: str) -> str:
    return re.sub(r"\\sqrt(\s*\w+)", r"\\sqrt{\1}", string)


def _fix_interval(expr: str) -> str:
    if "\\in " in expr:
        return expr.split("\\in ")[1].strip()
    return expr


def _inject_implicit_mixed_fraction(step: str) -> str:
    pattern = re.compile(r"(\d+) *\\frac{(\d+)}{(\d+)}")

    def replacer(match: re.Match[str]) -> str:
        whole_part = match.group(1)
        numerator = match.group(2)
        denominator = match.group(3)
        if whole_part:
            return f"{whole_part} + {numerator}/{denominator}"
        return f"{numerator}/{denominator}"

    return pattern.sub(replacer, step)


def normalize_answer_string(expr: str) -> str | None:
    if expr is None:
        return None

    expr = _remove_left_and_right(expr)
    expr = _process_and_or_inside_text(expr)
    expr = _remove_right_units(expr)
    expr = _fix_interval(expr)

    for surround_str in ["\\\\text", "\\\\mathrm", "\\\\mathcal", "\\\\textbf", "\\\\textit"]:
        expr = expr.replace(surround_str, "")
        pattern = f"^{surround_str}" + r"\{(?P<text>.+?)\}$"
        match = re.search(pattern, expr)
        if match is not None:
            expr = match.group("text")

    expr = expr.replace(r"\!", "")
    expr = expr.replace("\\%", "%")
    expr = expr.replace("\\$", "$")
    expr = expr.replace("$", "")
    expr = expr.replace("%", "")
    expr = expr.replace("^{\\circ}", "")
    expr = expr.replace(" or ", " , ")
    expr = expr.replace(" and ", " , ")
    expr = expr.replace("million", "*10^6")
    expr = expr.replace("billion", "*10^9")
    expr = expr.replace("trillion", "*10^12")

    for unit in [
        "degree",
        "cm",
        "centimeter",
        "meter",
        "mile",
        "second",
        "minute",
        "hour",
        "week",
        "month",
        "year",
        "foot",
        "feet",
        "inch",
        "yard",
        "p.m.",
        "PM",
    ]:
        expr = re.sub(rf"{unit}(es)?(s)? *(\^[0-9]+)?", "", expr)

    if "day" in expr:
        weekday_expressed = any(day in expr for day in [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ])
        if not weekday_expressed:
            expr = re.sub(r"day(s)?", "", expr)

    expr = re.sub(r"\^ *\\\\circ", "", expr)

    if len(expr) > 0 and expr[0] == "{" and expr[-1] == "}":
        expr = expr[1:-1]

    expr = _fix_sqrt(expr)
    expr = _fix_fracs(expr)
    expr = re.sub("- *", "-", expr)
    expr = _inject_implicit_mixed_number(expr)
    expr = _inject_implicit_mixed_fraction(expr)
    expr = expr.replace(" ", "")

    if _str_is_int(expr):
        expr = str(_str_to_int(expr))

    return expr


def is_digit(s: Union[bool, float, str]) -> tuple[bool, float | None]:
    try:
        if "{,}" in str(s):
            num = float(str(s).replace("{,}", ""))
            return True, num
        num = float(str(s).replace(",", ""))
        return True, num
    except ValueError:
        return False, None


def normalize(answer: Union[bool, float, str]) -> Union[bool, float, str]:
    if isinstance(answer, str) and bool(re.match(r"\$\d+(\.\d+)?", answer)):
        return answer[1:]
    if isinstance(answer, str) and (
        bool(re.match(r"^\d+(\.\d+)?%$", answer))
        or bool(re.match(r"^\d+(\.\d+)?\\%$", answer))
    ):
        return answer.replace("\\%", "").replace("%", "")
    return answer


class TimeoutException(Exception):
    pass


@contextlib.contextmanager
def time_limit(seconds: float):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


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


def symbolic_equal(a: str, b: str, tolerance: float, timeout: float = 10.0) -> bool:
    import sympy
    from sympy.parsing.latex import parse_latex
    from sympy.parsing.sympy_parser import parse_expr

    def _parse(s: str):
        for fn in [parse_expr, parse_latex]:
            try:
                with time_limit(timeout):
                    return fn(s)
            except Exception:
                pass
        return s

    a = _parse(a)
    b = _parse(b)

    try:
        with time_limit(timeout):
            if sympy.simplify(a - b) == 0:
                return True
    except Exception:
        pass

    try:
        with time_limit(timeout):
            if isclose(sympy.N(a), sympy.N(b), rel_tol=tolerance):
                return True
    except Exception:
        pass

    return False


def math_equal(
    prediction: Union[bool, float, str],
    reference: Union[float, str],
    include_percentage: bool = True,
    tolerance: float = 1e-4,
    timeout: float = 10.0,
    check_antlr_version: bool = True,
) -> bool:
    prediction = normalize(prediction)
    reference = normalize(reference)
    prediction = normalize_answer_string(prediction)
    reference = normalize_answer_string(reference)

    if isinstance(prediction, str) and len(prediction) > 1000:
        prediction = prediction[:1000]

    if isinstance(prediction, str) and isinstance(reference, str):
        if prediction.strip().lower() == reference.strip().lower():
            return True
        if prediction.replace(" ", "") == reference.replace(" ", ""):
            return True

    try:
        if is_digit(prediction)[0] and is_digit(reference)[0]:
            pred_val = is_digit(prediction)[1]
            ref_val = is_digit(reference)[1]
            gt_result = [ref_val / 100, ref_val, ref_val * 100] if include_percentage else [ref_val]
            for item in gt_result:
                try:
                    if isclose(item, pred_val, rel_tol=tolerance):
                        return True
                except Exception:
                    continue
            return False
    except Exception:
        pass

    if not prediction and prediction not in [0, False]:
        return False

    reference = str(reference).strip()
    prediction = str(prediction).strip()
    prediction = format_intervals(prediction)

    pred_str, ref_str = prediction, reference
    if (prediction.startswith("[") and prediction.endswith("]") and not reference.startswith("(")) or (
        prediction.startswith("(") and prediction.endswith(")") and not reference.startswith("[")
    ):
        pred_str = pred_str.strip("[]()")
        ref_str = ref_str.strip("[]()")
    for token in ["{", "}", "(", ")"]:
        ref_str = ref_str.replace(token, "")
        pred_str = pred_str.replace(token, "")
    if pred_str == ref_str:
        return True

    if (
        prediction
        and reference
        and prediction[0] in "(["
        and prediction[-1] in ")]"
        and prediction[0] == reference[0]
        and prediction[-1] == reference[-1]
    ):
        pred_parts = prediction[1:-1].split(",")
        ref_parts = reference[1:-1].split(",")
        if len(pred_parts) == len(ref_parts) and all(
            math_equal(pred_pt, ref_pt, include_percentage, tolerance, timeout, check_antlr_version)
            for pred_pt, ref_pt in zip(pred_parts, ref_parts)
        ):
            return True

    if "," in prediction and "," in reference:
        pred_parts = [item.strip() for item in prediction.split(",")]
        ref_parts = [item.strip() for item in reference.split(",")]
        if len(pred_parts) == len(ref_parts):
            return all(
                math_equal(pred_parts[i], ref_parts[i], include_percentage, tolerance, timeout, check_antlr_version)
                for i in range(len(pred_parts))
            )

    if prediction.startswith("Point") and reference.startswith("(") and reference.endswith(")"):
        pred_parts = prediction[prediction.find("(") + 1 : -1].split(",")
        ref_parts = reference[1:-1].split(",")
        if len(pred_parts) == len(ref_parts) and all(
            math_equal(pred_pt, ref_pt, include_percentage, tolerance, timeout, check_antlr_version)
            for pred_pt, ref_pt in zip(pred_parts, ref_parts)
        ):
            return True

    return symbolic_equal(prediction, reference, tolerance, timeout)


def extract_answer(
    string: str,
    extract_from_boxed: bool = True,
    extract_regex: str = r"The final answer is (.+)$",
) -> str | None:
    if not extract_from_boxed:
        match = re.search(extract_regex, string)
        if match:
            return match.group(1)
        return None

    if "\\boxed" not in string:
        return None

    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        return None

    retval = string[idx : right_brace_idx + 1]
    left = "\\boxed{"
    try:
        assert retval[: len(left)] == left
        assert retval[-1] == "}"
        return retval[len(left) : -1]
    except AssertionError:
        return None


def _normalize_completion(completion) -> str:
    if isinstance(completion, str):
        return completion
    if completion and isinstance(completion[0], dict):
        return completion[0].get("content", "")
    return str(completion)


def _score_completion(text: str, ground_truth: str, reward_style: str) -> float:
    solution_str = text
    if reward_style == "r1":
        if "</think>" in solution_str:
            solution_str = solution_str.split("</think>")[-1]
            format_correct = True
        else:
            format_correct = False
    elif reward_style == "boxed":
        format_correct = solution_str.count("\\boxed") == 1
    else:
        raise ValueError(f"Unsupported reward_style: {reward_style}")

    if not format_correct:
        return -1.0

    omi_correct = False
    mathv_correct = False

    try:
        omi_pred = extract_answer(solution_str, extract_from_boxed=True)
        if omi_pred is not None:
            omi_correct = math_equal(omi_pred, ground_truth, check_antlr_version=False)
    except Exception:
        omi_correct = False

    try:
        mathv_pred = parse(solution_str)
        mathv_gold = parse(f"\\boxed{{${ground_truth}$}}")
        mathv_correct = bool(verify(mathv_gold, mathv_pred, timeout_seconds=3))
    except Exception:
        mathv_correct = False

    return 1.0 if (omi_correct or mathv_correct) else -1.0


def deepmath_zero_reward(completions, ground_truth=None, solution=None, **kwargs):
    golds = ground_truth if ground_truth is not None else solution
    if golds is None:
        raise ValueError("deepmath_zero_reward requires `ground_truth` or `solution` in the dataset.")
    return [
        _score_completion(_normalize_completion(completion), gt, reward_style="boxed")
        for completion, gt in zip(completions, golds, strict=True)
    ]


def deepmath_r1_reward(completions, ground_truth=None, solution=None, **kwargs):
    golds = ground_truth if ground_truth is not None else solution
    if golds is None:
        raise ValueError("deepmath_r1_reward requires `ground_truth` or `solution` in the dataset.")
    return [
        _score_completion(_normalize_completion(completion), gt, reward_style="r1")
        for completion, gt in zip(completions, golds, strict=True)
    ]


def compute_score_deepmath_zero(data_source, solution_str, ground_truth, extra_info=None):
    score = _score_completion(solution_str, ground_truth, reward_style="boxed")
    return {"score": score, "acc": score > 0}


def compute_score_deepmath_r1(data_source, solution_str, ground_truth, extra_info=None):
    score = _score_completion(solution_str, ground_truth, reward_style="r1")
    return {"score": score, "acc": score > 0}
