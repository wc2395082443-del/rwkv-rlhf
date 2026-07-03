import re


def extract_choice(text: str):
    if text is None:
        return None
    text = str(text)
    boxed = re.findall(r"\\boxed\s*\{\s*([A-Ja-j])\s*\}", text)
    if boxed:
        return boxed[-1].upper()
    pats = [
        r"(?:final\s+answer|answer|choice|option)\s*(?:is|:)?\s*\(?\s*([A-Ja-j])\s*\)?",
        r"\b([A-Ja-j])\s*(?:is\s+correct|is\s+the\s+answer)\b",
    ]
    for pat in pats:
        ms = re.findall(pat, text, flags=re.IGNORECASE)
        if ms:
            return ms[-1].upper()
    tail = text[-200:]
    ms = re.findall(r"(?<![A-Za-z])([A-J])(?![A-Za-z])", tail)
    return ms[-1].upper() if ms else None


def _completion_text(c):
    if isinstance(c, str):
        return c
    if isinstance(c, list) and c and isinstance(c[0], dict):
        return c[0].get("content", "")
    return str(c)


def mcq_accuracy_reward(completions, solution=None, answer=None, **kwargs):
    golds = solution if solution is not None else answer
    rewards = []
    for comp, gold in zip(completions, golds):
        pred = extract_choice(_completion_text(comp))
        rewards.append(1.0 if pred == str(gold).strip().upper() else 0.0)
    return rewards
