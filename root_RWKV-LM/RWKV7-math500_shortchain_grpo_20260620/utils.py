#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import time
import random
from typing import List, Dict, Any, Optional

import torch


def calculate_pass_at_k(n, c, k):
    """
    ??????? Pass@k
    n: ????
    c: ?????
    k: ????
    """
    if c == 0:
        return 0.0
    if n < k:
        return c / n
    if n - c < k:
        return 1.0

    prob_all_wrong = 1.0
    for i in range(k):
        prob_all_wrong *= (n - c - i) / (n - i)
    return 1.0 - prob_all_wrong


def now_str() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def read_jsonl(path: str, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
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


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ProgressTracker:
    def __init__(self, total_steps: int, log_path: str):
        self.total_steps = total_steps
        self.log_path = log_path
        self.start_time = time.time()
        self.step_times = []

    def update(self, step: int, metrics: Dict[str, Any]):
        elapsed = time.time() - self.start_time
        self.step_times.append(elapsed)

        if len(self.step_times) > 1:
            avg_step_time = (self.step_times[-1] - self.step_times[0]) / len(self.step_times)
            eta_seconds = avg_step_time * (self.total_steps - step)
            eta_str = self._format_time(eta_seconds)
        else:
            eta_seconds = None
            eta_str = "calculating..."

        progress = step / self.total_steps
        bar_length = 40
        filled = int(bar_length * progress)
        bar = "#" * filled + "-" * (bar_length - filled)

        metric_str = " | ".join([
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in metrics.items()
        ])

        print(
            f"\r[{bar}] {progress*100:.1f}% | Step {step}/{self.total_steps} | "
            f"Elapsed: {self._format_time(elapsed)} | ETA: {eta_str} | {metric_str}",
            end="",
            flush=True,
        )

        log_entry = {
            "step": step,
            "progress": progress,
            "elapsed_seconds": elapsed,
            "eta_seconds": eta_seconds,
            **metrics,
        }
        append_jsonl(self.log_path, log_entry)

    def _format_time(self, seconds: float) -> str:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        if hours > 0:
            return f"{hours}h{minutes}m{secs}s"
        if minutes > 0:
            return f"{minutes}m{secs}s"
        return f"{secs}s"

    def finish(self):
        total_time = time.time() - self.start_time
        print(f"\n? Training completed in {self._format_time(total_time)}")


def build_prompt(problem: str, mode: str = "rwkv_boxed") -> str:
    p = (problem or "").strip()
    if mode == "short_math":
        return (
            f"User: {p}\n"
            "Solve concisely. Use only the necessary short steps, avoid repetition, "
            "and put the final answer in \boxed{...}.\n"
            f"Assistant: <think>\n"
        )
    if mode == "trl_doc":
        return f"User: {p}\n\nAssistant: <think></think"
    if mode == "question_only":
        return p
    return (
        f"User: {p}\n"
        f"Please put the final answer in \boxed{{...}} and output only that line. think step by step\n"
        f"Assistant: <think>\n"
    )
