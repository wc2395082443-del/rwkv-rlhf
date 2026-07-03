#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import time
import random
from typing import List, Dict, Any, Optional

import torch


def now_str() -> str:
    """返回当前时间字符串"""
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def read_jsonl(path: str, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    """读取JSONL文件，可选择最大样本数"""
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
    """追加JSON对象到JSONL文件"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def set_seed(seed: int):
    """设置所有随机种子"""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ProgressTracker:
    """实时进度跟踪和可视化"""
    
    def __init__(self, total_steps: int, log_path: str):
        self.total_steps = total_steps
        self.log_path = log_path
        self.start_time = time.time()
        self.step_times = []
        
    def update(self, step: int, metrics: Dict[str, Any]):
        """更新进度和当前指标"""
        elapsed = time.time() - self.start_time
        self.step_times.append(elapsed)
        
        # 计算ETA
        if len(self.step_times) > 1:
            avg_step_time = (self.step_times[-1] - self.step_times[0]) / len(self.step_times)
            eta_seconds = avg_step_time * (self.total_steps - step)
            eta_str = self._format_time(eta_seconds)
        else:
            eta_str = "calculating..."
        
        # 进度条
        progress = step / self.total_steps
        bar_length = 40
        filled = int(bar_length * progress)
        bar = "#" * filled + "-" * (bar_length - filled)
        
        # 格式化指标显示
        metric_str = " | ".join([f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" 
                                  for k, v in metrics.items()])
        
        # 打印进度
        print(f"\r[{bar}] {progress*100:.1f}% | Step {step}/{self.total_steps} | "
              f"Elapsed: {self._format_time(elapsed)} | ETA: {eta_str} | {metric_str}", 
              end="", flush=True)
        
        # 保存到日志
        log_entry = {
            "step": step,
            "progress": progress,
            "elapsed_seconds": elapsed,
            "eta_seconds": eta_seconds if len(self.step_times) > 1 else None,
            **metrics
        }
        append_jsonl(self.log_path, log_entry)
    
    def _format_time(self, seconds: float) -> str:
        """格式化秒数为可读时间字符串"""
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
        """打印完成消息"""
        total_time = time.time() - self.start_time
        print(f"\n✓ Training completed in {self._format_time(total_time)}")


def build_prompt(problem: str) -> str:
    """构建问题提示"""
    p = (problem or "").strip()
    return (
        f"User: {p}\n"
        f"Please put the final answer in \\boxed{{...}} and output only that line. think\n"
        f"Assistant: <think>\n"
    )
