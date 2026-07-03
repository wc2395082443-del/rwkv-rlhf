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
    计算无偏估计的 Pass@k
    n: 总样本数 (total samples)
    c: 正确样本数 (correct samples)
    k: 目标基准 (base size, e.g., 8)
    """
    # 1. 如果没对，概率为 0
    if c == 0:
        return 0.0
    
    # 2. 如果总数甚至不到 k，退化为普通的准确率 (或者报错，看你需求)
    # 在 GRPO 场景下，通常 n >= k (因为 k 是初始 group_size)
    if n < k:
        return c / n 

    # 3. 如果对的数量很多，使得 n-c < k，那必然抽到一个对的
    if n - c < k:
        return 1.0

    # 4. 计算全错概率: Prod((n-c-i)/(n-i))
    # 也就是从 n 个里抽 k 个，全都在 (n-c) 个错误样本里的概率
    prob_all_wrong = 1.0
    for i in range(k):
        prob_all_wrong *= (n - c - i) / (n - i)
        
    return 1.0 - prob_all_wrong


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
