#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import math
import random
import subprocess
import sys
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from utils import now_str, append_jsonl, build_prompt
from reward import calculate_reward, calculate_reward_details
from infer import AlbatrossBatchInference


@dataclass
class GRPOConfig:
    """GRPO训练配置"""
    # 采样配置
    num_questions: int = 24  # 每步从训练集随机抽取的题目数
    samples_per_question: int = 8  # 每道题采样的次数
    
    # 生成配置
    max_new_tokens: int = 2048
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: int = 0
    eval_temperature: float = 0.3
    eval_top_p: float = 0.4
    eval_top_k: int = 0
    
    # 训练配置
    ppo_epochs: int = 1
    micro_batch: int = 4
    lr: float = 1e-5
    grad_clip: float = 1.0
    
    # 奖励配置
    min_tokens: int = 50
    max_tokens: int = 2048
    length_weight: float = 0.5
    
    # KL散度配置
    kl_coef: float = 0.01
    
    # 正则化
    time_state_l2: float = 0.0
    time_state_clamp: float = 0.0
    
    # 日志和保存
    log_interval: int = 1
    save_interval: int = 50
    eval_interval: int = 5


def compute_unbiased_kl(ref_logp: torch.Tensor, policy_logp: torch.Tensor) -> torch.Tensor:
    """
    计算无偏KL散度: D_KL[ref || policy]
    
    latex公式:
    D_KL = (ref_prob / policy_prob) - log(ref_prob / policy_prob) - 1
         = exp(ref_logp - policy_logp) - (ref_logp - policy_logp) - 1
    
    Args:
        ref_logp: reference model的log概率
        policy_logp: 当前policy的log概率
    
    Returns:
        无偏KL散度
    """
    log_ratio = ref_logp - policy_logp
    return torch.exp(log_ratio) - log_ratio - 1.0


class GRPOTrainer:
    """GRPO训练器"""
    
    def __init__(
        self,
        train_model,
        ref_model,
        infer_engine: AlbatrossBatchInference,
        encode_fn,
        decode_fn,
        train_data: List[Dict[str, Any]],
        test_data: List[Dict[str, Any]],
        out_dir: str,
        device: str,
        cfg: GRPOConfig,
        seed: int = 42,
    ):
        self.model = train_model
        self.ref_model = ref_model
        self.infer = infer_engine
        self.encode = encode_fn
        self.decode = decode_fn
        self.train_data = train_data
        self.test_data = test_data
        self.out_dir = out_dir
        self.device = device
        self.cfg = cfg
        self.rng = random.Random(seed)
        
        os.makedirs(out_dir, exist_ok=True)
        
        # 日志路径
        self.log_path = os.path.join(out_dir, "train.log")
        self.metrics_path = os.path.join(out_dir, "metrics.jsonl")
        self.responses_dir = os.path.join(out_dir, "responses_by_step")
        self.eval_path = os.path.join(out_dir, "eval.jsonl")
        os.makedirs(self.responses_dir, exist_ok=True)
        
        # 优化器
        params = [p for p in self.model.parameters() if p.requires_grad]
        if not params:
            raise RuntimeError("No trainable parameters found")
        self.opt = torch.optim.Adam(params, lr=self.cfg.lr, betas=(0.9, 0.99), eps=1e-8)
        
        # 保存初始time_state (用于L2正则化)
        self._ts_init: Dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for n, p in self.model.named_parameters():
                if "time_state" in n:
                    self._ts_init[n] = p.detach().clone()
        self._ref_time_state = []
        with torch.no_grad():
            for block in self.model.blocks:
                self._ref_time_state.append(block.att.time_state.detach().clone())
    
    def _log(self, msg: str):
        """记录日志"""
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def _plot_metrics(self):
        plot_script = os.path.join(os.path.dirname(__file__), "plot_metrics.py")
        out_plot = os.path.join(self.out_dir, "metrics_plot.png")
        if os.path.isfile(plot_script) and os.path.isfile(self.metrics_path):
            try:
                subprocess.run([sys.executable, plot_script, "--metrics", self.metrics_path, "--out", out_plot], check=False)
            except Exception:
                pass

    def _time_state_stats(self):
        mx = 0.0
        rms_sum = 0.0
        cnt = 0
        bad = False
        with torch.no_grad():
            for n, p in self.model.named_parameters():
                if "time_state" not in n:
                    continue
                if torch.isnan(p).any() or torch.isinf(p).any():
                    bad = True
                v = p.detach().float()
                mx = max(mx, float(v.abs().max().item()))
                rms_sum += float((v * v).mean().sqrt().item())
                cnt += 1
        return {"absmax": mx, "rms_avg": (rms_sum / max(1, cnt)), "bad": bad}

    def _has_repeated_ngrams(self, text: str, n: int = 16, repeat: int = 5) -> bool:
        if not text or n <= 0 or repeat <= 1:
            return False
        tokens = re.findall(r"\w+|[^\w\s]", text)
        total = n * repeat
        if len(tokens) < total:
            return False
        counts = {}
        for i in range(len(tokens) - n + 1):
            ng = tuple(tokens[i:i + n])
            cnt = counts.get(ng, 0) + 1
            counts[ng] = cnt
            if cnt >= repeat:
                return True
        return False

    @torch.no_grad()
    def _get_answer(self, ex: Dict[str, Any]) -> str:
        """兼容 answer/solution 字段"""
        if "answer" in ex and ex.get("answer") is not None:
            return str(ex.get("answer"))
        if "solution" in ex and ex.get("solution") is not None:
            return str(ex.get("solution"))
        return ""

    def evaluate(self, step: int, tag: str = "eval", sample_ratio: float = 1.0) -> Optional[float]:
        """在测试集上评估（使用独立推理参数）"""
        if not self.test_data:
            return None
        t0 = time.time()
        data = self.test_data
        if sample_ratio < 1.0:
            k = max(1, int(len(data) * sample_ratio))
            idxs = self.rng.sample(range(len(data)), k)
            data = [data[i] for i in idxs]
        total = 0
        correct = 0
        total_len = 0
        total_trunc = 0
        trunc_wrong = 0
        repeat_ngram = 0
        format_correct = 0
        no_answer = 0
        sum_reward = 0.0
        sum_correct_reward = 0.0
        sum_format_reward = 0.0
        sum_length_reward = 0.0
        sum_length_lambda = 0.0
        chunk_size = len(data)
        for start in range(0, len(data), chunk_size):
            ex_list = data[start:start + chunk_size]
            problems = [ex.get("problem", "") for ex in ex_list]
            answers = [self._get_answer(ex) for ex in ex_list]
            prompt_strs = [build_prompt(p) for p in problems]
            prompt_tokens_list = []
            for ps in prompt_strs:
                ids = self.encode(ps)
                max_prompt_len = int(self.model.args.ctx_len) - int(self.cfg.max_new_tokens) - 4
                max_prompt_len = max(64, max_prompt_len)
                if len(ids) > max_prompt_len:
                    ids = ids[-max_prompt_len:]
                prompt_tokens_list.append(ids)

            comp_tokens_list, _, comp_texts_list, truncated_list = \
                self.infer.generate_group_parallel(
                    prompt_tokens_list=prompt_tokens_list,
                    group_size=1,
                    max_new_tokens=self.cfg.max_new_tokens,
                    temperature=self.cfg.eval_temperature,
                    top_p=self.cfg.eval_top_p,
                    top_k=self.cfg.eval_top_k,
                )

            for i in range(len(prompt_tokens_list)):
                problem = problems[i]
                answer = answers[i]
                comp_text = comp_texts_list[i]
                comp_tokens = comp_tokens_list[i]
                truncated = truncated_list[i]
                reward, is_correct, is_format_correct, reward_details = calculate_reward_details(
                    text=comp_text,
                    ground_truth=answer,
                    token_length=len(comp_tokens),
                    min_tokens=self.cfg.min_tokens,
                    max_tokens=self.cfg.max_tokens,
                    length_weight=self.cfg.length_weight,
                )
                append_jsonl(self.eval_path, {
                    "step": step,
                    "tag": tag,
                    "problem": problem,
                    "ground_truth": answer,
                    "response": comp_text,
                    "pred_extracted": reward_details.get("extracted_answer"),
                    "gt_extracted": reward_details.get("ground_truth_answer"),
                    "reward": reward,
                    "is_correct": is_correct,
                    "is_format_correct": is_format_correct,
                    "truncated": bool(truncated),
                    "gen_len": len(comp_tokens),
                    "reward_details": reward_details,
                    "eval_temperature": self.cfg.eval_temperature,
                    "eval_top_p": self.cfg.eval_top_p,
                    "eval_top_k": self.cfg.eval_top_k,
                    "repeat_16gram_5": self._has_repeated_ngrams(comp_text, n=16, repeat=5),
                })
                total += 1
                sum_reward += float(reward)
                sum_correct_reward += float(reward_details.get("correct_reward", 0.0))
                sum_format_reward += float(reward_details.get("format_reward", 0.0))
                sum_length_reward += float(reward_details.get("length_reward", 0.0))
                sum_length_lambda += float(reward_details.get("length_lambda", 0.0))
                total_len += len(comp_tokens)
                if truncated:
                    total_trunc += 1
                    if not is_correct:
                        trunc_wrong += 1
                if is_correct:
                    correct += 1
                if is_format_correct:
                    format_correct += 1
                if not reward_details.get("extracted_answer"):
                    no_answer += 1
                if self._has_repeated_ngrams(comp_text, n=16, repeat=5):
                    repeat_ngram += 1
        acc = correct / max(1, total)
        avg_len = total_len / max(1, total)
        trunc_rate = total_trunc / max(1, total)
        trunc_wrong_rate = trunc_wrong / max(1, total)
        repeat_ngram_rate = repeat_ngram / max(1, total)
        format_rate = format_correct / max(1, total)
        no_answer_rate = no_answer / max(1, total)
        avg_reward = sum_reward / max(1, total)
        avg_correct_reward = sum_correct_reward / max(1, total)
        avg_format_reward = sum_format_reward / max(1, total)
        avg_length_reward = sum_length_reward / max(1, total)
        avg_length_lambda = sum_length_lambda / max(1, total)
        eval_time = time.time() - t0
        append_jsonl(self.metrics_path, {
            "step": step,
            "accuracy": acc,
            "avg_length": avg_len,
            "trunc_rate": trunc_rate,
            "split": tag,
            "avg_reward": avg_reward,
            "trunc_wrong_rate": trunc_wrong_rate,
            "repeat_16gram_rate": repeat_ngram_rate,
            "format_rate": format_rate,
            "no_answer_rate": no_answer_rate,
            "avg_correct_reward": avg_correct_reward,
            "avg_format_reward": avg_format_reward,
            "avg_length_reward": avg_length_reward,
            "avg_length_lambda": avg_length_lambda,
            "eval_count": total,
            "eval_time": eval_time,
        })
        self._log(
            f"[EVAL step {step}] acc={acc:.3f} trunc={trunc_rate:.3f} trunc_wrong={trunc_wrong_rate:.3f} "
            f"repeat16@5={repeat_ngram_rate:.3f} fmt={format_rate:.3f} no_ans={no_answer_rate:.3f} "
            f"avg_len={avg_len:.1f} avg_r={avg_reward:.3f} time={eval_time:.1f}s "
            f"(temp={self.cfg.eval_temperature}, top_p={self.cfg.eval_top_p}, top_k={self.cfg.eval_top_k})"
        )
        return acc
    
    @torch.no_grad()
    def _init_ref_state(self, B: int):
        """用初始time_state构建reference推理状态"""
        state = self.infer.infer_model.generate_zero_state(B)
        for i, ts in enumerate(self._ref_time_state):
            state[1][i] = ts.unsqueeze(0).expand(B, -1, -1, -1).clone()
        return state
    
    def _pad_batch(self, seqs: List[List[int]], pad_id: int = 0):
        """填充batch"""
        max_len = max(len(s) for s in seqs)
        padded = []
        masks = []
        for s in seqs:
            pad_len = max_len - len(s)
            padded.append(s + [pad_id] * pad_len)
            masks.append([1] * len(s) + [0] * pad_len)
        return torch.tensor(padded, device=self.device, dtype=torch.long), \
               torch.tensor(masks, device=self.device, dtype=torch.bool)
    
    def train_step(self, step: int) -> Dict[str, Any]:
        """
        单步训练
        
        采样逻辑:
        1. 从训练集随机抽取num_questions道题
        2. 每道题做samples_per_question次采样
        3. 每道题的samples_per_question个样本作为一个group
        4. 在group内计算advantage
        """
        t0 = time.time()
        
        # 1. 随机采样题目
        if len(self.train_data) >= self.cfg.num_questions:
            sampled_indices = self.rng.sample(range(len(self.train_data)), self.cfg.num_questions)
        else:
            sampled_indices = [self.rng.randrange(len(self.train_data)) 
                             for _ in range(self.cfg.num_questions)]
        
        sampled_questions = [self.train_data[i] for i in sampled_indices]
        
        # 2. 为每道题生成多个响应（批量）
        prompt_strs = [build_prompt(q.get("problem", "")) for q in sampled_questions]
        prompt_tokens_list = []
        for ps in prompt_strs:
            ids = self.encode(ps)
            max_prompt_len = int(self.model.args.ctx_len) - int(self.cfg.max_new_tokens) - 4
            max_prompt_len = max(64, max_prompt_len)
            if len(ids) > max_prompt_len:
                ids = ids[-max_prompt_len:]
            prompt_tokens_list.append(ids)

        comp_tokens_list, old_logps_list, comp_texts_list, truncated_list = \
            self.infer.generate_group_parallel(
                prompt_tokens_list=prompt_tokens_list,
                group_size=self.cfg.samples_per_question,
                max_new_tokens=self.cfg.max_new_tokens,
                temperature=self.cfg.temperature,
                top_p=self.cfg.top_p,
                top_k=self.cfg.top_k,
            )

        # 3. 计算每个响应的奖励
        all_trajs = []
        stats = {
            "total_samples": 0,
            "correct_samples": 0,
            "total_reward": 0.0,
            "total_length": 0,
            "total_trunc": 0,
            "total_repeat": 0,
            "no_answer": 0,
            "sum_correct_reward": 0.0,
            "sum_format_reward": 0.0,
            "sum_length_reward": 0.0,
            "sum_length_lambda": 0.0,
        }
        adv_values = []
        groups_total = len(sampled_questions)
        groups_all_correct = 0
        groups_all_wrong = 0
        
        for q_idx, question in enumerate(sampled_questions):
            problem = question.get("problem", "")
            answer = self._get_answer(question)
            prompt_tokens = prompt_tokens_list[q_idx]
            
            # 3. 计算每个响应的奖励
            group_rewards = []
            group_trajs = []
            
            correct_in_group = 0
            for i in range(self.cfg.samples_per_question):
                flat_idx = q_idx * self.cfg.samples_per_question + i
                if flat_idx >= len(comp_tokens_list):
                    continue
                comp_tokens = comp_tokens_list[flat_idx]
                old_logps = old_logps_list[flat_idx]
                comp_text = comp_texts_list[flat_idx]
                truncated = truncated_list[flat_idx]
                
                # 计算奖励
                reward, is_correct, is_format_correct, reward_details = calculate_reward_details(
                    text=comp_text,
                    ground_truth=answer,
                    token_length=len(comp_tokens),
                    min_tokens=self.cfg.min_tokens,
                    max_tokens=self.cfg.max_tokens,
                    length_weight=self.cfg.length_weight,
                )
                
                
                # 保存轨迹
                traj = {
                    "prompt_tokens": prompt_tokens,
                    "comp_tokens": comp_tokens,
                    "old_logps": old_logps,
                    "reward": reward,
                    "text": comp_text,
                    "is_correct": is_correct,
                    "is_format_correct": is_format_correct,
                    "truncated": truncated,
                }
                
                group_rewards.append(reward)
                group_trajs.append(traj)
                
                # 更新统计
                stats["total_samples"] += 1
                stats["total_reward"] += reward
                stats["total_length"] += len(comp_tokens)
                if is_correct:
                    stats["correct_samples"] += 1
                    correct_in_group += 1
                if truncated:
                    stats["total_trunc"] += 1
                if self._has_repeated_ngrams(comp_text, n=16, repeat=5):
                    stats["total_repeat"] += 1
                if not reward_details.get("extracted_answer"):
                    stats["no_answer"] += 1
                stats["sum_correct_reward"] += float(reward_details.get("correct_reward", 0.0))
                stats["sum_format_reward"] += float(reward_details.get("format_reward", 0.0))
                stats["sum_length_reward"] += float(reward_details.get("length_reward", 0.0))
                stats["sum_length_lambda"] += float(reward_details.get("length_lambda", 0.0))
                
                # 记录响应
                record = {
                    "step": step,
                    "question_idx": q_idx,
                    "sample_idx": i,
                    "problem": problem,
                    "ground_truth": answer,
                    "response": comp_text,
                "pred_extracted": reward_details.get("extracted_answer"),
                "gt_extracted": reward_details.get("ground_truth_answer"),
                    "reward": reward,
                    "is_correct": is_correct,
                    "is_format_correct": is_format_correct,
                    "truncated": truncated,
                    "reward_details": reward_details,
                }
                step_path = os.path.join(self.responses_dir, f"step_{step}.jsonl")
                append_jsonl(step_path, record)

            if correct_in_group == 0:
                groups_all_wrong += 1
            elif correct_in_group == self.cfg.samples_per_question:
                groups_all_correct += 1

            # 4. 计算group内的advantage (均值-方差归一化)
            mean_reward = sum(group_rewards) / len(group_rewards)
            var_reward = sum((r - mean_reward) ** 2 for r in group_rewards) / len(group_rewards)
            std_reward = math.sqrt(var_reward) if var_reward > 1e-6 else 1.0
            
            for traj in group_trajs:
                traj["advantage"] = (traj["reward"] - mean_reward) / std_reward
                adv_values.append(traj["advantage"])
            
            all_trajs.extend(group_trajs)
        
        # 5. 训练
        loss_total = 0.0
        kl_total = 0.0
        grad_norm = 0.0
        batch_cnt = 0
        
        for epoch in range(self.cfg.ppo_epochs):
            self.model.train()
            self.opt.zero_grad(set_to_none=True)
            
            # 按长度排序 (提高效率)
            trajs_sorted = sorted(all_trajs, 
                                key=lambda x: len(x["prompt_tokens"]) + len(x["comp_tokens"]), 
                                reverse=True)
            
            # Mini-batch训练
            for start in range(0, len(trajs_sorted), self.cfg.micro_batch):
                batch = trajs_sorted[start:start + self.cfg.micro_batch]
                
                # 构建输入
                seqs = [traj["prompt_tokens"] + traj["comp_tokens"] for traj in batch]
                padded, _ = self._pad_batch(seqs, pad_id=0)
                
                inp = padded[:, :-1].contiguous()
                tgt = padded[:, 1:].contiguous()
                
                # 前向传播
                logits = self.model(inp)
                if torch.is_tensor(logits) and logits.dim() == 2:
                    logits = logits.unsqueeze(0)
                
                logp = F.log_softmax(logits.float(), dim=-1)
                picked_logp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
                # reference logits (batched, same as train_model forward)
                with torch.no_grad():
                    ref_logits = self.ref_model(inp)
                    if torch.is_tensor(ref_logits) and ref_logits.dim() == 2:
                        ref_logits = ref_logits.unsqueeze(0)
                    ref_logp_all = F.log_softmax(ref_logits.float(), dim=-1)
                    ref_picked_logp = ref_logp_all.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
                
                # 计算损失
                batch_loss = 0.0
                batch_kl = 0.0
                total_tokens = 0
                
                for bi, traj in enumerate(batch):
                    prompt_len = len(traj["prompt_tokens"])
                    comp_len = len(traj["comp_tokens"])
                    
                    # 只计算生成部分的损失
                    start_idx = prompt_len - 1
                    end_idx = start_idx + comp_len
                    
                    new_logp = picked_logp[bi, start_idx:end_idx]
                    ref_logp = ref_picked_logp[bi, start_idx:end_idx]
                    
                    # 对齐长度
                    min_len = min(new_logp.size(0), ref_logp.size(0))
                    if min_len == 0:
                        continue
                    new_logp = new_logp[:min_len]
                    ref_logp = ref_logp[:min_len]
                    
                    # 计算log比率 (ref - policy)
                    log_ratio = ref_logp - new_logp
                    
                    # GRPO目标: advantage * log_ratio
                    adv = torch.tensor(traj["advantage"], 
                                     device=self.device, dtype=torch.float32)
                    policy_loss = -(adv * log_ratio).sum()
                    
                    # 计算无偏KL散度
                    kl = compute_unbiased_kl(ref_logp, new_logp).sum()
                    
                    batch_loss += policy_loss
                    batch_kl += kl
                    total_tokens += min_len
                
                # 归一化损失 (除以token数和group size)
                if total_tokens > 0:
                    normalized_loss = batch_loss / (total_tokens * self.cfg.samples_per_question)
                    normalized_kl = batch_kl / (total_tokens * self.cfg.samples_per_question)
                    
                    # 总损失 = policy loss + KL惩罚
                    total_loss = normalized_loss + self.cfg.kl_coef * normalized_kl
                    
                    # L2正则化
                    if self.cfg.time_state_l2 > 0:
                        l2_reg = 0.0
                        for n, p in self.model.named_parameters():
                            if "time_state" in n:
                                l2_reg += (p.float() - self._ts_init[n].float()).pow(2).mean()
                        total_loss += self.cfg.time_state_l2 * l2_reg
                    
                    # 反向传播
                    total_loss.backward()
                    
                    loss_total += normalized_loss.item()
                    kl_total += normalized_kl.item()
                    batch_cnt += 1
            
            # 梯度裁剪
            if self.cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    self.cfg.grad_clip
                )
            
            # 计算梯度范数
            with torch.no_grad():
                g2 = 0.0
                for p in self.model.parameters():
                    if p.requires_grad and p.grad is not None:
                        g = p.grad.detach().float()
                        g2 += (g.norm(2) ** 2).item()
                grad_norm = math.sqrt(g2)
            
            # 更新参数
            self.opt.step()
            
            # 参数裁剪
            if self.cfg.time_state_clamp > 0:
                with torch.no_grad():
                    for n, p in self.model.named_parameters():
                        if "time_state" in n:
                            p.data.clamp_(-self.cfg.time_state_clamp, 
                                        self.cfg.time_state_clamp)
        
        # 计算统计
        dt = time.time() - t0
        avg_reward = stats["total_reward"] / max(1, stats["total_samples"])
        avg_length = stats["total_length"] / max(1, stats["total_samples"])
        accuracy = stats["correct_samples"] / max(1, stats["total_samples"])
        trunc_rate = stats["total_trunc"] / max(1, stats["total_samples"])
        repeat_rate = stats["total_repeat"] / max(1, stats["total_samples"])
        no_answer_rate = stats["no_answer"] / max(1, stats["total_samples"])
        avg_correct_reward = stats["sum_correct_reward"] / max(1, stats["total_samples"])
        avg_format_reward = stats["sum_format_reward"] / max(1, stats["total_samples"])
        avg_length_reward = stats["sum_length_reward"] / max(1, stats["total_samples"])
        avg_length_lambda = stats["sum_length_lambda"] / max(1, stats["total_samples"])
        if adv_values:
            adv_mean = sum(adv_values) / len(adv_values)
            adv_var = sum((a - adv_mean) ** 2 for a in adv_values) / len(adv_values)
            adv_std = math.sqrt(adv_var)
            pos_adv_ratio = sum(1 for a in adv_values if a > 0) / len(adv_values)
            neg_adv_ratio = sum(1 for a in adv_values if a < 0) / len(adv_values)
        else:
            adv_mean = 0.0
            adv_std = 0.0
            pos_adv_ratio = 0.0
            neg_adv_ratio = 0.0
        groups_used = groups_total
        groups_skipped = 0
        avg_kl = kl_total / max(1, batch_cnt)
        samples_per_sec = stats["total_samples"] / dt if dt > 0 else 0.0
        tokens_per_sec = stats["total_length"] / dt if dt > 0 else 0.0
        ts_stats = self._time_state_stats()
        
        metrics = {
            "step": step,
            "samples": stats["total_samples"],
            "accuracy": accuracy,
            "avg_reward": avg_reward,
            "avg_length": avg_length,
            "trunc_rate": trunc_rate,
            "repeat_rate": repeat_rate,
            "no_answer_rate": no_answer_rate,
            "avg_correct_reward": avg_correct_reward,
            "avg_format_reward": avg_format_reward,
            "avg_length_reward": avg_length_reward,
            "avg_length_lambda": avg_length_lambda,
            "adv_mean": adv_mean,
            "adv_std": adv_std,
            "pos_adv_ratio": pos_adv_ratio,
            "neg_adv_ratio": neg_adv_ratio,
            "groups_total": groups_total,
            "groups_used": groups_used,
            "groups_skipped": groups_skipped,
            "groups_all_correct": groups_all_correct,
            "groups_all_wrong": groups_all_wrong,
            "loss": loss_total,
            "kl": kl_total,
            "avg_kl": avg_kl,
            "clip_frac": None,
            "grad_norm": grad_norm,
            "time": dt,
            "samples_per_sec": samples_per_sec,
            "tokens_per_sec": tokens_per_sec,
            "ts_absmax": ts_stats["absmax"],
            "ts_rms": ts_stats["rms_avg"],
            "ts_bad": ts_stats["bad"],
        }
        
        return metrics
    
    def train(self, total_steps: int):
        """训练主循环"""
        self._log(f"开始训练: total_steps={total_steps}")
        self._log(f"配置: num_questions={self.cfg.num_questions}, "
                 f"samples_per_question={self.cfg.samples_per_question}")
        train_start = time.time()
        
        for step in range(1, total_steps + 1):
            metrics = self.train_step(step)
            metrics["elapsed"] = time.time() - train_start
            
            # 记录指标
            append_jsonl(self.metrics_path, metrics)
            
            # 打印日志
            if step % self.cfg.log_interval == 0:
                self._log(
                    f"[Step {step}/{total_steps}] "
                    f"samples={int(metrics['samples'])} "
                    f"acc={metrics['accuracy']:.3f} "
                    f"trunc={metrics['trunc_rate']:.3f} "
                    f"repeat={metrics['repeat_rate']:.3f} "
                    f"no_answer={metrics['no_answer_rate']:.3f} "
                    f"reward={metrics['avg_reward']:.4f} "
                    f"corr_r={metrics['avg_correct_reward']:.4f} fmt_r={metrics['avg_format_reward']:.4f} len_r={metrics['avg_length_reward']:.4f} "
                    f"len={metrics['avg_length']:.1f} "
                    f"loss={metrics['loss']:.4f} "
                    f"kl={metrics['avg_kl']:.6f} "
                    f"grad={metrics['grad_norm']:.3f} "
                    f"adv(m={metrics['adv_mean']:.3f},s={metrics['adv_std']:.3f},pos={metrics['pos_adv_ratio']:.2f},neg={metrics['neg_adv_ratio']:.2f}) "
                    f"groups(t={metrics['groups_total']},all0={metrics['groups_all_wrong']},all1={metrics['groups_all_correct']}) "
                    f"ts(absmax={metrics['ts_absmax']:.4f},rms={metrics['ts_rms']:.4f},bad={metrics['ts_bad']}) "
                    f"speed(samp/s={metrics['samples_per_sec']:.2f},tok/s={metrics['tokens_per_sec']:.1f}) "
                    f"step_time={metrics['time']:.1f}s elapsed={metrics['elapsed']:.1f}s"
                )
            
            # 保存检查点
            if step % self.cfg.save_interval == 0 or step == total_steps:
                ckpt_path = os.path.join(self.out_dir, f"ckpt_step{step}.pth")
                torch.save({
                    "step": step,
                    "time_state": {n: p.detach().cpu() 
                                 for n, p in self.model.named_parameters() 
                                 if "time_state" in n},
                    "optimizer": self.opt.state_dict(),
                }, ckpt_path)
                self._log(f"保存检查点: {ckpt_path}")

            # 评估
            full_eval = (step % self.cfg.save_interval == 0) or (step == total_steps)
            if self.cfg.eval_interval > 0 and self.test_data and (step % self.cfg.eval_interval == 0):
                if full_eval:
                    self.evaluate(step, tag="full_eval", sample_ratio=1.0)
                else:
                    self.evaluate(step, tag="eval", sample_ratio=0.2)
                self._plot_metrics()
            elif full_eval and self.test_data:
                self.evaluate(step, tag="full_eval", sample_ratio=1.0)
                self._plot_metrics()
        
        self._log("训练完成!")
        self._plot_metrics()
