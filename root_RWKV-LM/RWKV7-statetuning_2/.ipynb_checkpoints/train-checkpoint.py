#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import json
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
    eval_sample_ratio: float = 1.0

    # RFT sampling config
    rft_target_correct: int = 64
    rft_max_batch: int = 256
    rft_sigma: float = 0.30
    rft_min_attempts: int = 16
    rft_stop_p: float = 0.85
    rft_expand: float = 0.25


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
        self.eval_by_step_dir = os.path.join(out_dir, "eval_by_step")
        os.makedirs(self.eval_by_step_dir, exist_ok=True)
        
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

        # RFT history stats
        self.q_attempts = [0] * len(self.train_data)
        self.q_correct = [0] * len(self.train_data)
        self.global_attempts = 0
        self.global_correct = 0
    
    def _log(self, msg: str):
        # log message
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def _write_rft_table(self, step: int):
        path = os.path.join(self.out_dir, "rft_correctness.json")
        items = []
        for i, (a, c) in enumerate(zip(self.q_attempts, self.q_correct)):
            p = (c + 1) / (a + 2)
            items.append({"idx": i, "attempts": a, "correct": c, "p": p})
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"step": step, "items": items}, f)

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
        chunk_size = 192
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
                repeat_flag = self._has_repeated_ngrams(comp_text, n=16, repeat=5)
                reward, is_correct, is_format_correct, reward_details = calculate_reward_details(
                    text=comp_text,
                    ground_truth=answer,
                    token_length=len(comp_tokens),
                    min_tokens=self.cfg.min_tokens,
                    max_tokens=self.cfg.max_tokens,
                    length_weight=self.cfg.length_weight,
                    repeat_ngram=repeat_flag,
                    repeat_penalty=-0.5,
                )
                record = {
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
                }
                append_jsonl(self.eval_path, record)
                step_path = os.path.join(self.eval_by_step_dir, "%s_step_%s.jsonl" % (tag, step))
                append_jsonl(step_path, record)
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
                if repeat_flag:
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
            "repeat_rate": repeat_ngram_rate,
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
            f"repeat16@5={repeat_ngram_rate:.3f} fmt={format_rate:.3f} no_ans={no_answer_rate:.3f} corr_r={avg_correct_reward:.4f} fmt_r={avg_format_reward:.4f} "
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
        RFT sampling step.
        
        Sampling:
        1. p=(correct+1)/(attempts+2) with 1/2 prior
        2. skip if attempts>=rft_min_attempts and p>rft_stop_p
        3. weight by truncated Gaussian w=exp(-0.5*((p-0.5)/sigma)^2)
        4. batch size by global acc, capped by rft_max_batch
        5. keep only correct samples; adv=max(0,reward)*(1+2*(1-p)^2)
        6. stop when buffer reaches rft_target_correct
        """
        t0 = time.time()
        # 1. RFT sampling: truncated normal by historical accuracy
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
        groups_total = 0
        groups_all_correct = 0
        groups_all_wrong = 0
        groups_used = 0
        groups_skipped = 0

        def _p_from_counts(correct, attempts):
            return (correct + 1) / (attempts + 2)

        def _rft_weight(p):
            sigma = self.cfg.rft_sigma
            return math.exp(-0.5 * ((p - 0.5) / sigma) ** 2)

        def _eligible_indices():
            idxs = []
            for i in range(len(self.train_data)):
                attempts = self.q_attempts[i]
                correct = self.q_correct[i]
                p = _p_from_counts(correct, attempts)
                if attempts >= self.cfg.rft_min_attempts and p > self.cfg.rft_stop_p:
                    continue
                idxs.append(i)
            return idxs

        def _consume_sample(train_idx, prompt_tokens, comp_tokens, old_logps, comp_text, truncated, p_hist):
            repeat_flag = self._has_repeated_ngrams(comp_text, n=16, repeat=5)

            reward, is_correct, is_format_correct, reward_details = calculate_reward_details(
                text=comp_text,
                ground_truth=self._get_answer(self.train_data[train_idx]),
                token_length=len(comp_tokens),
                min_tokens=self.cfg.min_tokens,
                max_tokens=self.cfg.max_tokens,
                length_weight=self.cfg.length_weight,
                repeat_ngram=repeat_flag,
                repeat_penalty=-0.5,
            )

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

            stats["total_samples"] += 1
            stats["total_reward"] += reward
            stats["total_length"] += len(comp_tokens)
            if is_correct:
                stats["correct_samples"] += 1
            if truncated:
                stats["total_trunc"] += 1
            if repeat_flag:
                stats["total_repeat"] += 1
            if not reward_details.get("extracted_answer"):
                stats["no_answer"] += 1
            stats["sum_correct_reward"] += float(reward_details.get("correct_reward", 0.0))
            stats["sum_format_reward"] += float(reward_details.get("format_reward", 0.0))
            stats["sum_length_reward"] += float(reward_details.get("length_reward", 0.0))
            stats["sum_length_lambda"] += float(reward_details.get("length_lambda", 0.0))

            record = {
                "step": step,
                "question_idx": train_idx,
                "sample_idx": 0,
                "problem": self.train_data[train_idx].get("problem", ""),
                "ground_truth": self._get_answer(self.train_data[train_idx]),
                "response": comp_text,
                "pred_extracted": reward_details.get("extracted_answer"),
                "gt_extracted": reward_details.get("ground_truth_answer"),
                "reward": reward,
                "is_correct": is_correct,
                "is_format_correct": is_format_correct,
                "truncated": truncated,
                "reward_details": reward_details,
                "p_hist": p_hist,
            }
            step_path = os.path.join(self.responses_dir, f"step_{step}.jsonl")
            append_jsonl(step_path, record)

            return traj, is_correct, reward

        target_correct = self.cfg.rft_target_correct
        max_samples = target_correct * 15
        buffer_trajs = []
        while len(buffer_trajs) < target_correct and stats["total_samples"] < max_samples:
            eligible = _eligible_indices()
            if not eligible:
                break

            acc_est = _p_from_counts(self.global_correct, self.global_attempts)
            acc_est = max(acc_est, 1e-3)
            remaining = target_correct - len(buffer_trajs)
            n = math.ceil(remaining / acc_est)
            n = math.ceil(n * (1.0 + self.cfg.rft_expand))
            n = min(self.cfg.rft_max_batch, max(1, n))

            weights = [
                _rft_weight(_p_from_counts(self.q_correct[i], self.q_attempts[i]))
                for i in eligible
            ]
            if not weights or sum(weights) <= 0:
                weights = [1.0] * len(eligible)

            w_tensor = torch.tensor(weights, dtype=torch.float32)
            if n <= len(eligible):
                idxs = torch.multinomial(w_tensor, n, replacement=False).tolist()
            else:
                idxs = torch.multinomial(w_tensor, n, replacement=True).tolist()
            batch_indices = [eligible[j] for j in idxs]

            prompt_strs = [build_prompt(self.train_data[i].get("problem", "")) for i in batch_indices]
            prompt_tokens_list = []
            for ps in prompt_strs:
                ids = self.encode(ps)
                max_prompt_len = int(self.model.args.ctx_len) - int(self.cfg.max_new_tokens) - 4
                max_prompt_len = max(64, max_prompt_len)
                if len(ids) > max_prompt_len:
                    ids = ids[-max_prompt_len:]
                prompt_tokens_list.append(ids)

            comp_tokens_list, old_logps_list, comp_texts_list, truncated_list =                 self.infer.generate_group_parallel(
                    prompt_tokens_list=prompt_tokens_list,
                    group_size=1,
                    max_new_tokens=self.cfg.max_new_tokens,
                    temperature=self.cfg.temperature,
                    top_p=self.cfg.top_p,
                    top_k=self.cfg.top_k,
                )

            for bi, train_idx in enumerate(batch_indices):
                if bi >= len(comp_tokens_list):
                    continue
                p_hist = _p_from_counts(self.q_correct[train_idx], self.q_attempts[train_idx])
                traj, is_correct, reward = _consume_sample(
                    train_idx,
                    prompt_tokens_list[bi],
                    comp_tokens_list[bi],
                    old_logps_list[bi],
                    comp_texts_list[bi],
                    truncated_list[bi],
                    p_hist,
                )

                self.q_attempts[train_idx] += 1
                self.global_attempts += 1
                if is_correct:
                    self.q_correct[train_idx] += 1
                    self.global_correct += 1

                groups_total += 1
                if is_correct:
                    groups_all_correct += 1
                    groups_used += 1
                else:
                    groups_all_wrong += 1
                    groups_skipped += 1

                if is_correct and len(buffer_trajs) < target_correct:
                    adv = max(0.0, reward) * (1.0 + 2.0 * ((1.0 - p_hist) ** 2))
                    traj["advantage"] = adv
                    adv_values.append(adv)
                    buffer_trajs.append(traj)

            if len(buffer_trajs) >= target_correct:
                break

        all_trajs = buffer_trajs

        # 5. TRAIN
        effective_groups = max(1, groups_used)
        # --- [新增] 1. 训练前统计所有 Effective Groups 的 Token 总数 ---
        # 目的：实现 Global Token Level Normalization
        # 公式：Loss = Sum(All_Loss) / Sum(All_Tokens)
        # 在 Mini-batch 训练中，这就体现为：Batch_Loss / Global_Total_Tokens
        global_valid_tokens = sum(len(traj["comp_tokens"]) for traj in all_trajs)
        global_valid_tokens = max(1, global_valid_tokens)  # 防止除0安全阀
        loss_total = 0.0
        kl_total = 0.0
        grad_norm = 0.0
        batch_cnt = 0
        entropy_total =0.0
        clip_total = 0.0
        clip_total_tokens = 0
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
                seqs, _ = self._pad_batch(seqs, pad_id=0)
                
                inp = seqs[:, :-1].contiguous()
                tgt = seqs[:, 1:].contiguous()
                
                # 前向传播
                logits = self.model(inp)
                if torch.is_tensor(logits) and logits.dim() == 2:
                    logits = logits.unsqueeze(0)
                # --- [新增] 熵计算逻辑开始 ---
                # 计算当前分布的熵: H(x) = - sum(p * log_p)
                # logp: [batch, seq_len, vocab_size]
                logsumexp = torch.logsumexp(logits, dim=-1)
                logit_tgt = logits.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
                logsumexp = logsumexp.float()
                logit_tgt = logit_tgt.float()
                logp = logit_tgt - logsumexp
                # top-k entropy approximation (K=500)
                top_k = min(500, logits.size(-1))
                top_logits, _ = torch.topk(logits, k=top_k, dim=-1)
                logp_top = top_logits.float() - logsumexp.unsqueeze(-1)
                p_top = torch.exp(logp_top)
                entropy_per_token = -(p_top * logp_top).sum(dim=-1)
                del top_logits, logp_top, p_top, logits
                torch.cuda.empty_cache()
                
                # reference logits (batched, same as train_model forward)
                with torch.no_grad():
                    ref_logits = self.ref_model(inp)
                    if torch.is_tensor(ref_logits) and ref_logits.dim() == 2:
                        ref_logits = ref_logits.unsqueeze(0)
                    ref_logsumexp = torch.logsumexp(ref_logits, dim=-1)
                    ref_logit_tgt = ref_logits.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
                    ref_logsumexp = ref_logsumexp.float()
                    ref_logit_tgt = ref_logit_tgt.float()
                    ref_logp_all = ref_logit_tgt - ref_logsumexp
                    del ref_logits
                    torch.cuda.empty_cache()
                # 计算损失
                batch_loss = 0.0
                batch_kl = 0.0
                total_tokens = 0
                batch_entropy =0.0
                for bi, traj in enumerate(batch):
                    prompt_len = len(traj["prompt_tokens"])
                    comp_len = len(traj["comp_tokens"])
                    
                    # 只计算生成部分的损失
                    start_idx = prompt_len - 1
                    end_idx = start_idx + comp_len
                    
                    new_logp = logp[bi, start_idx:end_idx]
                    ref_logp = ref_logp_all[bi, start_idx:end_idx]
                    old_logp = torch.tensor(traj["old_logps"], device=self.device, dtype=torch.float32)
                    curr_entropy = entropy_per_token[bi, start_idx:end_idx]
                    # 对齐长度
                    min_len = min(new_logp.size(0), ref_logp.size(0), old_logp.size(0))
                    if min_len == 0:
                        continue
                    new_logp = new_logp[:min_len]
                    ref_logp = ref_logp[:min_len]
                    old_logp = old_logp[:min_len]
                    curr_entropy = curr_entropy[:min_len]
                    # log ratio for clip_frac stats
                    log_ratio = new_logp - old_logp
                    ratio = torch.exp(log_ratio)
                    clip_total += ((ratio < 0.8) | (ratio > 1.28)).sum().item()
                    clip_total_tokens += ratio.numel()

                    # RFT policy loss: maximize logp with advantage weight
                    adv = torch.tensor(traj["advantage"], 
                                     device=self.device, dtype=torch.float32)
                    policy_loss = -(adv * new_logp).sum()
                    
                    # 计算无偏KL散度
                    kl = compute_unbiased_kl(ref_logp, new_logp).sum()
                    batch_entropy += curr_entropy.sum()
                    batch_loss += policy_loss
                    batch_kl += kl
                    total_tokens += min_len
                
                # 归一化损失 (除以token数和group size)
                if total_tokens > 0:
                    normalized_loss = batch_loss / (global_valid_tokens )
                    normalized_kl = batch_kl / (global_valid_tokens )
                    normalized_entropy = batch_entropy / (global_valid_tokens)
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
                    entropy_total +=normalized_entropy.item()
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
        avg_kl = kl_total / max(1, batch_cnt)
        clip_frac = clip_total / max(1, clip_total_tokens)
        samples_per_sec = stats["total_samples"] / dt if dt > 0 else 0.0
        tokens_per_sec = stats["total_length"] / dt if dt > 0 else 0.0
        ts_stats = self._time_state_stats()
        
        metrics = {
            "step": step,
            "split": "train",
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
            "clip_frac": clip_frac,
            "grad_norm": grad_norm,
            "time": dt,
            "samples_per_sec": samples_per_sec,
            "tokens_per_sec": tokens_per_sec,
            "ts_absmax": ts_stats["absmax"],
            "ts_rms": ts_stats["rms_avg"],
            "ts_bad": ts_stats["bad"],
            "avg_entropy":entropy_total,
        }
        
        self._write_rft_table(step)

        return metrics
    
    def train(self, total_steps: int):
        """训练主循环"""
        self._log(f"开始训练: total_steps={total_steps}")
        self._log(
            f"config: rft_target_correct={self.cfg.rft_target_correct}, rft_max_batch={self.cfg.rft_max_batch}, "
            f"rft_sigma={self.cfg.rft_sigma}, rft_min_attempts={self.cfg.rft_min_attempts}, "
            f"rft_stop_p={self.cfg.rft_stop_p}, rft_expand={self.cfg.rft_expand}"
        )
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
                    self.evaluate(step, tag="eval", sample_ratio=self.cfg.eval_sample_ratio)
                self._plot_metrics()
            elif full_eval and self.test_data:
                self.evaluate(step, tag="full_eval", sample_ratio=1.0)
                self._plot_metrics()
        
        self._log("训练完成!")
        self._plot_metrics()
