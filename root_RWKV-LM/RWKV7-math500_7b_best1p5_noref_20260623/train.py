#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
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

from utils import now_str, append_jsonl, build_prompt,calculate_pass_at_k
from reward import calculate_reward, calculate_reward_details
from infer import AlbatrossBatchInference


@dataclass
class GRPOConfig:
    """GRPO训练配置"""
    # 采样配置
    num_questions: int = 24
    samples_per_question: int = 8
    
    # 生成配置
    max_new_tokens: int = 1500
    temperature: float = 1.0
    top_p: float = 0.28
    top_k: int = 32
    eval_temperature: float = 1.0
    eval_top_p: float = 0.28
    eval_top_k: int = 32
    eval_max_new_tokens: int = 1500
    
    # 训练配置
    ppo_epochs: int = 1
    micro_batch: int = 4
    rollout_forward_batch: int = 8
    lr: float = 1e-5
    grad_clip: float = 1.0
    optimizer: str = "adam"
    
    # 奖励配置
    min_tokens: int = 50
    max_tokens: int = 1024
    length_weight: float = 0.0
    
    # Zstd 和 Ngram 配置
    zstd_threshold: float = 2.5
    zstd_penalty_weight: float = 0.5
    ngram_penalty: float = 0.0
    
    # 负样本降权配置
    neg_adv_weight: float = 0.1
    
    # KL散度配置
    kl_coef: float = 0.05
    kl_mode: str = "k1_reward"
    
    # 正则化
    time_state_l2: float = 1e-4
    time_state_clamp: float = 0.0
    
    # 日志和保存
    log_interval: int = 1
    save_interval: int = 50
    eval_interval: int = 5
    eval_sample_ratio: float = 1.0
    save_last: bool = False
    final_full_eval: bool = False

    # Hard buffer extra-sampling config
    hard_buffer_ttl: int = 10
    hard_buffer_cooldown: int = 5
    hard_buffer_target_samples: int = 192
    hard_buffer_group_size: int = 8
    hard_buffer_extra_lr_scale: float = 0.5
    hard_buffer_adv_clip: float = 2.5
    tune_mode: str = "state"
    reward_mode: str = "trl_doc"
    prompt_mode: str = "trl_doc"
    save_responses: bool = True




class CPUAdamFP32:
    """Adam with fp32 master weights and optimizer states on CPU.

    This avoids GPU Adam-state memory for very large full-parameter runs.
    It is intentionally minimal: enough for GRPOTrainer's lr scaling,
    zero_grad, step, and optional checkpoint hooks.
    """
    def __init__(self, params, lr=1e-6, betas=(0.9, 0.99), eps=1e-8, chunk_size=8_000_000):
        self.params = list(params)
        self.param_groups = [{"params": self.params, "lr": float(lr)}]
        self.betas = betas
        self.eps = float(eps)
        self.chunk_size = int(chunk_size)
        self.state = {}
        self._step = 0

    def zero_grad(self, set_to_none=True):
        for p in self.params:
            if p.grad is not None:
                if set_to_none:
                    p.grad = None
                else:
                    p.grad.detach_()
                    p.grad.zero_()

    def state_dict(self):
        # Avoid serializing tens of GB of CPU optimizer states in experiment checkpoints.
        return {"type": "CPUAdamFP32", "step": self._step, "lr": self.param_groups[0]["lr"]}

    @torch.no_grad()
    def step(self):
        self._step += 1
        beta1, beta2 = self.betas
        lr = float(self.param_groups[0].get("lr", 0.0))
        bc1 = 1.0 - (beta1 ** self._step)
        bc2 = 1.0 - (beta2 ** self._step)
        step_size = lr * (bc2 ** 0.5) / bc1
        for p in self.params:
            if p.grad is None:
                continue
            st = self.state.get(p)
            if st is None:
                master = p.detach().float().cpu().contiguous()
                st = {
                    "master": master,
                    "exp_avg": torch.zeros_like(master, dtype=torch.float32, device="cpu"),
                    "exp_avg_sq": torch.zeros_like(master, dtype=torch.float32, device="cpu"),
                }
                self.state[p] = st
            master = st["master"].view(-1)
            exp_avg = st["exp_avg"].view(-1)
            exp_avg_sq = st["exp_avg_sq"].view(-1)
            g_gpu = p.grad.detach().view(-1)
            p_gpu = p.data.view(-1)
            n = master.numel()
            for start in range(0, n, self.chunk_size):
                end = min(n, start + self.chunk_size)
                g = g_gpu[start:end].float().cpu()
                m = exp_avg[start:end]
                v = exp_avg_sq[start:end]
                m.mul_(beta1).add_(g, alpha=1.0 - beta1)
                v.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)
                upd = m / (v.sqrt().add_(self.eps))
                master[start:end].add_(upd, alpha=-step_size)
                p_gpu[start:end].copy_(master[start:end].to(device=p.device, dtype=p.dtype, non_blocking=False))
                del g, upd
            p.grad = None


def compute_unbiased_kl(ref_logp: torch.Tensor, policy_logp: torch.Tensor) -> torch.Tensor:
    log_ratio = ref_logp - policy_logp
    return torch.exp(log_ratio) - log_ratio - 1.0


class GRPOTrainer:
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
        
        self.log_path = os.path.join(out_dir, "train.log")
        self.metrics_path = os.path.join(out_dir, "metrics.jsonl")
        self.responses_dir = os.path.join(out_dir, "responses_by_step")
        self.eval_path = os.path.join(out_dir, "eval.jsonl")
        os.makedirs(self.responses_dir, exist_ok=True)
        self.eval_by_step_dir = os.path.join(out_dir, "eval_by_step")
        os.makedirs(self.eval_by_step_dir, exist_ok=True)

        self._preeval_map = None
        self._preeval_loaded = False
        
        params = [p for p in self.model.parameters() if p.requires_grad]
        if not params:
            raise RuntimeError("No trainable parameters found")
        opt_name = str(getattr(self.cfg, "optimizer", "adam")).lower()
        if opt_name == "sgd":
            self.opt = torch.optim.SGD(params, lr=self.cfg.lr)
        elif opt_name in ("paged_adamw_8bit", "bnb_paged_adamw8bit", "bnb8bit"):
            import bitsandbytes as bnb
            self.opt = bnb.optim.PagedAdamW8bit(params, lr=self.cfg.lr, betas=(0.9, 0.99), eps=1e-8)
        elif opt_name in ("cpu_adam_fp32", "cpuadam", "cpu_adam"):
            self.opt = CPUAdamFP32(params, lr=self.cfg.lr, betas=(0.9, 0.99), eps=1e-8)
        else:
            self.opt = torch.optim.Adam(params, lr=self.cfg.lr, betas=(0.9, 0.99), eps=1e-8)
        print(f"[optimizer] {opt_name} lr={self.cfg.lr}", flush=True)
        
        self._ts_init: Dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for n, p in self.model.named_parameters():
                if "time_state" in n:
                    self._ts_init[n] = p.detach().clone()
        self._ref_time_state = []
        with torch.no_grad():
            for block in self.model.blocks:
                self._ref_time_state.append(block.att.time_state.detach().clone())

        # Buffer for all-wrong questions, consumed only when an extra batch can be filled.
        self._hard_buffer: List[Dict[str, Any]] = []
        self._hard_buffer_map: Dict[int, Dict[str, Any]] = {}
        self._hard_last_used: Dict[int, int] = {}
    
    def _log(self, msg: str):
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
        if "answer" in ex and ex.get("answer") is not None:
            return str(ex.get("answer"))
        if "solution" in ex and ex.get("solution") is not None:
            return str(ex.get("solution"))
        return ""

    def _load_preeval_map(self):
        if self._preeval_loaded:
            return
        self._preeval_loaded = True
        path = os.path.join(self.eval_by_step_dir, "pre_eval_step_0.jsonl")
        if not os.path.isfile(path):
            return
        m = {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        it = json.loads(line)
                    except Exception:
                        continue
                    prob = it.get("problem", "")
                    gt = it.get("ground_truth", "")
                    key = (prob, gt)
                    m[key] = bool(it.get("is_correct"))
        except Exception:
            return
        self._preeval_map = m

    def evaluate(self, step: int, tag: str = "eval", sample_ratio: float = 1.0) -> Optional[float]:
        if getattr(self.cfg, "prompt_mode", "") == "rlm_repl":
            return self.evaluate_rlm_repl(step=step, tag=tag, sample_ratio=sample_ratio)
        if getattr(self.cfg, "prompt_mode", "") == "recursive_math":
            return self.evaluate_recursive(step=step, tag=tag, sample_ratio=sample_ratio)
        if not self.test_data:
            return None
        t0 = time.time()
        data = self.test_data
        if sample_ratio < 1.0:
            k = max(1, int(len(data) * sample_ratio))
            idxs = self.rng.sample(range(len(data)), k)
            data = [data[i] for i in idxs]
        preeval_map = None
        preeval_total = 0
        preeval_correct = 0
        if tag != "pre_eval":
            self._load_preeval_map()
            preeval_map = self._preeval_map
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
        sum_zstd_penalty = 0.0 
        sum_zstd_ratio = 0.0 
        chunk_size = 192
        for start in range(0, len(data), chunk_size):
            ex_list = data[start:start + chunk_size]
            problems = [ex.get("problem", "") for ex in ex_list]
            answers = [self._get_answer(ex) for ex in ex_list]
            prompt_strs = [build_prompt(p, mode=getattr(self.cfg, "prompt_mode", "rwkv_boxed")) for p in problems]
            prompt_tokens_list = []
            for ps in prompt_strs:
                ids = self.encode(ps)
                max_prompt_len = int(self.model.args.ctx_len) - int(getattr(self.cfg, "eval_max_new_tokens", self.cfg.max_new_tokens)) - 4
                max_prompt_len = max(64, max_prompt_len)
                if len(ids) > max_prompt_len:
                    ids = ids[-max_prompt_len:]
                prompt_tokens_list.append(ids)

            comp_tokens_list, _, comp_texts_list, truncated_list = \
                self.infer.generate_group_parallel(
                    prompt_tokens_list=prompt_tokens_list,
                    group_size=1,
                    max_new_tokens=getattr(self.cfg, "eval_max_new_tokens", self.cfg.max_new_tokens),
                    temperature=self.cfg.eval_temperature,
                    top_p=self.cfg.eval_top_p,
                    top_k=self.cfg.eval_top_k,
                    stop_on_think_close=False,
                    stop_on_user=True,
                    stop_on_boxed=True,
                    stop_on_token_zero=True,
                    stop_on_repeat_ngram=False,
                    presence_penalty=0.0,
                    frequency_penalty=0.0,
                    alpha_decay=1.0,
                )

            for i in range(len(prompt_tokens_list)):
                problem = problems[i]
                answer = answers[i]
                if preeval_map is not None:
                    key = (problem, answer)
                    if key in preeval_map:
                        preeval_total += 1
                        if preeval_map[key]:
                            preeval_correct += 1
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
                    repeat_penalty=self.cfg.ngram_penalty,
                    zstd_threshold=self.cfg.zstd_threshold,
                    zstd_penalty_weight=self.cfg.zstd_penalty_weight,
                    reward_mode=getattr(self.cfg, "reward_mode", "rwkv")
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
                    "eval_max_new_tokens": getattr(self.cfg, "eval_max_new_tokens", self.cfg.max_new_tokens),
                    "repeat_16gram_5": repeat_flag,
                    "zstd_ratio": reward_details.get("zstd_ratio", 0.0),
                    "zstd_penalty": reward_details.get("zstd_penalty", 0.0),
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
                sum_zstd_penalty += float(reward_details.get("zstd_penalty", 0.0))
                sum_zstd_ratio +=  float(reward_details.get("zstd_ratio", 0.0))
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
        avg_zstd_penalty = sum_zstd_penalty / max(1, total)
        avg_zstd_ratio = sum_zstd_ratio / max(1, total)
        preeval_acc = None
        eval_acc_delta = None
        if preeval_total > 0:
            preeval_acc = preeval_correct / preeval_total
            eval_acc_delta = acc - preeval_acc
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
            "avg_zstd_penalty":avg_zstd_penalty,
            "avg_zstd_ratio":avg_zstd_ratio,
            "preeval_acc": preeval_acc,
            "eval_acc_delta": eval_acc_delta,
            "preeval_count": preeval_total,
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
        state = self.infer.infer_model.generate_zero_state(B)
        for i, ts in enumerate(self._ref_time_state):
            state[1][i] = ts.unsqueeze(0).expand(B, -1, -1, -1).clone()
        return state
    
    def _pad_batch(self, seqs: List[List[int]], pad_id: int = 0):
        max_len = max(len(s) for s in seqs)
        padded = []
        masks = []
        for s in seqs:
            pad_len = max_len - len(s)
            padded.append(s + [pad_id] * pad_len)
            masks.append([1] * len(s) + [0] * pad_len)
        return torch.tensor(padded, device=self.device, dtype=torch.long), \
               torch.tensor(masks, device=self.device, dtype=torch.bool)

    def _prune_hard_buffer(self, step: int):
        if not self._hard_buffer:
            return
        keep = []
        for item in self._hard_buffer:
            if int(item.get("ttl_left", 0)) <= 0:
                self._hard_buffer_map.pop(int(item.get("train_idx", -1)), None)
                continue
            keep.append(item)
        self._hard_buffer = keep

    def _remove_hard_candidate(self, train_idx: int):
        train_idx = int(train_idx)
        self._hard_buffer_map.pop(train_idx, None)
        self._hard_buffer = [it for it in self._hard_buffer if int(it.get("train_idx", -1)) != train_idx]

    def _push_hard_candidate(
        self,
        train_idx: int,
        prompt_tokens: List[int],
        problem: str,
        answer: str,
        step: int,
        ignore_cooldown: bool = False,
    ) -> bool:
        cooldown = max(0, int(self.cfg.hard_buffer_cooldown))
        last = self._hard_last_used.get(train_idx, -10**9)
        if (not ignore_cooldown) and (step - last <= cooldown):
            return False

        ttl = max(1, int(self.cfg.hard_buffer_ttl))
        item = self._hard_buffer_map.get(train_idx)
        if item is not None:
            item["prompt_tokens"] = prompt_tokens
            item["problem"] = problem
            item["answer"] = answer
            item["added_step"] = step
            # Keep ttl_left unchanged for existing buffered questions.
            return False

        item = {
            "train_idx": train_idx,
            "prompt_tokens": prompt_tokens,
            "problem": problem,
            "answer": answer,
            "added_step": step,
            "ttl_left": ttl,
        }
        self._hard_buffer.append(item)
        self._hard_buffer_map[train_idx] = item
        return True

    def _pop_hard_batch(self, step: int, needed_questions: int):
        self._prune_hard_buffer(step)
        cooldown = max(0, int(self.cfg.hard_buffer_cooldown))
        eligible = []
        for item in self._hard_buffer:
            train_idx = int(item.get("train_idx", -1))
            last = self._hard_last_used.get(train_idx, -10**9)
            if step - last <= cooldown:
                continue
            eligible.append(item)

        if len(eligible) < needed_questions:
            return [], len(eligible)

        selected = self.rng.sample(eligible, needed_questions)

        for item in selected:
            self._hard_last_used[int(item["train_idx"])] = step

        return selected, len(eligible)

    def _logp_with_sampling(self, logits: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        if torch.is_tensor(logits) and logits.dim() == 2:
            logits = logits.unsqueeze(0)
        logsumexp = torch.logsumexp(logits, dim=-1)
        logsumexp = logsumexp.float()
        logit_tgt = logits.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        logit_tgt = logit_tgt.float()
        logp = logit_tgt - logsumexp
        return logp

    def _optimize_trajs(self, trajs: List[Dict[str, Any]], lr_scale: float = 1.0, adv_clip: Optional[float] = None) -> Dict[str, float]:
        if not trajs:
            return {
                'loss_total': 0.0,
                'kl_total': 0.0,
                'entropy_total': 0.0,
                'batch_cnt': 0,
                'clip_total': 0.0,
                'clip_total_tokens': 0,
                'grad_norm': 0.0,
            }

        neg_w = self.cfg.neg_adv_weight
        pos_tokens = sum(len(traj['comp_tokens']) for traj in trajs if traj['is_correct'])
        neg_tokens = sum(len(traj['comp_tokens']) for traj in trajs if not traj['is_correct'])
        valid_tokens = pos_tokens + (neg_w * neg_tokens)
        valid_tokens = max(1.0, float(valid_tokens))

        loss_total = 0.0
        kl_total = 0.0
        entropy_total = 0.0
        batch_cnt = 0
        clip_total = 0.0
        clip_total_tokens = 0
        grad_norm = 0.0

        base_lrs = [float(pg.get('lr', self.cfg.lr)) for pg in self.opt.param_groups]
        if lr_scale != 1.0:
            for pg, base_lr in zip(self.opt.param_groups, base_lrs):
                pg['lr'] = base_lr * lr_scale

        try:
            for _ in range(self.cfg.ppo_epochs):
                self.model.train()
                self.opt.zero_grad(set_to_none=True)

                trajs_sorted = sorted(
                    trajs,
                    key=lambda x: len(x['prompt_tokens']) + len(x['comp_tokens']),
                    reverse=True,
                )

                for start in range(0, len(trajs_sorted), self.cfg.micro_batch):
                    batch = trajs_sorted[start:start + self.cfg.micro_batch]
                    seqs = [traj['prompt_tokens'] + traj['comp_tokens'] for traj in batch]
                    seqs, _ = self._pad_batch(seqs, pad_id=0)

                    inp = seqs[:, :-1].contiguous()
                    tgt = seqs[:, 1:].contiguous()

                    logits = self.model(inp)
                    if torch.is_tensor(logits) and logits.dim() == 2:
                        logits = logits.unsqueeze(0)

                    logp = self._logp_with_sampling(logits, tgt)
                    del logits
                    torch.cuda.empty_cache()

                    if self.ref_model is not None:
                        with torch.no_grad():
                            ref_logits = self.ref_model(inp)
                            if torch.is_tensor(ref_logits) and ref_logits.dim() == 2:
                                ref_logits = ref_logits.unsqueeze(0)
                            ref_logp_all = self._logp_with_sampling(ref_logits, tgt)
                            del ref_logits
                            torch.cuda.empty_cache()
                    else:
                        ref_logp_all = None

                    batch_loss = 0.0
                    batch_kl = 0.0
                    total_tokens = 0

                    for bi, traj in enumerate(batch):
                        prompt_len = len(traj['prompt_tokens'])
                        comp_len = len(traj['comp_tokens'])

                        start_idx = prompt_len - 1
                        end_idx = start_idx + comp_len

                        new_logp = logp[bi, start_idx:end_idx]
                        ref_logp = ref_logp_all[bi, start_idx:end_idx] if ref_logp_all is not None else None
                        old_logp = torch.tensor(traj['old_logps'], device=self.device, dtype=torch.float32)
                        min_len = min(new_logp.size(0), (ref_logp.size(0) if ref_logp is not None else new_logp.size(0)), old_logp.size(0))
                        if min_len == 0:
                            continue

                        new_logp = new_logp[:min_len]
                        if ref_logp is not None:
                            ref_logp = ref_logp[:min_len]
                        old_logp = old_logp[:min_len]
                        log_ratio = new_logp - old_logp

                        adv = torch.tensor(traj['advantage'], device=self.device, dtype=torch.float32)
                        if self.cfg.neg_adv_weight < 1.0:
                            neg_mask = adv < 0
                            adv[neg_mask] *= self.cfg.neg_adv_weight
                        if adv_clip is not None and adv_clip > 0:
                            adv = torch.clamp(adv, min=-float(adv_clip), max=float(adv_clip))

                        ratio = torch.exp(log_ratio)
                        ratio_clipped = torch.clamp(ratio, 0.8, 1.28)
                        clip_total += ((ratio < 0.8) | (ratio > 1.28)).sum().item()
                        clip_total_tokens += ratio.numel()

                        policy_loss = -(torch.min(ratio * adv, ratio_clipped * adv)).sum()
                        if ref_logp is None:
                            kl = new_logp.new_zeros(())
                        else:
                            kl = compute_unbiased_kl(ref_logp, new_logp).sum()

                        batch_loss += policy_loss
                        batch_kl += kl
                        total_tokens += min_len

                    if total_tokens > 0:
                        normalized_loss = batch_loss / valid_tokens
                        normalized_kl = batch_kl / valid_tokens
                        normalized_entropy = 0.0

                        if self.cfg.kl_mode == 'k3_loss':
                            total_loss = normalized_loss + self.cfg.kl_coef * normalized_kl
                        else:
                            total_loss = normalized_loss

                        if self.cfg.time_state_l2 > 0:
                            l2_reg = 0.0
                            for n, param in self.model.named_parameters():
                                if 'time_state' in n:
                                    l2_reg += (param.float() - self._ts_init[n].float()).pow(2).mean()
                            total_loss += self.cfg.time_state_l2 * l2_reg

                        total_loss.backward()

                        loss_total += normalized_loss.item()
                        kl_total += normalized_kl.item()
                        entropy_total += float(normalized_entropy)
                        batch_cnt += 1

                if self.cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad],
                        self.cfg.grad_clip,
                    )

                with torch.no_grad():
                    g2 = 0.0
                    for p in self.model.parameters():
                        if p.requires_grad and p.grad is not None:
                            g = p.grad.detach().float()
                            g2 += (g.norm(2) ** 2).item()
                    grad_norm = math.sqrt(g2)

                self.opt.step()
                # 7B full-parameter runs cannot keep gradient buffers while rebuilding rollout cache.
                self.opt.zero_grad(set_to_none=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                infer_model = getattr(self.infer, "infer_model", None)
                mark_dirty = getattr(infer_model, "mark_dirty", None)
                if mark_dirty is not None:
                    mark_dirty()

                if self.cfg.time_state_clamp > 0:
                    with torch.no_grad():
                        for n, p in self.model.named_parameters():
                            if 'time_state' in n:
                                p.data.clamp_(-self.cfg.time_state_clamp, self.cfg.time_state_clamp)
        finally:
            if lr_scale != 1.0:
                for pg, base_lr in zip(self.opt.param_groups, base_lrs):
                    pg['lr'] = base_lr

        return {
            'loss_total': loss_total,
            'kl_total': kl_total,
            'entropy_total': entropy_total,
            'batch_cnt': batch_cnt,
            'clip_total': clip_total,
            'clip_total_tokens': clip_total_tokens,
            'grad_norm': grad_norm,
        }

    def train_step(self, step: int) -> Dict[str, Any]:
        if getattr(self.cfg, "prompt_mode", "") == "rlm_repl":
            return self.train_step_rlm_repl(step)
        if getattr(self.cfg, "prompt_mode", "") == "recursive_math":
            return self.train_step_recursive(step)
        t0 = time.time()
        
        if len(self.train_data) >= self.cfg.num_questions:
            sampled_indices = self.rng.sample(range(len(self.train_data)), self.cfg.num_questions)
        else:
            sampled_indices = [self.rng.randrange(len(self.train_data)) 
                             for _ in range(self.cfg.num_questions)]
        
        sampled_questions = [self.train_data[i] for i in sampled_indices]
        
        prompt_strs = [build_prompt(q.get("problem", ""), mode=getattr(self.cfg, "prompt_mode", "rwkv_boxed")) for q in sampled_questions]
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
                stop_on_think_close=False,
                stop_on_user=True,
                stop_on_boxed=True,
                stop_on_repeat_ngram=False,
                presence_penalty=0.0,
                frequency_penalty=0.0,
                alpha_decay=1.0,
            )

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
            "sum_zstd_penalty": 0.0, 
            "sum_zstd_ratio": 0.0, 
        }
        adv_values = []
        groups_total = 0
        groups_all_correct = 0
        groups_all_wrong = 0
        groups_used = 0
        groups_skipped = 0
        extra_groups_total = 0
        extra_groups_all_correct = 0
        extra_groups_all_wrong = 0
        extra_groups_used = 0
        extra_groups_skipped = 0

        adv_component_keys = (
            'correct_reward',
            'format_reward',
            'length_reward',
            'repeat_penalty',
            'zstd_penalty',
        )
        # Preserve the historical full-scale reward ratio used when decoupling advantages:
        # correct : format : length : repeat : zstd =
        # 1 : 1 : min(0.25, |length_weight| * 0.5) : |ngram_penalty| : |zstd_penalty_weight|
        # residual_reward is intentionally excluded.
        adv_component_weights = {
            'correct_reward': 1.0,
            'format_reward': 0.0 if getattr(self.cfg, 'reward_mode', 'rwkv') == 'trl_doc' else 1.0,
            'length_reward': min(0.25, abs(float(self.cfg.length_weight)) * 0.5),
            'repeat_penalty': abs(float(self.cfg.ngram_penalty)),
            'zstd_penalty': abs(float(self.cfg.zstd_penalty_weight)),
        }
        def _consume_sample(group, comp_tokens, old_logps, comp_text, truncated, is_extra: bool = False):
            repeat_flag = self._has_repeated_ngrams(comp_text, n=16, repeat=5)

            reward, is_correct, is_format_correct, reward_details = calculate_reward_details(
                text=comp_text,
                ground_truth=group["answer"],
                token_length=len(comp_tokens),
                min_tokens=self.cfg.min_tokens,
                max_tokens=self.cfg.max_tokens,
                length_weight=self.cfg.length_weight,
                repeat_ngram=repeat_flag,
                repeat_penalty=self.cfg.ngram_penalty,
                zstd_threshold=self.cfg.zstd_threshold,
                zstd_penalty_weight=self.cfg.zstd_penalty_weight,
                reward_mode=getattr(self.cfg, "reward_mode", "rwkv")
            )

            sample_idx = group["sampled"]
            group["sampled"] += 1

            reward_components = {
                'correct_reward': float(reward_details.get('correct_reward', 0.0)),
                'format_reward': float(reward_details.get('format_reward', 0.0)),
                'length_reward': float(reward_details.get('length_reward', 0.0)),
                'repeat_penalty': float(reward_details.get('repeat_penalty', 0.0)),
                'zstd_penalty': float(reward_details.get('zstd_penalty', 0.0)),
            }
            traj = {
                "prompt_tokens": group["prompt_tokens"],
                "comp_tokens": comp_tokens,
                "old_logps": old_logps,
                "is_extra": bool(is_extra),
                "reward": reward,
                "text": comp_text,
                "is_correct": is_correct,
                "is_format_correct": is_format_correct,
                "truncated": truncated,
                "reward_components": reward_components,
            }

            group["group_rewards"].append(reward)
            group["group_trajs"].append(traj)

            stats["total_samples"] += 1
            stats["total_reward"] += reward
            stats["total_length"] += len(comp_tokens)
            if is_correct:
                stats["correct_samples"] += 1
                group["correct_in_group"] += 1
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
            stats["sum_zstd_penalty"] += float(reward_details.get("zstd_penalty", 0.0))
            stats["sum_zstd_ratio"] += float(reward_details.get("zstd_ratio", 0.0))
            
            record = {
                "step": step,
                "question_idx": group["q_idx"],
                "sample_idx": sample_idx,
                "problem": group["problem"],
                "ground_truth": group["answer"],
                "response": comp_text,
                "pred_extracted": reward_details.get("extracted_answer"),
                "gt_extracted": reward_details.get("ground_truth_answer"),
                "reward": reward,
                "is_correct": is_correct,
                "is_format_correct": is_format_correct,
                "truncated": truncated,
                "reward_details": reward_details,
            }
            if getattr(self.cfg, "save_responses", True):
                step_path = os.path.join(self.responses_dir, f"step_{step}.jsonl")
                append_jsonl(step_path, record)

        group_infos = []
        for q_idx, (question, train_idx) in enumerate(zip(sampled_questions, sampled_indices)):
            group_infos.append({
                "q_idx": q_idx,
                "train_idx": int(train_idx),
                "problem": question.get("problem", ""),
                "answer": self._get_answer(question),
                "prompt_tokens": prompt_tokens_list[q_idx],
                "group_rewards": [],
                "group_trajs": [],
                "correct_in_group": 0,
                "sampled": 0,
                "target_samples": self.cfg.samples_per_question,
                "is_extra": False,
            })

        for q_idx, group in enumerate(group_infos):
            for i in range(self.cfg.samples_per_question):
                flat_idx = q_idx * self.cfg.samples_per_question + i
                if flat_idx >= len(comp_tokens_list):
                    continue
                comp_tokens = comp_tokens_list[flat_idx]
                old_logps = old_logps_list[flat_idx]
                comp_text = comp_texts_list[flat_idx]
                truncated = truncated_list[flat_idx]
                _consume_sample(group, comp_tokens, old_logps, comp_text, truncated)

        hard_buffer_enabled = int(self.cfg.hard_buffer_target_samples) > 0
        hard_buffer_added = 0
        if hard_buffer_enabled:
            for group in group_infos:
                if group["correct_in_group"] == 0:
                    added = self._push_hard_candidate(
                        train_idx=int(group["train_idx"]),
                        prompt_tokens=group["prompt_tokens"],
                        problem=group["problem"],
                        answer=group["answer"],
                        step=step,
                    )
                    if added:
                        hard_buffer_added += 1

        extra_target_samples = 0 if not hard_buffer_enabled else max(self.cfg.samples_per_question, int(self.cfg.hard_buffer_target_samples))
        extra_group_size = max(1, int(self.cfg.hard_buffer_group_size))
        needed_questions = 0 if extra_target_samples <= 0 else max(1, int(math.ceil(float(extra_target_samples) / float(extra_group_size))))

        # Match the older hard-buffer behavior: only trigger when a full hard batch can be formed.
        if hard_buffer_enabled and needed_questions > 0:
            hard_selected, hard_eligible = self._pop_hard_batch(step, needed_questions)
        else:
            hard_selected, hard_eligible = [], 0
        hard_triggered = len(hard_selected) > 0

        if hard_triggered:
            extra_items = []
            for it in hard_selected:
                tid = int(it["train_idx"])
                extra_items.append({
                    "train_idx": tid,
                    "problem": it["problem"],
                    "answer": it["answer"],
                    "prompt_tokens": it["prompt_tokens"],
                    "extra_source": "hard",
                })

            base_q_idx = len(group_infos)
            extra_groups = []
            for i, it in enumerate(extra_items):
                extra_groups.append({
                    "q_idx": base_q_idx + i,
                    "train_idx": int(it["train_idx"]),
                    "problem": it["problem"],
                    "answer": it["answer"],
                    "prompt_tokens": it["prompt_tokens"],
                    "group_rewards": [],
                    "group_trajs": [],
                    "correct_in_group": 0,
                    "sampled": 0,
                    "target_samples": extra_group_size,
                    "is_extra": True,
                    "extra_source": it.get("extra_source", "hard"),
                })

            group_infos.extend(extra_groups)

            extra_comp_tokens_list, extra_old_logps_list, extra_comp_texts_list, extra_truncated_list = \
                self.infer.generate_group_parallel(
                    prompt_tokens_list=[g[ prompt_tokens] for g in extra_groups],
                    group_size=extra_group_size,
                    max_new_tokens=self.cfg.max_new_tokens,
                    temperature=self.cfg.temperature,
                    top_p=self.cfg.top_p,
                    top_k=self.cfg.top_k,
                    stop_on_think_close=False,
                    stop_on_repeat_ngram=False,
                )

            for bi, g in enumerate(extra_groups):
                base = bi * extra_group_size
                for j in range(extra_group_size):
                    flat_idx = base + j
                    if flat_idx >= len(extra_comp_tokens_list):
                        continue
                    comp_tokens = extra_comp_tokens_list[flat_idx]
                    old_logps = extra_old_logps_list[flat_idx]
                    comp_text = extra_comp_texts_list[flat_idx]
                    truncated = extra_truncated_list[flat_idx]
                    _consume_sample(g, comp_tokens, old_logps, comp_text, truncated, is_extra=True)

                source = str(g.get("extra_source", "hard"))
                tid = int(g["train_idx"])
                if source == "hard":
                    if g["correct_in_group"] == 0:
                        item = self._hard_buffer_map.get(tid)
                        if item is not None:
                            item["ttl_left"] = int(item.get("ttl_left", max(1, int(self.cfg.hard_buffer_ttl)))) - 1
                            if int(item.get("ttl_left", 0)) <= 0:
                                self._remove_hard_candidate(tid)
                    else:
                        # If not all-0, leave buffer immediately.
                        self._remove_hard_candidate(tid)
                else:
                    if g["correct_in_group"] == 0:
                        self._push_hard_candidate(
                            train_idx=tid,
                            prompt_tokens=g["prompt_tokens"],
                            problem=g["problem"],
                            answer=g["answer"],
                            step=step,
                            ignore_cooldown=False,
                        )

        # K1-as-reward component: k1 = logp_old - logp_ref, reward contribution uses -k1.
        # We compute it once per sampled trajectory before advantage normalization.
        all_group_trajs = []
        for g in group_infos:
            all_group_trajs.extend(g.get('group_trajs', []))

        if self.cfg.kl_mode == 'k1_reward' and all_group_trajs and abs(float(self.cfg.kl_coef)) > 0.0:
            self.ref_model.eval()
            ref_bs = max(1, int(self.cfg.micro_batch))
            for s in range(0, len(all_group_trajs), ref_bs):
                ref_batch = all_group_trajs[s:s + ref_bs]
                seqs = [traj['prompt_tokens'] + traj['comp_tokens'] for traj in ref_batch]
                seqs, _ = self._pad_batch(seqs, pad_id=0)
                inp = seqs[:, :-1].contiguous()
                tgt = seqs[:, 1:].contiguous()

                with torch.no_grad():
                    ref_logits = self.ref_model(inp)
                    if torch.is_tensor(ref_logits) and ref_logits.dim() == 2:
                        ref_logits = ref_logits.unsqueeze(0)
                    ref_logp_all = self._logp_with_sampling(ref_logits, tgt)
                    del ref_logits

                for bi, traj in enumerate(ref_batch):
                    prompt_len = len(traj['prompt_tokens'])
                    comp_len = len(traj['comp_tokens'])
                    start_idx = prompt_len - 1
                    end_idx = start_idx + comp_len
                    ref_logp = ref_logp_all[bi, start_idx:end_idx]
                    old_logps = traj.get('old_logps', [])
                    min_len = min(ref_logp.size(0), len(old_logps))
                    if min_len <= 0:
                        traj['k1_seq'] = 0.0
                        continue
                    ref_sum = float(ref_logp[:min_len].float().sum().item())
                    old_sum = float(sum(float(x) for x in old_logps[:min_len]))
                    # Use token-average K1 to avoid sequence-length-dependent scale explosion.
                    k1_seq = (old_sum - ref_sum) / float(min_len)
                    traj['k1_seq'] = k1_seq
                    # Fold K1 reward into the correctness component so it shares the same relative-advantage normalization.
                    traj['reward_components']['correct_reward'] += (-float(self.cfg.kl_coef) * k1_seq)

        groups_total = len(group_infos)

        for group in group_infos:
            is_extra_group = bool(group.get("is_extra", False))
            if is_extra_group:
                extra_groups_total += 1
            group_all_wrong = (group["correct_in_group"] == 0)
            group_all_correct = (group["correct_in_group"] == group.get("target_samples", self.cfg.samples_per_question))

            if group_all_wrong:
                groups_all_wrong += 1
                if is_extra_group:
                    extra_groups_all_wrong += 1
            elif group_all_correct:
                groups_all_correct += 1
                if is_extra_group:
                    extra_groups_all_correct += 1

            if group_all_wrong or group_all_correct:
                groups_skipped += 1
                if is_extra_group:
                    extra_groups_skipped += 1
                continue

            groups_used += 1
            if is_extra_group:
                extra_groups_used += 1

            if not group["group_rewards"]:
                continue

            n = len(group["group_trajs"])

            comp_stats = {}
            for key in adv_component_keys:
                vals = [float(traj.get("reward_components", {}).get(key, 0.0)) for traj in group["group_trajs"]]
                if not vals:
                    comp_stats[key] = (0.0, 0.0)
                    continue
                mean_v = sum(vals) / len(vals)
                var_v = sum((v - mean_v) ** 2 for v in vals) / len(vals)
                std_v = math.sqrt(var_v) if var_v > 1e-6 else 0.0
                comp_stats[key] = (mean_v, std_v)

            # 2. ????????
            c = group["correct_in_group"]
            if getattr(self.cfg, "reward_mode", "rwkv") == "trl_doc":
                acc_weight = 1.0
            else:
                acc_weight = calculate_pass_at_k(n, c, 1) * self.cfg.samples_per_question
                acc_weight = max(1e-6, float(acc_weight))

            weight_norm = sum(abs(w) for w in adv_component_weights.values())
            weight_norm = max(1e-6, float(weight_norm))

            for traj in group["group_trajs"]:
                decoupled_adv = 0.0
                comps = traj.get("reward_components", {})
                for key in adv_component_keys:
                    w = float(adv_component_weights.get(key, 0.0))
                    if w == 0.0:
                        continue
                    mean_v, std_v = comp_stats.get(key, (0.0, 0.0))
                    if std_v <= 1e-6:
                        rel_adv = 0.0
                    else:
                        rel_adv = (float(comps.get(key, 0.0)) - mean_v) / std_v
                    decoupled_adv += w * rel_adv

                traj["advantage"] = (decoupled_adv / weight_norm) / acc_weight
                adv_values.append(traj["advantage"])

            all_trajs.extend(group["group_trajs"])

        # 5. 训练
        # 5. training (normal step + extra hard-buffer step)
        normal_trajs = [t for t in all_trajs if not bool(t.get("is_extra", False))]
        extra_trajs = [t for t in all_trajs if bool(t.get("is_extra", False))]

        infer_model = getattr(self.infer, "infer_model", None)
        cleanup_rollout = getattr(infer_model, "cleanup_stateful_rollout", None)
        if cleanup_rollout is not None:
            cleanup_rollout()

        normal_opt = self._optimize_trajs(normal_trajs, lr_scale=1.0, adv_clip=None)

        extra_lr_scale = float(self.cfg.hard_buffer_extra_lr_scale)
        extra_adv_clip = float(self.cfg.hard_buffer_adv_clip)
        extra_opt = {
            'loss_total': 0.0,
            'kl_total': 0.0,
            'entropy_total': 0.0,
            'batch_cnt': 0,
            'clip_total': 0.0,
            'clip_total_tokens': 0,
            'grad_norm': 0.0,
        }

        extra_step_ran = int(len(extra_trajs) > 0)
        if extra_step_ran:
            extra_opt = self._optimize_trajs(
                extra_trajs,
                lr_scale=extra_lr_scale,
                adv_clip=extra_adv_clip,
            )

        loss_total = float(normal_opt['loss_total'] + extra_opt['loss_total'])
        kl_total = float(normal_opt['kl_total'] + extra_opt['kl_total'])
        entropy_total = float(normal_opt['entropy_total'] + extra_opt['entropy_total'])
        batch_cnt = int(normal_opt['batch_cnt'] + extra_opt['batch_cnt'])
        clip_total = float(normal_opt['clip_total'] + extra_opt['clip_total'])
        clip_total_tokens = int(normal_opt['clip_total_tokens'] + extra_opt['clip_total_tokens'])
        grad_norm = max(float(normal_opt['grad_norm']), float(extra_opt['grad_norm']))

        extra_samples = sum(1 for t in all_trajs if bool(t.get("is_extra", False)))

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
        avg_zstd_penalty = stats["sum_zstd_penalty"] / max(1, stats["total_samples"])
        avg_zstd_ratio = stats["sum_zstd_ratio"] / max(1, stats["total_samples"])

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
            "avg_zstd_penalty": avg_zstd_penalty,
            "avg_zstd_ratio": avg_zstd_ratio,
            "hard_buffer_size": len(self._hard_buffer),
            "hard_buffer_added": hard_buffer_added,
            "hard_buffer_eligible": hard_eligible,
            "hard_buffer_selected": len(hard_selected),
            "hard_buffer_triggered": int(hard_triggered),
            "extra_step_ran": extra_step_ran,
            "extra_samples": extra_samples,
            "extra_groups_total": extra_groups_total,
            "extra_groups_used": extra_groups_used,
            "extra_groups_skipped": extra_groups_skipped,
            "extra_groups_all_correct": extra_groups_all_correct,
            "extra_groups_all_wrong": extra_groups_all_wrong,
            "extra_loss": float(extra_opt['loss_total']),
            "extra_avg_kl": (float(extra_opt['kl_total']) / max(1, int(extra_opt['batch_cnt']))),
            "extra_grad_norm": float(extra_opt['grad_norm']),
            "extra_lr_scale": extra_lr_scale,
            "extra_adv_clip": extra_adv_clip,
        }
        
        return metrics
    


    def _rlm_system_prompt(self) -> str:
        return (
            "You are a Recursive Language Model (RLM) with a persistent Python REPL. "
            "Solve the task by writing Python code inside ```repl blocks. The REPL has variables/functions: "
            "context (the problem), llm_query(prompt), llm_query_batched(prompts), SHOW_VARS(), and answer. "
            "To submit the final answer, execute: answer[\"content\"] = \"\\\\boxed{...}\"; answer[\"ready\"] = True. "
            "Do not answer outside the REPL. Use at least one ```repl block."
        )

    def _rlm_build_prompt(self, history: List[Dict[str, str]]) -> List[int]:
        text_parts = ["System: " + self._rlm_system_prompt()]
        for m in history:
            role = m.get("role", "user")
            content = m.get("content", "")
            text_parts.append(("Assistant: " if role == "assistant" else "User: ") + content)
        text_parts.append("Assistant:")
        text = "\n\n".join(text_parts)
        ids = self.encode(text)
        max_prompt_len = int(self.model.args.ctx_len) - int(self.cfg.max_new_tokens) - 4
        max_prompt_len = max(64, max_prompt_len)
        if len(ids) > max_prompt_len:
            ids = ids[-max_prompt_len:]
        return ids

    def _rlm_subquery_stub(self, prompt: str) -> str:
        # Official RLM routes this back to the trainer inference server. For the
        # local RWKV trainer we keep the hook deterministic and cheap first; once
        # the orchestrator learns to call it, this can be replaced with a real
        # one-shot RWKV generation path.
        return "Sub-LLM unavailable in this local smoke; solve directly and return a boxed answer."

    def _rlm_rollout_items(self, questions: List[Dict[str, Any]], indices: List[int], eval_mode: bool = False):
        from rlm_repl_runtime import LocalRLMRepl, find_repl_blocks, format_repl_outputs
        group_size = 1 if eval_mode else int(self.cfg.samples_per_question)
        max_turns = 4
        turn_tokens = min(512, int(getattr(self.cfg, "eval_max_new_tokens", self.cfg.max_new_tokens) if eval_mode else self.cfg.max_new_tokens))
        temperature = float(self.cfg.eval_temperature if eval_mode else self.cfg.temperature)
        top_p = float(self.cfg.eval_top_p if eval_mode else self.cfg.top_p)
        top_k = int(self.cfg.eval_top_k if eval_mode else self.cfg.top_k)
        items = []
        for q_idx, (q, train_idx) in enumerate(zip(questions, indices)):
            for sample_idx in range(group_size):
                problem = q.get("problem", "")
                answer = self._get_answer(q)
                items.append({
                    "q_idx": q_idx, "train_idx": int(train_idx), "sample_idx": sample_idx,
                    "problem": problem, "answer": answer,
                    "history": [{"role": "user", "content": f"context is the math problem. Solve it.\nProblem: {problem}\nTurn 1/{max_turns}:"}],
                    "segments": [], "final_answer": None, "done": False,
                    "repl_calls": 0, "sub_llm_calls": 0, "truncated": False,
                    "repl": LocalRLMRepl(context=problem, llm_query=self._rlm_subquery_stub),
                })
        for turn in range(max_turns):
            active_items = [it for it in items if not it["done"]]
            if not active_items:
                break
            prompts = [self._rlm_build_prompt(it["history"]) for it in active_items]
            comp_tokens, old_logps, comp_texts, truncs = self.infer.generate_group_parallel(
                prompt_tokens_list=prompts, group_size=1, max_new_tokens=turn_tokens,
                temperature=temperature, top_p=top_p, top_k=top_k,
                stop_on_think_close=False, stop_on_user=True, stop_on_boxed=False,
                stop_on_repeat_ngram=True, presence_penalty=0.0, frequency_penalty=0.0,
                alpha_decay=1.0, stop_strings=["\n```"],
            )
            for i, it in enumerate(active_items):
                text = comp_texts[i]
                it["segments"].append({
                    "prompt_tokens": prompts[i], "comp_tokens": comp_tokens[i],
                    "old_logps": old_logps[i], "stage": f"turn{turn+1}",
                })
                it["history"].append({"role": "assistant", "content": text})
                it["truncated"] = bool(it["truncated"] or truncs[i])
                outputs = []
                for code in find_repl_blocks(text):
                    outputs.append(it["repl"].execute(code))
                    it["repl_calls"] += 1
                    if outputs[-1].get("final_answer") is not None:
                        it["final_answer"] = outputs[-1].get("final_answer")
                        it["done"] = True
                        break
                if outputs:
                    it["history"].append({"role": "user", "content": format_repl_outputs(outputs)})
                else:
                    it["history"].append({"role": "user", "content": "No REPL code block found. Use a ```repl block and set answer when done."})
                if turn + 1 < max_turns and not it["done"]:
                    it["history"].append({"role": "user", "content": f"Turn {turn+2}/{max_turns}: continue. Use REPL and submit answer if ready."})
        for it in items:
            if it["final_answer"] is None:
                # Fallback for logging/verifier: use last assistant text, but official
                # stop criterion remains answer[ready].
                last = ""
                for m in reversed(it["history"]):
                    if m.get("role") == "assistant":
                        last = m.get("content", "")
                        break
                it["final_answer"] = last
        return items

    def train_step_rlm_repl(self, step: int) -> Dict[str, Any]:
        t0 = time.time()
        if len(self.train_data) >= self.cfg.num_questions:
            sampled_indices = self.rng.sample(range(len(self.train_data)), self.cfg.num_questions)
        else:
            sampled_indices = [self.rng.randrange(len(self.train_data)) for _ in range(self.cfg.num_questions)]
        sampled_questions = [self.train_data[i] for i in sampled_indices]
        items = self._rlm_rollout_items(sampled_questions, sampled_indices, eval_mode=False)
        groups = [{"q_idx": i, "answer": self._get_answer(q), "records": [], "correct_in_group": 0, "target_samples": int(self.cfg.samples_per_question)} for i, q in enumerate(sampled_questions)]
        stats = {"total_samples": 0, "correct_samples": 0, "total_reward": 0.0, "total_length": 0, "total_trunc": 0, "total_repeat": 0, "no_answer": 0, "sum_correct_reward": 0.0, "sum_format_reward": 0.0, "sum_length_reward": 0.0, "sum_length_lambda": 0.0, "sum_zstd_penalty": 0.0, "sum_zstd_ratio": 0.0, "repl_calls": 0}
        for it in items:
            final_text = str(it.get("final_answer") or "")
            total_len = sum(len(seg["comp_tokens"]) for seg in it["segments"])
            repeat_flag = self._has_repeated_ngrams(final_text, n=16, repeat=5)
            reward, is_correct, is_format_correct, details = calculate_reward_details(
                text=final_text, ground_truth=it["answer"], token_length=total_len,
                min_tokens=self.cfg.min_tokens, max_tokens=self.cfg.max_tokens,
                length_weight=self.cfg.length_weight, repeat_ngram=repeat_flag,
                repeat_penalty=self.cfg.ngram_penalty, zstd_threshold=self.cfg.zstd_threshold,
                zstd_penalty_weight=self.cfg.zstd_penalty_weight,
                reward_mode=getattr(self.cfg, "reward_mode", "rwkv"),
            )
            comps = {"correct_reward": float(details.get("correct_reward", 0.0)), "format_reward": float(details.get("format_reward", 0.0)), "length_reward": float(details.get("length_reward", 0.0)), "repeat_penalty": float(details.get("repeat_penalty", 0.0)), "zstd_penalty": float(details.get("zstd_penalty", 0.0))}
            segs = []
            for seg in it["segments"]:
                if not seg["comp_tokens"]:
                    continue
                tr = {"prompt_tokens": seg["prompt_tokens"], "comp_tokens": seg["comp_tokens"], "old_logps": seg["old_logps"], "is_extra": False, "reward": reward, "text": final_text, "is_correct": is_correct, "is_format_correct": is_format_correct, "truncated": bool(it["truncated"]), "reward_components": comps}
                segs.append(tr)
            rec = {"reward_components": comps, "reward": reward, "is_correct": is_correct, "segments": segs}
            g = groups[int(it["q_idx"])]
            g["records"].append(rec)
            g["correct_in_group"] += int(bool(is_correct))
            stats["total_samples"] += 1; stats["correct_samples"] += int(bool(is_correct)); stats["total_reward"] += float(reward); stats["total_length"] += total_len; stats["total_trunc"] += int(bool(it["truncated"])); stats["total_repeat"] += int(bool(repeat_flag)); stats["no_answer"] += int(not details.get("extracted_answer")); stats["sum_correct_reward"] += float(details.get("correct_reward",0.0)); stats["sum_format_reward"] += float(details.get("format_reward",0.0)); stats["sum_length_reward"] += float(details.get("length_reward",0.0)); stats["sum_length_lambda"] += float(details.get("length_lambda",0.0)); stats["sum_zstd_penalty"] += float(details.get("zstd_penalty",0.0)); stats["sum_zstd_ratio"] += float(details.get("zstd_ratio",0.0)); stats["repl_calls"] += int(it.get("repl_calls",0))
            if getattr(self.cfg, "save_responses", True):
                append_jsonl(os.path.join(self.responses_dir, f"step_{step}.jsonl"), {"step": step, "question_idx": it["q_idx"], "sample_idx": it["sample_idx"], "problem": it["problem"], "ground_truth": it["answer"], "response": final_text, "history": it["history"], "reward": reward, "is_correct": is_correct, "truncated": bool(it["truncated"]), "repl_calls": it.get("repl_calls",0), "reward_details": details})
        keys = ("correct_reward", "format_reward", "length_reward", "repeat_penalty", "zstd_penalty")
        weights = {"correct_reward":1.0,"format_reward":0.0 if getattr(self.cfg,"reward_mode","rwkv")=="trl_doc" else 1.0,"length_reward":min(0.25,abs(float(self.cfg.length_weight))*0.5),"repeat_penalty":abs(float(self.cfg.ngram_penalty)),"zstd_penalty":abs(float(self.cfg.zstd_penalty_weight))}
        weight_norm=max(1e-6,sum(abs(w) for w in weights.values()))
        all_trajs=[]; adv_values=[]; groups_all_wrong=groups_all_correct=groups_used=groups_skipped=0
        for g in groups:
            c=int(g["correct_in_group"]); n=int(g["target_samples"])
            groups_all_wrong += int(c==0); groups_all_correct += int(c==n)
            if c==0 or c==n:
                groups_skipped += 1; continue
            groups_used += 1
            recs=g["records"]; comp_stats={}
            for k in keys:
                vals=[float(r["reward_components"].get(k,0.0)) for r in recs]
                mean=sum(vals)/max(1,len(vals)); var=sum((v-mean)**2 for v in vals)/max(1,len(vals)); comp_stats[k]=(mean, math.sqrt(var) if var>1e-6 else 0.0)
            for r in recs:
                adv=0.0
                for k in keys:
                    w=float(weights.get(k,0.0))
                    if w==0: continue
                    mean,std=comp_stats[k]; rel=0.0 if std<=1e-6 else (float(r["reward_components"].get(k,0.0))-mean)/std
                    adv += w*rel
                adv /= weight_norm; adv_values.append(adv)
                for seg in r["segments"]:
                    seg["advantage"] = adv; all_trajs.append(seg)
        cleanup_rollout=getattr(getattr(self.infer,"infer_model",None),"cleanup_stateful_rollout",None)
        if cleanup_rollout is not None: cleanup_rollout()
        opt=self._optimize_trajs(all_trajs, lr_scale=1.0, adv_clip=None)
        dt=time.time()-t0; ts=self._time_state_stats(); adv_mean=sum(adv_values)/len(adv_values) if adv_values else 0.0; adv_std=math.sqrt(sum((a-adv_mean)**2 for a in adv_values)/len(adv_values)) if adv_values else 0.0
        return {"step":step,"split":"train","samples":stats["total_samples"],"accuracy":stats["correct_samples"]/max(1,stats["total_samples"]),"avg_reward":stats["total_reward"]/max(1,stats["total_samples"]),"avg_length":stats["total_length"]/max(1,stats["total_samples"]),"trunc_rate":stats["total_trunc"]/max(1,stats["total_samples"]),"repeat_rate":stats["total_repeat"]/max(1,stats["total_samples"]),"no_answer_rate":stats["no_answer"]/max(1,stats["total_samples"]),"avg_correct_reward":stats["sum_correct_reward"]/max(1,stats["total_samples"]),"avg_format_reward":stats["sum_format_reward"]/max(1,stats["total_samples"]),"avg_length_reward":stats["sum_length_reward"]/max(1,stats["total_samples"]),"avg_length_lambda":stats["sum_length_lambda"]/max(1,stats["total_samples"]),"adv_mean":adv_mean,"adv_std":adv_std,"pos_adv_ratio":sum(1 for a in adv_values if a>0)/max(1,len(adv_values)),"neg_adv_ratio":sum(1 for a in adv_values if a<0)/max(1,len(adv_values)),"groups_total":len(groups),"groups_used":groups_used,"groups_skipped":groups_skipped,"groups_all_correct":groups_all_correct,"groups_all_wrong":groups_all_wrong,"loss":float(opt["loss_total"]),"kl":float(opt["kl_total"]),"avg_kl":float(opt["kl_total"])/max(1,int(opt["batch_cnt"])),"clip_frac":float(opt["clip_total"])/max(1,int(opt["clip_total_tokens"])),"grad_norm":float(opt["grad_norm"]),"time":dt,"samples_per_sec":stats["total_samples"]/dt if dt>0 else 0.0,"tokens_per_sec":stats["total_length"]/dt if dt>0 else 0.0,"ts_absmax":ts["absmax"],"ts_rms":ts["rms_avg"],"ts_bad":ts["bad"],"avg_entropy":float(opt["entropy_total"]),"avg_zstd_penalty":stats["sum_zstd_penalty"]/max(1,stats["total_samples"]),"avg_zstd_ratio":stats["sum_zstd_ratio"]/max(1,stats["total_samples"]),"hard_buffer_size":0,"hard_buffer_added":0,"hard_buffer_eligible":0,"hard_buffer_selected":0,"hard_buffer_triggered":0,"extra_step_ran":0,"extra_samples":0,"extra_groups_total":0,"extra_groups_used":0,"extra_groups_skipped":0,"extra_groups_all_correct":0,"extra_groups_all_wrong":0,"extra_loss":0.0,"extra_avg_kl":0.0,"extra_grad_norm":0.0,"extra_lr_scale":float(self.cfg.hard_buffer_extra_lr_scale),"extra_adv_clip":float(self.cfg.hard_buffer_adv_clip),"rlm_repl_calls":stats["repl_calls"]}

    def evaluate_rlm_repl(self, step: int, tag: str = "eval", sample_ratio: float = 1.0) -> Optional[float]:
        if not self.test_data: return None
        t0=time.time(); data=self.test_data
        if sample_ratio < 1.0:
            k=max(1,int(len(data)*sample_ratio)); idxs=self.rng.sample(range(len(data)),k); data=[data[i] for i in idxs]
        total=correct=total_len=total_trunc=repeat_ngram=format_correct=no_answer=repl_calls=0; sum_reward=sum_correct_reward=sum_format_reward=sum_length_reward=sum_length_lambda=sum_zstd_penalty=sum_zstd_ratio=0.0
        for start in range(0,len(data),32):
            ex_list=data[start:start+32]; idxs=list(range(start,start+len(ex_list))); items=self._rlm_rollout_items(ex_list,idxs,eval_mode=True)
            for it in items:
                txt=str(it.get("final_answer") or ""); total_tokens=sum(len(seg["comp_tokens"]) for seg in it["segments"]); repeat=self._has_repeated_ngrams(txt,n=16,repeat=5)
                reward,is_correct,is_format,details=calculate_reward_details(text=txt,ground_truth=it["answer"],token_length=total_tokens,min_tokens=self.cfg.min_tokens,max_tokens=self.cfg.max_tokens,length_weight=self.cfg.length_weight,repeat_ngram=repeat,repeat_penalty=self.cfg.ngram_penalty,zstd_threshold=self.cfg.zstd_threshold,zstd_penalty_weight=self.cfg.zstd_penalty_weight,reward_mode=getattr(self.cfg,"reward_mode","rwkv"))
                append_jsonl(self.eval_path,{"step":step,"tag":tag,"problem":it["problem"],"ground_truth":it["answer"],"response":txt,"history":it["history"],"reward":reward,"is_correct":is_correct,"is_format_correct":is_format,"truncated":bool(it["truncated"]),"gen_len":total_tokens,"reward_details":details,"repl_calls":it.get("repl_calls",0),"eval_temperature":self.cfg.eval_temperature,"eval_top_p":self.cfg.eval_top_p,"eval_top_k":self.cfg.eval_top_k,"eval_max_new_tokens":getattr(self.cfg,"eval_max_new_tokens",self.cfg.max_new_tokens),"repeat_16gram_5":repeat})
                total+=1; correct+=int(bool(is_correct)); format_correct+=int(bool(is_format)); total_len+=total_tokens; total_trunc+=int(bool(it["truncated"])); repeat_ngram+=int(bool(repeat)); no_answer+=int(not details.get("extracted_answer")); repl_calls+=int(it.get("repl_calls",0)); sum_reward+=float(reward); sum_correct_reward+=float(details.get("correct_reward",0.0)); sum_format_reward+=float(details.get("format_reward",0.0)); sum_length_reward+=float(details.get("length_reward",0.0)); sum_length_lambda+=float(details.get("length_lambda",0.0)); sum_zstd_penalty+=float(details.get("zstd_penalty",0.0)); sum_zstd_ratio+=float(details.get("zstd_ratio",0.0))
        acc=correct/max(1,total); avg_len=total_len/max(1,total); trunc_rate=total_trunc/max(1,total); rep=repeat_ngram/max(1,total); fmt=format_correct/max(1,total); noans=no_answer/max(1,total); et=time.time()-t0
        append_jsonl(self.metrics_path,{"step":step,"accuracy":acc,"avg_length":avg_len,"trunc_rate":trunc_rate,"split":tag,"avg_reward":sum_reward/max(1,total),"trunc_wrong_rate":0.0,"repeat_16gram_rate":rep,"repeat_rate":rep,"format_rate":fmt,"no_answer_rate":noans,"avg_correct_reward":sum_correct_reward/max(1,total),"avg_format_reward":sum_format_reward/max(1,total),"avg_length_reward":sum_length_reward/max(1,total),"avg_length_lambda":sum_length_lambda/max(1,total),"eval_count":total,"eval_time":et,"avg_zstd_penalty":sum_zstd_penalty/max(1,total),"avg_zstd_ratio":sum_zstd_ratio/max(1,total),"preeval_acc":None,"eval_acc_delta":None,"preeval_count":0,"rlm_repl_calls":repl_calls})
        self._log(f"[EVAL step {step}] acc={acc:.3f} trunc={trunc_rate:.3f} repeat16@5={rep:.3f} fmt={fmt:.3f} no_ans={noans:.3f} avg_len={avg_len:.1f} repl_calls={repl_calls/max(1,total):.2f} time={et:.1f}s (rlm_repl)")
        return acc

    def _recursive_token_budgets(self, total_tokens: int):
        # In the recursive environment, plan/solve are bounded tool calls. They
        # should be short scaffolds, while the finalizer owns the answer budget.
        total_tokens = max(384, int(total_tokens))
        plan_tokens = min(64, max(32, total_tokens // 12))
        solve_tokens = min(128, max(64, total_tokens // 6))
        final_tokens = min(320, max(160, total_tokens - plan_tokens - solve_tokens))
        return plan_tokens, solve_tokens, final_tokens

    def _encode_trim_for_gen(self, text: str, gen_tokens: int):
        ids = self.encode(text)
        max_prompt_len = int(self.model.args.ctx_len) - int(gen_tokens) - 4
        max_prompt_len = max(64, max_prompt_len)
        if len(ids) > max_prompt_len:
            ids = ids[-max_prompt_len:]
        return ids

    def _recursive_prompt(self, problem: str, stage: str, plan: str = "", memo: str = "") -> str:
        problem = (problem or "").strip()
        if stage == "plan":
            return (
                f"User: {problem}\n"
                "Make a very short recursive decomposition. Do not solve. "
                "Output at most 3 subproblems and an order. End with <END_PLAN>.\n"
                "Assistant: <think>\n"
            )
        if stage == "solve":
            return (
                f"User: Original problem:\n{problem}\n\n"
                f"Plan:\n{plan}\n\n"
                "Solve the subproblems only. Write a compact memo of useful intermediate results. "
                "End with <END_MEMO>.\n"
                "Assistant: <think>\n"
            )
        return (
            f"User: Original problem:\n{problem}\n\n"
            f"Subproblem memo:\n{memo}\n\n"
            "Use the memo if helpful. Output exactly one final answer in the form \\boxed{...}. "
            "No long explanation. Stop after the box.\n"
            "Assistant: <think>\n"
        )

    def _generate_recursive_samples(self, sampled_questions: List[Dict[str, Any]], sampled_indices: List[int], eval_mode: bool = False):
        group_size = 1 if eval_mode else int(self.cfg.samples_per_question)
        total_budget = int(getattr(self.cfg, "eval_max_new_tokens", self.cfg.max_new_tokens) if eval_mode else self.cfg.max_new_tokens)
        plan_max, solve_max, final_max = self._recursive_token_budgets(total_budget)
        temperature = float(self.cfg.eval_temperature if eval_mode else self.cfg.temperature)
        top_p = float(self.cfg.eval_top_p if eval_mode else self.cfg.top_p)
        top_k = int(self.cfg.eval_top_k if eval_mode else self.cfg.top_k)

        items = []
        plan_prompts = []
        for q_idx, (question, train_idx) in enumerate(zip(sampled_questions, sampled_indices)):
            problem = question.get("problem", "")
            answer = self._get_answer(question)
            for sample_idx in range(group_size):
                prompt = self._recursive_prompt(problem, "plan")
                items.append({
                    "q_idx": q_idx,
                    "train_idx": int(train_idx),
                    "sample_idx": sample_idx,
                    "problem": problem,
                    "answer": answer,
                    "plan_prompt_tokens": self._encode_trim_for_gen(prompt, plan_max),
                })
                plan_prompts.append(items[-1]["plan_prompt_tokens"])

        plan_tokens, plan_logps, plan_texts, plan_trunc = self.infer.generate_group_parallel(
            prompt_tokens_list=plan_prompts,
            group_size=1,
            max_new_tokens=plan_max,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop_on_think_close=False,
            stop_on_user=True,
            stop_on_boxed=False,
            stop_on_repeat_ngram=False,
            stop_strings=["<END_PLAN>"],
            presence_penalty=0.0,
            frequency_penalty=0.0,
            alpha_decay=1.0,
        )

        solve_prompts = []
        for i, it in enumerate(items):
            it["plan_tokens"] = plan_tokens[i]
            it["plan_old_logps"] = plan_logps[i]
            it["plan_text"] = plan_texts[i]
            it["plan_truncated"] = bool(plan_trunc[i])
            prompt = self._recursive_prompt(it["problem"], "solve", plan=it["plan_text"])
            it["solve_prompt_tokens"] = self._encode_trim_for_gen(prompt, solve_max)
            solve_prompts.append(it["solve_prompt_tokens"])

        solve_tokens, solve_logps, solve_texts, solve_trunc = self.infer.generate_group_parallel(
            prompt_tokens_list=solve_prompts,
            group_size=1,
            max_new_tokens=solve_max,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop_on_think_close=False,
            stop_on_user=True,
            stop_on_boxed=False,
            stop_on_repeat_ngram=False,
            stop_strings=["<END_MEMO>"],
            presence_penalty=0.0,
            frequency_penalty=0.0,
            alpha_decay=1.0,
        )

        final_prompts = []
        for i, it in enumerate(items):
            it["solve_tokens"] = solve_tokens[i]
            it["solve_old_logps"] = solve_logps[i]
            it["solve_text"] = solve_texts[i]
            it["solve_truncated"] = bool(solve_trunc[i])
            prompt = self._recursive_prompt(it["problem"], "final", plan=it["plan_text"], memo=it["solve_text"])
            it["final_prompt_tokens"] = self._encode_trim_for_gen(prompt, final_max)
            final_prompts.append(it["final_prompt_tokens"])

        final_tokens, final_logps, final_texts, final_trunc = self.infer.generate_group_parallel(
            prompt_tokens_list=final_prompts,
            group_size=1,
            max_new_tokens=final_max,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop_on_think_close=False,
            stop_on_user=True,
            stop_on_boxed=True,
            stop_on_repeat_ngram=False,
            presence_penalty=0.0,
            frequency_penalty=0.0,
            alpha_decay=1.0,
        )

        for i, it in enumerate(items):
            it["final_tokens"] = final_tokens[i]
            it["final_old_logps"] = final_logps[i]
            it["final_text"] = final_texts[i]
            it["final_truncated"] = bool(final_trunc[i])
            it["combined_text"] = (
                "[FINAL]\n" + it["final_text"] +
                "\n[PLAN]\n" + it["plan_text"] +
                "\n[SUBPROBLEM_MEMO]\n" + it["solve_text"]
            )
        return items

    def train_step_recursive(self, step: int) -> Dict[str, Any]:
        t0 = time.time()
        if len(self.train_data) >= self.cfg.num_questions:
            sampled_indices = self.rng.sample(range(len(self.train_data)), self.cfg.num_questions)
        else:
            sampled_indices = [self.rng.randrange(len(self.train_data)) for _ in range(self.cfg.num_questions)]
        sampled_questions = [self.train_data[i] for i in sampled_indices]
        items = self._generate_recursive_samples(sampled_questions, sampled_indices, eval_mode=False)

        stats = {
            "total_samples": 0, "correct_samples": 0, "total_reward": 0.0, "total_length": 0,
            "total_trunc": 0, "total_repeat": 0, "no_answer": 0,
            "sum_correct_reward": 0.0, "sum_format_reward": 0.0, "sum_length_reward": 0.0,
            "sum_length_lambda": 0.0, "sum_zstd_penalty": 0.0, "sum_zstd_ratio": 0.0,
        }
        groups = []
        for q_idx, (question, train_idx) in enumerate(zip(sampled_questions, sampled_indices)):
            groups.append({
                "q_idx": q_idx, "train_idx": int(train_idx), "problem": question.get("problem", ""),
                "answer": self._get_answer(question), "sample_records": [], "correct_in_group": 0,
                "target_samples": int(self.cfg.samples_per_question),
            })

        for it in items:
            group = groups[int(it["q_idx"])]
            total_len = len(it["plan_tokens"]) + len(it["solve_tokens"]) + len(it["final_tokens"])
            truncated = bool(it["final_truncated"])
            repeat_flag = self._has_repeated_ngrams(it["combined_text"], n=16, repeat=5)
            reward, is_correct, is_format_correct, reward_details = calculate_reward_details(
                text=it["combined_text"],
                ground_truth=group["answer"],
                token_length=total_len,
                min_tokens=self.cfg.min_tokens,
                max_tokens=self.cfg.max_tokens,
                length_weight=self.cfg.length_weight,
                repeat_ngram=repeat_flag,
                repeat_penalty=self.cfg.ngram_penalty,
                zstd_threshold=self.cfg.zstd_threshold,
                zstd_penalty_weight=self.cfg.zstd_penalty_weight,
                reward_mode=getattr(self.cfg, "reward_mode", "rwkv"),
            )
            reward_components = {
                "correct_reward": float(reward_details.get("correct_reward", 0.0)),
                "format_reward": float(reward_details.get("format_reward", 0.0)),
                "length_reward": float(reward_details.get("length_reward", 0.0)),
                "repeat_penalty": float(reward_details.get("repeat_penalty", 0.0)),
                "zstd_penalty": float(reward_details.get("zstd_penalty", 0.0)),
            }
            segs = [
                {"stage": "plan", "prompt_tokens": it["plan_prompt_tokens"], "comp_tokens": it["plan_tokens"], "old_logps": it["plan_old_logps"]},
                {"stage": "solve", "prompt_tokens": it["solve_prompt_tokens"], "comp_tokens": it["solve_tokens"], "old_logps": it["solve_old_logps"]},
                {"stage": "final", "prompt_tokens": it["final_prompt_tokens"], "comp_tokens": it["final_tokens"], "old_logps": it["final_old_logps"]},
            ]
            for seg in segs:
                seg.update({
                    "is_extra": False, "reward": reward, "text": it["combined_text"],
                    "is_correct": is_correct, "is_format_correct": is_format_correct,
                    "truncated": truncated, "reward_components": reward_components,
                    "sample_id": int(it["sample_idx"]),
                })
            rec = {"reward_components": reward_components, "reward": reward, "is_correct": is_correct, "segments": segs}
            group["sample_records"].append(rec)
            if is_correct:
                group["correct_in_group"] += 1
            stats["total_samples"] += 1
            stats["total_reward"] += float(reward)
            stats["total_length"] += int(total_len)
            stats["correct_samples"] += int(bool(is_correct))
            stats["total_trunc"] += int(bool(truncated))
            stats["total_repeat"] += int(bool(repeat_flag))
            stats["no_answer"] += int(not reward_details.get("extracted_answer"))
            stats["sum_correct_reward"] += float(reward_details.get("correct_reward", 0.0))
            stats["sum_format_reward"] += float(reward_details.get("format_reward", 0.0))
            stats["sum_length_reward"] += float(reward_details.get("length_reward", 0.0))
            stats["sum_length_lambda"] += float(reward_details.get("length_lambda", 0.0))
            stats["sum_zstd_penalty"] += float(reward_details.get("zstd_penalty", 0.0))
            stats["sum_zstd_ratio"] += float(reward_details.get("zstd_ratio", 0.0))
            if getattr(self.cfg, "save_responses", True):
                append_jsonl(os.path.join(self.responses_dir, f"step_{step}.jsonl"), {
                    "step": step, "question_idx": group["q_idx"], "sample_idx": it["sample_idx"],
                    "problem": group["problem"], "ground_truth": group["answer"],
                    "response": it["combined_text"], "pred_extracted": reward_details.get("extracted_answer"),
                    "gt_extracted": reward_details.get("ground_truth_answer"), "reward": reward,
                    "is_correct": is_correct, "is_format_correct": is_format_correct,
                    "truncated": truncated, "reward_details": reward_details,
                })

        adv_component_keys = ("correct_reward", "format_reward", "length_reward", "repeat_penalty", "zstd_penalty")
        adv_component_weights = {
            "correct_reward": 1.0,
            "format_reward": 0.0 if getattr(self.cfg, "reward_mode", "rwkv") == "trl_doc" else 1.0,
            "length_reward": min(0.25, abs(float(self.cfg.length_weight)) * 0.5),
            "repeat_penalty": abs(float(self.cfg.ngram_penalty)),
            "zstd_penalty": abs(float(self.cfg.zstd_penalty_weight)),
        }
        weight_norm = max(1e-6, sum(abs(w) for w in adv_component_weights.values()))
        all_trajs, adv_values = [], []
        groups_total = len(groups)
        groups_all_correct = groups_all_wrong = groups_used = groups_skipped = 0
        for group in groups:
            c = int(group["correct_in_group"])
            n = int(group["target_samples"])
            if c == 0:
                groups_all_wrong += 1
            elif c == n:
                groups_all_correct += 1
            if c == 0 or c == n:
                groups_skipped += 1
                continue
            groups_used += 1
            recs = group["sample_records"]
            comp_stats = {}
            for key in adv_component_keys:
                vals = [float(r["reward_components"].get(key, 0.0)) for r in recs]
                mean_v = sum(vals) / max(1, len(vals))
                var_v = sum((v - mean_v) ** 2 for v in vals) / max(1, len(vals))
                comp_stats[key] = (mean_v, math.sqrt(var_v) if var_v > 1e-6 else 0.0)
            for rec in recs:
                decoupled_adv = 0.0
                for key in adv_component_keys:
                    w = float(adv_component_weights.get(key, 0.0))
                    if w == 0.0:
                        continue
                    mean_v, std_v = comp_stats.get(key, (0.0, 0.0))
                    rel_adv = 0.0 if std_v <= 1e-6 else (float(rec["reward_components"].get(key, 0.0)) - mean_v) / std_v
                    decoupled_adv += w * rel_adv
                adv = decoupled_adv / weight_norm
                adv_values.append(adv)
                for seg in rec["segments"]:
                    seg["advantage"] = adv
                    all_trajs.append(seg)

        infer_model = getattr(self.infer, "infer_model", None)
        cleanup_rollout = getattr(infer_model, "cleanup_stateful_rollout", None)
        if cleanup_rollout is not None:
            cleanup_rollout()
        opt = self._optimize_trajs(all_trajs, lr_scale=1.0, adv_clip=None)

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
        avg_zstd_penalty = stats["sum_zstd_penalty"] / max(1, stats["total_samples"])
        avg_zstd_ratio = stats["sum_zstd_ratio"] / max(1, stats["total_samples"])
        if adv_values:
            adv_mean = sum(adv_values) / len(adv_values)
            adv_std = math.sqrt(sum((a - adv_mean) ** 2 for a in adv_values) / len(adv_values))
            pos_adv_ratio = sum(1 for a in adv_values if a > 0) / len(adv_values)
            neg_adv_ratio = sum(1 for a in adv_values if a < 0) / len(adv_values)
        else:
            adv_mean = adv_std = pos_adv_ratio = neg_adv_ratio = 0.0
        batch_cnt = int(opt["batch_cnt"])
        avg_kl = float(opt["kl_total"]) / max(1, batch_cnt)
        clip_frac = float(opt["clip_total"]) / max(1, int(opt["clip_total_tokens"]))
        ts_stats = self._time_state_stats()
        return {
            "step": step, "split": "train", "samples": stats["total_samples"], "accuracy": accuracy,
            "avg_reward": avg_reward, "avg_length": avg_length, "trunc_rate": trunc_rate,
            "repeat_rate": repeat_rate, "no_answer_rate": no_answer_rate,
            "avg_correct_reward": avg_correct_reward, "avg_format_reward": avg_format_reward,
            "avg_length_reward": avg_length_reward, "avg_length_lambda": avg_length_lambda,
            "adv_mean": adv_mean, "adv_std": adv_std, "pos_adv_ratio": pos_adv_ratio, "neg_adv_ratio": neg_adv_ratio,
            "groups_total": groups_total, "groups_used": groups_used, "groups_skipped": groups_skipped,
            "groups_all_correct": groups_all_correct, "groups_all_wrong": groups_all_wrong,
            "loss": float(opt["loss_total"]), "kl": float(opt["kl_total"]), "avg_kl": avg_kl,
            "clip_frac": clip_frac, "grad_norm": float(opt["grad_norm"]), "time": dt,
            "samples_per_sec": stats["total_samples"] / dt if dt > 0 else 0.0,
            "tokens_per_sec": stats["total_length"] / dt if dt > 0 else 0.0,
            "ts_absmax": ts_stats["absmax"], "ts_rms": ts_stats["rms_avg"], "ts_bad": ts_stats["bad"],
            "avg_entropy": float(opt["entropy_total"]), "avg_zstd_penalty": avg_zstd_penalty,
            "avg_zstd_ratio": avg_zstd_ratio, "hard_buffer_size": 0, "hard_buffer_added": 0,
            "hard_buffer_eligible": 0, "hard_buffer_selected": 0, "hard_buffer_triggered": 0,
            "extra_step_ran": 0, "extra_samples": 0, "extra_groups_total": 0, "extra_groups_used": 0,
            "extra_groups_skipped": 0, "extra_groups_all_correct": 0, "extra_groups_all_wrong": 0,
            "extra_loss": 0.0, "extra_avg_kl": 0.0, "extra_grad_norm": 0.0,
            "extra_lr_scale": float(self.cfg.hard_buffer_extra_lr_scale), "extra_adv_clip": float(self.cfg.hard_buffer_adv_clip),
        }

    def evaluate_recursive(self, step: int, tag: str = "eval", sample_ratio: float = 1.0) -> Optional[float]:
        if not self.test_data:
            return None
        t0 = time.time()
        data = self.test_data
        if sample_ratio < 1.0:
            k = max(1, int(len(data) * sample_ratio))
            idxs = self.rng.sample(range(len(data)), k)
            data = [data[i] for i in idxs]
        total = correct = total_len = total_trunc = trunc_wrong = repeat_ngram = format_correct = no_answer = 0
        sum_reward = sum_correct_reward = sum_format_reward = sum_length_reward = sum_length_lambda = 0.0
        sum_zstd_penalty = sum_zstd_ratio = 0.0
        chunk_size = 64
        for start in range(0, len(data), chunk_size):
            ex_list = data[start:start + chunk_size]
            idxs = list(range(start, start + len(ex_list)))
            items = self._generate_recursive_samples(ex_list, idxs, eval_mode=True)
            for it in items:
                answer = it["answer"]
                total_tokens = len(it["plan_tokens"]) + len(it["solve_tokens"]) + len(it["final_tokens"])
                truncated = bool(it["final_truncated"])
                repeat_flag = self._has_repeated_ngrams(it["combined_text"], n=16, repeat=5)
                reward, is_correct, is_format_correct, reward_details = calculate_reward_details(
                    text=it["combined_text"], ground_truth=answer, token_length=total_tokens,
                    min_tokens=self.cfg.min_tokens, max_tokens=self.cfg.max_tokens,
                    length_weight=self.cfg.length_weight, repeat_ngram=repeat_flag,
                    repeat_penalty=self.cfg.ngram_penalty, zstd_threshold=self.cfg.zstd_threshold,
                    zstd_penalty_weight=self.cfg.zstd_penalty_weight,
                    reward_mode=getattr(self.cfg, "reward_mode", "rwkv"),
                )
                record = {
                    "step": step, "tag": tag, "problem": it["problem"], "ground_truth": answer,
                    "response": it["combined_text"], "pred_extracted": reward_details.get("extracted_answer"),
                    "gt_extracted": reward_details.get("ground_truth_answer"), "reward": reward,
                    "is_correct": is_correct, "is_format_correct": is_format_correct,
                    "truncated": bool(truncated), "gen_len": total_tokens, "reward_details": reward_details,
                    "eval_temperature": self.cfg.eval_temperature, "eval_top_p": self.cfg.eval_top_p,
                    "eval_top_k": self.cfg.eval_top_k,
                    "eval_max_new_tokens": getattr(self.cfg, "eval_max_new_tokens", self.cfg.max_new_tokens),
                    "repeat_16gram_5": repeat_flag, "zstd_ratio": reward_details.get("zstd_ratio", 0.0),
                    "zstd_penalty": reward_details.get("zstd_penalty", 0.0),
                }
                append_jsonl(self.eval_path, record)
                append_jsonl(os.path.join(self.eval_by_step_dir, "%s_step_%s.jsonl" % (tag, step)), record)
                total += 1
                correct += int(bool(is_correct))
                format_correct += int(bool(is_format_correct))
                total_len += total_tokens
                total_trunc += int(bool(truncated))
                trunc_wrong += int(bool(truncated) and not bool(is_correct))
                repeat_ngram += int(bool(repeat_flag))
                no_answer += int(not reward_details.get("extracted_answer"))
                sum_reward += float(reward)
                sum_correct_reward += float(reward_details.get("correct_reward", 0.0))
                sum_format_reward += float(reward_details.get("format_reward", 0.0))
                sum_length_reward += float(reward_details.get("length_reward", 0.0))
                sum_length_lambda += float(reward_details.get("length_lambda", 0.0))
                sum_zstd_penalty += float(reward_details.get("zstd_penalty", 0.0))
                sum_zstd_ratio += float(reward_details.get("zstd_ratio", 0.0))
        acc = correct / max(1, total)
        avg_len = total_len / max(1, total)
        trunc_rate = total_trunc / max(1, total)
        trunc_wrong_rate = trunc_wrong / max(1, total)
        repeat_ngram_rate = repeat_ngram / max(1, total)
        format_rate = format_correct / max(1, total)
        no_answer_rate = no_answer / max(1, total)
        eval_time = time.time() - t0
        append_jsonl(self.metrics_path, {
            "step": step, "accuracy": acc, "avg_length": avg_len, "trunc_rate": trunc_rate,
            "split": tag, "avg_reward": sum_reward / max(1, total),
            "trunc_wrong_rate": trunc_wrong_rate, "repeat_16gram_rate": repeat_ngram_rate,
            "repeat_rate": repeat_ngram_rate, "format_rate": format_rate, "no_answer_rate": no_answer_rate,
            "avg_correct_reward": sum_correct_reward / max(1, total),
            "avg_format_reward": sum_format_reward / max(1, total),
            "avg_length_reward": sum_length_reward / max(1, total),
            "avg_length_lambda": sum_length_lambda / max(1, total), "eval_count": total,
            "eval_time": eval_time, "avg_zstd_penalty": sum_zstd_penalty / max(1, total),
            "avg_zstd_ratio": sum_zstd_ratio / max(1, total), "preeval_acc": None,
            "eval_acc_delta": None, "preeval_count": 0,
        })
        self._log(
            f"[EVAL step {step}] acc={acc:.3f} trunc={trunc_rate:.3f} trunc_wrong={trunc_wrong_rate:.3f} "
            f"repeat16@5={repeat_ngram_rate:.3f} fmt={format_rate:.3f} no_ans={no_answer_rate:.3f} "
            f"corr_r={sum_correct_reward / max(1, total):.4f} fmt_r={sum_format_reward / max(1, total):.4f} "
            f"avg_len={avg_len:.1f} avg_r={sum_reward / max(1, total):.3f} time={eval_time:.1f}s "
            f"(recursive, temp={self.cfg.eval_temperature}, top_p={self.cfg.eval_top_p}, top_k={self.cfg.eval_top_k})"
        )
        return acc

    def train(self, total_steps: int):
        self._log(f"开始训练: total_steps={total_steps}")
        self._log(f"配置: num_questions={self.cfg.num_questions}, "
                 f"samples_per_question={self.cfg.samples_per_question}")
        train_start = time.time()
        
        for step in range(1, total_steps + 1):
            metrics = self.train_step(step)
            metrics["elapsed"] = time.time() - train_start
            
            append_jsonl(self.metrics_path, metrics)
            
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
            
            should_save = ((self.cfg.save_interval > 0) and (step % self.cfg.save_interval == 0)) or (bool(self.cfg.save_last) and step == total_steps)
            if should_save:
                ckpt_path = os.path.join(self.out_dir, f"ckpt_step{step}.pth")
                ckpt_payload = {"step": step}
                if self.cfg.tune_mode == "full":
                    ckpt_payload["model"] = {
                        n: p.detach().cpu()
                        for n, p in self.model.named_parameters()
                        if p.requires_grad
                    }
                else:
                    ckpt_payload["time_state"] = {
                        n: p.detach().cpu()
                        for n, p in self.model.named_parameters()
                        if "time_state" in n
                    }
                    ckpt_payload["optimizer"] = self.opt.state_dict()
                torch.save(ckpt_payload, ckpt_path)
                self._log(f"保存检查点: {ckpt_path}")

            full_eval = ((self.cfg.save_interval > 0) and (step % self.cfg.save_interval == 0)) or (bool(self.cfg.final_full_eval) and step == total_steps)
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




