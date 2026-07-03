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
from typing import List, Dict, Any, Optional, Tuple
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
    rollout_forward_batch: int = 8
    lr: float = 1e-5
    grad_clip: float = 1.0
    
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
    reward_mode: str = "rwkv"
    prompt_mode: str = "rwkv_boxed"
    save_responses: bool = True
    opd_enabled: bool = False
    opd_weight: float = 0.0
    opd_max_new_tokens: int = 512
    opd_temperature: float = 0.7
    opd_top_p: float = 0.9
    opd_micro_batch: int = 2


def compute_unbiased_kl(ref_logp: torch.Tensor, policy_logp: torch.Tensor) -> torch.Tensor:
    log_ratio = ref_logp - policy_logp
    return torch.exp(log_ratio) - log_ratio - 1.0


def _sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


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
        teacher_model=None,
        teacher_tokenizer=None,
        teacher_kind: str = "hf",
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
        self.teacher_model = teacher_model
        self.teacher_tokenizer = teacher_tokenizer
        self.teacher_kind = str(teacher_kind or "hf").lower()
        
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
        self.opt = torch.optim.Adam(params, lr=self.cfg.lr, betas=(0.9, 0.99), eps=1e-8)
        
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

    def _teacher_ready(self) -> bool:
        if (not self.cfg.opd_enabled) or self.teacher_model is None:
            return False
        if self.teacher_kind == "rwkv":
            return True
        return self.teacher_tokenizer is not None

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
                    stop_on_think_close=False,
                    stop_on_user=True,
                    stop_on_boxed=False,
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
    
    def _opd_generate_teacher_targets(self, problems: List[str]) -> List[Tuple[str, str]]:
        if not self._teacher_ready():
            return []
        self.teacher_model.eval()
        prompt_texts = [build_prompt(p, mode=getattr(self.cfg, "prompt_mode", "rwkv_boxed")) for p in problems]
        toks = self.teacher_tokenizer(prompt_texts, return_tensors='pt', padding=True, truncation=True)
        toks = {k: v.to(self.device) for k, v in toks.items()}
        with torch.no_grad():
            out = self.teacher_model.generate(
                **toks,
                max_new_tokens=int(self.cfg.opd_max_new_tokens),
                do_sample=True,
                temperature=float(self.cfg.opd_temperature),
                top_p=float(self.cfg.opd_top_p),
                pad_token_id=self.teacher_tokenizer.eos_token_id,
                eos_token_id=self.teacher_tokenizer.eos_token_id,
            )
        input_lens = toks['attention_mask'].sum(dim=1).tolist()
        pairs = []
        for i, prompt in enumerate(prompt_texts):
            gen_ids = out[i, int(input_lens[i]):].tolist()
            gen_text = self.teacher_tokenizer.decode(gen_ids, skip_special_tokens=True)
            pairs.append((prompt, gen_text))
        return pairs

    def _optimize_opd_sft(self, teacher_pairs: List[Tuple[str, str]]) -> Dict[str, float]:
        if not teacher_pairs:
            return {'opd_loss': 0.0, 'opd_batches': 0}
        self.model.train()
        total_loss = 0.0
        total_batches = 0
        self.opt.zero_grad(set_to_none=True)
        batch_size = max(1, int(self.cfg.opd_micro_batch))
        for start in range(0, len(teacher_pairs), batch_size):
            batch = teacher_pairs[start:start + batch_size]
            seqs = []
            loss_masks = []
            for prompt_text, target_text in batch:
                prompt_ids = self.encode(prompt_text)
                target_ids = self.encode(target_text)
                seq = prompt_ids + target_ids
                seqs.append(seq)
                loss_masks.append([0] * len(prompt_ids) + [1] * len(target_ids))
            seqs, _ = self._pad_batch(seqs, pad_id=0)
            max_len = seqs.size(1)
            padded_masks = []
            for m in loss_masks:
                padded_masks.append(m + [0] * (max_len - len(m)))
            loss_mask = torch.tensor(padded_masks, device=self.device, dtype=torch.float32)
            inp = seqs[:, :-1].contiguous()
            tgt = seqs[:, 1:].contiguous()
            tok_mask = loss_mask[:, 1:].contiguous()
            logits = self.model(inp)
            if torch.is_tensor(logits) and logits.dim() == 2:
                logits = logits.unsqueeze(0)
            ce = F.cross_entropy(logits.transpose(1, 2).float(), tgt, reduction='none')
            denom = tok_mask.sum().clamp_min(1.0)
            loss = (ce * tok_mask).sum() / denom
            loss = loss * float(self.cfg.opd_weight)
            loss.backward()
            total_loss += float(loss.item())
            total_batches += 1
        if self.cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad], self.cfg.grad_clip)
        self.opt.step()
        infer_model = getattr(self.infer, 'infer_model', None)
        mark_dirty = getattr(infer_model, 'mark_dirty', None)
        if mark_dirty is not None:
            mark_dirty()
        return {'opd_loss': total_loss, 'opd_batches': total_batches}

    @torch.no_grad()
    def _opd_score_rollouts(self, prompt_response_pairs: List[Tuple[str, str]]) -> List[Dict[str, float]]:
        if not self._teacher_ready():
            return []
        self.teacher_model.eval()
        results: List[Dict[str, float]] = []
        batch_size = max(1, int(self.cfg.opd_micro_batch))
        pad_id = 0
        for start in range(0, len(prompt_response_pairs), batch_size):
            batch = prompt_response_pairs[start:start + batch_size]
            seqs = []
            labels = []
            for prompt_text, response_text in batch:
                if self.teacher_kind == "rwkv":
                    prompt_ids = self.encode(prompt_text)
                    response_ids = self.encode(response_text)
                else:
                    prompt_ids = self.teacher_tokenizer.encode(prompt_text, add_special_tokens=False)
                    response_ids = self.teacher_tokenizer.encode(response_text, add_special_tokens=False)
                    pad_id = self.teacher_tokenizer.pad_token_id
                    if pad_id is None:
                        pad_id = self.teacher_tokenizer.eos_token_id
                if int(self.cfg.opd_max_new_tokens) > 0:
                    response_ids = response_ids[: int(self.cfg.opd_max_new_tokens)]
                seq = prompt_ids + response_ids
                lab = [-100] * len(prompt_ids) + response_ids
                if len(seq) < 2 or len(response_ids) == 0:
                    seq = [pad_id, pad_id]
                    lab = [-100, -100]
                seqs.append(seq)
                labels.append(lab)
            max_len = max(len(s) for s in seqs)
            input_ids = []
            label_ids = []
            for seq, lab in zip(seqs, labels):
                pad_len = max_len - len(seq)
                input_ids.append(seq + [pad_id] * pad_len)
                label_ids.append(lab + [-100] * pad_len)
            input_ids = torch.tensor(input_ids, device=self.device, dtype=torch.long)
            label_ids = torch.tensor(label_ids, device=self.device, dtype=torch.long)
            if self.teacher_kind == "rwkv":
                logits = self.teacher_model(input_ids)
            else:
                attention_mask = (input_ids != pad_id).long()
                logits = self.teacher_model(input_ids=input_ids, attention_mask=attention_mask).logits
            if torch.is_tensor(logits) and logits.dim() == 2:
                logits = logits.unsqueeze(0)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = label_ids[:, 1:].contiguous()
            ce = F.cross_entropy(shift_logits.transpose(1, 2).float(), shift_labels, reduction='none', ignore_index=-100)
            mask = (shift_labels != -100).float()
            seq_sum = -(ce * mask).sum(dim=1)
            seq_cnt = mask.sum(dim=1)
            seq_mean = seq_sum / seq_cnt.clamp_min(1.0)
            for ssum, scnt, smean in zip(seq_sum.tolist(), seq_cnt.tolist(), seq_mean.tolist()):
                results.append({
                    'teacher_logp_sum': float(ssum),
                    'teacher_token_count': float(scnt),
                    'teacher_logp_mean': float(smean if scnt > 0 else 0.0),
                })
        return results

    def _assign_opd_advantages(self, group_infos: List[Dict[str, Any]]) -> Dict[str, float]:
        if not self._teacher_ready():
            return {'opd_pairs': 0, 'opd_score_mean': 0.0, 'opd_score_std': 0.0}
        pairs: List[Tuple[str, str]] = []
        traj_refs = []
        for group in group_infos:
            if bool(group.get('is_extra', False)):
                continue
            prompt_text = build_prompt(group.get('problem', ''), mode=getattr(self.cfg, 'prompt_mode', 'rwkv_boxed'))
            for traj in group.get('group_trajs', []):
                if bool(traj.get('is_extra', False)):
                    continue
                pairs.append((prompt_text, str(traj.get('text', ''))))
                traj_refs.append((group, traj))
        if not pairs:
            return {'opd_pairs': 0, 'opd_score_mean': 0.0, 'opd_score_std': 0.0}
        score_infos = self._opd_score_rollouts(pairs)
        for (_, traj), score in zip(traj_refs, score_infos):
            traj['teacher_logp_mean'] = float(score.get('teacher_logp_mean', 0.0))
            traj['teacher_logp_sum'] = float(score.get('teacher_logp_sum', 0.0))
            traj['opd_advantage'] = 0.0
        all_scores = [float(s.get('teacher_logp_mean', 0.0)) for s in score_infos]
        offset = 0
        for group in group_infos:
            trajs = [t for t in group.get('group_trajs', []) if not bool(t.get('is_extra', False))]
            if not trajs:
                continue
            group_scores = all_scores[offset: offset + len(trajs)]
            offset += len(trajs)
            if len(group_scores) <= 1:
                for traj in trajs:
                    traj['opd_advantage'] = 0.0
                continue
            mean_v = sum(group_scores) / len(group_scores)
            var_v = sum((x - mean_v) ** 2 for x in group_scores) / len(group_scores)
            std_v = math.sqrt(max(var_v, 0.0))
            for traj, score in zip(trajs, group_scores):
                adv = 0.0 if std_v <= 1e-6 else (float(score) - mean_v) / std_v
                traj['opd_advantage'] = max(-3.0, min(3.0, float(adv)))
        global_mean = sum(all_scores) / len(all_scores)
        global_var = sum((x - global_mean) ** 2 for x in all_scores) / len(all_scores)
        return {'opd_pairs': len(pairs), 'opd_score_mean': float(global_mean), 'opd_score_std': float(math.sqrt(max(global_var, 0.0)))}

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
        opd_loss_total = 0.0
        phase_times = {
            'zero_grad_time': 0.0,
            'pad_time': 0.0,
            'policy_forward_time': 0.0,
            'ref_forward_time': 0.0,
            'loss_time': 0.0,
            'backward_time': 0.0,
            'grad_clip_time': 0.0,
            'opt_step_time': 0.0,
        }

        base_lrs = [float(pg.get('lr', self.cfg.lr)) for pg in self.opt.param_groups]
        if lr_scale != 1.0:
            for pg, base_lr in zip(self.opt.param_groups, base_lrs):
                pg['lr'] = base_lr * lr_scale

        try:
            for _ in range(self.cfg.ppo_epochs):
                self.model.train()
                _t = time.time()
                self.opt.zero_grad(set_to_none=True)
                _sync_cuda()
                phase_times['zero_grad_time'] += time.time() - _t

                trajs_sorted = sorted(
                    trajs,
                    key=lambda x: len(x['prompt_tokens']) + len(x['comp_tokens']),
                    reverse=True,
                )

                for start in range(0, len(trajs_sorted), self.cfg.micro_batch):
                    batch = trajs_sorted[start:start + self.cfg.micro_batch]
                    seqs = [traj['prompt_tokens'] + traj['comp_tokens'] for traj in batch]
                    _t = time.time()
                    seqs, _ = self._pad_batch(seqs, pad_id=0)
                    phase_times['pad_time'] += time.time() - _t

                    inp = seqs[:, :-1].contiguous()
                    tgt = seqs[:, 1:].contiguous()

                    _sync_cuda()
                    _t = time.time()
                    logits = self.model(inp)
                    _sync_cuda()
                    phase_times['policy_forward_time'] += time.time() - _t
                    if torch.is_tensor(logits) and logits.dim() == 2:
                        logits = logits.unsqueeze(0)

                    logp = self._logp_with_sampling(logits, tgt)
                    del logits
                    torch.cuda.empty_cache()

                    with torch.no_grad():
                        _sync_cuda()
                        _t = time.time()
                        ref_logits = self.ref_model(inp)
                        _sync_cuda()
                        phase_times['ref_forward_time'] += time.time() - _t
                        if torch.is_tensor(ref_logits) and ref_logits.dim() == 2:
                            ref_logits = ref_logits.unsqueeze(0)
                        ref_logp_all = self._logp_with_sampling(ref_logits, tgt)
                        del ref_logits
                        torch.cuda.empty_cache()

                    batch_loss = 0.0
                    batch_kl = 0.0
                    batch_opd_loss = 0.0
                    batch_opd_items = 0
                    total_tokens = 0

                    _t = time.time()
                    for bi, traj in enumerate(batch):
                        prompt_len = len(traj['prompt_tokens'])
                        comp_len = len(traj['comp_tokens'])

                        start_idx = prompt_len - 1
                        end_idx = start_idx + comp_len

                        new_logp = logp[bi, start_idx:end_idx]
                        ref_logp = ref_logp_all[bi, start_idx:end_idx]
                        old_logp = torch.tensor(traj['old_logps'], device=self.device, dtype=torch.float32)
                        min_len = min(new_logp.size(0), ref_logp.size(0), old_logp.size(0))
                        if min_len == 0:
                            continue

                        new_logp = new_logp[:min_len]
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
                        kl = compute_unbiased_kl(ref_logp, new_logp).sum()

                        batch_loss += policy_loss
                        batch_kl += kl
                        total_tokens += min_len

                        opd_adv = float(traj.get('opd_advantage', 0.0))
                        if self.cfg.opd_enabled and abs(opd_adv) > 1e-8:
                            batch_opd_loss += -(new_logp.mean() * opd_adv)
                            batch_opd_items += 1
                    phase_times['loss_time'] += time.time() - _t

                    if total_tokens > 0:
                        normalized_loss = batch_loss / valid_tokens
                        normalized_kl = batch_kl / valid_tokens
                        normalized_entropy = 0.0

                        normalized_opd = (batch_opd_loss / max(1, batch_opd_items)) if batch_opd_items > 0 else 0.0
                        if self.cfg.kl_mode == 'k3_loss':
                            total_loss = normalized_loss + self.cfg.kl_coef * normalized_kl + float(self.cfg.opd_weight) * normalized_opd
                        else:
                            total_loss = normalized_loss + float(self.cfg.opd_weight) * normalized_opd

                        if self.cfg.time_state_l2 > 0:
                            l2_reg = 0.0
                            for n, param in self.model.named_parameters():
                                if 'time_state' in n:
                                    l2_reg += (param.float() - self._ts_init[n].float()).pow(2).mean()
                            total_loss += self.cfg.time_state_l2 * l2_reg

                        _sync_cuda()
                        _t = time.time()
                        total_loss.backward()
                        _sync_cuda()
                        phase_times['backward_time'] += time.time() - _t

                        loss_total += normalized_loss.item()
                        kl_total += normalized_kl.item()
                        entropy_total += float(normalized_entropy)
                        batch_cnt += 1

                if self.cfg.grad_clip > 0:
                    _sync_cuda()
                    _t = time.time()
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad],
                        self.cfg.grad_clip,
                    )
                    _sync_cuda()
                    phase_times['grad_clip_time'] += time.time() - _t

                with torch.no_grad():
                    g2 = 0.0
                    for p in self.model.parameters():
                        if p.requires_grad and p.grad is not None:
                            g = p.grad.detach().float()
                            g2 += (g.norm(2) ** 2).item()
                    grad_norm = math.sqrt(g2)

                _sync_cuda()
                _t = time.time()
                self.opt.step()
                _sync_cuda()
                phase_times['opt_step_time'] += time.time() - _t

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
            'opd_loss_total': opd_loss_total,
            'phase_times': phase_times,
        }

    def _optimize_opd_only(self, trajs: List[Dict[str, Any]], lr_scale: float = 1.0) -> Dict[str, float]:
        opd_trajs = [t for t in trajs if abs(float(t.get('opd_advantage', 0.0))) > 1e-8]
        if not self.cfg.opd_enabled or not opd_trajs:
            return {
                'opd_loss_total': 0.0,
                'batch_cnt': 0,
                'grad_norm': 0.0,
                'phase_times': {
                    'zero_grad_time': 0.0,
                    'pad_time': 0.0,
                    'policy_forward_time': 0.0,
                    'loss_time': 0.0,
                    'backward_time': 0.0,
                    'grad_clip_time': 0.0,
                    'opt_step_time': 0.0,
                },
            }

        total_loss = 0.0
        batch_cnt = 0
        grad_norm = 0.0
        phase_times = {
            'zero_grad_time': 0.0,
            'pad_time': 0.0,
            'policy_forward_time': 0.0,
            'loss_time': 0.0,
            'backward_time': 0.0,
            'grad_clip_time': 0.0,
            'opt_step_time': 0.0,
        }
        base_lrs = [float(pg.get('lr', self.cfg.lr)) for pg in self.opt.param_groups]
        if lr_scale != 1.0:
            for pg, base_lr in zip(self.opt.param_groups, base_lrs):
                pg['lr'] = base_lr * lr_scale

        try:
            self.model.train()
            _t = time.time()
            self.opt.zero_grad(set_to_none=True)
            _sync_cuda()
            phase_times['zero_grad_time'] += time.time() - _t

            trajs_sorted = sorted(
                opd_trajs,
                key=lambda x: len(x['prompt_tokens']) + len(x['comp_tokens']),
                reverse=True,
            )
            for start in range(0, len(trajs_sorted), self.cfg.micro_batch):
                batch = trajs_sorted[start:start + self.cfg.micro_batch]
                seqs = [traj['prompt_tokens'] + traj['comp_tokens'] for traj in batch]
                _t = time.time()
                seqs, _ = self._pad_batch(seqs, pad_id=0)
                phase_times['pad_time'] += time.time() - _t

                inp = seqs[:, :-1].contiguous()
                tgt = seqs[:, 1:].contiguous()

                _sync_cuda()
                _t = time.time()
                logits = self.model(inp)
                _sync_cuda()
                phase_times['policy_forward_time'] += time.time() - _t
                if torch.is_tensor(logits) and logits.dim() == 2:
                    logits = logits.unsqueeze(0)
                logp = self._logp_with_sampling(logits, tgt)
                del logits
                torch.cuda.empty_cache()

                batch_loss = 0.0
                batch_items = 0
                _t = time.time()
                for bi, traj in enumerate(batch):
                    prompt_len = len(traj['prompt_tokens'])
                    comp_len = len(traj['comp_tokens'])
                    start_idx = prompt_len - 1
                    end_idx = start_idx + comp_len
                    new_logp = logp[bi, start_idx:end_idx]
                    if new_logp.numel() == 0:
                        continue
                    opd_adv = float(traj.get('opd_advantage', 0.0))
                    if abs(opd_adv) <= 1e-8:
                        continue
                    batch_loss += -(new_logp.mean() * opd_adv)
                    batch_items += 1
                phase_times['loss_time'] += time.time() - _t

                if batch_items > 0:
                    loss = float(self.cfg.opd_weight) * (batch_loss / batch_items)
                    _sync_cuda()
                    _t = time.time()
                    loss.backward()
                    _sync_cuda()
                    phase_times['backward_time'] += time.time() - _t
                    total_loss += float(loss.item())
                    batch_cnt += 1

            if batch_cnt > 0:
                if self.cfg.grad_clip > 0:
                    _sync_cuda()
                    _t = time.time()
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad],
                        self.cfg.grad_clip,
                    )
                    _sync_cuda()
                    phase_times['grad_clip_time'] += time.time() - _t

                with torch.no_grad():
                    g2 = 0.0
                    for p in self.model.parameters():
                        if p.requires_grad and p.grad is not None:
                            g = p.grad.detach().float()
                            g2 += (g.norm(2) ** 2).item()
                    grad_norm = math.sqrt(g2)

                _sync_cuda()
                _t = time.time()
                self.opt.step()
                _sync_cuda()
                phase_times['opt_step_time'] += time.time() - _t

                infer_model = getattr(self.infer, 'infer_model', None)
                mark_dirty = getattr(infer_model, 'mark_dirty', None)
                if mark_dirty is not None:
                    mark_dirty()
        finally:
            if lr_scale != 1.0:
                for pg, base_lr in zip(self.opt.param_groups, base_lrs):
                    pg['lr'] = base_lr

        return {
            'opd_loss_total': total_loss,
            'batch_cnt': batch_cnt,
            'grad_norm': grad_norm,
            'phase_times': phase_times,
        }

    def train_step(self, step: int) -> Dict[str, Any]:
        t0 = time.time()
        phase_times = {
            'sample_questions_time': 0.0,
            'prompt_encode_time': 0.0,
            'rollout_time': 0.0,
            'reward_parse_time': 0.0,
            'reward_calc_time': 0.0,
            'response_write_time': 0.0,
            'extra_rollout_time': 0.0,
            'k1_ref_time': 0.0,
            'advantage_time': 0.0,
            'cleanup_time': 0.0,
            'normal_opt_time': 0.0,
            'extra_opt_time': 0.0,
            'opd_only_opt_time': 0.0,
        }
        _t = time.time()

        if len(self.train_data) >= self.cfg.num_questions:
            sampled_indices = self.rng.sample(range(len(self.train_data)), self.cfg.num_questions)
        else:
            sampled_indices = [self.rng.randrange(len(self.train_data)) 
                             for _ in range(self.cfg.num_questions)]
        phase_times['sample_questions_time'] += time.time() - _t

        sampled_questions = [self.train_data[i] for i in sampled_indices]

        _t = time.time()
        prompt_strs = [build_prompt(q.get("problem", ""), mode=getattr(self.cfg, "prompt_mode", "rwkv_boxed")) for q in sampled_questions]
        prompt_tokens_list = []
        for ps in prompt_strs:
            ids = self.encode(ps)
            max_prompt_len = int(self.model.args.ctx_len) - int(self.cfg.max_new_tokens) - 4
            max_prompt_len = max(64, max_prompt_len)
            if len(ids) > max_prompt_len:
                ids = ids[-max_prompt_len:]
            prompt_tokens_list.append(ids)
        phase_times['prompt_encode_time'] += time.time() - _t

        _sync_cuda()
        _t = time.time()
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
                stop_on_boxed=False,
                stop_on_repeat_ngram=False,
                presence_penalty=0.0,
                frequency_penalty=0.0,
                alpha_decay=1.0,
            )
        _sync_cuda()
        phase_times['rollout_time'] += time.time() - _t

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
            _reward_t0 = time.time()
            repeat_flag = self._has_repeated_ngrams(comp_text, n=16, repeat=5)

            _calc_t0 = time.time()
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
            phase_times['reward_calc_time'] += time.time() - _calc_t0

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
                _write_t0 = time.time()
                step_path = os.path.join(self.responses_dir, f"step_{step}.jsonl")
                append_jsonl(step_path, record)
                phase_times['response_write_time'] += time.time() - _write_t0
            phase_times['reward_parse_time'] += time.time() - _reward_t0

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

            _sync_cuda()
            _t = time.time()
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
            _sync_cuda()
            phase_times['extra_rollout_time'] += time.time() - _t

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
            _sync_cuda()
            _t = time.time()
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
            _sync_cuda()
            phase_times['k1_ref_time'] += time.time() - _t

        groups_total = len(group_infos)

        _t = time.time()
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
        phase_times['advantage_time'] += time.time() - _t

        # 5. 训练
        # 5. training (normal step + extra hard-buffer step)
        normal_trajs = [t for t in all_trajs if not bool(t.get("is_extra", False))]
        extra_trajs = [t for t in all_trajs if bool(t.get("is_extra", False))]

        infer_model = getattr(self.infer, "infer_model", None)
        cleanup_rollout = getattr(infer_model, "cleanup_stateful_rollout", None)
        if cleanup_rollout is not None:
            _sync_cuda()
            _t = time.time()
            cleanup_rollout()
            _sync_cuda()
            phase_times['cleanup_time'] += time.time() - _t

        opd_stats = {'opd_pairs': 0, 'opd_score_mean': 0.0, 'opd_score_std': 0.0}
        if self.cfg.opd_enabled and self.teacher_model is not None and self.teacher_tokenizer is not None:
            opd_stats = self._assign_opd_advantages(group_infos)

        _sync_cuda()
        _t = time.time()
        normal_opt = self._optimize_trajs(normal_trajs, lr_scale=1.0, adv_clip=None)
        _sync_cuda()
        phase_times['normal_opt_time'] += time.time() - _t

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
            _sync_cuda()
            _t = time.time()
            extra_opt = self._optimize_trajs(
                extra_trajs,
                lr_scale=extra_lr_scale,
                adv_clip=extra_adv_clip,
            )
            _sync_cuda()
            phase_times['extra_opt_time'] += time.time() - _t

        _sync_cuda()
        _t = time.time()
        opd_only_opt = self._optimize_opd_only(normal_trajs, lr_scale=1.0)
        _sync_cuda()
        phase_times['opd_only_opt_time'] += time.time() - _t

        loss_total = float(normal_opt['loss_total'] + extra_opt['loss_total'])
        opd_loss = float(normal_opt.get('opd_loss_total', 0.0) + extra_opt.get('opd_loss_total', 0.0) + opd_only_opt.get('opd_loss_total', 0.0))
        kl_total = float(normal_opt['kl_total'] + extra_opt['kl_total'])
        entropy_total = float(normal_opt['entropy_total'] + extra_opt['entropy_total'])
        batch_cnt = int(normal_opt['batch_cnt'] + extra_opt['batch_cnt'])
        clip_total = float(normal_opt['clip_total'] + extra_opt['clip_total'])
        clip_total_tokens = int(normal_opt['clip_total_tokens'] + extra_opt['clip_total_tokens'])
        grad_norm = max(float(normal_opt['grad_norm']), float(extra_opt['grad_norm']), float(opd_only_opt.get('grad_norm', 0.0)))

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
            "opd_only_loss": float(opd_only_opt.get('opd_loss_total', 0.0)),
            "opd_only_batches": int(opd_only_opt.get('batch_cnt', 0)),
            "opd_only_grad_norm": float(opd_only_opt.get('grad_norm', 0.0)),
            "extra_lr_scale": extra_lr_scale,
            "extra_adv_clip": extra_adv_clip,
            "opd_enabled": int(self.cfg.opd_enabled),
            "opd_pairs": int(opd_stats.get("opd_pairs", 0)),
            "opd_batches": 0,
            "opd_loss": opd_loss,
            "opd_score_mean": float(opd_stats.get("opd_score_mean", 0.0)),
            "opd_score_std": float(opd_stats.get("opd_score_std", 0.0)),
            "sample_questions_time": phase_times['sample_questions_time'],
            "prompt_encode_time": phase_times['prompt_encode_time'],
            "rollout_time": phase_times['rollout_time'],
            "reward_parse_time": phase_times['reward_parse_time'],
            "reward_calc_time": phase_times['reward_calc_time'],
            "response_write_time": phase_times['response_write_time'],
            "extra_rollout_time": phase_times['extra_rollout_time'],
            "k1_ref_time": phase_times['k1_ref_time'],
            "advantage_time": phase_times['advantage_time'],
            "cleanup_time": phase_times['cleanup_time'],
            "normal_opt_time": phase_times['normal_opt_time'],
            "extra_opt_time": phase_times['extra_opt_time'],
            "opd_only_opt_time": phase_times['opd_only_opt_time'],
            "opt_zero_grad_time": float(normal_opt.get('phase_times', {}).get('zero_grad_time', 0.0) + extra_opt.get('phase_times', {}).get('zero_grad_time', 0.0) + opd_only_opt.get('phase_times', {}).get('zero_grad_time', 0.0)),
            "opt_pad_time": float(normal_opt.get('phase_times', {}).get('pad_time', 0.0) + extra_opt.get('phase_times', {}).get('pad_time', 0.0) + opd_only_opt.get('phase_times', {}).get('pad_time', 0.0)),
            "opt_policy_forward_time": float(normal_opt.get('phase_times', {}).get('policy_forward_time', 0.0) + extra_opt.get('phase_times', {}).get('policy_forward_time', 0.0) + opd_only_opt.get('phase_times', {}).get('policy_forward_time', 0.0)),
            "opt_ref_forward_time": float(normal_opt.get('phase_times', {}).get('ref_forward_time', 0.0) + extra_opt.get('phase_times', {}).get('ref_forward_time', 0.0)),
            "opt_loss_time": float(normal_opt.get('phase_times', {}).get('loss_time', 0.0) + extra_opt.get('phase_times', {}).get('loss_time', 0.0) + opd_only_opt.get('phase_times', {}).get('loss_time', 0.0)),
            "opt_backward_time": float(normal_opt.get('phase_times', {}).get('backward_time', 0.0) + extra_opt.get('phase_times', {}).get('backward_time', 0.0) + opd_only_opt.get('phase_times', {}).get('backward_time', 0.0)),
            "opt_grad_clip_time": float(normal_opt.get('phase_times', {}).get('grad_clip_time', 0.0) + extra_opt.get('phase_times', {}).get('grad_clip_time', 0.0) + opd_only_opt.get('phase_times', {}).get('grad_clip_time', 0.0)),
            "opt_step_time": float(normal_opt.get('phase_times', {}).get('opt_step_time', 0.0) + extra_opt.get('phase_times', {}).get('opt_step_time', 0.0)),
        }
        
        return metrics
    
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
                    f"opd(en={metrics['opd_enabled']},pairs={metrics['opd_pairs']},loss={metrics['opd_loss']:.4f}) "
                    f"phases(roll={metrics['rollout_time']:.1f}s,bwd={metrics['opt_backward_time']:.1f}s,opt={metrics['opt_step_time']:.1f}s,ref={metrics['opt_ref_forward_time'] + metrics['k1_ref_time']:.1f}s) "
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




