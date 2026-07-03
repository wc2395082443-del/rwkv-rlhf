#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import math
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from infer import AlbatrossBatchInference
from main import (
    cast_trainable_time_state_fp32,
    freeze_except_time_state,
    load_infer_model_albatross,
    load_time_state_only,
    load_train_model_rwkv7_cuda,
    normalize_model_arg,
)
from reward import calculate_reward_details
from utils import append_jsonl, build_prompt, read_jsonl, set_seed


def compute_unbiased_kl(ref_logp: torch.Tensor, policy_logp: torch.Tensor) -> torch.Tensor:
    log_ratio = ref_logp - policy_logp
    return torch.exp(log_ratio) - log_ratio - 1.0


@dataclass
class DirectGRPOConfig:
    num_questions: int = 24
    samples_per_question: int = 8
    max_new_tokens: int = 1024
    temperature: float = 1.0
    top_p: float = 0.6
    top_k: int = 0
    eval_temperature: float = 0.3
    eval_top_p: float = 0.4
    eval_top_k: int = 500
    ppo_epochs: int = 1
    micro_batch: int = 4
    lr: float = 6e-5
    grad_clip: float = 1.0
    clip_eps: float = 0.2
    kl_coef: float = 0.0
    neg_adv_weight: float = 1.0
    min_tokens: int = 200
    max_tokens: int = 1024
    length_weight: float = 0.0
    zstd_threshold: float = 2.5
    zstd_penalty_weight: float = 0.0
    ngram_penalty: float = 0.0
    time_state_l2: float = 0.0
    time_state_clamp: float = 10.0
    log_interval: int = 1
    save_interval: int = 50
    eval_interval: int = 5
    eval_sample_ratio: float = 1.0


class DirectGRPOTrainer:
    def __init__(self, train_model, ref_model, infer_engine: AlbatrossBatchInference, encode_fn, decode_fn, train_data: List[Dict[str, Any]], test_data: List[Dict[str, Any]], out_dir: str, device: str, cfg: DirectGRPOConfig, seed: int = 42):
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
        os.makedirs(self.out_dir, exist_ok=True)
        self.log_path = os.path.join(out_dir, 'train.log')
        self.metrics_path = os.path.join(out_dir, 'metrics.jsonl')
        self.responses_dir = os.path.join(out_dir, 'responses_by_step')
        self.eval_by_step_dir = os.path.join(out_dir, 'eval_by_step')
        os.makedirs(self.responses_dir, exist_ok=True)
        os.makedirs(self.eval_by_step_dir, exist_ok=True)
        params = [p for p in self.model.parameters() if p.requires_grad]
        if not params:
            raise RuntimeError('No trainable parameters found')
        self.opt = torch.optim.Adam(params, lr=self.cfg.lr, betas=(0.9, 0.99), eps=1e-8)
        self._ts_init = {}
        with torch.no_grad():
            for n, p in self.model.named_parameters():
                if 'time_state' in n:
                    self._ts_init[n] = p.detach().clone()

    def _log(self, msg: str):
        ts = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        line = f'[{ts}] {msg}'
        print(line, flush=True)
        with open(self.log_path, 'a', encoding='utf-8') as f:
            f.write(line + '\n')

    def _plot_metrics(self):
        plot_script = os.path.join(os.path.dirname(__file__), 'plot_metrics.py')
        out_plot = os.path.join(self.out_dir, 'metrics_plot.png')
        if os.path.isfile(plot_script) and os.path.isfile(self.metrics_path):
            try:
                subprocess.run([sys.executable, plot_script, '--metrics', self.metrics_path, '--out', out_plot], check=False)
            except Exception:
                pass

    @staticmethod
    def _has_repeated_ngrams(text: str, n: int = 16, repeat: int = 5) -> bool:
        import re
        if not text or n <= 0 or repeat <= 1:
            return False
        tokens = re.findall(r'\w+|[^\w\s]', text)
        if len(tokens) < n * repeat:
            return False
        counts = {}
        for idx in range(len(tokens) - n + 1):
            gram = tuple(tokens[idx: idx + n])
            counts[gram] = counts.get(gram, 0) + 1
            if counts[gram] >= repeat:
                return True
        return False

    @torch.no_grad()
    def _get_answer(self, ex: Dict[str, Any]) -> str:
        if 'answer' in ex and ex.get('answer') is not None:
            return str(ex['answer'])
        if 'solution' in ex and ex.get('solution') is not None:
            return str(ex['solution'])
        return ''

    @staticmethod
    def _pad_batch(seqs: List[List[int]], pad_id: int = 0):
        max_len = max(len(s) for s in seqs)
        padded, masks = [], []
        for seq in seqs:
            pad_len = max_len - len(seq)
            padded.append(seq + [pad_id] * pad_len)
            masks.append([1] * len(seq) + [0] * pad_len)
        return torch.tensor(padded, dtype=torch.long), torch.tensor(masks, dtype=torch.bool)

    @staticmethod
    def _gather_logp(logits: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        if logits.dim() == 2:
            logits = logits.unsqueeze(0)
        logsumexp = torch.logsumexp(logits, dim=-1).float()
        logit_tgt = logits.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).float()
        return logit_tgt - logsumexp

    def _time_state_stats(self):
        mx = 0.0
        rms_sum = 0.0
        cnt = 0
        bad = False
        with torch.no_grad():
            for n, p in self.model.named_parameters():
                if 'time_state' not in n:
                    continue
                if torch.isnan(p).any() or torch.isinf(p).any():
                    bad = True
                value = p.detach().float()
                mx = max(mx, float(value.abs().max().item()))
                rms_sum += float((value * value).mean().sqrt().item())
                cnt += 1
        return {'absmax': mx, 'rms_avg': rms_sum / max(1, cnt), 'bad': bad}

    def _encode_prompts(self, questions: List[Dict[str, Any]]) -> List[List[int]]:
        prompt_tokens_list = []
        for question in questions:
            prompt = build_prompt(question.get('problem', ''))
            ids = self.encode(prompt)
            max_prompt_len = int(self.model.args.ctx_len) - int(self.cfg.max_new_tokens) - 4
            max_prompt_len = max(64, max_prompt_len)
            if len(ids) > max_prompt_len:
                ids = ids[-max_prompt_len:]
            prompt_tokens_list.append(ids)
        return prompt_tokens_list

    def _collect_rollouts(self, step: int):
        if len(self.train_data) >= self.cfg.num_questions:
            sampled_indices = self.rng.sample(range(len(self.train_data)), self.cfg.num_questions)
        else:
            sampled_indices = [self.rng.randrange(len(self.train_data)) for _ in range(self.cfg.num_questions)]
        questions = [self.train_data[i] for i in sampled_indices]
        prompt_tokens_list = self._encode_prompts(questions)
        comp_tokens_list, old_logps_list, comp_texts_list, truncated_list = self.infer.generate_group_parallel(
            prompt_tokens_list=prompt_tokens_list,
            group_size=self.cfg.samples_per_question,
            max_new_tokens=self.cfg.max_new_tokens,
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            top_k=self.cfg.top_k,
        )
        groups = []
        flat_idx = 0
        stats = {'total_samples': 0, 'correct_samples': 0, 'total_reward': 0.0, 'total_length': 0, 'total_trunc': 0, 'total_repeat': 0, 'no_answer': 0, 'sum_correct_reward': 0.0, 'sum_format_reward': 0.0, 'sum_length_reward': 0.0, 'sum_length_lambda': 0.0, 'sum_zstd_penalty': 0.0, 'sum_zstd_ratio': 0.0}
        for q_idx, question in enumerate(questions):
            group = {'q_idx': q_idx, 'problem': question.get('problem', ''), 'answer': self._get_answer(question), 'prompt_tokens': prompt_tokens_list[q_idx], 'trajs': []}
            for sample_idx in range(self.cfg.samples_per_question):
                comp_tokens = comp_tokens_list[flat_idx]
                old_logps = old_logps_list[flat_idx]
                comp_text = comp_texts_list[flat_idx]
                truncated = truncated_list[flat_idx]
                flat_idx += 1
                repeat_flag = self._has_repeated_ngrams(comp_text, n=16, repeat=5)
                reward, is_correct, is_format_correct, reward_details = calculate_reward_details(text=comp_text, ground_truth=group['answer'], token_length=len(comp_tokens), min_tokens=self.cfg.min_tokens, max_tokens=self.cfg.max_tokens, length_weight=self.cfg.length_weight, repeat_ngram=repeat_flag, repeat_penalty=self.cfg.ngram_penalty, zstd_threshold=self.cfg.zstd_threshold, zstd_penalty_weight=self.cfg.zstd_penalty_weight)
                traj = {'prompt_tokens': group['prompt_tokens'], 'comp_tokens': comp_tokens, 'old_logps': old_logps, 'reward': float(reward), 'text': comp_text, 'is_correct': bool(is_correct), 'is_format_correct': bool(is_format_correct), 'truncated': bool(truncated), 'reward_details': reward_details}
                group['trajs'].append(traj)
                stats['total_samples'] += 1
                stats['total_reward'] += float(reward)
                stats['total_length'] += len(comp_tokens)
                stats['total_trunc'] += int(bool(truncated))
                stats['total_repeat'] += int(bool(repeat_flag))
                stats['correct_samples'] += int(bool(is_correct))
                stats['no_answer'] += int(not reward_details.get('extracted_answer'))
                stats['sum_correct_reward'] += float(reward_details.get('correct_reward', 0.0))
                stats['sum_format_reward'] += float(reward_details.get('format_reward', 0.0))
                stats['sum_length_reward'] += float(reward_details.get('length_reward', 0.0))
                stats['sum_length_lambda'] += float(reward_details.get('length_lambda', 0.0))
                stats['sum_zstd_penalty'] += float(reward_details.get('zstd_penalty', 0.0))
                stats['sum_zstd_ratio'] += float(reward_details.get('zstd_ratio', 0.0))
                append_jsonl(os.path.join(self.responses_dir, f'step_{step}.jsonl'), {'step': step, 'question_idx': q_idx, 'sample_idx': sample_idx, 'problem': group['problem'], 'ground_truth': group['answer'], 'response': comp_text, 'reward': reward, 'is_correct': is_correct, 'is_format_correct': is_format_correct, 'truncated': truncated, 'reward_details': reward_details})
            groups.append(group)
        return groups, stats

    def _compute_advantages(self, groups: List[Dict[str, Any]]):
        all_trajs, adv_values = [], []
        groups_all_correct = 0
        groups_all_wrong = 0
        groups_zero_std = 0
        for group in groups:
            rewards = [traj['reward'] for traj in group['trajs']]
            correct_count = sum(1 for traj in group['trajs'] if traj['is_correct'])
            if correct_count == 0:
                groups_all_wrong += 1
            if correct_count == len(group['trajs']):
                groups_all_correct += 1
            if not rewards:
                continue
            mean_reward = sum(rewards) / len(rewards)
            variance = sum((r - mean_reward) ** 2 for r in rewards) / len(rewards)
            std_reward = math.sqrt(variance)
            if std_reward < 1e-8:
                groups_zero_std += 1
                for traj in group['trajs']:
                    traj['advantage'] = 0.0
            else:
                denom = std_reward + 1e-8
                for traj in group['trajs']:
                    traj['advantage'] = (traj['reward'] - mean_reward) / denom
                    adv_values.append(float(traj['advantage']))
            all_trajs.extend(group['trajs'])
        return all_trajs, adv_values, groups_all_correct, groups_all_wrong, groups_zero_std

    def _apply_loss(self, batch: List[Dict[str, Any]]):
        seqs = [traj['prompt_tokens'] + traj['comp_tokens'] for traj in batch]
        seqs, _ = self._pad_batch(seqs, pad_id=0)
        seqs = seqs.to(self.device)
        inp = seqs[:, :-1].contiguous()
        tgt = seqs[:, 1:].contiguous()
        logits = self.model(inp)
        new_logp_all = self._gather_logp(logits, tgt)
        del logits
        with torch.no_grad():
            ref_logits = self.ref_model(inp)
            ref_logp_all = self._gather_logp(ref_logits, tgt)
            del ref_logits
        batch_policy_loss = 0.0
        batch_kl = 0.0
        batch_tokens = 0
        clip_hits = 0
        clip_total = 0
        lo = 1.0 - self.cfg.clip_eps
        hi = 1.0 + self.cfg.clip_eps
        for bi, traj in enumerate(batch):
            prompt_len = len(traj['prompt_tokens'])
            comp_len = len(traj['comp_tokens'])
            start_idx = prompt_len - 1
            end_idx = start_idx + comp_len
            new_logp = new_logp_all[bi, start_idx:end_idx]
            ref_logp = ref_logp_all[bi, start_idx:end_idx]
            old_logp = torch.tensor(traj['old_logps'], device=self.device, dtype=torch.float32)
            min_len = min(new_logp.size(0), ref_logp.size(0), old_logp.size(0))
            if min_len <= 0:
                continue
            new_logp = new_logp[:min_len]
            ref_logp = ref_logp[:min_len]
            old_logp = old_logp[:min_len]
            advantage = torch.tensor(float(traj['advantage']), device=self.device, dtype=torch.float32)
            if advantage.item() < 0:
                advantage = advantage * float(self.cfg.neg_adv_weight)
            ratio = torch.exp(new_logp - old_logp)
            ratio_clipped = torch.clamp(ratio, lo, hi)
            clip_hits += int(((ratio < lo) | (ratio > hi)).sum().item())
            clip_total += ratio.numel()
            policy_loss = -(torch.min(ratio * advantage, ratio_clipped * advantage)).sum()
            kl = compute_unbiased_kl(ref_logp, new_logp).sum()
            batch_policy_loss = batch_policy_loss + policy_loss
            batch_kl = batch_kl + kl
            batch_tokens += min_len
        if batch_tokens <= 0:
            return None
        total_loss = (batch_policy_loss / batch_tokens) + self.cfg.kl_coef * (batch_kl / batch_tokens)
        if self.cfg.time_state_l2 > 0:
            l2_reg = 0.0
            for name, param in self.model.named_parameters():
                if 'time_state' in name:
                    l2_reg = l2_reg + (param.float() - self._ts_init[name].float()).pow(2).mean()
            total_loss = total_loss + self.cfg.time_state_l2 * l2_reg
        total_loss.backward()
        return {'loss': float((batch_policy_loss / batch_tokens).detach().item()), 'avg_kl': float((batch_kl / batch_tokens).detach().item()), 'clip_hits': clip_hits, 'clip_total': clip_total}

    def _optimize(self, all_trajs: List[Dict[str, Any]]):
        loss_total = 0.0
        kl_total = 0.0
        batch_cnt = 0
        clip_hits = 0
        clip_total = 0
        grad_norm = 0.0
        if not all_trajs:
            return loss_total, kl_total, batch_cnt, clip_hits, clip_total, grad_norm
        for _ in range(self.cfg.ppo_epochs):
            self.model.train()
            self.opt.zero_grad(set_to_none=True)
            trajs = list(all_trajs)
            self.rng.shuffle(trajs)
            for start in range(0, len(trajs), self.cfg.micro_batch):
                result = self._apply_loss(trajs[start:start + self.cfg.micro_batch])
                if result is None:
                    continue
                loss_total += result['loss']
                kl_total += result['avg_kl']
                batch_cnt += 1
                clip_hits += result['clip_hits']
                clip_total += result['clip_total']
            if self.cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad], self.cfg.grad_clip)
            with torch.no_grad():
                g2 = 0.0
                for p in self.model.parameters():
                    if p.requires_grad and p.grad is not None:
                        g = p.grad.detach().float()
                        g2 += (g.norm(2) ** 2).item()
                grad_norm = math.sqrt(g2)
            self.opt.step()
            if self.cfg.time_state_clamp > 0:
                with torch.no_grad():
                    for name, param in self.model.named_parameters():
                        if 'time_state' in name:
                            param.data.clamp_(-self.cfg.time_state_clamp, self.cfg.time_state_clamp)
        return loss_total, kl_total, batch_cnt, clip_hits, clip_total, grad_norm

    def train_step(self, step: int) -> Dict[str, Any]:
        t0 = time.time()
        groups, stats = self._collect_rollouts(step)
        all_trajs, adv_values, groups_all_correct, groups_all_wrong, groups_zero_std = self._compute_advantages(groups)
        loss_total, kl_total, batch_cnt, clip_hits, clip_total, grad_norm = self._optimize(all_trajs)
        dt = time.time() - t0
        if adv_values:
            adv_mean = sum(adv_values) / len(adv_values)
            adv_var = sum((x - adv_mean) ** 2 for x in adv_values) / len(adv_values)
            adv_std = math.sqrt(adv_var)
            pos_adv_ratio = sum(1 for x in adv_values if x > 0) / len(adv_values)
            neg_adv_ratio = sum(1 for x in adv_values if x < 0) / len(adv_values)
        else:
            adv_mean = adv_std = pos_adv_ratio = neg_adv_ratio = 0.0
        total_samples = max(1, stats['total_samples'])
        ts_stats = self._time_state_stats()
        return {'step': step, 'split': 'train', 'samples': stats['total_samples'], 'accuracy': stats['correct_samples'] / total_samples, 'avg_reward': stats['total_reward'] / total_samples, 'avg_length': stats['total_length'] / total_samples, 'trunc_rate': stats['total_trunc'] / total_samples, 'repeat_rate': stats['total_repeat'] / total_samples, 'no_answer_rate': stats['no_answer'] / total_samples, 'avg_correct_reward': stats['sum_correct_reward'] / total_samples, 'avg_format_reward': stats['sum_format_reward'] / total_samples, 'avg_length_reward': stats['sum_length_reward'] / total_samples, 'avg_length_lambda': stats['sum_length_lambda'] / total_samples, 'avg_zstd_penalty': stats['sum_zstd_penalty'] / total_samples, 'avg_zstd_ratio': stats['sum_zstd_ratio'] / total_samples, 'adv_mean': adv_mean, 'adv_std': adv_std, 'pos_adv_ratio': pos_adv_ratio, 'neg_adv_ratio': neg_adv_ratio, 'groups_total': len(groups), 'groups_used': len(groups) - groups_zero_std, 'groups_zero_std': groups_zero_std, 'groups_all_correct': groups_all_correct, 'groups_all_wrong': groups_all_wrong, 'loss': loss_total, 'kl': kl_total, 'avg_kl': kl_total / max(1, batch_cnt), 'clip_frac': clip_hits / max(1, clip_total), 'grad_norm': grad_norm, 'time': dt, 'samples_per_sec': stats['total_samples'] / dt if dt > 0 else 0.0, 'tokens_per_sec': stats['total_length'] / dt if dt > 0 else 0.0, 'ts_absmax': ts_stats['absmax'], 'ts_rms': ts_stats['rms_avg'], 'ts_bad': ts_stats['bad']}

    @torch.no_grad()
    def evaluate(self, step: int, tag: str = 'eval', sample_ratio: float = 1.0) -> Optional[float]:
        if not self.test_data:
            return None
        t0 = time.time()
        data = self.test_data
        if sample_ratio < 1.0:
            count = max(1, int(len(data) * sample_ratio))
            ids = self.rng.sample(range(len(data)), count)
            data = [data[i] for i in ids]
        total = correct = total_len = total_trunc = total_repeat = total_no_answer = 0
        total_zstd = 0.0
        chunk_size = 192
        for start in range(0, len(data), chunk_size):
            ex_list = data[start:start + chunk_size]
            prompts = self._encode_prompts(ex_list)
            comp_tokens_list, _, comp_texts_list, truncated_list = self.infer.generate_group_parallel(prompt_tokens_list=prompts, group_size=1, max_new_tokens=self.cfg.max_new_tokens, temperature=self.cfg.eval_temperature, top_p=self.cfg.eval_top_p, top_k=self.cfg.eval_top_k)
            for idx, ex in enumerate(ex_list):
                answer = self._get_answer(ex)
                comp_text = comp_texts_list[idx]
                comp_tokens = comp_tokens_list[idx]
                truncated = truncated_list[idx]
                repeat_flag = self._has_repeated_ngrams(comp_text, n=16, repeat=5)
                reward, is_correct, is_format_correct, reward_details = calculate_reward_details(text=comp_text, ground_truth=answer, token_length=len(comp_tokens), min_tokens=self.cfg.min_tokens, max_tokens=self.cfg.max_tokens, length_weight=self.cfg.length_weight, repeat_ngram=repeat_flag, repeat_penalty=self.cfg.ngram_penalty, zstd_threshold=self.cfg.zstd_threshold, zstd_penalty_weight=self.cfg.zstd_penalty_weight)
                append_jsonl(os.path.join(self.eval_by_step_dir, f'{tag}_step_{step}.jsonl'), {'step': step, 'tag': tag, 'problem': ex.get('problem', ''), 'ground_truth': answer, 'response': comp_text, 'reward': reward, 'is_correct': is_correct, 'is_format_correct': is_format_correct, 'truncated': truncated, 'gen_len': len(comp_tokens), 'zstd_ratio': reward_details.get('zstd_ratio', 0.0), 'repeat_16gram_5': repeat_flag})
                total += 1
                correct += int(bool(is_correct))
                total_len += len(comp_tokens)
                total_trunc += int(bool(truncated))
                total_repeat += int(bool(repeat_flag))
                total_no_answer += int(not reward_details.get('extracted_answer'))
                total_zstd += float(reward_details.get('zstd_ratio', 0.0))
        acc = correct / max(1, total)
        metrics = {'step': step, 'split': tag, 'accuracy': acc, 'acc': acc, 'total': total, 'correct': correct, 'avg_length': total_len / max(1, total), 'trunc_rate': total_trunc / max(1, total), 'repeat_rate': total_repeat / max(1, total), 'repeat_16gram_rate': total_repeat / max(1, total), 'no_answer_rate': total_no_answer / max(1, total), 'avg_zstd_ratio': total_zstd / max(1, total), 'eval_time': time.time() - t0}
        append_jsonl(self.metrics_path, metrics)
        self._log(f"[EVAL step {step}] acc={metrics['acc']:.3f} trunc={metrics['trunc_rate']:.3f} repeat={metrics['repeat_rate']:.3f} no_ans={metrics['no_answer_rate']:.3f} avg_len={metrics['avg_length']:.1f} zstd={metrics['avg_zstd_ratio']:.3f} time={metrics['eval_time']:.1f}s")
        return acc

    def train(self, total_steps: int):
        self._log(f'start direct grpo training total_steps={total_steps}')
        train_start = time.time()
        for step in range(1, total_steps + 1):
            metrics = self.train_step(step)
            metrics['elapsed'] = time.time() - train_start
            append_jsonl(self.metrics_path, metrics)
            if step % self.cfg.log_interval == 0:
                self._log(f"[Step {step}/{total_steps}] samples={int(metrics['samples'])} acc={metrics['accuracy']:.3f} reward={metrics['avg_reward']:.4f} len={metrics['avg_length']:.1f} trunc={metrics['trunc_rate']:.3f} repeat={metrics['repeat_rate']:.3f} no_answer={metrics['no_answer_rate']:.3f} loss={metrics['loss']:.4f} kl={metrics['avg_kl']:.6f} adv(m={metrics['adv_mean']:.3f},s={metrics['adv_std']:.3f}) groups(zero_std={metrics['groups_zero_std']},all0={metrics['groups_all_wrong']},all1={metrics['groups_all_correct']}) ts(absmax={metrics['ts_absmax']:.4f},rms={metrics['ts_rms']:.4f},bad={metrics['ts_bad']}) speed(samp/s={metrics['samples_per_sec']:.2f},tok/s={metrics['tokens_per_sec']:.1f}) step_time={metrics['time']:.1f}s")
            if step % self.cfg.save_interval == 0 or step == total_steps:
                ckpt_path = os.path.join(self.out_dir, f'ckpt_step{step}.pth')
                torch.save({'step': step, 'time_state': {n: p.detach().cpu() for n, p in self.model.named_parameters() if 'time_state' in n}, 'optimizer': self.opt.state_dict()}, ckpt_path)
                self._log(f'saved checkpoint: {ckpt_path}')
            full_eval = (step % self.cfg.save_interval == 0) or (step == total_steps)
            if self.cfg.eval_interval > 0 and self.test_data and (step % self.cfg.eval_interval == 0):
                if full_eval:
                    self.evaluate(step, tag='full_eval', sample_ratio=1.0)
                else:
                    self.evaluate(step, tag='eval', sample_ratio=self.cfg.eval_sample_ratio)
                self._plot_metrics()
            elif full_eval and self.test_data:
                self.evaluate(step, tag='full_eval', sample_ratio=1.0)
                self._plot_metrics()


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train_jsonl', type=str, required=True)
    ap.add_argument('--eval_jsonl', type=str, default=None)
    ap.add_argument('--max_data_samples', type=int, default=None)
    ap.add_argument('--model', type=str, required=True)
    ap.add_argument('--tokenizer', type=str, required=True)
    ap.add_argument('--state_init', type=str, default=None)
    ap.add_argument('--ctx_len', type=int, default=8192)
    ap.add_argument('--num_questions', type=int, default=24)
    ap.add_argument('--samples_per_question', type=int, default=8)
    ap.add_argument('--max_new_tokens', type=int, default=1024)
    ap.add_argument('--temperature', type=float, default=1.0)
    ap.add_argument('--top_p', type=float, default=0.6)
    ap.add_argument('--top_k', type=int, default=0)
    ap.add_argument('--eval_temperature', type=float, default=0.3)
    ap.add_argument('--eval_top_p', type=float, default=0.4)
    ap.add_argument('--eval_top_k', type=int, default=500)
    ap.add_argument('--min_tokens', type=int, default=200)
    ap.add_argument('--length_weight', type=float, default=0.0)
    ap.add_argument('--zstd_threshold', type=float, default=2.5)
    ap.add_argument('--zstd_penalty_weight', type=float, default=0.0)
    ap.add_argument('--ngram_penalty', type=float, default=0.0)
    ap.add_argument('--total_steps', type=int, default=200)
    ap.add_argument('--ppo_epochs', type=int, default=1)
    ap.add_argument('--micro_batch', type=int, default=4)
    ap.add_argument('--lr', type=float, default=6e-5)
    ap.add_argument('--grad_clip', type=float, default=1.0)
    ap.add_argument('--clip_eps', type=float, default=0.2)
    ap.add_argument('--kl_coef', type=float, default=0.0)
    ap.add_argument('--neg_adv_weight', type=float, default=1.0)
    ap.add_argument('--time_state_l2', type=float, default=0.0)
    ap.add_argument('--time_state_clamp', type=float, default=10.0)
    ap.add_argument('--out_dir', type=str, required=True)
    ap.add_argument('--log_interval', type=int, default=1)
    ap.add_argument('--save_interval', type=int, default=50)
    ap.add_argument('--eval_interval', type=int, default=5)
    ap.add_argument('--eval_sample_ratio', type=float, default=1.0)
    ap.add_argument('--seed', type=int, default=42)
    return ap.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.environ['RWKV_HEAD_SIZE_A'] = '64'
    os.environ['RWKV_MY_TESTING'] = 'x070'
    os.environ['RWKV_TRAIN_TYPE'] = 'state'
    os.environ['RWKV_CTXLEN'] = str(int(args.ctx_len))
    os.environ['FUSED_KERNEL'] = '0'
    os.environ['WKV'] = 'cuda'
    os.makedirs(args.out_dir, exist_ok=True)
    train_data = read_jsonl(args.train_jsonl, max_samples=args.max_data_samples)
    if not train_data:
        raise RuntimeError('??????')
    if args.eval_jsonl:
        test_data = read_jsonl(args.eval_jsonl)
    else:
        split = min(128, len(train_data) // 5)
        test_data = train_data[:split]
        train_data = train_data[split:]
    from reference.utils import TRIE_TOKENIZER
    tok = TRIE_TOKENIZER(args.tokenizer)
    encode = lambda s: tok.encode(s)
    def safe_decode(ids):
        try:
            return tok.decode(ids, utf8_errors='replace')
        except Exception:
            try:
                return tok.decode(ids)
            except Exception:
                b = tok.decodeBytes(ids)
                return b.decode('utf-8', errors='replace')
    base_name, pth_path = normalize_model_arg(args.model)
    train_model, _ = load_train_model_rwkv7_cuda(pth_path, device=device, ctx_len=int(args.ctx_len))
    if args.state_init:
        load_time_state_only(train_model, args.state_init)
    import copy
    ref_model = copy.deepcopy(train_model)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False
    infer_model, _ = load_infer_model_albatross(base_name)
    trainable = freeze_except_time_state(train_model)
    if trainable <= 0:
        raise RuntimeError('??????time_state??')
    cast_trainable_time_state_fp32(train_model)
    cfg = DirectGRPOConfig(num_questions=int(args.num_questions), samples_per_question=int(args.samples_per_question), max_new_tokens=int(args.max_new_tokens), temperature=float(args.temperature), top_p=float(args.top_p), top_k=int(args.top_k), eval_temperature=float(args.eval_temperature), eval_top_p=float(args.eval_top_p), eval_top_k=int(args.eval_top_k), ppo_epochs=int(args.ppo_epochs), micro_batch=int(args.micro_batch), lr=float(args.lr), grad_clip=float(args.grad_clip), clip_eps=float(args.clip_eps), kl_coef=float(args.kl_coef), neg_adv_weight=float(args.neg_adv_weight), min_tokens=int(args.min_tokens), max_tokens=int(args.max_new_tokens), length_weight=float(args.length_weight), zstd_threshold=float(args.zstd_threshold), zstd_penalty_weight=float(args.zstd_penalty_weight), ngram_penalty=float(args.ngram_penalty), time_state_l2=float(args.time_state_l2), time_state_clamp=float(args.time_state_clamp), log_interval=int(args.log_interval), save_interval=int(args.save_interval), eval_interval=int(args.eval_interval), eval_sample_ratio=float(args.eval_sample_ratio))
    infer_engine = AlbatrossBatchInference(infer_model=infer_model, train_model=train_model, encode_fn=encode, decode_fn=safe_decode, device=device, cfg=cfg)
    trainer = DirectGRPOTrainer(train_model=train_model, ref_model=ref_model, infer_engine=infer_engine, encode_fn=encode, decode_fn=safe_decode, train_data=train_data, test_data=test_data, out_dir=args.out_dir, device=device, cfg=cfg, seed=int(args.seed))
    pre_acc = trainer.evaluate(step=0, tag='pre_eval', sample_ratio=1.0)
    if pre_acc is not None:
        print(f'[pre_eval] acc={pre_acc:.4f}')
    trainer.train(total_steps=int(args.total_steps))
    post_acc = trainer.evaluate(step=int(args.total_steps), tag='post_eval', sample_ratio=1.0)
    if post_acc is not None:
        print(f'[post_eval] acc={post_acc:.4f}')


if __name__ == '__main__':
    main()


