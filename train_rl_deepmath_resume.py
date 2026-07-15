#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import copy
import contextlib
import math
import time
import types
import argparse
import logging
import importlib.util
from pathlib import Path
from typing import List, Dict, Any, Optional

logging.basicConfig(level=logging.INFO)

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import deepspeed
from pytorch_lightning import Trainer

BASE_DIR = Path(__file__).resolve().parent
BASELINE_DIR = Path('/root/RWKV-LM/RWKV7-statetuning_hardbuffer_v1')

# src.model reads these at import time. Keep defaults compatible with stage3-offload smoke.
os.environ.setdefault('RWKV_MY_TESTING', 'x070')
os.environ.setdefault('RWKV_CTXLEN', '8192')
os.environ.setdefault('RWKV_HEAD_SIZE', '64')
os.environ.setdefault('RWKV_FLOAT_MODE', 'bf16')
os.environ.setdefault('RWKV_JIT_ON', '0')
for _ninja_dir in ['/root/miniconda3/bin', '/usr/bin', '/bin']:
    if os.path.isfile(os.path.join(_ninja_dir, 'ninja')) and _ninja_dir not in os.environ.get('PATH', ''):
        os.environ['PATH'] = _ninja_dir + os.pathsep + os.environ.get('PATH', '')
        break
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
if str(BASELINE_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINE_DIR))

from utils import read_jsonl, set_seed, append_jsonl
from infer import AlbatrossBatchInference
from reference.utils import TRIE_TOKENIZER
from reference.rwkv7 import RWKV_x070, RWKV_x070_TMix_seq_batch, RWKV_x070_CMix_seq_batch
from src.model import RWKV

_baseline_spec = importlib.util.spec_from_file_location('baseline_grpo_train', BASELINE_DIR / 'train.py')
_baseline_mod = importlib.util.module_from_spec(_baseline_spec)
_baseline_spec.loader.exec_module(_baseline_mod)
GRPOConfig = _baseline_mod.GRPOConfig
BaseGRPOTrainer = _baseline_mod.GRPOTrainer
compute_unbiased_kl = _baseline_mod.compute_unbiased_kl


class TrainTempBatchInference(AlbatrossBatchInference):
    def init_state_with_time_state(self, B: int):
        state = self.infer_model.generate_zero_state(B)
        blocks = getattr(self.train_model, 'blocks', None)
        if blocks is None:
            return state
        for i, block in enumerate(blocks):
            ts = getattr(getattr(block, 'att', None), 'time_state', None)
            if ts is None:
                continue
            state[1][i] = ts.unsqueeze(0).expand(B, -1, -1, -1).clone()
        return state

    @torch.no_grad()
    def generate_group_parallel(self, *args, **kwargs):
        prep = getattr(self.infer_model, 'prepare_stateful_rollout', None)
        if prep is not None:
            prep()
        return super().generate_group_parallel(*args, **kwargs)


def _torch_load_weights(path: str):
    try:
        return torch.load(path, map_location='cpu', weights_only=True)
    except TypeError:
        return torch.load(path, map_location='cpu')


def _normalize_state_dict(sd):
    load_keys = list(sd.keys())
    for k in load_keys:
        if k.startswith('_forward_module.'):
            sd[k.replace('_forward_module.', '')] = sd[k]
            del sd[k]
    return sd


def _infer_arch(sd):
    n_embd = sd['emb.weight'].shape[1]
    vocab_size = sd['emb.weight'].shape[0]
    n_layer = max(int(k.split('.')[1]) for k in sd if k.startswith('blocks.')) + 1
    dim_ffn = sd.get('blocks.0.ffn.key.weight', torch.zeros(n_embd * 4, n_embd)).shape[0]
    return n_layer, n_embd, vocab_size, dim_ffn


def _cast_ref_model_dtype(model):
    mode = os.environ.get('RWKV_FLOAT_MODE', 'bf16')
    if mode == 'fp16':
        return model.half()
    if mode == 'bf16':
        return model.bfloat16()
    return model.float()


def _rwkv_float_dtype():
    mode = os.environ.get('RWKV_FLOAT_MODE', 'bf16')
    if mode == 'fp16':
        return torch.float16
    if mode == 'bf16':
        return torch.bfloat16
    return torch.float32


def _rollout_kernel_dtype():
    mode = os.environ.get('RWKV_FLOAT_MODE', 'bf16')
    if mode == 'fp16':
        return torch.float16
    return torch.bfloat16


def _rollout_logit_dtype():
    return torch.float32


class DummyStepDataset(Dataset):
    def __init__(self, total_steps: int, micro_bsz: int):
        self.total_steps = int(total_steps)
        self.micro_bsz = int(max(1, micro_bsz))

    def __len__(self):
        return max(1, self.total_steps * self.micro_bsz)

    def __getitem__(self, idx):
        return torch.tensor(idx, dtype=torch.long)


class PaddedRWKV(RWKV):
    def forward(self, idx):
        orig_t = idx.size(1)
        pad_len = (-orig_t) % 16
        if pad_len:
            idx = F.pad(idx, (0, pad_len), value=0)
        logits = super().forward(idx)
        if pad_len:
            logits = logits[:, :orig_t, :].contiguous()
        return logits


class LightningGRPOTrainer(BaseGRPOTrainer):
    def __init__(
        self,
        pl_module,
        train_model,
        ref_model,
        infer_engine,
        encode_fn,
        decode_fn,
        train_data: List[Dict[str, Any]],
        test_data: List[Dict[str, Any]],
        out_dir: str,
        device: str,
        cfg: GRPOConfig,
        seed: int = 42,
        full_test_data: Optional[List[Dict[str, Any]]] = None,
    ):
        self.pl_module = pl_module
        self.model = train_model
        self.ref_model = ref_model
        self.infer = infer_engine
        self.encode = encode_fn
        self.decode = decode_fn
        self.train_data = train_data
        self.small_eval_data = test_data
        self.full_test_data = full_test_data if full_test_data is not None else test_data
        self.test_data = self.small_eval_data
        self.out_dir = out_dir
        self.device = device
        self.cfg = cfg
        self.rng = __import__('random').Random(seed)

        os.makedirs(out_dir, exist_ok=True)
        self.log_path = os.path.join(out_dir, 'train.log')
        self.metrics_path = os.path.join(out_dir, 'metrics.jsonl')
        self.responses_dir = os.path.join(out_dir, 'responses_by_step')
        self.eval_path = os.path.join(out_dir, 'eval.jsonl')
        self.eval_by_step_dir = os.path.join(out_dir, 'eval_by_step')
        os.makedirs(self.responses_dir, exist_ok=True)
        os.makedirs(self.eval_by_step_dir, exist_ok=True)

        self._preeval_map = None
        self._preeval_loaded = False
        self._ts_init = {}
        self._ref_time_state = []
        self._hard_buffer = []
        self._hard_buffer_map = {}
        self._hard_last_used = {}
        self._pending_extra_batch = None
        self.plot_script = str(BASELINE_DIR / 'plot_metrics.py')

    def evaluate(self, step: int, tag: str = 'eval', sample_ratio: float = 1.0):
        prev_test_data = self.test_data
        try:
            if tag == 'full_eval':
                self.test_data = self.full_test_data
            else:
                self.test_data = self.small_eval_data
            return super().evaluate(step=step, tag=tag, sample_ratio=sample_ratio)
        finally:
            self.test_data = prev_test_data


    def _extra_schedule(self, step: int) -> Dict[str, Any]:
        mode = str(getattr(self.cfg, 'extra_curriculum', 'off'))
        base = {
            'hard_ratio': 0.5,
            'extra_lr_scale': float(self.cfg.hard_buffer_extra_lr_scale),
            'hard_ttl': int(self.cfg.hard_buffer_ttl),
            'hard_cooldown': int(self.cfg.hard_buffer_cooldown),
            'disable_extra': False,
        }
        if mode == 'pure_hard':
            return {
                'hard_ratio': 1.0,
                'extra_lr_scale': float(self.cfg.hard_buffer_extra_lr_scale),
                'hard_ttl': int(self.cfg.hard_buffer_ttl),
                'hard_cooldown': int(self.cfg.hard_buffer_cooldown),
                'disable_extra': False,
            }
        if mode != 'staged_v1':
            return base
        if step <= 150:
            return {'hard_ratio': 1.0, 'extra_lr_scale': 0.5, 'hard_ttl': 10, 'hard_cooldown': 5, 'disable_extra': False}
        if step <= 250:
            return {'hard_ratio': 0.5, 'extra_lr_scale': 0.35, 'hard_ttl': 8, 'hard_cooldown': 5, 'disable_extra': False}
        if step <= 350:
            return {'hard_ratio': 0.25, 'extra_lr_scale': 0.2, 'hard_ttl': 4, 'hard_cooldown': 4, 'disable_extra': False}
        return {'hard_ratio': 0.0, 'extra_lr_scale': 0.0, 'hard_ttl': 2, 'hard_cooldown': 4, 'disable_extra': True}

    def _plot_metrics(self):
        out_plot = os.path.join(self.out_dir, 'metrics_plot.png')
        if os.path.isfile(self.plot_script) and os.path.isfile(self.metrics_path):
            try:
                import subprocess
                subprocess.run([sys.executable, self.plot_script, '--metrics', self.metrics_path, '--out', out_plot], check=False)
            except Exception:
                pass

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
                v = p.detach().float()
                mx = max(mx, float(v.abs().max().item()))
                rms_sum += float((v * v).mean().sqrt().item())
                cnt += 1
        return {'absmax': mx, 'rms_avg': (rms_sum / max(1, cnt)), 'bad': bad}

    def _get_pl_optimizer(self):
        opt = self.pl_module.optimizers()
        if isinstance(opt, (list, tuple)):
            opt = opt[0]
        raw_opt = getattr(opt, 'optimizer', opt)
        return opt, raw_opt

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

        pl_opt, raw_opt = self._get_pl_optimizer()
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

        base_lrs = [float(pg.get('lr', self.cfg.lr)) for pg in raw_opt.param_groups]
        if lr_scale != 1.0:
            for pg, base_lr in zip(raw_opt.param_groups, base_lrs):
                pg['lr'] = base_lr * lr_scale

        try:
            for _ in range(self.cfg.ppo_epochs):
                self.model.train()
                epoch_had_backward = False
                try:
                    pl_opt.zero_grad(set_to_none=True)
                except TypeError:
                    pl_opt.zero_grad()

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

                    logsumexp = torch.logsumexp(logits, dim=-1).float()
                    logp = self._logp_with_sampling(logits, tgt)

                    # Avoid materializing a full fp32 vocab tensor here. This block is only
                    # for logging entropy, so we stream top-k selection over vocab chunks.
                    top_k = min(500, logits.size(-1))
                    top_logits = None
                    vocab_chunk = 2048
                    for v_start in range(0, logits.size(-1), vocab_chunk):
                        chunk = logits[..., v_start:v_start + vocab_chunk].float()
                        k_chunk = min(top_k, chunk.size(-1))
                        chunk_top, _ = torch.topk(chunk, k=k_chunk, dim=-1)
                        if top_logits is None:
                            top_logits = chunk_top
                        else:
                            merged_top = torch.cat((top_logits, chunk_top), dim=-1)
                            top_logits, _ = torch.topk(merged_top, k=top_k, dim=-1)
                            del merged_top
                        del chunk, chunk_top
                    logp_top = top_logits - logsumexp.unsqueeze(-1)
                    p_top = torch.exp(logp_top)
                    entropy_per_token = -(p_top * logp_top).sum(dim=-1)
                    del top_logits, logp_top, p_top, logits
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    with torch.no_grad():
                        ref_logits = self.ref_model(inp)
                        if torch.is_tensor(ref_logits) and ref_logits.dim() == 2:
                            ref_logits = ref_logits.unsqueeze(0)
                        ref_logp_all = self._logp_with_sampling(ref_logits, tgt)
                        del ref_logits
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                    batch_loss = 0.0
                    batch_kl = 0.0
                    batch_entropy = 0.0
                    total_tokens = 0

                    for bi, traj in enumerate(batch):
                        prompt_len = len(traj['prompt_tokens'])
                        comp_len = len(traj['comp_tokens'])
                        start_idx = prompt_len - 1
                        end_idx = start_idx + comp_len
                        new_logp = logp[bi, start_idx:end_idx]
                        ref_logp = ref_logp_all[bi, start_idx:end_idx]
                        old_logp = torch.tensor(traj['old_logps'], device=self.device, dtype=torch.float32)
                        curr_entropy = entropy_per_token[bi, start_idx:end_idx]

                        min_len = min(new_logp.size(0), ref_logp.size(0), old_logp.size(0))
                        if min_len == 0:
                            continue

                        new_logp = new_logp[:min_len]
                        ref_logp = ref_logp[:min_len]
                        old_logp = old_logp[:min_len]
                        curr_entropy = curr_entropy[:min_len]
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

                        batch_entropy += curr_entropy.sum()
                        batch_loss += policy_loss
                        batch_kl += kl
                        total_tokens += min_len

                    if total_tokens > 0:
                        normalized_loss = batch_loss / valid_tokens
                        normalized_kl = batch_kl / valid_tokens
                        normalized_entropy = batch_entropy / valid_tokens
                        if self.cfg.kl_mode == 'k3_loss':
                            total_loss = normalized_loss + self.cfg.kl_coef * normalized_kl
                        else:
                            total_loss = normalized_loss

                        self.pl_module.manual_backward(total_loss)
                        epoch_had_backward = True
                        loss_total += normalized_loss.item()
                        kl_total += normalized_kl.item()
                        entropy_total += normalized_entropy.item()
                        batch_cnt += 1

                if not epoch_had_backward:
                    dummy = self.model.head.weight.view(-1)[0].float() * 0.0
                    self.pl_module.manual_backward(dummy)

                if self.cfg.grad_clip > 0:
                    try:
                        self.pl_module.clip_gradients(pl_opt, gradient_clip_val=self.cfg.grad_clip, gradient_clip_algorithm='norm')
                    except Exception:
                        torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad], self.cfg.grad_clip)

                with torch.no_grad():
                    g2 = 0.0
                    for p in self.model.parameters():
                        if p.requires_grad and p.grad is not None:
                            g = p.grad.detach().float()
                            g2 += (g.norm(2) ** 2).item()
                    grad_norm = math.sqrt(g2)

                pl_opt.step()
        finally:
            if lr_scale != 1.0:
                for pg, base_lr in zip(raw_opt.param_groups, base_lrs):
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
        t0 = time.time()
        pending_extra = getattr(self, '_pending_extra_batch', None)
        stage_cfg = self._extra_schedule(step)
        self.cfg.hard_buffer_ttl = int(stage_cfg.get('hard_ttl', self.cfg.hard_buffer_ttl))
        self.cfg.hard_buffer_cooldown = int(stage_cfg.get('hard_cooldown', self.cfg.hard_buffer_cooldown))
        disable_extra_step = bool(getattr(self.cfg, 'disable_extra_step', False)) or bool(stage_cfg.get('disable_extra', False))
        run_extra_only = (not disable_extra_step) and bool(pending_extra and pending_extra.get('items'))
        if pending_extra is not None and (disable_extra_step or run_extra_only):
            self._pending_extra_batch = None

        stats = {
            'total_samples': 0,
            'correct_samples': 0,
            'total_reward': 0.0,
            'total_length': 0,
            'total_trunc': 0,
            'total_repeat': 0,
            'no_answer': 0,
            'sum_correct_reward': 0.0,
            'sum_format_reward': 0.0,
            'sum_length_reward': 0.0,
            'sum_length_lambda': 0.0,
            'sum_zstd_penalty': 0.0,
            'sum_zstd_ratio': 0.0,
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
        adv_component_weights = {
            'correct_reward': 1.0,
            'format_reward': 1.0,
            'length_reward': min(0.25, abs(float(self.cfg.length_weight)) * 0.5),
            'repeat_penalty': abs(float(self.cfg.ngram_penalty)),
            'zstd_penalty': abs(float(self.cfg.zstd_penalty_weight)),
        }
        step_path = os.path.join(self.responses_dir, f'step_{step}.jsonl')
        step_type = 'extra' if run_extra_only else 'normal'

        def _encode_problem(problem: str):
            prompt = _baseline_mod.build_prompt(problem)
            ids = self.encode(prompt)
            max_prompt_len = int(self.model.args.ctx_len) - int(self.cfg.max_new_tokens) - 4
            max_prompt_len = max(64, max_prompt_len)
            if len(ids) > max_prompt_len:
                ids = ids[-max_prompt_len:]
            return ids

        def _make_group(q_idx: int, train_idx: int, problem: str, answer: str, prompt_tokens, target_samples: int, is_extra: bool, extra_source: str = 'hard'):
            return {
                'q_idx': q_idx,
                'train_idx': int(train_idx),
                'problem': problem,
                'answer': answer,
                'prompt_tokens': prompt_tokens,
                'group_rewards': [],
                'group_trajs': [],
                'correct_in_group': 0,
                'sampled': 0,
                'target_samples': int(target_samples),
                'is_extra': bool(is_extra),
                'extra_source': extra_source,
            }

        def _consume_sample(group, comp_tokens, old_logps, comp_text, truncated, is_extra: bool = False):
            repeat_flag = self._has_repeated_ngrams(comp_text, n=16, repeat=5)
            reward, is_correct, is_format_correct, reward_details = _baseline_mod.calculate_reward_details(
                text=comp_text,
                ground_truth=group['answer'],
                token_length=len(comp_tokens),
                min_tokens=self.cfg.min_tokens,
                max_tokens=self.cfg.max_tokens,
                length_weight=self.cfg.length_weight,
                repeat_ngram=repeat_flag,
                repeat_penalty=self.cfg.ngram_penalty,
                zstd_threshold=self.cfg.zstd_threshold,
                zstd_penalty_weight=self.cfg.zstd_penalty_weight,
            )
            sample_idx = group['sampled']
            group['sampled'] += 1
            reward_components = {
                'correct_reward': float(reward_details.get('correct_reward', 0.0)),
                'format_reward': float(reward_details.get('format_reward', 0.0)),
                'length_reward': float(reward_details.get('length_reward', 0.0)),
                'repeat_penalty': float(reward_details.get('repeat_penalty', 0.0)),
                'zstd_penalty': float(reward_details.get('zstd_penalty', 0.0)),
            }
            traj = {
                'prompt_tokens': group['prompt_tokens'],
                'comp_tokens': comp_tokens,
                'old_logps': old_logps,
                'is_extra': bool(is_extra),
                'reward': reward,
                'text': comp_text,
                'is_correct': is_correct,
                'is_format_correct': is_format_correct,
                'truncated': truncated,
                'reward_components': reward_components,
            }
            group['group_rewards'].append(reward)
            group['group_trajs'].append(traj)
            stats['total_samples'] += 1
            stats['total_reward'] += reward
            stats['total_length'] += len(comp_tokens)
            if is_correct:
                stats['correct_samples'] += 1
                group['correct_in_group'] += 1
            if truncated:
                stats['total_trunc'] += 1
            if repeat_flag:
                stats['total_repeat'] += 1
            if not reward_details.get('extracted_answer'):
                stats['no_answer'] += 1
            stats['sum_correct_reward'] += float(reward_details.get('correct_reward', 0.0))
            stats['sum_format_reward'] += float(reward_details.get('format_reward', 0.0))
            stats['sum_length_reward'] += float(reward_details.get('length_reward', 0.0))
            stats['sum_length_lambda'] += float(reward_details.get('length_lambda', 0.0))
            stats['sum_zstd_penalty'] += float(reward_details.get('zstd_penalty', 0.0))
            stats['sum_zstd_ratio'] += float(reward_details.get('zstd_ratio', 0.0))
            record = {
                'step': step,
                'step_type': step_type,
                'question_idx': group['q_idx'],
                'sample_idx': sample_idx,
                'problem': group['problem'],
                'ground_truth': group['answer'],
                'response': comp_text,
                'pred_extracted': reward_details.get('extracted_answer'),
                'gt_extracted': reward_details.get('ground_truth_answer'),
                'reward': reward,
                'is_correct': is_correct,
                'is_format_correct': is_format_correct,
                'truncated': truncated,
                'is_extra': bool(is_extra),
                'extra_source': group.get('extra_source'),
                'reward_details': reward_details,
            }
            append_jsonl(step_path, record)

        group_infos = []
        hard_buffer_enabled = int(self.cfg.hard_buffer_target_samples) > 0
        hard_buffer_added = 0
        hard_eligible = 0
        hard_selected = []
        hard_triggered = 0
        extra_group_size = max(1, int(self.cfg.hard_buffer_group_size))
        extra_lr_scale = float(stage_cfg.get('extra_lr_scale', self.cfg.hard_buffer_extra_lr_scale))
        extra_adv_clip = float(self.cfg.hard_buffer_adv_clip)

        if run_extra_only:
            queued_items = list(pending_extra.get('items', []))
            extra_group_size = max(1, int(pending_extra.get('group_size', extra_group_size)))
            hard_selected = queued_items
            hard_eligible = len(self._hard_buffer)
            for i, item in enumerate(queued_items):
                prompt_tokens = item.get('prompt_tokens') or _encode_problem(item.get('problem', ''))
                group_infos.append(_make_group(i, int(item['train_idx']), item.get('problem', ''), item.get('answer', ''), prompt_tokens, extra_group_size, True, str(item.get('extra_source', 'hard'))))
            comp_tokens_list, old_logps_list, comp_texts_list, truncated_list = self.infer.generate_group_parallel(
                prompt_tokens_list=[g['prompt_tokens'] for g in group_infos],
                group_size=extra_group_size,
                max_new_tokens=self.cfg.max_new_tokens,
                temperature=self.cfg.temperature,
                top_p=self.cfg.top_p,
                top_k=self.cfg.top_k,
            )
            for bi, group in enumerate(group_infos):
                base = bi * extra_group_size
                for j in range(extra_group_size):
                    flat_idx = base + j
                    if flat_idx >= len(comp_tokens_list):
                        continue
                    _consume_sample(group, comp_tokens_list[flat_idx], old_logps_list[flat_idx], comp_texts_list[flat_idx], truncated_list[flat_idx], is_extra=True)
                tid = int(group['train_idx'])
                source = str(group.get('extra_source', 'hard'))
                if source == 'hard':
                    if group['correct_in_group'] == 0:
                        item = self._hard_buffer_map.get(tid)
                        if item is not None:
                            item['ttl_left'] = int(item.get('ttl_left', max(1, int(self.cfg.hard_buffer_ttl)))) - 1
                            if int(item.get('ttl_left', 0)) <= 0:
                                self._remove_hard_candidate(tid)
                    else:
                        self._remove_hard_candidate(tid)
                elif group['correct_in_group'] == 0:
                    self._push_hard_candidate(train_idx=tid, prompt_tokens=group['prompt_tokens'], problem=group['problem'], answer=group['answer'], step=step, ignore_cooldown=False)
        else:
            if len(self.train_data) >= self.cfg.num_questions:
                sampled_indices = self.rng.sample(range(len(self.train_data)), self.cfg.num_questions)
            else:
                sampled_indices = [self.rng.randrange(len(self.train_data)) for _ in range(self.cfg.num_questions)]
            sampled_questions = [self.train_data[i] for i in sampled_indices]
            prompt_tokens_list = [_encode_problem(q.get('problem', '')) for q in sampled_questions]
            for q_idx, (question, train_idx) in enumerate(zip(sampled_questions, sampled_indices)):
                group_infos.append(_make_group(q_idx, int(train_idx), question.get('problem', ''), self._get_answer(question), prompt_tokens_list[q_idx], self.cfg.samples_per_question, False, 'normal'))
            comp_tokens_list, old_logps_list, comp_texts_list, truncated_list = self.infer.generate_group_parallel(
                prompt_tokens_list=prompt_tokens_list,
                group_size=self.cfg.samples_per_question,
                max_new_tokens=self.cfg.max_new_tokens,
                temperature=self.cfg.temperature,
                top_p=self.cfg.top_p,
                top_k=self.cfg.top_k,
            )
            for q_idx, group in enumerate(group_infos):
                for i in range(self.cfg.samples_per_question):
                    flat_idx = q_idx * self.cfg.samples_per_question + i
                    if flat_idx >= len(comp_tokens_list):
                        continue
                    _consume_sample(group, comp_tokens_list[flat_idx], old_logps_list[flat_idx], comp_texts_list[flat_idx], truncated_list[flat_idx], is_extra=False)
            if hard_buffer_enabled:
                for group in group_infos:
                    if group['correct_in_group'] == 0:
                        added = self._push_hard_candidate(train_idx=int(group['train_idx']), prompt_tokens=group['prompt_tokens'], problem=group['problem'], answer=group['answer'], step=step)
                        if added:
                            hard_buffer_added += 1
            if hard_buffer_enabled and (not disable_extra_step):
                extra_target_samples = max(self.cfg.samples_per_question, int(self.cfg.hard_buffer_target_samples))
                needed_questions = max(1, int(math.ceil(float(extra_target_samples) / float(extra_group_size))))
                hard_ratio = max(0.0, min(1.0, float(stage_cfg.get('hard_ratio', 0.5))))
                hard_quota = int(round(needed_questions * hard_ratio))
                if hard_ratio > 0.0 and needed_questions > 0 and hard_quota <= 0:
                    hard_quota = 1
                hard_selected, hard_eligible = (self._pop_hard_batch(step, hard_quota) if hard_quota > 0 else ([], len(self._hard_buffer)))
                hard_triggered = int(len(hard_selected) > 0)
                if hard_triggered:
                    hard_items = [
                        {
                            'train_idx': int(item['train_idx']),
                            'problem': item['problem'],
                            'answer': item['answer'],
                            'prompt_tokens': item['prompt_tokens'],
                            'extra_source': 'hard',
                        }
                        for item in hard_selected
                    ]
                    random_needed = max(0, needed_questions - len(hard_items))
                    random_items = []
                    if random_needed > 0 and self.train_data:
                        hard_train_ids = {int(item['train_idx']) for item in hard_items}
                        candidate_indices = [i for i in range(len(self.train_data)) if i not in hard_train_ids]
                        if len(candidate_indices) >= random_needed:
                            random_indices = self.rng.sample(candidate_indices, random_needed)
                        else:
                            random_indices = list(candidate_indices)
                            while len(random_indices) < random_needed:
                                random_indices.append(self.rng.randrange(len(self.train_data)))
                        for train_idx in random_indices[:random_needed]:
                            question = self.train_data[int(train_idx)]
                            problem = question.get('problem', '')
                            random_items.append({
                                'train_idx': int(train_idx),
                                'problem': problem,
                                'answer': self._get_answer(question),
                                'prompt_tokens': _encode_problem(problem),
                                'extra_source': 'random',
                            })
                    self._pending_extra_batch = {
                        'queued_at_step': int(step),
                        'group_size': int(extra_group_size),
                        'items': hard_items + random_items,
                    }
                else:
                    self._pending_extra_batch = None
            else:
                hard_selected, hard_eligible = [], 0
                hard_triggered = 0
                self._pending_extra_batch = None

        all_group_trajs = []
        for group in group_infos:
            all_group_trajs.extend(group.get('group_trajs', []))
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
                    k1_seq = (old_sum - ref_sum) / float(min_len)
                    traj['k1_seq'] = k1_seq
                    traj['reward_components']['correct_reward'] += (-float(self.cfg.kl_coef) * k1_seq)

        groups_total = len(group_infos)
        all_trajs = []
        for group in group_infos:
            is_extra_group = bool(group.get('is_extra', False))
            if is_extra_group:
                extra_groups_total += 1
            group_all_wrong = (group['correct_in_group'] == 0)
            group_all_correct = (group['correct_in_group'] == group.get('target_samples', self.cfg.samples_per_question))
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
            if not group['group_rewards']:
                continue
            n = len(group['group_trajs'])
            comp_stats = {}
            for key in adv_component_keys:
                vals = [float(traj.get('reward_components', {}).get(key, 0.0)) for traj in group['group_trajs']]
                if not vals:
                    comp_stats[key] = (0.0, 0.0)
                    continue
                mean_v = sum(vals) / len(vals)
                var_v = sum((v - mean_v) ** 2 for v in vals) / len(vals)
                std_v = math.sqrt(var_v) if var_v > 1e-6 else 0.0
                comp_stats[key] = (mean_v, std_v)
            c = group['correct_in_group']
            acc_weight = _baseline_mod.calculate_pass_at_k(n, c, 1) * self.cfg.samples_per_question
            acc_weight = max(1e-6, float(acc_weight))
            weight_norm = max(1e-6, float(sum(abs(w) for w in adv_component_weights.values())))
            for traj in group['group_trajs']:
                decoupled_adv = 0.0
                comps = traj.get('reward_components', {})
                for key in adv_component_keys:
                    w = float(adv_component_weights.get(key, 0.0))
                    if w == 0.0:
                        continue
                    mean_v, std_v = comp_stats.get(key, (0.0, 0.0))
                    rel_adv = 0.0 if std_v <= 1e-6 else (float(comps.get(key, 0.0)) - mean_v) / std_v
                    decoupled_adv += w * rel_adv
                traj['advantage'] = (decoupled_adv / weight_norm) / acc_weight
                adv_values.append(traj['advantage'])
            all_trajs.extend(group['group_trajs'])

        opt_lr_scale = extra_lr_scale if run_extra_only else 1.0
        opt_adv_clip = extra_adv_clip if run_extra_only else None
        opt_stats = self._optimize_trajs(all_trajs, lr_scale=opt_lr_scale, adv_clip=opt_adv_clip)
        loss_total = float(opt_stats['loss_total'])
        kl_total = float(opt_stats['kl_total'])
        entropy_total = float(opt_stats['entropy_total'])
        batch_cnt = int(opt_stats['batch_cnt'])
        clip_total = float(opt_stats['clip_total'])
        clip_total_tokens = int(opt_stats['clip_total_tokens'])
        grad_norm = float(opt_stats['grad_norm'])

        dt = time.time() - t0
        avg_reward = stats['total_reward'] / max(1, stats['total_samples'])
        avg_length = stats['total_length'] / max(1, stats['total_samples'])
        accuracy = stats['correct_samples'] / max(1, stats['total_samples'])
        trunc_rate = stats['total_trunc'] / max(1, stats['total_samples'])
        repeat_rate = stats['total_repeat'] / max(1, stats['total_samples'])
        no_answer_rate = stats['no_answer'] / max(1, stats['total_samples'])
        avg_correct_reward = stats['sum_correct_reward'] / max(1, stats['total_samples'])
        avg_format_reward = stats['sum_format_reward'] / max(1, stats['total_samples'])
        avg_length_reward = stats['sum_length_reward'] / max(1, stats['total_samples'])
        avg_length_lambda = stats['sum_length_lambda'] / max(1, stats['total_samples'])
        avg_zstd_penalty = stats['sum_zstd_penalty'] / max(1, stats['total_samples'])
        avg_zstd_ratio = stats['sum_zstd_ratio'] / max(1, stats['total_samples'])
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
        samples_per_sec = stats['total_samples'] / dt if dt > 0 else 0.0
        tokens_per_sec = stats['total_length'] / dt if dt > 0 else 0.0
        ts_stats = self._time_state_stats()
        queued_extra = getattr(self, '_pending_extra_batch', None)
        queued_extra_questions = len(queued_extra.get('items', [])) if queued_extra else 0
        metrics = {
            'step': step,
            'split': 'train',
            'step_type': step_type,
            'samples': stats['total_samples'],
            'accuracy': accuracy,
            'avg_reward': avg_reward,
            'avg_length': avg_length,
            'trunc_rate': trunc_rate,
            'repeat_rate': repeat_rate,
            'no_answer_rate': no_answer_rate,
            'avg_correct_reward': avg_correct_reward,
            'avg_format_reward': avg_format_reward,
            'avg_length_reward': avg_length_reward,
            'avg_length_lambda': avg_length_lambda,
            'adv_mean': adv_mean,
            'adv_std': adv_std,
            'pos_adv_ratio': pos_adv_ratio,
            'neg_adv_ratio': neg_adv_ratio,
            'groups_total': groups_total,
            'groups_used': groups_used,
            'groups_skipped': groups_skipped,
            'groups_all_correct': groups_all_correct,
            'groups_all_wrong': groups_all_wrong,
            'loss': loss_total,
            'kl': kl_total,
            'avg_kl': avg_kl,
            'clip_frac': clip_frac,
            'grad_norm': grad_norm,
            'time': dt,
            'samples_per_sec': samples_per_sec,
            'tokens_per_sec': tokens_per_sec,
            'ts_absmax': ts_stats['absmax'],
            'ts_rms': ts_stats['rms_avg'],
            'ts_bad': ts_stats['bad'],
            'avg_entropy': entropy_total,
            'avg_zstd_penalty': avg_zstd_penalty,
            'avg_zstd_ratio': avg_zstd_ratio,
            'hard_buffer_size': len(self._hard_buffer),
            'hard_buffer_added': hard_buffer_added,
            'hard_buffer_eligible': hard_eligible,
            'hard_buffer_selected': len(hard_selected),
            'hard_buffer_triggered': int(hard_triggered),
            'extra_step_ran': int(run_extra_only),
            'extra_samples': (stats['total_samples'] if run_extra_only else 0),
            'extra_groups_total': extra_groups_total,
            'extra_groups_used': extra_groups_used,
            'extra_groups_skipped': extra_groups_skipped,
            'extra_groups_all_correct': extra_groups_all_correct,
            'extra_groups_all_wrong': extra_groups_all_wrong,
            'extra_loss': (float(opt_stats['loss_total']) if run_extra_only else 0.0),
            'extra_avg_kl': ((float(opt_stats['kl_total']) / max(1, int(opt_stats['batch_cnt']))) if run_extra_only else 0.0),
            'extra_grad_norm': (float(opt_stats['grad_norm']) if run_extra_only else 0.0),
            'extra_lr_scale': (opt_lr_scale if run_extra_only else 0.0),
            'extra_adv_clip': (float(opt_adv_clip) if run_extra_only and opt_adv_clip is not None else 0.0),
            'queued_extra_questions': queued_extra_questions,
        }
        return metrics


class RWKVGRPOModel(PaddedRWKV):
    def __init__(self, args, rl_cfg, train_data, test_data, full_test_data=None):
        super().__init__(args)
        self.args = args
        self.rl_cfg = rl_cfg
        self.train_data_json = train_data
        self.test_data_json = test_data
        self.full_test_data_json = full_test_data if full_test_data is not None else test_data
        self.automatic_optimization = False
        self.fit_start_time = None
        self.__dict__['_ref_model'] = None
        self.__dict__['_rollout_model'] = None
        self.__dict__['_rl_trainer'] = None
        self._optimizer_bootstrapped = False
        self._best_full_eval_acc = float('-inf')
        self._full_eval_bad_count = 0
        self._load_from_checkpoint_path(args.load_model)

    def configure_optimizers(self):
        if int(getattr(self.args, 'quiet_optimizer_log', 1)) == 1:
            with open(os.devnull, 'w') as devnull, contextlib.redirect_stdout(devnull):
                return super().configure_optimizers()
        return super().configure_optimizers()

    def _load_from_checkpoint_path(self, path: str):
        sd = _normalize_state_dict(_torch_load_weights(path))
        self.load_state_dict(sd, strict=True)

    def _save_checkpoint_to(self, out_path: str) -> str:
        state = {}
        for name, param in self.named_parameters():
            with deepspeed.zero.GatheredParameters([param], modifier_rank=None):
                state[name] = param.detach().cpu().clone()
        for name, buf in self.named_buffers():
            state[name] = buf.detach().cpu().clone()
        torch.save(state, out_path)
        return out_path

    def _save_final_checkpoint(self, step: int) -> str:
        out_path = os.path.join(self.args.proj_dir, f'final_step_{int(step)}.pth')
        return self._save_checkpoint_to(out_path)

    def _save_step_checkpoint(self, step: int) -> str:
        out_path = os.path.join(self.args.proj_dir, f'checkpoint_step_{int(step)}.pth')
        return self._save_checkpoint_to(out_path)

    def _mem_stats(self, reset_peak: bool = False):
        if not torch.cuda.is_available():
            return None
        if reset_peak:
            torch.cuda.reset_peak_memory_stats()
        alloc = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        peak_alloc = torch.cuda.max_memory_allocated() / (1024 ** 2)
        peak_reserved = torch.cuda.max_memory_reserved() / (1024 ** 2)
        return {
            'alloc_mb': alloc,
            'reserved_mb': reserved,
            'peak_alloc_mb': peak_alloc,
            'peak_reserved_mb': peak_reserved,
        }

    def _log_mem(self, tag: str, reset_peak: bool = False):
        stats = self._mem_stats(reset_peak=reset_peak)
        if stats is None or self._rl_trainer is None:
            return
        self._rl_trainer._log(
            f"[MEM {tag}] alloc={stats['alloc_mb']:.1f}MB reserved={stats['reserved_mb']:.1f}MB peak_alloc={stats['peak_alloc_mb']:.1f}MB peak_reserved={stats['peak_reserved_mb']:.1f}MB"
        )

    def _move_rollout_model(self, device: str):
        if self._rollout_model is None:
            return
        target = torch.device(device)
        try:
            src = next(self._rollout_model.parameters()).device
        except StopIteration:
            src = target
        if src == target:
            return
        self._rollout_model.to(target)
        self._rollout_model.eval()
        if target.type == 'cpu' and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _move_ref_model(self, device: str):
        if self._ref_model is None:
            return
        target = torch.device(device)
        try:
            src = next(self._ref_model.parameters()).device
        except StopIteration:
            src = target
        if src == target:
            return
        self._ref_model.to(target)
        self._ref_model.eval()
        if target.type == 'cpu' and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _prepare_rollout_model(self):
        self._move_rollout_model(str(self.device))

    def _offload_rollout_model(self):
        self._move_rollout_model('cpu')

    def _prepare_ref_model(self):
        self._move_ref_model(str(self.device))

    def _offload_ref_model(self):
        self._move_ref_model('cpu')

    def _bootstrap_optimizer_states(self):
        if self._optimizer_bootstrapped:
            return
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        opt = self.optimizers()
        if isinstance(opt, (list, tuple)):
            opt = opt[0]
        raw_opt = getattr(opt, 'optimizer', opt)
        base_lrs = [float(pg.get('lr', 0.0)) for pg in raw_opt.param_groups]
        try:
            for pg in raw_opt.param_groups:
                pg['lr'] = 0.0
            self.train()
            try:
                opt.zero_grad(set_to_none=True)
            except TypeError:
                opt.zero_grad()
            seq_len = 16
            idx = torch.zeros((1, seq_len), device=self.device, dtype=torch.long)
            logits = self(idx)
            if torch.is_tensor(logits) and logits.dim() == 2:
                logits = logits.unsqueeze(0)
            loss = logits.float().sum() * 0.0
            self.manual_backward(loss)
            opt.step()
            try:
                opt.zero_grad(set_to_none=True)
            except TypeError:
                opt.zero_grad()
            for st in raw_opt.state.values():
                if not isinstance(st, dict):
                    continue
                if 'step' in st:
                    if torch.is_tensor(st['step']):
                        st['step'].zero_()
                    else:
                        st['step'] = 0
                if 'exp_avg' in st and torch.is_tensor(st['exp_avg']):
                    st['exp_avg'].zero_()
                if 'exp_avg_sq' in st and torch.is_tensor(st['exp_avg_sq']):
                    st['exp_avg_sq'].zero_()
        finally:
            for pg, base_lr in zip(raw_opt.param_groups, base_lrs):
                pg['lr'] = base_lr
        self._optimizer_bootstrapped = True

    def _build_tokenizer(self):
        tok = TRIE_TOKENIZER(self.args.tokenizer)
        encode = lambda s: tok.encode(s)
        def safe_decode(ids):
            try:
                return tok.decode(ids, utf8_errors='replace')
            except Exception:
                try:
                    return tok.decode(ids)
                except Exception:
                    try:
                        b = tok.decodeBytes(ids)
                        return b.decode('utf-8', errors='replace')
                    except Exception:
                        return ''.join(chr(int(x) % 256) for x in ids)
        return encode, safe_decode

    def _clone_zero_param(self, param, *, dtype=None, transpose=False, flatten=False, squeeze=True):
        if dtype is None:
            dtype = _rollout_kernel_dtype()
        ds_status = getattr(param, 'ds_status', None)
        if ds_status is not None and 'INFLIGHT' in str(ds_status):
            x = param.detach()
            if transpose:
                x = x.t()
            if squeeze:
                x = x.squeeze()
            if flatten:
                x = x.flatten()
            return x.to(device=self.device, dtype=dtype).contiguous()
        with deepspeed.zero.GatheredParameters([param], modifier_rank=None):
            x = param.detach()
            if transpose:
                x = x.t()
            if squeeze:
                x = x.squeeze()
            if flatten:
                x = x.flatten()
            return x.to(device=self.device, dtype=dtype).contiguous()

    def prepare_stateful_rollout(self):
        if getattr(self, '_stateful_rollout_cache', None) is not None:
            return
        dtype = _rwkv_float_dtype()
        z = {}
        ln0 = self.blocks[0].ln0
        emb = self._clone_zero_param(self.emb.weight, dtype=dtype)
        ln0_w = self._clone_zero_param(ln0.weight, dtype=dtype)
        ln0_b = self._clone_zero_param(ln0.bias, dtype=dtype)
        z['emb.weight'] = F.layer_norm(emb, (self.args.n_embd,), weight=ln0_w, bias=ln0_b)
        for i, block in enumerate(self.blocks):
            bbb = f'blocks.{i}.'
            att = f'{bbb}att.'
            ffn = f'{bbb}ffn.'
            z[bbb+'ln1.weight'] = self._clone_zero_param(block.ln1.weight, dtype=dtype)
            z[bbb+'ln1.bias'] = self._clone_zero_param(block.ln1.bias, dtype=dtype)
            z[bbb+'ln2.weight'] = self._clone_zero_param(block.ln2.weight, dtype=dtype)
            z[bbb+'ln2.bias'] = self._clone_zero_param(block.ln2.bias, dtype=dtype)
            a = block.att
            z[att+'x_r'] = self._clone_zero_param(a.x_r, dtype=dtype)
            z[att+'x_w'] = self._clone_zero_param(a.x_w, dtype=dtype)
            z[att+'x_k'] = self._clone_zero_param(a.x_k, dtype=dtype)
            z[att+'x_v'] = self._clone_zero_param(a.x_v, dtype=dtype)
            z[att+'x_a'] = self._clone_zero_param(a.x_a, dtype=dtype)
            z[att+'x_g'] = self._clone_zero_param(a.x_g, dtype=dtype)
            z[att+'w0'] = self._clone_zero_param(a.w0, dtype=dtype)
            z[att+'w1'] = self._clone_zero_param(a.w1, dtype=dtype)
            z[att+'w2'] = self._clone_zero_param(a.w2, dtype=dtype)
            z[att+'a0'] = self._clone_zero_param(a.a0, dtype=dtype)
            z[att+'a1'] = self._clone_zero_param(a.a1, dtype=dtype)
            z[att+'a2'] = self._clone_zero_param(a.a2, dtype=dtype)
            z[att+'v0'] = self._clone_zero_param(a.v0, dtype=dtype)
            z[att+'v1'] = self._clone_zero_param(a.v1, dtype=dtype)
            z[att+'v2'] = self._clone_zero_param(a.v2, dtype=dtype)
            z[att+'g1'] = self._clone_zero_param(a.g1, dtype=dtype)
            z[att+'g2'] = self._clone_zero_param(a.g2, dtype=dtype)
            z[att+'k_k'] = self._clone_zero_param(a.k_k, dtype=dtype)
            z[att+'k_a'] = self._clone_zero_param(a.k_a, dtype=dtype)
            z[att+'r_k'] = self._clone_zero_param(a.r_k, dtype=dtype, flatten=True)
            z[att+'receptance.weight'] = self._clone_zero_param(a.receptance.weight, dtype=dtype, transpose=True)
            z[att+'key.weight'] = self._clone_zero_param(a.key.weight, dtype=dtype, transpose=True)
            z[att+'value.weight'] = self._clone_zero_param(a.value.weight, dtype=dtype, transpose=True)
            z[att+'output.weight'] = self._clone_zero_param(a.output.weight, dtype=dtype, transpose=True)
            z[att+'ln_x.weight'] = self._clone_zero_param(a.ln_x.weight, dtype=dtype)
            z[att+'ln_x.bias'] = self._clone_zero_param(a.ln_x.bias, dtype=dtype)
            f = block.ffn
            z[ffn+'x_k'] = self._clone_zero_param(f.x_k, dtype=dtype)
            z[ffn+'key.weight'] = self._clone_zero_param(f.key.weight, dtype=dtype, transpose=True)
            z[ffn+'value.weight'] = self._clone_zero_param(f.value.weight, dtype=dtype, transpose=True)
        z['ln_out.weight'] = self._clone_zero_param(self.ln_out.weight, dtype=torch.float32)
        z['ln_out.bias'] = self._clone_zero_param(self.ln_out.bias, dtype=torch.float32)
        z['head.weight'] = self._clone_zero_param(self.head.weight, dtype=torch.float32, transpose=True)
        self._stateful_rollout_cache = z

    def cleanup_stateful_rollout(self):
        if getattr(self, '_stateful_rollout_cache', None) is not None:
            self._stateful_rollout_cache = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def generate_zero_state(self, bsz):
        args = self.args
        dev = self.device
        state = [None, None]
        state[0] = torch.zeros((args.n_layer, 2, bsz, args.n_embd), dtype=_rollout_kernel_dtype(), requires_grad=False, device=dev)
        state[1] = torch.zeros((args.n_layer, bsz, args.n_embd // args.head_size, args.head_size, args.head_size), dtype=torch.float32, requires_grad=False, device=dev)
        return state

    @torch.no_grad()
    def _forward_batch_same_length_stateful(self, tokens, state, full_output=False):
        z = self._stateful_rollout_cache
        idx = torch.tensor(tokens, device=self.device, dtype=torch.long)
        x = z['emb.weight'][idx]
        v_first = torch.empty_like(x)
        for i in range(self.args.n_layer):
            bbb = f'blocks.{i}.'
            att = f'{bbb}att.'
            ffn = f'{bbb}ffn.'
            xx = F.layer_norm(x, (self.args.n_embd,), weight=z[bbb+'ln1.weight'], bias=z[bbb+'ln1.bias'])
            xx, v_first = RWKV_x070_TMix_seq_batch(i, self.blocks[i].att.n_head, self.blocks[i].att.head_size, xx, state[0][i], v_first, state[1][i],
                z[att+'x_r'], z[att+'x_w'], z[att+'x_k'], z[att+'x_v'], z[att+'x_a'], z[att+'x_g'],
                z[att+'w0'], z[att+'w1'], z[att+'w2'], z[att+'a0'], z[att+'a1'], z[att+'a2'], z[att+'v0'], z[att+'v1'], z[att+'v2'],
                z[att+'g1'], z[att+'g2'], z[att+'k_k'], z[att+'k_a'], z[att+'r_k'],
                z[att+'receptance.weight'], z[att+'key.weight'], z[att+'value.weight'], z[att+'output.weight'],
                z[att+'ln_x.weight'], z[att+'ln_x.bias'])
            x = x + xx
            xx = F.layer_norm(x, (self.args.n_embd,), weight=z[bbb+'ln2.weight'], bias=z[bbb+'ln2.bias'])
            xx = RWKV_x070_CMix_seq_batch(xx, state[0][i], z[ffn+'x_k'], z[ffn+'key.weight'], z[ffn+'value.weight'])
            x = x + xx
        if not full_output:
            x = x[:, -1, :]
        x = F.layer_norm(x.float(), (self.args.n_embd,), weight=z['ln_out.weight'], bias=z['ln_out.bias'])
        x = x @ z['head.weight']
        return x

    @torch.no_grad()
    def forward_batch(self, tokens, state, full_output=False):
        lengths = [len(x) for x in tokens]
        if len(set(lengths)) == 1:
            return self._forward_batch_same_length_stateful(tokens, state, full_output=full_output)
        bsz = len(tokens)
        pos = [0] * bsz
        z = self._stateful_rollout_cache
        if not full_output:
            out = torch.empty((bsz, self.args.vocab_size), dtype=_rollout_logit_dtype(), requires_grad=False, device=self.device)
        else:
            out = [torch.empty((0, self.args.vocab_size), dtype=_rollout_logit_dtype(), requires_grad=False, device=self.device) for _ in range(bsz)]
        while True:
            active = [i for i in range(bsz) if pos[i] < len(tokens[i])]
            if not active:
                break
            min_len = min(len(tokens[i]) - pos[i] for i in active)
            batch_tokens = [tokens[i][pos[i]:pos[i]+min_len] for i in active]
            batch_state = [state[0][:, :, active], state[1][:, active]]
            new_out = self._forward_batch_same_length_stateful(batch_tokens, batch_state, full_output=full_output)
            for k, i in enumerate(active):
                pos[i] += min_len
                state[0][:, :, i] = batch_state[0][:, :, k]
                state[1][:, i] = batch_state[1][:, k]
                if not full_output:
                    out[i] = new_out[k]
                else:
                    out[i] = torch.cat((out[i], new_out[k]), dim=0)
        return out

    def _build_ref_model(self):
        ref_args = copy.deepcopy(self.args)
        ref_model = PaddedRWKV(ref_args)
        ref_model.load_state_dict(_normalize_state_dict(_torch_load_weights(self.args.load_model)), strict=True)
        ref_model = _cast_ref_model_dtype(ref_model.to(self.device))
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False
        return ref_model

    def _build_stateful_rollout_model(self):
        rollout_args = types.SimpleNamespace(
            MODEL_NAME=str(Path(self.args.load_model).with_suffix('')),
            vocab_size=int(self.args.vocab_size),
        )
        rollout_model = RWKV_x070(rollout_args)
        return rollout_model

    def _build_rollout_model(self):
        rollout_args = copy.deepcopy(self.args)
        rollout_args.grad_cp = 0
        rollout_model = PaddedRWKV(rollout_args)
        rollout_model.load_state_dict(_normalize_state_dict(_torch_load_weights(self.args.load_model)), strict=True)
        rollout_model = _cast_ref_model_dtype(rollout_model.to(self.device))
        rollout_model.eval()
        for p in rollout_model.parameters():
            p.requires_grad = False
        return rollout_model

    @torch.no_grad()
    def _debug_rollout_sync_check(self, step: int):
        if self._rollout_model is None or not hasattr(self._rollout_model, 'z'):
            return
        if self._rl_trainer is None or not self._rl_trainer.train_data:
            return
        try:
            q = self._rl_trainer.train_data[0]
            prompt = _baseline_mod.build_prompt(q.get('problem', ''))
            seq = self._rl_trainer.encode(prompt) + self._rl_trainer.encode(' Let us solve this step by step.')[:24]
            if len(seq) < 2:
                return
            inp = torch.tensor([seq[:-1]], dtype=torch.long, device=self.device)
            tgt = torch.tensor(seq[1:], dtype=torch.long, device=self.device)
            logits_t = self(inp)
            if torch.is_tensor(logits_t) and logits_t.dim() == 2:
                logits_t = logits_t.unsqueeze(0)
            train_lp = F.log_softmax(logits_t.float(), dim=-1).gather(-1, tgt.view(1, -1, 1)).squeeze(0).squeeze(-1)
            state = self._rollout_model.generate_zero_state(1)
            logits_r = self._rollout_model.forward_batch([seq[:-1]], state, full_output=True)
            if isinstance(logits_r, list):
                logits_r = logits_r[0].unsqueeze(0)
            if torch.is_tensor(logits_r) and logits_r.dim() == 2:
                logits_r = logits_r.unsqueeze(0)
            roll_lp = F.log_softmax(logits_r.float(), dim=-1).gather(-1, tgt.view(1, -1, 1)).squeeze(0).squeeze(-1)
            min_len = min(train_lp.numel(), roll_lp.numel())
            diff = (train_lp[:min_len] - roll_lp[:min_len]).abs()
            self._rl_trainer._log(
                f'[SYNC_CHECK step {step}] n={min_len} mean_abs={diff.mean().item():.6f} '
                f'max_abs={diff.max().item():.6f} train_mean={train_lp[:min_len].mean().item():.6f} '
                f'roll_mean={roll_lp[:min_len].mean().item():.6f}'
            )
            z = self._rollout_model.z
            head_src = self._clone_zero_param(self.head.weight, transpose=True)
            head_diff = (head_src - z['head.weight']).abs().max().item()
            emb_src = F.layer_norm(
                self._clone_zero_param(self.emb.weight),
                (self.args.n_embd,),
                weight=self._clone_zero_param(self.blocks[0].ln0.weight),
                bias=self._clone_zero_param(self.blocks[0].ln0.bias),
            )
            emb_diff = (emb_src - z['emb.weight']).abs().max().item()
            xr_src = self._clone_zero_param(self.blocks[0].att.x_r)
            xr_diff = (xr_src - z['blocks.0.att.x_r']).abs().max().item()
            self._rl_trainer._log(
                f'[SYNC_PARAM step {step}] head_max={head_diff:.6f} emb_max={emb_diff:.6f} xr_max={xr_diff:.6f} '
                f"head_src00={float(head_src.flatten()[0]):.6f} head_roll00={float(z['head.weight'].flatten()[0]):.6f}"
            )
            del inp, tgt, logits_t, logits_r, train_lp, roll_lp, diff
        except Exception as e:
            if self._rl_trainer is not None:
                self._rl_trainer._log(f'[SYNC_CHECK step {step}] ERROR {type(e).__name__}: {e}')

    def _sync_rollout_model(self):
        if self._rollout_model is None:
            return
        with torch.no_grad():
            if hasattr(self._rollout_model, 'z'):
                z = self._rollout_model.z
                dtype = _rwkv_float_dtype()

                def _copy_z(name, src):
                    src = src.contiguous()
                    if name in z and torch.is_tensor(z[name]) and tuple(z[name].shape) == tuple(src.shape):
                        z[name].copy_(src)
                    else:
                        z[name] = src

                ln0 = self.blocks[0].ln0
                emb = self._clone_zero_param(self.emb.weight, dtype=dtype)
                ln0_w = self._clone_zero_param(ln0.weight, dtype=dtype)
                ln0_b = self._clone_zero_param(ln0.bias, dtype=dtype)
                _copy_z('emb.weight', F.layer_norm(emb, (self.args.n_embd,), weight=ln0_w, bias=ln0_b))
                for i, block in enumerate(self.blocks):
                    bbb = f'blocks.{i}.'
                    att = f'{bbb}att.'
                    ffn = f'{bbb}ffn.'
                    _copy_z(bbb+'ln1.weight', self._clone_zero_param(block.ln1.weight, dtype=dtype))
                    _copy_z(bbb+'ln1.bias', self._clone_zero_param(block.ln1.bias, dtype=dtype))
                    _copy_z(bbb+'ln2.weight', self._clone_zero_param(block.ln2.weight, dtype=dtype))
                    _copy_z(bbb+'ln2.bias', self._clone_zero_param(block.ln2.bias, dtype=dtype))
                    a = block.att
                    _copy_z(att+'x_r', self._clone_zero_param(a.x_r, dtype=dtype))
                    _copy_z(att+'x_w', self._clone_zero_param(a.x_w, dtype=dtype))
                    _copy_z(att+'x_k', self._clone_zero_param(a.x_k, dtype=dtype))
                    _copy_z(att+'x_v', self._clone_zero_param(a.x_v, dtype=dtype))
                    _copy_z(att+'x_a', self._clone_zero_param(a.x_a, dtype=dtype))
                    _copy_z(att+'x_g', self._clone_zero_param(a.x_g, dtype=dtype))
                    _copy_z(att+'w0', self._clone_zero_param(a.w0, dtype=dtype))
                    _copy_z(att+'w1', self._clone_zero_param(a.w1, dtype=dtype))
                    _copy_z(att+'w2', self._clone_zero_param(a.w2, dtype=dtype))
                    _copy_z(att+'a0', self._clone_zero_param(a.a0, dtype=dtype))
                    _copy_z(att+'a1', self._clone_zero_param(a.a1, dtype=dtype))
                    _copy_z(att+'a2', self._clone_zero_param(a.a2, dtype=dtype))
                    if i == 0:
                        _copy_z(att+'v0', self._clone_zero_param(a.a0, dtype=dtype))
                        _copy_z(att+'v1', self._clone_zero_param(a.a1, dtype=dtype))
                        _copy_z(att+'v2', self._clone_zero_param(a.a2, dtype=dtype))
                    else:
                        _copy_z(att+'v0', self._clone_zero_param(a.v0, dtype=dtype))
                        _copy_z(att+'v1', self._clone_zero_param(a.v1, dtype=dtype))
                        _copy_z(att+'v2', self._clone_zero_param(a.v2, dtype=dtype))
                    _copy_z(att+'g1', self._clone_zero_param(a.g1, dtype=dtype))
                    _copy_z(att+'g2', self._clone_zero_param(a.g2, dtype=dtype))
                    _copy_z(att+'k_k', self._clone_zero_param(a.k_k, dtype=dtype))
                    _copy_z(att+'k_a', self._clone_zero_param(a.k_a, dtype=dtype))
                    _copy_z(att+'r_k', self._clone_zero_param(a.r_k, dtype=dtype, flatten=True))
                    _copy_z(att+'receptance.weight', self._clone_zero_param(a.receptance.weight, dtype=dtype, transpose=True))
                    _copy_z(att+'key.weight', self._clone_zero_param(a.key.weight, dtype=dtype, transpose=True))
                    _copy_z(att+'value.weight', self._clone_zero_param(a.value.weight, dtype=dtype, transpose=True))
                    _copy_z(att+'output.weight', self._clone_zero_param(a.output.weight, dtype=dtype, transpose=True))
                    _copy_z(att+'ln_x.weight', self._clone_zero_param(a.ln_x.weight, dtype=dtype))
                    _copy_z(att+'ln_x.bias', self._clone_zero_param(a.ln_x.bias, dtype=dtype))
                    f = block.ffn
                    _copy_z(ffn+'x_k', self._clone_zero_param(f.x_k, dtype=dtype))
                    _copy_z(ffn+'key.weight', self._clone_zero_param(f.key.weight, dtype=dtype, transpose=True))
                    _copy_z(ffn+'value.weight', self._clone_zero_param(f.value.weight, dtype=dtype, transpose=True))
                _copy_z('ln_out.weight', self._clone_zero_param(self.ln_out.weight, dtype=torch.float32))
                _copy_z('ln_out.bias', self._clone_zero_param(self.ln_out.bias, dtype=torch.float32))
                _copy_z('head.weight', self._clone_zero_param(self.head.weight, dtype=torch.float32, transpose=True))
                return
            rollout_params = dict(self._rollout_model.named_parameters())
            for name, param in self.named_parameters():
                target = rollout_params.get(name)
                if target is None:
                    continue
                with deepspeed.zero.GatheredParameters([param], modifier_rank=None):
                    target.data.copy_(param.data.to(device=target.device, dtype=target.dtype))
            rollout_buffers = dict(self._rollout_model.named_buffers())
            for name, buf in self.named_buffers():
                target = rollout_buffers.get(name)
                if target is None:
                    continue
                target.data.copy_(buf.data.to(device=target.device, dtype=target.dtype))
            self._rollout_model.eval()

    def on_fit_start(self):
        if self._rl_trainer is not None:
            return
        self._rl_step_idx = int(getattr(self.args, 'step_offset', 0))
        self.fit_start_time = time.time()
        self._bootstrap_optimizer_states()
        encode, decode = self._build_tokenizer()
        ref_model = self._build_ref_model()
        self.__dict__['_ref_model'] = ref_model
        if int(getattr(self.args, 'use_stateful_rollout', 1)) == 1:
            rollout_model = self._build_stateful_rollout_model()
            self.__dict__['_rollout_model'] = rollout_model
            infer_engine = TrainTempBatchInference(
                infer_model=rollout_model,
                train_model=self,
                encode_fn=encode,
                decode_fn=decode,
                device=str(self.device),
                cfg=self.rl_cfg,
            )
        else:
            rollout_model = self._build_rollout_model()
            self.__dict__['_rollout_model'] = rollout_model
            infer_engine = AlbatrossBatchInference(
                infer_model=None,
                train_model=rollout_model,
                encode_fn=encode,
                decode_fn=decode,
                device=str(self.device),
                cfg=self.rl_cfg,
            )
        trainer_core = LightningGRPOTrainer(
            pl_module=self,
            train_model=self,
            ref_model=ref_model,
            infer_engine=infer_engine,
            encode_fn=encode,
            decode_fn=decode,
            train_data=self.train_data_json,
            test_data=self.test_data_json,
            full_test_data=self.full_test_data_json,
            out_dir=self.args.proj_dir,
            device=str(self.device),
            cfg=self.rl_cfg,
            seed=int(self.args.random_seed),
        )
        self.__dict__['_rl_trainer'] = trainer_core
        self._log_mem('after_deepspeed_setup')
        if self._rollout_model is not None:
            self._sync_rollout_model()
            self._log_mem('after_rollout_sync')
        else:
            self._log_mem('after_stateful_infer_setup')
        if int(self.args.skip_preeval) != 1:
            acc = trainer_core.evaluate(step=0, tag='pre_eval', sample_ratio=float(self.args.preeval_sample_ratio))
            if acc is not None:
                trainer_core._log(f'[pre_eval] acc={acc:.4f}')

    def on_train_batch_start(self, batch, batch_idx):
        if self._rollout_model is None and batch_idx > 0:
            self.cleanup_stateful_rollout()
            self.prepare_stateful_rollout()

    def on_train_batch_end(self, outputs, batch, batch_idx):
        step = getattr(self, '_pending_post_step_sync', None)
        if step is not None:
            self._pending_post_step_sync = None
            if self._rollout_model is not None:
                self._sync_rollout_model()
                self._log_mem(f'after_post_step_sync_{step}')
                if step <= 2:
                    self._debug_rollout_sync_check(step)
        pending_eval = getattr(self, '_pending_eval_request', None)
        if pending_eval is not None:
            self._pending_eval_request = None
            step = int(pending_eval['step'])
            if self.test_data_json:
                if pending_eval['tag'] == 'full_eval':
                    acc = self._rl_trainer.evaluate(step, tag='full_eval', sample_ratio=1.0)
                    if int(getattr(self.args, 'save_eval_checkpoint', 0)) == 1 and (int(self.args.save_interval) > 0) and (step % int(self.args.save_interval) == 0):
                        ckpt_path = self._save_step_checkpoint(step)
                        self._rl_trainer._log(f'[step_ckpt] saved={ckpt_path}')
                    patience = int(getattr(self.args, 'full_eval_early_stop_patience', 0))
                    if acc is not None:
                        if float(acc) > float(self._best_full_eval_acc) + 1e-12:
                            self._best_full_eval_acc = float(acc)
                            self._full_eval_bad_count = 0
                        else:
                            self._full_eval_bad_count += 1
                            if patience > 0 and self._full_eval_bad_count >= patience:
                                self._rl_trainer._log(f'[early_stop] patience={patience} best_full_eval={self._best_full_eval_acc:.6f} current={float(acc):.6f} at_step={step}')
                                self.trainer.should_stop = True
                else:
                    acc = self._rl_trainer.evaluate(step, tag='eval', sample_ratio=float(pending_eval['sample_ratio']))
                self._rl_trainer._plot_metrics()

    def on_train_end(self):
        if self._rl_trainer is None:
            return
        if self._rollout_model is not None:
            self._sync_rollout_model()
        else:
            self.cleanup_stateful_rollout()
        if int(self.args.skip_posteval) != 1:
            target_step = int(self.args.step_offset) + int(self.args.total_steps)
            acc = self._rl_trainer.evaluate(step=target_step, tag='post_eval', sample_ratio=float(self.args.posteval_sample_ratio))
            if acc is not None:
                self._rl_trainer._log(f'[post_eval] acc={acc:.4f}')
            self._rl_trainer._plot_metrics()
        if int(getattr(self.args, 'save_final_checkpoint', 0)) == 1:
            target_step = int(self.args.step_offset) + int(self.args.total_steps)
            ckpt_path = self._save_final_checkpoint(int(getattr(self, '_rl_step_idx', target_step)))
            self._rl_trainer._log(f'[final_ckpt] saved={ckpt_path}')

    def training_step(self, batch, batch_idx):
        if self._rl_trainer is None:
            raise RuntimeError('RL trainer not initialized')
        step = int(getattr(self, '_rl_step_idx', int(self.args.step_offset))) + 1
        self._rl_step_idx = step
        target_step = int(self.args.step_offset) + int(self.args.total_steps)
        self._log_mem(f'before_step_{step}', reset_peak=True)
        if self._rollout_model is not None:
            self._log_mem(f'after_sync_{step}')
        else:
            self._log_mem(f'after_stateful_ready_{step}')
        metrics = self._rl_trainer.train_step(step)
        self._log_mem(f'after_step_{step}')
        if self._rollout_model is not None:
            self._pending_post_step_sync = step
        need_eval = bool(self.test_data_json) and (((self.rl_cfg.eval_interval > 0) and (step % self.rl_cfg.eval_interval == 0)) or (((self.rl_cfg.save_interval > 0) and (step % self.rl_cfg.save_interval == 0)) or (bool(self.rl_cfg.final_full_eval) and step == target_step)))
        metrics['elapsed'] = time.time() - (self.fit_start_time or time.time())
        append_jsonl(self._rl_trainer.metrics_path, metrics)
        self.log('loss', float(metrics['loss']), prog_bar=True, on_step=True, logger=False)
        self.log('acc', float(metrics['accuracy']), prog_bar=True, on_step=True, logger=False)

        if step % self.rl_cfg.log_interval == 0:
            self._rl_trainer._log(
                f"[Step {step}/{target_step}] samples={int(metrics['samples'])} acc={metrics['accuracy']:.3f} "
                f"trunc={metrics['trunc_rate']:.3f} repeat={metrics['repeat_rate']:.3f} no_answer={metrics['no_answer_rate']:.3f} "
                f"reward={metrics['avg_reward']:.4f} corr_r={metrics['avg_correct_reward']:.4f} fmt_r={metrics['avg_format_reward']:.4f} "
                f"len_r={metrics['avg_length_reward']:.4f} len={metrics['avg_length']:.1f} loss={metrics['loss']:.4f} "
                f"kl={metrics['avg_kl']:.6f} grad={metrics['grad_norm']:.3f} "
                f"adv(m={metrics['adv_mean']:.3f},s={metrics['adv_std']:.3f},pos={metrics['pos_adv_ratio']:.2f},neg={metrics['neg_adv_ratio']:.2f}) "
                f"groups(t={metrics['groups_total']},all0={metrics['groups_all_wrong']},all1={metrics['groups_all_correct']}) "
                f"speed(samp/s={metrics['samples_per_sec']:.2f},tok/s={metrics['tokens_per_sec']:.1f}) "
                f"step_time={metrics['time']:.1f}s elapsed={metrics['elapsed']:.1f}s"
            )

        full_eval = ((self.rl_cfg.save_interval > 0) and (step % self.rl_cfg.save_interval == 0)) or (bool(self.rl_cfg.final_full_eval) and step == target_step)
        if self._rollout_model is not None:
            if self.rl_cfg.eval_interval > 0 and self.test_data_json and (step % self.rl_cfg.eval_interval == 0):
                self._pending_eval_request = {
                    'step': step,
                    'tag': ('full_eval' if full_eval else 'eval'),
                    'sample_ratio': (1.0 if full_eval else self.rl_cfg.eval_sample_ratio),
                }
            elif full_eval and self.test_data_json:
                self._pending_eval_request = {
                    'step': step,
                    'tag': 'full_eval',
                    'sample_ratio': 1.0,
                }
        else:
            if self.rl_cfg.eval_interval > 0 and self.test_data_json and (step % self.rl_cfg.eval_interval == 0):
                if full_eval:
                    self._rl_trainer.evaluate(step, tag='full_eval', sample_ratio=1.0)
                else:
                    self._rl_trainer.evaluate(step, tag='eval', sample_ratio=self.rl_cfg.eval_sample_ratio)
                self._rl_trainer._plot_metrics()
            elif full_eval and self.test_data_json:
                self._rl_trainer.evaluate(step, tag='full_eval', sample_ratio=1.0)
                self._rl_trainer._plot_metrics()
        return None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_model', type=str, required=True)
    parser.add_argument('--proj_dir', type=str, required=True)
    parser.add_argument('--tokenizer', type=str, required=True)
    parser.add_argument('--train_jsonl', type=str, required=True)
    parser.add_argument('--eval_jsonl', type=str, default='')
    parser.add_argument('--max_data_samples', type=int, default=None)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--ctx_len', type=int, default=8192)
    parser.add_argument('--micro_bsz', type=int, default=1)
    parser.add_argument('--devices', type=int, default=1)
    parser.add_argument('--num_nodes', type=int, default=1)
    parser.add_argument('--accelerator', type=str, default='gpu')
    parser.add_argument('--strategy', type=str, default='deepspeed_stage_3_offload')
    parser.add_argument('--precision', type=str, default='bf16')
    parser.add_argument('--grad_cp', type=int, default=1)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.99)
    parser.add_argument('--adam_eps', type=float, default=1e-8)
    parser.add_argument('--head_size', type=int, default=64)
    parser.add_argument('--my_testing', type=str, default='x070')
    parser.add_argument('--enable_progress_bar', type=int, default=1)
    parser.add_argument('--quiet_optimizer_log', type=int, default=1)
    parser.add_argument('--ds_bucket_mb', type=int, default=200)
    parser.add_argument('--ds_contiguous_gradients', type=int, default=0)
    parser.add_argument('--total_steps', type=int, default=200)
    parser.add_argument('--step_offset', type=int, default=0)

    parser.add_argument('--num_questions', type=int, default=24)
    parser.add_argument('--samples_per_question', type=int, default=8)
    parser.add_argument('--max_new_tokens', type=int, default=1024)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top_p', type=float, default=0.6)
    parser.add_argument('--top_k', type=int, default=0)
    parser.add_argument('--eval_temperature', type=float, default=0.3)
    parser.add_argument('--eval_top_p', type=float, default=0.4)
    parser.add_argument('--eval_top_k', type=int, default=500)
    parser.add_argument('--ppo_epochs', type=int, default=1)
    parser.add_argument('--micro_batch', type=int, default=4)
    parser.add_argument('--rollout_forward_batch', type=int, default=8)
    parser.add_argument('--use_stateful_rollout', type=int, default=1)
    parser.add_argument('--lr', type=float, default=6e-5)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--min_tokens', type=int, default=200)
    parser.add_argument('--length_weight', type=float, default=0.0)
    parser.add_argument('--zstd_threshold', type=float, default=2.5)
    parser.add_argument('--zstd_penalty_weight', type=float, default=0.0)
    parser.add_argument('--ngram_penalty', type=float, default=0.0)
    parser.add_argument('--neg_adv_weight', type=float, default=0.6)
    parser.add_argument('--kl_coef', type=float, default=0.05)
    parser.add_argument('--kl_mode', type=str, default='k3_loss', choices=['k1_reward', 'k3_loss'])
    parser.add_argument('--hard_buffer_ttl', type=int, default=10)
    parser.add_argument('--hard_buffer_cooldown', type=int, default=5)
    parser.add_argument('--hard_buffer_target_samples', type=int, default=192)
    parser.add_argument('--hard_buffer_group_size', type=int, default=8)
    parser.add_argument('--hard_buffer_extra_lr_scale', type=float, default=0.5)
    parser.add_argument('--hard_buffer_adv_clip', type=float, default=2.5)
    parser.add_argument('--log_interval', type=int, default=1)
    parser.add_argument('--save_interval', type=int, default=50)
    parser.add_argument('--eval_interval', type=int, default=5)
    parser.add_argument('--eval_sample_ratio', type=float, default=1.0)
    parser.add_argument('--full_eval_jsonl', type=str, default='')
    parser.add_argument('--disable_extra_step', type=int, default=0)
    parser.add_argument('--extra_curriculum', type=str, default='off', choices=['off', 'staged_v1', 'pure_hard'])
    parser.add_argument('--save_eval_checkpoint', type=int, default=0)
    parser.add_argument('--full_eval_early_stop_patience', type=int, default=0)
    parser.add_argument('--save_final_checkpoint', type=int, default=0)
    parser.add_argument('--preeval_sample_ratio', type=float, default=1.0)
    parser.add_argument('--posteval_sample_ratio', type=float, default=1.0)
    parser.add_argument('--skip_preeval', type=int, default=0)
    parser.add_argument('--skip_posteval', type=int, default=0)
    parser.add_argument('--final_full_eval', type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(int(args.random_seed))

    rwkv_precision = {'32': 'fp32', 32: 'fp32', '16': 'fp16', 16: 'fp16'}.get(args.precision, args.precision)
    trainer_precision = {'fp32': '32', 'fp16': '16'}.get(args.precision, args.precision)

    os.environ['RWKV_MY_TESTING'] = args.my_testing
    os.environ['RWKV_CTXLEN'] = str(int(args.ctx_len))
    os.environ['RWKV_HEAD_SIZE'] = str(int(args.head_size))
    os.environ['RWKV_FLOAT_MODE'] = rwkv_precision
    os.environ['RWKV_JIT_ON'] = '0' if 'deepspeed_stage_3' in str(args.strategy) else '1'

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    if rwkv_precision == 'fp32':
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    else:
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True

    args.my_timestamp = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
    args.lr_init = float(args.lr)
    args.lr_final = float(args.lr)
    args.betas = (float(args.beta1), float(args.beta2))
    args.train_stage = 0
    args.real_bsz = int(args.num_nodes) * int(args.devices) * int(args.micro_bsz)

    sd = _normalize_state_dict(_torch_load_weights(args.load_model))
    args.n_layer, args.n_embd, args.vocab_size, args.dim_ffn = _infer_arch(sd)
    args.dim_att = args.n_embd

    os.makedirs(args.proj_dir, exist_ok=True)
    train_data = read_jsonl(args.train_jsonl, max_samples=args.max_data_samples)
    if not train_data:
        raise RuntimeError('empty train_data')
    test_data = read_jsonl(args.eval_jsonl) if args.eval_jsonl else []
    full_test_data = read_jsonl(args.full_eval_jsonl) if args.full_eval_jsonl else test_data

    rl_cfg = GRPOConfig(
        num_questions=int(args.num_questions),
        samples_per_question=int(args.samples_per_question),
        max_new_tokens=int(args.max_new_tokens),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        top_k=int(args.top_k),
        eval_temperature=float(args.eval_temperature),
        eval_top_p=float(args.eval_top_p),
        eval_top_k=int(args.eval_top_k),
        ppo_epochs=int(args.ppo_epochs),
        micro_batch=int(args.micro_batch),
        rollout_forward_batch=int(args.rollout_forward_batch),
        lr=float(args.lr),
        grad_clip=float(args.grad_clip),
        min_tokens=int(args.min_tokens),
        max_tokens=int(args.max_new_tokens),
        length_weight=float(args.length_weight),
        zstd_threshold=float(args.zstd_threshold),
        zstd_penalty_weight=float(args.zstd_penalty_weight),
        ngram_penalty=float(args.ngram_penalty),
        neg_adv_weight=float(args.neg_adv_weight),
        kl_coef=float(args.kl_coef),
        kl_mode=str(args.kl_mode),
        time_state_l2=0.0,
        time_state_clamp=0.0,
        log_interval=int(args.log_interval),
        save_interval=int(args.save_interval),
        eval_interval=int(args.eval_interval),
        eval_sample_ratio=float(args.eval_sample_ratio),
        save_last=False,
        final_full_eval=bool(int(args.final_full_eval)),
        hard_buffer_ttl=int(args.hard_buffer_ttl),
        hard_buffer_cooldown=int(args.hard_buffer_cooldown),
        hard_buffer_target_samples=int(args.hard_buffer_target_samples),
        hard_buffer_group_size=int(args.hard_buffer_group_size),
        hard_buffer_extra_lr_scale=float(args.hard_buffer_extra_lr_scale),
        hard_buffer_adv_clip=float(args.hard_buffer_adv_clip),
        tune_mode=('state' if int(args.use_stateful_rollout) == 1 else 'full'),
    )
    rl_cfg.disable_extra_step = bool(int(args.disable_extra_step))
    rl_cfg.extra_curriculum = str(args.extra_curriculum)

    model = RWKVGRPOModel(args=args, rl_cfg=rl_cfg, train_data=train_data, test_data=test_data, full_test_data=full_test_data)
    dataset = DummyStepDataset(total_steps=int(args.total_steps), micro_bsz=int(args.micro_bsz))
    data_loader = DataLoader(dataset, shuffle=False, pin_memory=True, batch_size=int(args.micro_bsz), num_workers=0, drop_last=False)

    trainer = Trainer(
        accelerator=args.accelerator,
        devices=int(args.devices),
        num_nodes=int(args.num_nodes),
        strategy=args.strategy,
        precision=trainer_precision,
        max_steps=int(args.total_steps),
        max_epochs=1,
        logger=False,
        enable_checkpointing=False,
        num_sanity_val_steps=0,
        log_every_n_steps=1,
        enable_progress_bar=bool(int(args.enable_progress_bar)),
        enable_model_summary=False,
    )
    if 'deepspeed' in str(args.strategy):
        trainer.strategy.config['zero_optimization']['allgather_bucket_size'] = int(args.ds_bucket_mb) * 1000 * 1000
        trainer.strategy.config['zero_optimization']['reduce_bucket_size'] = int(args.ds_bucket_mb) * 1000 * 1000
        trainer.strategy.config['zero_optimization']['contiguous_gradients'] = bool(int(args.ds_contiguous_gradients))
    trainer.fit(model, train_dataloaders=data_loader)


if __name__ == '__main__':
    main()
