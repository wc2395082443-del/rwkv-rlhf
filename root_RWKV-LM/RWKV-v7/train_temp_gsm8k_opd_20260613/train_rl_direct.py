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
BASELINE_DIR = Path('/root/RWKV-LM/RWKV7-statetuning')

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

_baseline_spec = importlib.util.spec_from_file_location('baseline_grpo_train', BASELINE_DIR / 'grpo_direct.py')
_baseline_mod = importlib.util.module_from_spec(_baseline_spec)
_baseline_spec.loader.exec_module(_baseline_mod)
GRPOConfig = _baseline_mod.DirectGRPOConfig
BaseGRPOTrainer = _baseline_mod.DirectGRPOTrainer
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


def _make_cfg(cfg_cls, **kwargs):
    import inspect
    sig = inspect.signature(cfg_cls)
    allowed = {k: v for k, v in kwargs.items() if k in sig.parameters}
    cfg = cfg_cls(**allowed)
    for k, v in kwargs.items():
        if not hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


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
    ):
        self.pl_module = pl_module
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

                    top_k = min(500, logits.size(-1))
                    top_logits, _ = torch.topk(logits, k=top_k, dim=-1)
                    logp_top = top_logits.float() - logsumexp.unsqueeze(-1)
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




    def _apply_loss(self, batch: List[Dict[str, Any]]):
        seqs = [traj['prompt_tokens'] + traj['comp_tokens'] for traj in batch]
        seqs, _ = self._pad_batch(seqs, pad_id=0)
        seqs = seqs.to(self.device)
        inp = seqs[:, :-1].contiguous()
        tgt = seqs[:, 1:].contiguous()

        logits = self.model(inp)
        new_logp_all = self._gather_logp(logits, tgt)
        del logits
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        with torch.no_grad():
            ref_logits = self.ref_model(inp)
            ref_logp_all = self._gather_logp(ref_logits, tgt)
            del ref_logits
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        batch_policy_loss = 0.0
        batch_kl = 0.0
        batch_tokens = 0
        clip_hits = 0
        clip_total = 0
        lo = 1.0 - float(self.cfg.clip_eps)
        hi = 1.0 + float(self.cfg.clip_eps)
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
        total_loss = (batch_policy_loss / batch_tokens) + float(self.cfg.kl_coef) * (batch_kl / batch_tokens)
        if float(getattr(self.cfg, 'time_state_l2', 0.0)) > 0:
            l2_reg = 0.0
            for name, param in self.model.named_parameters():
                if 'time_state' in name and name in self._ts_init:
                    l2_reg = l2_reg + (param.float() - self._ts_init[name].float()).pow(2).mean()
            total_loss = total_loss + float(self.cfg.time_state_l2) * l2_reg
        self.pl_module.manual_backward(total_loss)
        return {
            'loss': float((batch_policy_loss / batch_tokens).detach().item()),
            'avg_kl': float((batch_kl / batch_tokens).detach().item()),
            'clip_hits': clip_hits,
            'clip_total': clip_total,
        }

    def _optimize(self, all_trajs: List[Dict[str, Any]]):
        loss_total = 0.0
        kl_total = 0.0
        batch_cnt = 0
        clip_hits = 0
        clip_total = 0
        grad_norm = 0.0
        if not all_trajs:
            return loss_total, kl_total, batch_cnt, clip_hits, clip_total, grad_norm

        pl_opt, raw_opt = self._get_pl_optimizer()
        try:
            for _ in range(self.cfg.ppo_epochs):
                self.model.train()
                try:
                    pl_opt.zero_grad(set_to_none=True)
                except TypeError:
                    pl_opt.zero_grad()
                trajs = list(all_trajs)
                self.rng.shuffle(trajs)
                epoch_had_backward = False
                for start in range(0, len(trajs), self.cfg.micro_batch):
                    result = self._apply_loss(trajs[start:start + self.cfg.micro_batch])
                    if result is None:
                        continue
                    epoch_had_backward = True
                    loss_total += result['loss']
                    kl_total += result['avg_kl']
                    batch_cnt += 1
                    clip_hits += result['clip_hits']
                    clip_total += result['clip_total']
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
                if float(getattr(self.cfg, 'time_state_clamp', 0.0)) > 0:
                    with torch.no_grad():
                        clamp_v = float(self.cfg.time_state_clamp)
                        for name, param in self.model.named_parameters():
                            if 'time_state' in name:
                                param.data.clamp_(-clamp_v, clamp_v)
        finally:
            pass
        return loss_total, kl_total, batch_cnt, clip_hits, clip_total, grad_norm


class RWKVGRPOModel(PaddedRWKV):
    def __init__(self, args, rl_cfg, train_data, test_data):
        super().__init__(args)
        self.args = args
        self.rl_cfg = rl_cfg
        self.train_data_json = train_data
        self.test_data_json = test_data
        self.automatic_optimization = False
        self.fit_start_time = None
        self.__dict__['_ref_model'] = None
        self.__dict__['_rollout_model'] = None
        self.__dict__['_rl_trainer'] = None
        self._optimizer_bootstrapped = False
        self._load_from_checkpoint_path(args.load_model)

    def configure_optimizers(self):
        if int(getattr(self.args, 'quiet_optimizer_log', 1)) == 1:
            with open(os.devnull, 'w') as devnull, contextlib.redirect_stdout(devnull):
                return super().configure_optimizers()
        return super().configure_optimizers()

    def _load_from_checkpoint_path(self, path: str):
        sd = _normalize_state_dict(_torch_load_weights(path))
        self.load_state_dict(sd, strict=True)

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

    def _clone_zero_param(self, param, *, dtype=torch.float16, transpose=False, flatten=False, squeeze=True):
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
        dtype = torch.float16
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
        z['ln_out.weight'] = self._clone_zero_param(self.ln_out.weight, dtype=dtype)
        z['ln_out.bias'] = self._clone_zero_param(self.ln_out.bias, dtype=dtype)
        z['head.weight'] = self._clone_zero_param(self.head.weight, dtype=dtype, transpose=True)
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
        state[0] = torch.zeros((args.n_layer, 2, bsz, args.n_embd), dtype=torch.float16, requires_grad=False, device=dev)
        state[1] = torch.zeros((args.n_layer, bsz, args.n_embd // args.head_size, args.head_size, args.head_size), dtype=torch.float, requires_grad=False, device=dev)
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
        x = F.layer_norm(x, (self.args.n_embd,), weight=z['ln_out.weight'], bias=z['ln_out.bias'])
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
            out = torch.empty((bsz, self.args.vocab_size), dtype=torch.float16, requires_grad=False, device=self.device)
        else:
            out = [torch.empty((0, self.args.vocab_size), dtype=torch.float16, requires_grad=False, device=self.device) for _ in range(bsz)]
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
                f'head_src00={float(head_src.flatten()[0]):.6f} head_roll00={float(z['head.weight'].flatten()[0]):.6f}'
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
                dtype = torch.float16

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
                _copy_z('ln_out.weight', self._clone_zero_param(self.ln_out.weight, dtype=dtype))
                _copy_z('ln_out.bias', self._clone_zero_param(self.ln_out.bias, dtype=dtype))
                _copy_z('head.weight', self._clone_zero_param(self.head.weight, dtype=dtype, transpose=True))
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
                    self._rl_trainer.evaluate(step, tag='full_eval', sample_ratio=1.0)
                else:
                    self._rl_trainer.evaluate(step, tag='eval', sample_ratio=float(pending_eval['sample_ratio']))
                self._rl_trainer._plot_metrics()

    def on_train_end(self):
        if self._rl_trainer is None:
            return
        if self._rollout_model is not None:
            self._sync_rollout_model()
        else:
            self.cleanup_stateful_rollout()
        if int(self.args.skip_posteval) != 1:
            acc = self._rl_trainer.evaluate(step=int(self.args.total_steps), tag='post_eval', sample_ratio=float(self.args.posteval_sample_ratio))
            if acc is not None:
                self._rl_trainer._log(f'[post_eval] acc={acc:.4f}')
            self._rl_trainer._plot_metrics()

    def training_step(self, batch, batch_idx):
        if self._rl_trainer is None:
            raise RuntimeError('RL trainer not initialized')
        step = int(getattr(self, '_rl_step_idx', 0)) + 1
        self._rl_step_idx = step
        self._log_mem(f'before_step_{step}', reset_peak=True)
        if self._rollout_model is not None:
            self._log_mem(f'after_sync_{step}')
        else:
            self._log_mem(f'after_stateful_ready_{step}')
        metrics = self._rl_trainer.train_step(step)
        self._log_mem(f'after_step_{step}')
        if self._rollout_model is not None:
            self._pending_post_step_sync = step
        need_eval = bool(self.test_data_json) and (((self.rl_cfg.eval_interval > 0) and (step % self.rl_cfg.eval_interval == 0)) or (((self.rl_cfg.save_interval > 0) and (step % self.rl_cfg.save_interval == 0)) or (bool(self.rl_cfg.final_full_eval) and step == int(self.args.total_steps))))
        metrics['elapsed'] = time.time() - (self.fit_start_time or time.time())
        append_jsonl(self._rl_trainer.metrics_path, metrics)
        self.log('loss', float(metrics['loss']), prog_bar=True, on_step=True, logger=False)
        self.log('acc', float(metrics['accuracy']), prog_bar=True, on_step=True, logger=False)

        if step % self.rl_cfg.log_interval == 0:
            self._rl_trainer._log(
                f"[Step {step}/{self.args.total_steps}] samples={int(metrics['samples'])} acc={metrics['accuracy']:.3f} "
                f"trunc={metrics['trunc_rate']:.3f} repeat={metrics['repeat_rate']:.3f} no_answer={metrics['no_answer_rate']:.3f} "
                f"reward={metrics['avg_reward']:.4f} corr_r={metrics['avg_correct_reward']:.4f} fmt_r={metrics['avg_format_reward']:.4f} "
                f"len_r={metrics['avg_length_reward']:.4f} len={metrics['avg_length']:.1f} loss={metrics['loss']:.4f} "
                f"kl={metrics['avg_kl']:.6f} grad={metrics['grad_norm']:.3f} "
                f"adv(m={metrics['adv_mean']:.3f},s={metrics['adv_std']:.3f},pos={metrics['pos_adv_ratio']:.2f},neg={metrics['neg_adv_ratio']:.2f}) "
                f"groups(t={metrics['groups_total']},all0={metrics['groups_all_wrong']},all1={metrics['groups_all_correct']}) "
                f"speed(samp/s={metrics['samples_per_sec']:.2f},tok/s={metrics['tokens_per_sec']:.1f}) "
                f"step_time={metrics['time']:.1f}s elapsed={metrics['elapsed']:.1f}s"
            )

        full_eval = ((self.rl_cfg.save_interval > 0) and (step % self.rl_cfg.save_interval == 0)) or (bool(self.rl_cfg.final_full_eval) and step == int(self.args.total_steps))
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
    parser.add_argument('--clip_eps', type=float, default=0.2)
    parser.add_argument('--time_state_l2', type=float, default=0.0)
    parser.add_argument('--time_state_clamp', type=float, default=10.0)
    parser.add_argument('--log_interval', type=int, default=1)
    parser.add_argument('--save_interval', type=int, default=50)
    parser.add_argument('--eval_interval', type=int, default=5)
    parser.add_argument('--eval_sample_ratio', type=float, default=1.0)
    parser.add_argument('--preeval_sample_ratio', type=float, default=1.0)
    parser.add_argument('--posteval_sample_ratio', type=float, default=1.0)
    parser.add_argument('--skip_preeval', type=int, default=0)
    parser.add_argument('--skip_posteval', type=int, default=0)
    parser.add_argument('--final_full_eval', type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(int(args.random_seed))

    os.environ['RWKV_MY_TESTING'] = args.my_testing
    os.environ['RWKV_CTXLEN'] = str(int(args.ctx_len))
    os.environ['RWKV_HEAD_SIZE'] = str(int(args.head_size))
    os.environ['RWKV_FLOAT_MODE'] = args.precision
    os.environ['RWKV_JIT_ON'] = '0' if 'deepspeed_stage_3' in str(args.strategy) else '1'

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    if args.precision == 'fp32':
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

    rl_cfg = _make_cfg(GRPOConfig,
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
        clip_eps=float(args.clip_eps),
        time_state_l2=float(args.time_state_l2),
        time_state_clamp=float(args.time_state_clamp),
        log_interval=int(args.log_interval),
        save_interval=int(args.save_interval),
        eval_interval=int(args.eval_interval),
        eval_sample_ratio=float(args.eval_sample_ratio),
        hard_buffer_ttl=int(args.hard_buffer_ttl),
        hard_buffer_cooldown=int(args.hard_buffer_cooldown),
        hard_buffer_target_samples=int(args.hard_buffer_target_samples),
        hard_buffer_group_size=int(args.hard_buffer_group_size),
        hard_buffer_extra_lr_scale=float(args.hard_buffer_extra_lr_scale),
        hard_buffer_adv_clip=float(args.hard_buffer_adv_clip),
        tune_mode=('state' if int(args.use_stateful_rollout) == 1 else 'full'),
        save_last=False,
        final_full_eval=bool(int(args.final_full_eval)),
    )

    model = RWKVGRPOModel(args=args, rl_cfg=rl_cfg, train_data=train_data, test_data=test_data)
    dataset = DummyStepDataset(total_steps=int(args.total_steps), micro_bsz=int(args.micro_bsz))
    data_loader = DataLoader(dataset, shuffle=False, pin_memory=True, batch_size=int(args.micro_bsz), num_workers=0, drop_last=False)

    trainer = Trainer(
        accelerator=args.accelerator,
        devices=int(args.devices),
        num_nodes=int(args.num_nodes),
        strategy=args.strategy,
        precision=args.precision,
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
