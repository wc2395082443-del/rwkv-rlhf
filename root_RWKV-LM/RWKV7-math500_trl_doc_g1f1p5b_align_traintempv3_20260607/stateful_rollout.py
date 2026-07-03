#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List

import os
import torch
import torch.nn.functional as F


def _ensure_ninja_in_path():
    if os.path.isfile('/root/miniconda3/bin/ninja') and '/root/miniconda3/bin' not in os.environ.get('PATH', ''):
        os.environ['PATH'] = '/root/miniconda3/bin:' + os.environ.get('PATH', '')


_ensure_ninja_in_path()

from reference.rwkv7 import RWKV_x070_TMix_seq_batch, RWKV_x070_CMix_seq_batch


def _model_float_dtype(model) -> torch.dtype:
    dtype = getattr(getattr(model, 'emb', None), 'weight', None)
    dtype = getattr(dtype, 'dtype', None)
    if dtype == torch.float16:
        return torch.float16
    if dtype == torch.float32:
        return torch.float32
    return torch.bfloat16


def _rollout_kernel_dtype(model) -> torch.dtype:
    return torch.float16


class StatefulTrainRollout:
    def __init__(self, train_model, device: str):
        self.train_model = train_model
        self.blocks = train_model.blocks
        self.emb = train_model.emb
        self.ln_out = train_model.ln_out
        self.head = train_model.head
        self.args = train_model.args
        self.device = device
        self._stateful_rollout_cache = None
        self._cache_dirty = True

    def mark_dirty(self):
        self._cache_dirty = True

    def _clone_param(self, param, *, dtype: torch.dtype, transpose: bool = False, squeeze: bool = False, flatten: bool = False):
        x = param.detach()
        if transpose:
            x = x.t()
        if squeeze:
            x = x.squeeze()
        if flatten:
            x = x.flatten()
        return x.to(device=self.device, dtype=dtype).contiguous()

    def prepare_stateful_rollout(self, force: bool = False):
        if self._stateful_rollout_cache is not None and (not self._cache_dirty) and (not force):
            return

        kernel_dtype = _rollout_kernel_dtype(self.train_model)
        z = {}
        ln0 = self.blocks[0].ln0
        emb = self._clone_param(self.emb.weight, dtype=kernel_dtype)
        ln0_w = self._clone_param(ln0.weight, dtype=kernel_dtype)
        ln0_b = self._clone_param(ln0.bias, dtype=kernel_dtype)
        z['emb.weight'] = F.layer_norm(emb, (self.args.n_embd,), weight=ln0_w, bias=ln0_b)

        for i, block in enumerate(self.blocks):
            bbb = f'blocks.{i}.'
            att = f'{bbb}att.'
            ffn = f'{bbb}ffn.'
            z[bbb + 'ln1.weight'] = self._clone_param(block.ln1.weight, dtype=kernel_dtype)
            z[bbb + 'ln1.bias'] = self._clone_param(block.ln1.bias, dtype=kernel_dtype)
            z[bbb + 'ln2.weight'] = self._clone_param(block.ln2.weight, dtype=kernel_dtype)
            z[bbb + 'ln2.bias'] = self._clone_param(block.ln2.bias, dtype=kernel_dtype)

            a = block.att
            z[att + 'x_r'] = self._clone_param(a.x_r, dtype=kernel_dtype)
            z[att + 'x_w'] = self._clone_param(a.x_w, dtype=kernel_dtype)
            z[att + 'x_k'] = self._clone_param(a.x_k, dtype=kernel_dtype)
            z[att + 'x_v'] = self._clone_param(a.x_v, dtype=kernel_dtype)
            z[att + 'x_a'] = self._clone_param(a.x_a, dtype=kernel_dtype)
            z[att + 'x_g'] = self._clone_param(a.x_g, dtype=kernel_dtype)
            z[att + 'w0'] = self._clone_param(a.w0, dtype=kernel_dtype)
            z[att + 'w1'] = self._clone_param(a.w1, dtype=kernel_dtype)
            z[att + 'w2'] = self._clone_param(a.w2, dtype=kernel_dtype)
            z[att + 'a0'] = self._clone_param(a.a0, dtype=kernel_dtype)
            z[att + 'a1'] = self._clone_param(a.a1, dtype=kernel_dtype)
            z[att + 'a2'] = self._clone_param(a.a2, dtype=kernel_dtype)
            z[att + 'v0'] = self._clone_param(a.v0, dtype=kernel_dtype)
            z[att + 'v1'] = self._clone_param(a.v1, dtype=kernel_dtype)
            z[att + 'v2'] = self._clone_param(a.v2, dtype=kernel_dtype)
            z[att + 'g1'] = self._clone_param(a.g1, dtype=kernel_dtype)
            z[att + 'g2'] = self._clone_param(a.g2, dtype=kernel_dtype)
            z[att + 'k_k'] = self._clone_param(a.k_k, dtype=kernel_dtype)
            z[att + 'k_a'] = self._clone_param(a.k_a, dtype=kernel_dtype)
            z[att + 'r_k'] = self._clone_param(a.r_k, dtype=kernel_dtype, flatten=True)
            z[att + 'receptance.weight'] = self._clone_param(a.receptance.weight, dtype=kernel_dtype, transpose=True)
            z[att + 'key.weight'] = self._clone_param(a.key.weight, dtype=kernel_dtype, transpose=True)
            z[att + 'value.weight'] = self._clone_param(a.value.weight, dtype=kernel_dtype, transpose=True)
            z[att + 'output.weight'] = self._clone_param(a.output.weight, dtype=kernel_dtype, transpose=True)
            z[att + 'ln_x.weight'] = self._clone_param(a.ln_x.weight, dtype=kernel_dtype)
            z[att + 'ln_x.bias'] = self._clone_param(a.ln_x.bias, dtype=kernel_dtype)

            f = block.ffn
            z[ffn + 'x_k'] = self._clone_param(f.x_k, dtype=kernel_dtype)
            z[ffn + 'key.weight'] = self._clone_param(f.key.weight, dtype=kernel_dtype, transpose=True)
            z[ffn + 'value.weight'] = self._clone_param(f.value.weight, dtype=kernel_dtype, transpose=True)

        z['ln_out.weight'] = self._clone_param(self.ln_out.weight, dtype=torch.float32)
        z['ln_out.bias'] = self._clone_param(self.ln_out.bias, dtype=torch.float32)
        z['head.weight'] = self._clone_param(self.head.weight, dtype=torch.float32, transpose=True)
        self._stateful_rollout_cache = z
        self._cache_dirty = False

    def cleanup_stateful_rollout(self):
        if self._stateful_rollout_cache is not None:
            self._stateful_rollout_cache = None
            self._cache_dirty = True
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def generate_zero_state(self, bsz: int):
        args = self.args
        state = [None, None]
        state[0] = torch.zeros((args.n_layer, 2, bsz, args.n_embd), dtype=_rollout_kernel_dtype(self.train_model), requires_grad=False, device=self.device)
        state[1] = torch.zeros((args.n_layer, bsz, args.n_embd // args.head_size_a, args.head_size_a, args.head_size_a), dtype=torch.float32, requires_grad=False, device=self.device)
        return state

    @torch.no_grad()
    def _forward_batch_same_length_stateful(self, tokens: List[List[int]], state, full_output: bool = False):
        self.prepare_stateful_rollout()
        z = self._stateful_rollout_cache
        idx = torch.tensor(tokens, device=self.device, dtype=torch.long)
        x = z['emb.weight'][idx]
        v_first = torch.empty_like(x)
        for i in range(self.args.n_layer):
            bbb = f'blocks.{i}.'
            att = f'{bbb}att.'
            ffn = f'{bbb}ffn.'
            xx = F.layer_norm(x, (self.args.n_embd,), weight=z[bbb + 'ln1.weight'], bias=z[bbb + 'ln1.bias'])
            xx, v_first = RWKV_x070_TMix_seq_batch(
                i,
                self.blocks[i].att.n_head,
                self.blocks[i].att.head_size,
                xx,
                state[0][i],
                v_first,
                state[1][i],
                z[att + 'x_r'], z[att + 'x_w'], z[att + 'x_k'], z[att + 'x_v'], z[att + 'x_a'], z[att + 'x_g'],
                z[att + 'w0'], z[att + 'w1'], z[att + 'w2'], z[att + 'a0'], z[att + 'a1'], z[att + 'a2'], z[att + 'v0'], z[att + 'v1'], z[att + 'v2'],
                z[att + 'g1'], z[att + 'g2'], z[att + 'k_k'], z[att + 'k_a'], z[att + 'r_k'],
                z[att + 'receptance.weight'], z[att + 'key.weight'], z[att + 'value.weight'], z[att + 'output.weight'],
                z[att + 'ln_x.weight'], z[att + 'ln_x.bias'],
            )
            x = x + xx
            xx = F.layer_norm(x, (self.args.n_embd,), weight=z[bbb + 'ln2.weight'], bias=z[bbb + 'ln2.bias'])
            xx = RWKV_x070_CMix_seq_batch(xx, state[0][i], z[ffn + 'x_k'], z[ffn + 'key.weight'], z[ffn + 'value.weight'])
            x = x + xx
        if not full_output:
            x = x[:, -1, :]
        x = F.layer_norm(x.float(), (self.args.n_embd,), weight=z['ln_out.weight'], bias=z['ln_out.bias'])
        x = x @ z['head.weight']
        return x

    @torch.no_grad()
    def forward_batch(self, tokens: List[List[int]], state, full_output: bool = False):
        self.prepare_stateful_rollout()
        lengths = [len(x) for x in tokens]
        if len(set(lengths)) == 1:
            return self._forward_batch_same_length_stateful(tokens, state, full_output=full_output)

        bsz = len(tokens)
        pos = [0] * bsz
        if not full_output:
            out = torch.empty((bsz, self.args.vocab_size), dtype=torch.float32, requires_grad=False, device=self.device)
        else:
            out = [torch.empty((0, self.args.vocab_size), dtype=torch.float32, requires_grad=False, device=self.device) for _ in range(bsz)]

        while True:
            active = [i for i in range(bsz) if pos[i] < len(tokens[i])]
            if not active:
                break
            min_len = min(len(tokens[i]) - pos[i] for i in active)
            batch_tokens = [tokens[i][pos[i]:pos[i] + min_len] for i in active]
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
