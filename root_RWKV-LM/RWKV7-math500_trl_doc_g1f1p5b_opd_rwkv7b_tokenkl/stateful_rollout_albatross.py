#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List, Any
import os
import sys
from pathlib import Path

import torch

THIS_DIR = Path(__file__).resolve().parent
ALBATROSS_DIR = Path('/root/Albatross/faster3a_2605')
if str(ALBATROSS_DIR) not in sys.path:
    sys.path.insert(0, str(ALBATROSS_DIR))

import rwkv7_fast_v3a as v3a


def _ensure_loaded():
    v3a.WKV_MODE = 'fp16'
    v3a.EMB_DEVICE = 'gpu'
    v3a.RKV_MODE = 'off'
    v3a.CMIX_SPARSE = 'no-fc'
    v3a.LOWRANK_WEIGHT = 'both'
    v3a.ORIG_LINEAR_GROUPS = v3a.parse_orig_linear_groups('att_c2c,ffn_key,head')
    v3a.PP_DEVICES = []
    v3a.load_extensions(v3a.WKV_MODE)


class AlbatrossTrainRollout:
    def __init__(self, train_model, device: str):
        self.train_model = train_model
        self.blocks = train_model.blocks
        self.emb = train_model.emb
        self.ln_out = train_model.ln_out
        self.head = train_model.head
        self.args = train_model.args
        self.device = device
        self._backend = None
        self._cache_dirty = True
        _ensure_loaded()

    def mark_dirty(self):
        self._cache_dirty = True

    def _clone_param(self, param, *, dtype: torch.dtype, transpose: bool = False, squeeze: bool = False, flatten: bool = False):
        x = param.detach()
        if squeeze:
            x = x.squeeze()
        if transpose:
            x = x.t()
        if flatten:
            x = x.flatten()
        return x.to(device=self.device, dtype=dtype).contiguous()

    def _build_cache(self):
        v3a.H = self.blocks[0].att.n_head
        v3a.N = self.blocks[0].att.head_size
        v3a.C = self.args.n_embd
        v3a.V = self.emb.weight.shape[0]
        v3a.L = len(self.blocks)
        z = {}

        emb_dev = torch.device(self.device)
        emb_src = self.emb.weight.detach().to(device=emb_dev, dtype=torch.bfloat16).contiguous()
        ln0 = self.blocks[0].ln0
        ln0_w = ln0.weight.detach().to(device=emb_dev, dtype=torch.bfloat16).contiguous()
        ln0_b = ln0.bias.detach().to(device=emb_dev, dtype=torch.bfloat16).contiguous()
        z['emb.weight'] = torch.ops.rwkv7_v3a_ops.emb_ln0_bf16_to_f16(emb_src, ln0_w, ln0_b)

        for i, block in enumerate(self.blocks):
            bbb = f'blocks.{i}.'
            att = f'{bbb}att.'
            ffn = f'{bbb}ffn.'
            z[bbb + 'ln1.weight'] = self._clone_param(block.ln1.weight, dtype=v3a.DTYPE)
            z[bbb + 'ln1.bias'] = self._clone_param(block.ln1.bias, dtype=v3a.DTYPE)
            z[bbb + 'ln2.weight'] = self._clone_param(block.ln2.weight, dtype=v3a.DTYPE)
            z[bbb + 'ln2.bias'] = self._clone_param(block.ln2.bias, dtype=v3a.DTYPE)

            a = block.att
            z[att + 'x_r'] = self._clone_param(a.x_r, dtype=v3a.DTYPE, squeeze=True)
            z[att + 'x_w'] = self._clone_param(a.x_w, dtype=v3a.DTYPE, squeeze=True)
            z[att + 'x_k'] = self._clone_param(a.x_k, dtype=v3a.DTYPE, squeeze=True)
            z[att + 'x_v'] = self._clone_param(a.x_v, dtype=v3a.DTYPE, squeeze=True)
            z[att + 'x_a'] = self._clone_param(a.x_a, dtype=v3a.DTYPE, squeeze=True)
            z[att + 'x_g'] = self._clone_param(a.x_g, dtype=v3a.DTYPE, squeeze=True)
            z[att + 'w0'] = self._clone_param(a.w0, dtype=v3a.DTYPE, squeeze=True)
            z[att + 'w1'] = self._clone_param(a.w1, dtype=v3a.DTYPE)
            z[att + 'w2'] = self._clone_param(a.w2, dtype=v3a.DTYPE)
            z[att + 'a0'] = self._clone_param(a.a0, dtype=v3a.DTYPE, squeeze=True)
            z[att + 'a1'] = self._clone_param(a.a1, dtype=v3a.DTYPE)
            z[att + 'a2'] = self._clone_param(a.a2, dtype=v3a.DTYPE)
            z[att + 'v0'] = self._clone_param(a.v0, dtype=v3a.DTYPE, squeeze=True)
            z[att + 'v1'] = self._clone_param(a.v1, dtype=v3a.DTYPE)
            z[att + 'v2'] = self._clone_param(a.v2, dtype=v3a.DTYPE)
            z[att + 'g1'] = self._clone_param(a.g1, dtype=v3a.DTYPE)
            z[att + 'g2'] = self._clone_param(a.g2, dtype=v3a.DTYPE)
            z[att + 'k_k'] = self._clone_param(a.k_k, dtype=v3a.DTYPE, squeeze=True)
            z[att + 'k_a'] = self._clone_param(a.k_a, dtype=v3a.DTYPE, squeeze=True)
            z[att + 'r_k'] = self._clone_param(a.r_k, dtype=v3a.DTYPE, flatten=True)
            z[att + 'receptance.weight'] = self._clone_param(a.receptance.weight, dtype=v3a.DTYPE)
            z[att + 'key.weight'] = self._clone_param(a.key.weight, dtype=v3a.DTYPE)
            z[att + 'value.weight'] = self._clone_param(a.value.weight, dtype=v3a.DTYPE)
            z[att + 'output.weight'] = self._clone_param(a.output.weight, dtype=v3a.DTYPE)
            z[att + 'ln_x.weight'] = self._clone_param(a.ln_x.weight, dtype=v3a.DTYPE)
            z[att + 'ln_x.bias'] = self._clone_param(a.ln_x.bias, dtype=v3a.DTYPE)

            f = block.ffn
            z[ffn + 'x_k'] = self._clone_param(f.x_k, dtype=v3a.DTYPE, squeeze=True)
            z[ffn + 'key.weight'] = self._clone_param(f.key.weight, dtype=v3a.DTYPE)
            z[ffn + 'value.weight'] = self._clone_param(f.value.weight, dtype=v3a.DTYPE)

        z['ln_out.weight'] = self._clone_param(self.ln_out.weight, dtype=v3a.DTYPE)
        z['ln_out.bias'] = self._clone_param(self.ln_out.bias, dtype=v3a.DTYPE)
        z['head.weight'] = self._clone_param(self.head.weight, dtype=v3a.DTYPE)

        # Match Albatross expected layout / extra caches.
        for key in list(z.keys()):
            value = z[key]
            is_lowrank = v3a.is_lowrank_weight(key)
            if (
                not is_lowrank
                and (("key.weight" in key and not v3a.is_orig_linear_weight(key))
                or ("value.weight" in key and not v3a.is_orig_linear_weight(key))
                or ("receptance.weight" in key and not v3a.is_orig_linear_weight(key))
                or ("output.weight" in key and not v3a.is_orig_linear_weight(key))
                or ("head.weight" in key and not v3a.is_orig_linear_weight(key)))
            ):
                value = value.t().contiguous()
                z[key] = value
            if is_lowrank:
                z[key + '.t'] = value.t().contiguous()
        if v3a.RKV_MODE != 'off' and not v3a.use_orig_linear('att_c2c'):
            for layer in range(v3a.L):
                p = f'blocks.{layer}.att.'
                z[p+'rkv.weight'] = torch.stack((z[p+'receptance.weight'], z[p+'key.weight'], z[p+'value.weight'])).contiguous()

        backend = v3a.RWKV7.__new__(v3a.RWKV7)
        backend.z = z
        backend.emb_cpu = False
        backend.emb_cache = {}
        self._backend = backend
        self._cache_dirty = False

    def prepare_stateful_rollout(self, force: bool = False):
        if self._backend is not None and (not self._cache_dirty) and (not force):
            return
        self._build_cache()

    def cleanup_stateful_rollout(self):
        self._backend = None
        self._cache_dirty = True
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def generate_zero_state(self, bsz: int):
        self.prepare_stateful_rollout()
        return self._backend.zero_state(bsz)

    @torch.no_grad()
    def _forward_same_length(self, tokens, state, full_output: bool = False):
        self.prepare_stateful_rollout()
        if not torch.is_tensor(tokens):
            tokens = torch.tensor(tokens, dtype=torch.long, device=self.device)
        else:
            tokens = tokens.to(device=self.device, dtype=torch.long)
        if len(state) == 2:
            state = [state[0], state[1], torch.zeros((tokens.size(0),), dtype=torch.int32, device=self.device)]
        if full_output:
            return self._backend.forward_all_logits(tokens, state).float()
        return self._backend.forward(tokens, state).float()

    @torch.no_grad()
    def forward_batch(self, tokens: List[List[int]], state, full_output: bool = False):
        self.prepare_stateful_rollout()
        if torch.is_tensor(tokens):
            return self._forward_same_length(tokens, state, full_output=full_output)
        lengths = [len(x) for x in tokens]
        if len(set(lengths)) == 1:
            return self._forward_same_length(tokens, state, full_output=full_output)

        bsz = len(tokens)
        pos = [0] * bsz
        if not full_output:
            out = torch.empty((bsz, v3a.V), dtype=torch.float32, device=self.device)
        else:
            out = [torch.empty((0, v3a.V), dtype=torch.float32, device=self.device) for _ in range(bsz)]

        while True:
            active = [i for i in range(bsz) if pos[i] < len(tokens[i])]
            if not active:
                break
            min_len = min(len(tokens[i]) - pos[i] for i in active)
            batch_tokens = [tokens[i][pos[i]:pos[i] + min_len] for i in active]
            batch_state = [state[0][:, :, active], state[1][:, active], state[2][active]]
            new_out = self._forward_same_length(batch_tokens, batch_state, full_output=full_output)
            for k, i in enumerate(active):
                pos[i] += min_len
                state[0][:, :, i] = batch_state[0][:, :, k]
                state[1][:, i] = batch_state[1][:, k]
                state[2][i] = batch_state[2][k]
                if not full_output:
                    out[i] = new_out[k]
                else:
                    out[i] = torch.cat((out[i], new_out[k]), dim=0)
        return out
