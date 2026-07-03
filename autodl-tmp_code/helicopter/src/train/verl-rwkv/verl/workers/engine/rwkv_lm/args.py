# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Argument bridge for native rwkv-lm.

Defaults in ``RWKV_LM_TRAIN_ARG_DEFAULTS`` are copied from
the native flat-layout ``rwkv-lm/train.py``. This module only builds an
argparse-like namespace for the native upstream code.
"""

from __future__ import annotations

import datetime
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

RWKV_LM_TRAIN_ARG_DEFAULTS: dict[str, Any] = {
    "load_model": "",
    "wandb": "",
    "proj_dir": "out",
    "random_seed": -1,
    "data_file": "",
    "data_type": "utf-8",
    "vocab_size": 0,
    "ctx_len": 1024,
    "epoch_steps": 1000,
    "epoch_count": 500,
    "epoch_begin": 0,
    "epoch_save": 5,
    "micro_bsz": 12,
    "n_layer": 6,
    "n_embd": 512,
    "dim_att": 0,
    "dim_ffn": 0,
    "decay_lora": 0,
    "aaa_lora": 0,
    "mv_lora": 0,
    "gate_lora": 0,
    "lr_init": 6e-4,
    "lr_final": 1e-5,
    "warmup_steps": -1,
    "beta1": 0.9,
    "beta2": 0.99,
    "adam_eps": 1e-18,
    "grad_cp": 0,
    "weight_decay": 0,
    "grad_clip": 1.0,
    "train_stage": 0,
    "ds_bucket_mb": 200,
    "head_size": 64,
    "head_chunk": 0,
    "load_partial": 0,
    "magic_prime": 0,
    "my_testing": "x070",
    "kernel": "",
    "my_exit_tokens": 0,
}

RWKV_LM_LIGHTNING_ARG_DEFAULTS: dict[str, Any] = {
    "num_nodes": 1,
    "devices": 1,
    "accelerator": "gpu",
    "strategy": "auto",
}


def _get(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _set_if_not_none(values: dict[str, Any], key: str, value: Any) -> None:
    if value is not None:
        values[key] = value


def _load_checkpoint_metadata(path: str) -> Any:
    import torch

    load_kwargs = {"map_location": "cpu", "mmap": True}
    try:
        from torch._subclasses.fake_tensor import FakeTensorMode

        with FakeTensorMode():
            return torch.load(path, **load_kwargs)
    except TypeError:
        load_kwargs.pop("mmap", None)
        return torch.load(path, **load_kwargs)
    except Exception:
        return torch.load(path, **load_kwargs)


def _state_dict_items(checkpoint: Any) -> dict[str, Any]:
    if not isinstance(checkpoint, dict):
        return {}
    for key in ("state_dict", "model", "module"):
        nested = checkpoint.get(key)
        if isinstance(nested, dict) and any(isinstance(name, str) for name in nested):
            checkpoint = nested
            break
    return {key.removeprefix("module."): value for key, value in checkpoint.items() if isinstance(key, str)}


def infer_rwkv_lm_args_from_checkpoint(path: str) -> dict[str, int]:
    """Infer native rwkv-lm construction args from a ``.pth`` state dict."""

    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        return {}

    state = _state_dict_items(_load_checkpoint_metadata(str(checkpoint_path)))
    inferred: dict[str, int] = {}

    emb = state.get("emb.weight")
    head = state.get("head.weight")
    if emb is not None and len(emb.shape) == 2:
        inferred["vocab_size"] = int(emb.shape[0])
        inferred["n_embd"] = int(emb.shape[1])
    elif head is not None and len(head.shape) == 2:
        inferred["vocab_size"] = int(head.shape[0])
        inferred["n_embd"] = int(head.shape[1])

    block_ids = []
    for name in state:
        match = re.match(r"blocks\.(\d+)\.", name)
        if match:
            block_ids.append(int(match.group(1)))
    if block_ids:
        inferred["n_layer"] = max(block_ids) + 1

    r_k = state.get("blocks.0.att.r_k")
    if r_k is not None and len(r_k.shape) == 2:
        inferred["head_size"] = int(r_k.shape[1])
        inferred["dim_att"] = int(r_k.shape[0] * r_k.shape[1])
    att_key = state.get("blocks.0.att.key.weight")
    if att_key is not None and len(att_key.shape) == 2:
        inferred.setdefault("dim_att", int(att_key.shape[0]))
    ffn_key = state.get("blocks.0.ffn.key.weight")
    if ffn_key is not None and len(ffn_key.shape) == 2:
        inferred["dim_ffn"] = int(ffn_key.shape[0])

    lora_shapes = {
        "decay_lora": "blocks.0.att.w1",
        "aaa_lora": "blocks.0.att.a1",
        "mv_lora": "blocks.0.att.v1",
        "gate_lora": "blocks.0.att.g1",
    }
    for arg_name, tensor_name in lora_shapes.items():
        tensor = state.get(tensor_name)
        if tensor is not None and len(tensor.shape) == 2:
            inferred[arg_name] = int(tensor.shape[1])

    ctx_match = re.search(r"ctx(\d+)", checkpoint_path.name)
    if ctx_match:
        inferred["ctx_len"] = int(ctx_match.group(1))

    return inferred


def _apply_checkpoint_metadata(values: dict[str, Any], model_config: Any, path: str) -> None:
    inferred = infer_rwkv_lm_args_from_checkpoint(path)
    for key, value in inferred.items():
        if _get(model_config, key) is None:
            values[key] = value


def _apply_model_config(values: dict[str, Any], model_config: Any) -> None:
    path = _get(model_config, "path")
    if path and path != "???":
        values["load_model"] = path
        _apply_checkpoint_metadata(values, model_config, path)
    for key in ("ctx_len", "n_layer", "n_embd", "head_size", "vocab_size"):
        _set_if_not_none(values, key, _get(model_config, key))


def _apply_engine_config(values: dict[str, Any], engine_config: Any) -> None:
    _set_if_not_none(values, "ctx_len", _get(engine_config, "ctx_len"))
    _set_if_not_none(values, "head_size", _get(engine_config, "head_size"))
    _set_if_not_none(values, "precision", _get(engine_config, "precision"))
    _set_if_not_none(values, "grad_cp", _get(engine_config, "grad_cp"))


def _apply_optimizer_config(values: dict[str, Any], optimizer_config: Any) -> None:
    _set_if_not_none(values, "lr_init", _get(optimizer_config, "lr"))
    _set_if_not_none(values, "weight_decay", _get(optimizer_config, "weight_decay"))
    _set_if_not_none(values, "warmup_steps", _get(optimizer_config, "lr_warmup_steps"))
    _set_if_not_none(values, "grad_clip", _get(optimizer_config, "clip_grad"))
    _set_if_not_none(values, "adam_eps", _get(optimizer_config, "adam_eps"))
    betas = _get(optimizer_config, "betas")
    if betas is not None:
        values["beta1"], values["beta2"] = betas
    _set_if_not_none(values, "beta1", _get(optimizer_config, "beta1"))
    _set_if_not_none(values, "beta2", _get(optimizer_config, "beta2"))
    native_kwargs = _get(optimizer_config, "native_optimizer_kwargs", {}) or {}
    values.update(native_kwargs)


def finalize_rwkv_lm_args(args: SimpleNamespace) -> SimpleNamespace:
    """Apply native rwkv-lm post-parse fields copied from ``train.py``."""

    args.my_timestamp = datetime.datetime.today().strftime("%Y-%m-%d-%H-%M-%S")
    args.enable_checkpointing = False
    args.replace_sampler_ddp = False
    args.logger = False
    args.gradient_clip_val = args.grad_clip
    args.num_sanity_val_steps = 0
    args.check_val_every_n_epoch = int(1e20)
    args.log_every_n_steps = int(1e20)
    args.max_epochs = -1
    args.betas = (args.beta1, args.beta2)
    args.real_bsz = int(args.num_nodes) * int(args.devices) * args.micro_bsz
    if args.dim_att <= 0:
        args.dim_att = args.n_embd
    if args.dim_ffn <= 0:
        args.dim_ffn = int((args.n_embd * 3.5) // 32 * 32)
    args.run_name = f"{args.vocab_size} ctx{args.ctx_len} L{args.n_layer} D{args.n_embd}"
    return args


def build_rwkv_lm_args(
    *,
    model_config: Any = None,
    engine_config: Any = None,
    optimizer_config: Any = None,
    overrides: dict[str, Any] | None = None,
) -> SimpleNamespace:
    """Build an argparse-like namespace for native rwkv-lm code."""

    values = dict(RWKV_LM_TRAIN_ARG_DEFAULTS)
    values.update(RWKV_LM_LIGHTNING_ARG_DEFAULTS)
    _apply_model_config(values, model_config)
    _apply_engine_config(values, engine_config)
    _apply_optimizer_config(values, optimizer_config)
    if overrides:
        values.update(overrides)
    values.setdefault("precision", _get(engine_config, "precision", "bf16"))
    return finalize_rwkv_lm_args(SimpleNamespace(**values))
