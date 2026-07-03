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

import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace


def _load_engine_module(name: str):
    path = Path(f"verl/workers/engine/rwkv_lm/{name}.py")
    spec = importlib.util.spec_from_file_location(f"rwkv_lm_{name}_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rwkv_lm_args_apply_verl_config_overrides_without_rewriting_native_names():
    args_module = _load_engine_module("args")
    model_config = SimpleNamespace(path="/models/rwkv.pth", ctx_len=2048, n_layer=12, n_embd=768, head_size=64)
    engine_config = SimpleNamespace(precision="bf16", ctx_len=4096, head_size=64, grad_cp=1)
    optimizer_config = SimpleNamespace(
        lr=1e-4,
        weight_decay=0.01,
        lr_warmup_steps=10,
        clip_grad=0.7,
        betas=(0.8, 0.95),
        beta1=None,
        beta2=0.99,
        adam_eps=1e-8,
        native_optimizer_kwargs={"lr_final": 2e-5, "head_chunk": 4096},
    )

    args = args_module.build_rwkv_lm_args(
        model_config=model_config,
        engine_config=engine_config,
        optimizer_config=optimizer_config,
        overrides={"devices": 2, "micro_bsz": 4},
    )

    assert args.load_model == "/models/rwkv.pth"
    assert args.ctx_len == 4096
    assert args.n_layer == 12
    assert args.n_embd == 768
    assert args.lr_init == 1e-4
    assert args.lr_final == 2e-5
    assert args.weight_decay == 0.01
    assert args.warmup_steps == 10
    assert args.grad_clip == 0.7
    assert args.beta1 == 0.8
    assert args.beta2 == 0.99
    assert args.adam_eps == 1e-8
    assert args.head_chunk == 4096
    assert args.grad_cp == 1
    assert args.real_bsz == 8


def test_rwkv_lm_args_infer_native_shape_from_checkpoint(tmp_path, monkeypatch):
    args_module = _load_engine_module("args")
    checkpoint = tmp_path / "rwkv7-test-ctx8192.pth"
    checkpoint.touch()

    class TensorMeta:
        def __init__(self, shape):
            self.shape = shape

    monkeypatch.setattr(
        args_module,
        "_load_checkpoint_metadata",
        lambda path: {
            "emb.weight": TensorMeta((65536, 2048)),
            "head.weight": TensorMeta((65536, 2048)),
            "blocks.0.att.r_k": TensorMeta((32, 64)),
            "blocks.0.att.key.weight": TensorMeta((2048, 2048)),
            "blocks.0.att.w1": TensorMeta((2048, 96)),
            "blocks.0.att.a1": TensorMeta((2048, 96)),
            "blocks.0.att.v1": TensorMeta((2048, 64)),
            "blocks.0.att.g1": TensorMeta((2048, 256)),
            "blocks.0.ffn.key.weight": TensorMeta((8192, 2048)),
            "blocks.1.ln1.weight": TensorMeta((2048,)),
        },
    )
    model_config = SimpleNamespace(
        path=str(checkpoint),
        ctx_len=None,
        n_layer=None,
        n_embd=None,
        head_size=None,
        vocab_size=None,
    )

    args = args_module.build_rwkv_lm_args(model_config=model_config)

    assert args.load_model == str(checkpoint)
    assert args.ctx_len == 8192
    assert args.n_layer == 2
    assert args.n_embd == 2048
    assert args.vocab_size == 65536
    assert args.head_size == 64
    assert args.dim_att == 2048
    assert args.dim_ffn == 8192
    assert args.decay_lora == 96
    assert args.aaa_lora == 96
    assert args.mv_lora == 64
    assert args.gate_lora == 256


def test_rwkv_lm_env_matches_native_train_py_assignments(monkeypatch):
    env_module = _load_engine_module("env")
    args = SimpleNamespace(
        my_testing="x070",
        kernel="@rwkv3",
        ctx_len=1024,
        head_size=64,
        head_chunk=0,
        precision="bf16",
        strategy="auto",
    )

    env = env_module.build_rwkv_lm_env(args)

    assert env == {
        "RWKV_MY_TESTING": "x070",
        "RWKV_KERNEL": "@rwkv3",
        "RWKV_CTXLEN": "1024",
        "RWKV_HEAD_SIZE": "64",
        "RWKV_HEAD_L2WRAP_CE_CHUNK": "0",
        "RWKV_FLOAT_MODE": "bf16",
        "RWKV_JIT_ON": "1",
    }

    monkeypatch.delenv("RWKV_CTXLEN", raising=False)
    with env_module.rwkv_lm_env(args) as applied:
        assert applied["RWKV_CTXLEN"] == "1024"
        assert os.environ["RWKV_CTXLEN"] == "1024"
    assert "RWKV_CTXLEN" not in os.environ


def test_rwkv_lm_env_disables_jit_for_deepspeed_stage_3():
    env_module = _load_engine_module("env")
    args = SimpleNamespace(
        my_testing="x070",
        kernel="",
        ctx_len=1024,
        head_size=64,
        head_chunk=0,
        precision="bf16",
        strategy="deepspeed_stage_3",
    )

    assert env_module.build_rwkv_lm_env(args)["RWKV_JIT_ON"] == "0"
