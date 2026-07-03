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

"""Tokenizer construction shared by PPO trainer entrypoints."""

from __future__ import annotations

from typing import Any

from omegaconf import DictConfig


RWKV_NATIVE_MODEL_TARGET = "verl.models.rwkv.RWKVNativeModelConfig"


def _get(config: Any, key: str, default: Any = None) -> Any:
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


def is_rwkv_native_model_config(model_config: Any) -> bool:
    """Return whether a model config describes native RWKV weights."""

    return _get(model_config, "_target_") == RWKV_NATIVE_MODEL_TARGET


def build_ppo_tokenizer_and_processor(config: DictConfig) -> tuple[Any, Any]:
    """Build tokenizer and processor for PPO datasets.

    Native RWKV checkpoints are `.pth` files, so they cannot be passed to
    Hugging Face `AutoTokenizer.from_pretrained`. Use the native tokenizer
    bridge instead and leave the processor unset because RWKV is text-only here.
    """

    model_config = config.actor_rollout_ref.model
    if is_rwkv_native_model_config(model_config):
        from verl.models.rwkv import build_rwkv_tokenizer

        tokenizer = build_rwkv_tokenizer(
            tokenizer_path=_get(model_config, "tokenizer_path"),
            pickleable=True,
        )
        return tokenizer, None

    from verl.utils import hf_processor, hf_tokenizer
    from verl.utils.fs import copy_to_local

    local_path = copy_to_local(model_config.path, use_shm=model_config.get("use_shm", False))
    trust_remote_code = config.data.get("trust_remote_code", False)
    tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
    processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)
    return tokenizer, processor
