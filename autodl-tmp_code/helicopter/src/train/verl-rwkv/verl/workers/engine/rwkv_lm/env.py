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
"""Environment bridge for native rwkv-lm.

The base ``RWKV_*`` assignments are copied from
the native flat-layout ``rwkv-lm/train.py``.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any


RWKV_LM_ENV_KEYS = (
    "RWKV_MY_TESTING",
    "RWKV_KERNEL",
    "RWKV_CTXLEN",
    "RWKV_HEAD_SIZE",
    "RWKV_HEAD_L2WRAP_CE_CHUNK",
    "RWKV_FLOAT_MODE",
    "RWKV_JIT_ON",
)


def build_rwkv_lm_env(args: Any, extra_env: Mapping[str, str] | None = None) -> dict[str, str]:
    """Build native ``RWKV_*`` env values from an rwkv-lm args namespace."""

    env = {
        "RWKV_MY_TESTING": str(args.my_testing),
        "RWKV_KERNEL": str(args.kernel),
        "RWKV_CTXLEN": str(args.ctx_len),
        "RWKV_HEAD_SIZE": str(args.head_size),
        "RWKV_HEAD_L2WRAP_CE_CHUNK": str(args.head_chunk),
        "RWKV_FLOAT_MODE": str(args.precision),
        "RWKV_JIT_ON": "0" if "deepspeed_stage_3" in str(args.strategy) else "1",
    }
    if extra_env:
        env.update({key: str(value) for key, value in extra_env.items()})
    return env


@contextmanager
def rwkv_lm_env(args: Any, extra_env: Mapping[str, str] | None = None) -> Iterator[dict[str, str]]:
    """Temporarily apply native rwkv-lm environment values."""

    env = build_rwkv_lm_env(args, extra_env=extra_env)
    old_values = {key: os.environ.get(key) for key in env}
    os.environ.update(env)
    try:
        yield env
    finally:
        for key, old_value in old_values.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value
