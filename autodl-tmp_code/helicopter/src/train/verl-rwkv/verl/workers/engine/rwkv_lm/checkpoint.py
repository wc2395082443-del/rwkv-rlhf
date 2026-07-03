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

from pathlib import Path
from typing import Any

RWKV_LM_DEEPSPEED_PREFIX = "_forward_module."


def normalize_rwkv_lm_state_dict(state: dict[str, Any]) -> dict[str, Any]:
    """Normalize native rwkv-lm checkpoint keys.

    Source: native flat-layout ``rwkv-lm/train.py`` strips the ``_forward_module.``
    prefix after ``torch.load``.
    """

    normalized = dict(state)
    for key in list(normalized.keys()):
        if key.startswith(RWKV_LM_DEEPSPEED_PREFIX):
            normalized[key.replace(RWKV_LM_DEEPSPEED_PREFIX, "", 1)] = normalized[key]
            del normalized[key]
    return normalized


def load_rwkv_lm_checkpoint(path: str | Path, *, torch_module: Any = None) -> dict[str, Any]:
    """Load a native rwkv-lm ``.pth`` checkpoint."""

    if torch_module is None:
        import torch as torch_module

    state = torch_module.load(path, map_location="cpu", weights_only=True, mmap=True)
    return normalize_rwkv_lm_state_dict(state)
