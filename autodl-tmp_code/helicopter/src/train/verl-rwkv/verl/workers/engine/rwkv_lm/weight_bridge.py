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

from collections.abc import Generator, Iterable
from typing import Any

import torch


RWKV_LM_ROLLOUT_WEIGHT_DTYPE = torch.bfloat16


def _to_rollout_weight_dtype(weight: Any) -> Any:
    if isinstance(weight, torch.Tensor) and torch.is_floating_point(weight):
        return weight.to(dtype=RWKV_LM_ROLLOUT_WEIGHT_DTYPE)
    return weight


def export_rwkv_lm_weights(weights: Iterable[tuple[str, Any]]) -> Generator[tuple[str, Any], None, None]:
    """Export native rwkv-lm weights for rollout synchronization."""

    from verl.models.rwkv.weight_mapping import map_verl_to_rwkv_lm

    for name, weight in map_verl_to_rwkv_lm(weights):
        yield name, _to_rollout_weight_dtype(weight)


def iter_rwkv_lm_state_dict_weights(model_or_state: Any) -> Generator[tuple[str, Any], None, None]:
    """Iterate native rwkv-lm ``state_dict`` weights without layout changes."""

    state = model_or_state.state_dict() if hasattr(model_or_state, "state_dict") else model_or_state
    yield from export_rwkv_lm_weights(state.items())
