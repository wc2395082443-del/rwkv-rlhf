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
"""Batch boundary for native rwkv-lm model compute.

This module does not implement RWKV forward or RL losses. It only preserves
the Verl tensor fields that the native RWKV compute path and Verl loss bridge
need to exchange.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RWKVLMForwardBatch:
    """Verl batch fields needed by the future native RWKV forward adapter."""

    input_ids: Any
    attention_mask: Any = None
    position_ids: Any = None
    responses: Any = None
    response_mask: Any = None


def _has(data: Any, key: str) -> bool:
    if isinstance(data, dict):
        return key in data
    try:
        data[key]
    except KeyError:
        return False
    return True


def _get(data: Any, key: str, default: Any = None) -> Any:
    if isinstance(data, dict):
        return data.get(key, default)
    return data[key] if _has(data, key) else default


def extract_rwkv_lm_forward_batch(data: Any) -> RWKVLMForwardBatch:
    """Extract Verl tensors without changing objective or sequence semantics."""

    if not _has(data, "input_ids"):
        raise KeyError("input_ids is required for native rwkv-lm forward")
    return RWKVLMForwardBatch(
        input_ids=_get(data, "input_ids"),
        attention_mask=_get(data, "attention_mask"),
        position_ids=_get(data, "position_ids"),
        responses=_get(data, "responses"),
        response_mask=_get(data, "response_mask"),
    )


def build_verl_loss_model_output(
    *,
    log_probs: Any,
    entropy: Any = None,
    values: Any = None,
    **extra: Any,
) -> dict[str, Any]:
    """Build ``model_output`` for Verl's existing loss functions."""

    model_output: dict[str, Any] = {"log_probs": log_probs}
    if entropy is not None:
        model_output["entropy"] = entropy
    if values is not None:
        model_output["values"] = values
    model_output.update(extra)
    return model_output
