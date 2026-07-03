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
from pathlib import Path

import pytest


def _load_batch_bridge():
    path = Path("verl/workers/engine/rwkv_lm/batch_bridge.py")
    spec = importlib.util.spec_from_file_location("rwkv_lm_batch_bridge_test", path)
    module = importlib.util.module_from_spec(spec)
    import sys

    sys.modules["rwkv_lm_batch_bridge_test"] = module
    spec.loader.exec_module(module)
    return module


def test_extract_rwkv_lm_forward_batch_preserves_verl_tensor_fields():
    bridge = _load_batch_bridge()
    data = {
        "input_ids": "input_ids",
        "attention_mask": "attention_mask",
        "position_ids": "position_ids",
        "responses": "responses",
        "response_mask": "response_mask",
    }

    batch = bridge.extract_rwkv_lm_forward_batch(data)

    assert batch.input_ids == "input_ids"
    assert batch.attention_mask == "attention_mask"
    assert batch.position_ids == "position_ids"
    assert batch.responses == "responses"
    assert batch.response_mask == "response_mask"


def test_extract_rwkv_lm_forward_batch_requires_input_ids():
    bridge = _load_batch_bridge()

    with pytest.raises(KeyError, match="input_ids"):
        bridge.extract_rwkv_lm_forward_batch({"responses": "responses"})


def test_build_verl_loss_model_output_keeps_existing_loss_keys():
    bridge = _load_batch_bridge()

    model_output = bridge.build_verl_loss_model_output(
        log_probs="log_probs",
        entropy="entropy",
        values="values",
        distillation_losses="distillation_losses",
    )

    assert model_output == {
        "log_probs": "log_probs",
        "entropy": "entropy",
        "values": "values",
        "distillation_losses": "distillation_losses",
    }
