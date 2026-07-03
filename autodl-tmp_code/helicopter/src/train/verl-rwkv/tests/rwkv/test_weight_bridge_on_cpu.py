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
import sys
import types
from pathlib import Path

import torch


def _install_weight_mapping_module():
    for name in ("verl", "verl.models", "verl.models.rwkv"):
        module = types.ModuleType(name)
        module.__path__ = []
        sys.modules[name] = module
    path = Path("verl/models/rwkv/weight_mapping.py")
    spec = importlib.util.spec_from_file_location("verl.models.rwkv.weight_mapping", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["verl.models.rwkv.weight_mapping"] = module
    spec.loader.exec_module(module)


def _load_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeModel:
    def state_dict(self):
        return {
            "_forward_module.emb.weight": "emb",
            "head.weight": "head",
        }


def test_rwkv_lm_weight_bridge_exports_state_dict_items_with_native_key_mapping():
    _install_weight_mapping_module()
    bridge = _load_module("rwkv_lm_weight_bridge_test", "verl/workers/engine/rwkv_lm/weight_bridge.py")

    assert list(bridge.iter_rwkv_lm_state_dict_weights(FakeModel())) == [
        ("emb.weight", "emb"),
        ("head.weight", "head"),
    ]


def test_rwkv_lm_weight_bridge_exports_floating_tensors_as_bf16():
    _install_weight_mapping_module()
    bridge = _load_module("rwkv_lm_weight_bridge_dtype_test", "verl/workers/engine/rwkv_lm/weight_bridge.py")

    weights = dict(bridge.export_rwkv_lm_weights([("emb.weight", torch.ones(2, dtype=torch.float32))]))

    assert weights["emb.weight"].dtype is torch.bfloat16
