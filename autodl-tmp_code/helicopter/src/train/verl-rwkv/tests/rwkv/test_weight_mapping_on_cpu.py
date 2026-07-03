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


def _load_weight_mapping_module():
    path = Path("verl/models/rwkv/weight_mapping.py")
    spec = importlib.util.spec_from_file_location("rwkv_weight_mapping_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_strip_rwkv_lm_deepspeed_prefix_matches_native_checkpoint_rule():
    mapping = _load_weight_mapping_module()

    assert mapping.strip_rwkv_lm_deepspeed_prefix("_forward_module.blocks.0.att.key.weight") == (
        "blocks.0.att.key.weight"
    )
    assert mapping.strip_rwkv_lm_deepspeed_prefix("emb.weight") == "emb.weight"


def test_map_verl_to_rwkv_lm_keeps_native_names_without_explicit_override():
    mapping = _load_weight_mapping_module()
    weights = [
        ("_forward_module.blocks.0.att.key.weight", "key"),
        ("head.weight", "head"),
    ]

    assert list(mapping.map_verl_to_rwkv_lm(weights)) == [
        ("blocks.0.att.key.weight", "key"),
        ("head.weight", "head"),
    ]


def test_map_verl_to_rwkv_lm_uses_explicit_table_only_when_present():
    mapping = _load_weight_mapping_module()
    mapping.VERL_TO_RWKV_LM_WEIGHT_MAP["actor.emb.weight"] = "emb.weight"

    assert list(mapping.map_verl_to_rwkv_lm([("actor.emb.weight", "tensor"), ("head.weight", "head")])) == [
        ("emb.weight", "tensor"),
        ("head.weight", "head"),
    ]
