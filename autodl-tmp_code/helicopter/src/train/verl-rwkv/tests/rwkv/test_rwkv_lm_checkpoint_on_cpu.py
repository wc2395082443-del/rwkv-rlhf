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


def _load_checkpoint_module():
    path = Path("verl/workers/engine/rwkv_lm/checkpoint.py")
    spec = importlib.util.spec_from_file_location("rwkv_lm_checkpoint_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeTorch:
    def __init__(self, state=None):
        self.state = state or {}
        self.load_kwargs = None

    def load(self, path, **kwargs):
        self.load_kwargs = {"path": path, **kwargs}
        return dict(self.state)


def test_normalize_rwkv_lm_state_dict_strips_deepspeed_forward_prefix():
    checkpoint = _load_checkpoint_module()

    normalized = checkpoint.normalize_rwkv_lm_state_dict(
        {
            "_forward_module.blocks.0.att.key.weight": "prefixed",
            "emb.weight": "plain",
        }
    )

    assert normalized == {
        "blocks.0.att.key.weight": "prefixed",
        "emb.weight": "plain",
    }


def test_load_rwkv_lm_checkpoint_uses_native_torch_load_arguments():
    checkpoint = _load_checkpoint_module()
    fake_torch = FakeTorch({"_forward_module.head.weight": "tensor"})

    loaded = checkpoint.load_rwkv_lm_checkpoint("/tmp/rwkv.pth", torch_module=fake_torch)

    assert fake_torch.load_kwargs == {
        "path": "/tmp/rwkv.pth",
        "map_location": "cpu",
        "weights_only": True,
        "mmap": True,
    }
    assert loaded == {"head.weight": "tensor"}
