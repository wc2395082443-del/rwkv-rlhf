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

from omegaconf import OmegaConf
import pytest


def test_rwkv_native_rollout_model_config_uses_vllm_pth_config(tmp_path):
    pytest.importorskip("vllm")

    from verl.workers.rollout.vllm_rollout.vllm_async_server import _init_rwkv_rollout_model_config

    checkpoint = tmp_path / "rwkv7-g1f-1.5b-20260419-ctx8192.pth"
    checkpoint.touch()
    model_config = OmegaConf.create(
        {
            "_target_": "verl.models.rwkv.RWKVNativeModelConfig",
            "path": str(checkpoint),
            "load_tokenizer": False,
            "rwkv_lm_path": "/workspace/Projects/MachineLearning/rwkv-lm",
            "lora": {},
        }
    )

    rollout_model_config = _init_rwkv_rollout_model_config(model_config)

    assert rollout_model_config.local_path == str(checkpoint)
    assert rollout_model_config.tokenizer is None
    assert rollout_model_config.processor is None
    assert rollout_model_config.lora == {}
    assert rollout_model_config.lora_rank == 0
    assert rollout_model_config.trust_remote_code is False
    assert rollout_model_config.hf_config.model_type == "rwkv7"
    assert rollout_model_config.hf_config.architectures == ["RWKV7ForCausalLM"]
    assert rollout_model_config.hf_config.hidden_size == 2048
    assert rollout_model_config.hf_config.num_hidden_layers == 24
    assert rollout_model_config.hf_config.max_position_embeddings == 8192
