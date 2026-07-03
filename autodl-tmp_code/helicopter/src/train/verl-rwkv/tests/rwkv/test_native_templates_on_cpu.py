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

import pytest
import torch
from hydra import compose, initialize_config_dir


def test_rwkv_model_config_exposes_generic_worker_compat_fields():
    from verl.models.rwkv import RWKVNativeModelConfig

    model = RWKVNativeModelConfig(path="/models/rwkv", load_tokenizer=False)

    assert model.get("use_remove_padding", False) is False
    assert model.get("use_fused_kernels", False) is False
    assert model.get("tokenizer") is None
    assert model.get("processor") is None
    assert model.get("custom_chat_template") is None
    assert model.lora.get("merge", False) is False

    model.model_type = "value_model"
    assert model.model_type == "value_model"


def test_rwkv_grpo_vllm_hydra_entrypoint_composes():
    from verl.utils.config import validate_config

    config_dir = str(Path("verl/trainer/config").resolve())
    overrides = [
        "algorithm.adv_estimator=grpo",
        "algorithm.use_kl_in_reward=False",
        "actor@actor_rollout_ref.actor=rwkv_lm",
        "ref@actor_rollout_ref.ref=rwkv_lm",
        "model@actor_rollout_ref.model=rwkv_native",
        "data.train_batch_size=2",
        "actor_rollout_ref.model.path=/models/rwkv.pth",
        "actor_rollout_ref.model.rwkv_lm_path=/src/rwkv-lm",
        "actor_rollout_ref.actor.engine.rwkv_lm_path=/src/rwkv-lm",
        "actor_rollout_ref.actor.ppo_mini_batch_size=2",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1",
        "actor_rollout_ref.actor.use_dynamic_bsz=True",
        "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=2048",
        "actor_rollout_ref.actor.use_kl_loss=True",
        "actor_rollout_ref.ref.engine.rwkv_lm_path=/src/rwkv-lm",
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1",
        "actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True",
        "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=2048",
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.load_format=auto",
        "+actor_rollout_ref.rollout.engine_kwargs.vllm.tokenizer_mode=rwkv",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1",
        "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True",
        "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=2048",
        "critic.enable=False",
        "trainer.logger=['console']",
        "trainer.nnodes=1",
        "trainer.n_gpus_per_node=1",
    ]

    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="ppo_trainer", overrides=overrides)

    assert cfg.actor_rollout_ref.actor.strategy == "rwkv_lm"
    assert cfg.actor_rollout_ref.actor.engine.strategy == "rwkv_lm"
    assert cfg.actor_rollout_ref.actor.optim._target_ == "verl.workers.config.RWKVLMOptimizerConfig"
    assert cfg.actor_rollout_ref.ref.strategy == "rwkv_lm"
    assert cfg.actor_rollout_ref.ref.engine.forward_only is True
    assert cfg.actor_rollout_ref.model._target_ == "verl.models.rwkv.RWKVNativeModelConfig"
    assert cfg.actor_rollout_ref.rollout.name == "vllm"
    assert cfg.actor_rollout_ref.rollout.engine_kwargs.vllm.tokenizer_mode == "rwkv"
    assert "nano_vllm_rwkv" not in cfg.actor_rollout_ref.rollout.engine_kwargs
    assert cfg.actor_rollout_ref.rollout.val_kwargs.do_sample is False
    assert cfg.actor_rollout_ref.rollout.multi_turn.enable is False
    skip_config = cfg.actor_rollout_ref.rollout.get("skip")
    assert skip_config is None or skip_config.enable is False
    assert cfg.actor_rollout_ref.rollout.disaggregation.enabled is False
    assert cfg.critic.enable is False
    validate_config(cfg, use_reference_policy=True, use_critic=False)


def test_rwkv_grpo_reward_delegates_to_math_verify(monkeypatch):
    from examples.rwkv_trainer import math_verify_reward

    calls = []

    def fake_compute_score(model_output, ground_truth):
        calls.append((model_output, ground_truth))
        return 1.0

    monkeypatch.setattr(math_verify_reward.math_verify, "compute_score", fake_compute_score)

    assert (
        math_verify_reward.compute_score(
            data_source="gsm8k",
            solution_str="<think>...</think>\\boxed{2}",
            ground_truth="2",
            extra_info={"index": 0},
        )
        == 1.0
    )
    assert calls == [("<think>...</think>\\boxed{2}", "2")]


def test_rwkv_engine_is_registered_and_exposes_native_methods():
    from verl.workers.config import RWKVLMEngineConfig, RWKVLMOptimizerConfig
    from verl.workers.engine.base import BaseEngine, EngineRegistry
    from verl.workers.engine.rwkv_lm import RWKVLMEngine, RWKVLMEngineWithLMHead

    assert issubclass(RWKVLMEngine, BaseEngine)
    assert issubclass(RWKVLMEngineWithLMHead, RWKVLMEngine)
    assert "rwkv_lm" in EngineRegistry._engines.get("language_model", {})

    class MissingRunner:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def build_model(self):
            raise FileNotFoundError("/opt/rwkv-lm")

    engine = RWKVLMEngine(
        model_config=None,
        engine_config=RWKVLMEngineConfig(rwkv_lm_path="/opt/rwkv-lm"),
        optimizer_config=RWKVLMOptimizerConfig(lr=1e-4),
        checkpoint_config=None,
        runner_cls=MissingRunner,
    )

    with pytest.raises(FileNotFoundError):
        engine.initialize()


def test_rwkv_engine_initializes_model_on_current_device(monkeypatch):
    from verl.workers.config import RWKVLMEngineConfig, RWKVLMOptimizerConfig
    from verl.workers.engine.rwkv_lm import RWKVLMEngine, transformer_impl

    class FakeModel:
        def __init__(self):
            self.trainer = None
            self.devices = []
            self.dtypes = []

        def to(self, *args, **kwargs):
            device = kwargs.get("device", args[0] if args else None)
            self.devices.append(device)
            self.dtypes.append(kwargs.get("dtype"))
            return self

    class FakeRunner:
        model = FakeModel()

        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def build_model(self):
            return self.model

        def build_train_callback(self):
            return object()

    monkeypatch.setattr(transformer_impl, "get_device_name", lambda: "cuda")

    engine = RWKVLMEngine(
        model_config=None,
        engine_config=RWKVLMEngineConfig(rwkv_lm_path="/opt/rwkv-lm", param_offload=False),
        optimizer_config=RWKVLMOptimizerConfig(lr=1e-4),
        checkpoint_config=None,
        runner_cls=FakeRunner,
    )

    engine.initialize()

    assert FakeRunner.model.devices == ["cuda"]
    assert FakeRunner.model.dtypes == [torch.bfloat16]


def test_rwkv_engine_weight_generator_accepts_common_update_kwargs():
    from verl.workers.config import RWKVLMEngineConfig, RWKVLMOptimizerConfig
    from verl.workers.engine.rwkv_lm import RWKVLMEngine

    class FakeModel:
        def state_dict(self):
            return {"_forward_module.emb.weight": "emb"}

    engine = RWKVLMEngine(
        model_config=None,
        engine_config=RWKVLMEngineConfig(rwkv_lm_path="/opt/rwkv-lm"),
        optimizer_config=RWKVLMOptimizerConfig(lr=1e-4),
        checkpoint_config=None,
    )
    engine.model = FakeModel()

    weights, peft_config = engine.get_per_tensor_param(layered_summon=False, base_sync_done=True)

    assert list(weights) == [("emb.weight", "emb")]
    assert peft_config is None


def test_rwkv_engine_reports_torch_distributed_data_parallel(monkeypatch):
    from verl.workers.config import RWKVLMEngineConfig, RWKVLMOptimizerConfig
    from verl.workers.engine.rwkv_lm import RWKVLMEngine, transformer_impl

    monkeypatch.setattr(transformer_impl.torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(transformer_impl.torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(transformer_impl.torch.distributed, "get_world_size", lambda: 8)
    monkeypatch.setattr(transformer_impl.torch.distributed, "get_rank", lambda: 3)

    engine = RWKVLMEngine(
        model_config=None,
        engine_config=RWKVLMEngineConfig(rwkv_lm_path="/opt/rwkv-lm"),
        optimizer_config=RWKVLMOptimizerConfig(lr=1e-4),
        checkpoint_config=None,
    )

    assert engine.get_data_parallel_size() == 8
    assert engine.get_data_parallel_rank() == 3
    assert engine.get_data_parallel_group() is transformer_impl.torch.distributed.group.WORLD


def test_rwkv_rollout_uses_canonical_vllm_registration():
    pytest.importorskip("vllm")

    from verl.workers.rollout.base import _ROLLOUT_REGISTRY, BaseRollout
    from verl.workers.rollout.replica import RolloutReplica, RolloutReplicaRegistry
    from verl.workers.rollout.vllm_rollout import ServerAdapter
    from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMReplica

    assert issubclass(ServerAdapter, BaseRollout)
    assert issubclass(vLLMReplica, RolloutReplica)
    assert _ROLLOUT_REGISTRY[("vllm", "async")] == "verl.workers.rollout.vllm_rollout.ServerAdapter"
    assert ("nano_vllm_rwkv", "async") not in _ROLLOUT_REGISTRY
    assert RolloutReplicaRegistry.get("vllm") is vLLMReplica
    with pytest.raises(ValueError, match="Unknown rollout mode"):
        RolloutReplicaRegistry.get("nano_vllm_rwkv")
