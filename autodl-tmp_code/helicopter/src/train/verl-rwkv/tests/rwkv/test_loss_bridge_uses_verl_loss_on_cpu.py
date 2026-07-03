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

import pytest


def _load_loss_bridge():
    path = Path("verl/workers/engine/rwkv_lm/loss_bridge.py")
    spec = importlib.util.spec_from_file_location("rwkv_lm_loss_bridge_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _install_verl_loss_modules():
    calls = []
    for name in (
        "verl",
        "verl.workers",
        "verl.workers.utils",
        "verl.trainer",
        "verl.trainer.distillation",
        "verl.trainer.ppo",
    ):
        module = types.ModuleType(name)
        module.__path__ = []
        sys.modules[name] = module

    worker_losses = types.ModuleType("verl.workers.utils.losses")
    distillation_losses = types.ModuleType("verl.trainer.distillation.losses")
    core_algos = types.ModuleType("verl.trainer.ppo.core_algos")

    def ppo_loss(*args, **kwargs):
        calls.append(("ppo_loss", args, kwargs))
        return "ppo-loss"

    def distillation_ppo_loss(*args, **kwargs):
        calls.append(("distillation_ppo_loss", args, kwargs))
        return "distillation-loss"

    def get_policy_loss_fn(name):
        calls.append(("get_policy_loss_fn", (name,), {}))
        return f"policy-loss-fn:{name}"

    worker_losses.ppo_loss = ppo_loss
    distillation_losses.distillation_ppo_loss = distillation_ppo_loss
    core_algos.get_policy_loss_fn = get_policy_loss_fn

    sys.modules["verl.workers.utils.losses"] = worker_losses
    sys.modules["verl.trainer.distillation.losses"] = distillation_losses
    sys.modules["verl.trainer.ppo.core_algos"] = core_algos
    return calls


def test_policy_loss_bridge_delegates_to_verl_ppo_loss():
    calls = _install_verl_loss_modules()
    bridge = _load_loss_bridge()

    result = bridge.compute_verl_policy_loss("config", "model_output", "data", dp_group="dp")

    assert result == "ppo-loss"
    assert calls == [
        (
            "ppo_loss",
            ("config", "model_output", "data", "dp"),
            {},
        )
    ]


def test_distillation_loss_bridge_delegates_to_verl_distillation_ppo_loss():
    calls = _install_verl_loss_modules()
    bridge = _load_loss_bridge()

    result = bridge.compute_verl_distillation_loss(
        "actor_config",
        "distillation_config",
        model_output="model_output",
        data="data",
        dp_group="dp",
        student_logits="student_logits",
        data_format="bshd",
    )

    assert result == "distillation-loss"
    assert calls == [
        (
            "distillation_ppo_loss",
            ("actor_config", "distillation_config"),
            {
                "model_output": "model_output",
                "data": "data",
                "dp_group": "dp",
                "student_logits": "student_logits",
                "data_format": "bshd",
            },
        )
    ]


def test_policy_loss_fn_lookup_uses_verl_registry():
    calls = _install_verl_loss_modules()
    bridge = _load_loss_bridge()

    assert bridge.get_verl_policy_loss_fn("gspo") == "policy-loss-fn:gspo"
    assert calls == [("get_policy_loss_fn", ("gspo",), {})]


def test_rwkv_lm_ce_loss_is_rejected_for_rl_objectives():
    bridge = _load_loss_bridge()

    with pytest.raises(RuntimeError, match="rwkv-lm CE training_step is not a Verl RL objective"):
        bridge.reject_rwkv_lm_ce_for_rl_objective()
