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
"""Loss boundary for the native RWKV-LM engine.

RWKV model compute is delegated to rwkv-lm, but reward-conditioned actor
objectives must stay on Verl's native loss path. This file is intentionally a
thin bridge: it imports Verl loss entrypoints lazily and calls them directly.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "compute_verl_distillation_loss",
    "compute_verl_policy_loss",
    "get_verl_policy_loss_fn",
    "reject_rwkv_lm_ce_for_rl_objective",
]


def compute_verl_policy_loss(config: Any, model_output: Any, data: Any, dp_group: Any = None):
    """Call Verl's existing PPO/GRPO policy loss entrypoint.

    Source path: verl/workers/utils/losses.py::ppo_loss.
    """

    from verl.workers.utils.losses import ppo_loss

    return ppo_loss(config, model_output, data, dp_group)


def compute_verl_distillation_loss(
    config: Any,
    distillation_config: Any,
    model_output: Any = None,
    data: Any = None,
    dp_group: Any = None,
    student_logits: Any = None,
    data_format: str = "thd",
):
    """Call Verl's existing on-policy distillation loss entrypoint.

    Source path: verl/trainer/distillation/losses.py::distillation_ppo_loss.
    """

    from verl.trainer.distillation.losses import distillation_ppo_loss

    return distillation_ppo_loss(
        config,
        distillation_config,
        model_output=model_output,
        data=data,
        dp_group=dp_group,
        student_logits=student_logits,
        data_format=data_format,
    )


def get_verl_policy_loss_fn(loss_mode: str):
    """Return a policy-loss implementation from Verl's existing registry.

    Source path: verl/trainer/ppo/core_algos.py::get_policy_loss_fn.
    """

    from verl.trainer.ppo.core_algos import get_policy_loss_fn

    return get_policy_loss_fn(loss_mode)


def reject_rwkv_lm_ce_for_rl_objective() -> None:
    """Fail closed when a caller attempts to use rwkv-lm CE as an RL objective."""

    raise RuntimeError(
        "rwkv-lm CE training_step is not a Verl RL objective; use Verl "
        "PPO/GRPO/distillation loss through loss_bridge instead."
    )
