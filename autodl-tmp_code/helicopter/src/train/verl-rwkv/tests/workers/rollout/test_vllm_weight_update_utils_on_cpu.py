# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

import asyncio
import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch
from packaging import version

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_weight_update_utils():
    module_path = _REPO_ROOT / "verl/workers/rollout/vllm_rollout/weight_update_utils.py"
    spec = importlib.util.spec_from_file_location("weight_update_utils", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


_weight_update_utils = _load_weight_update_utils()
apply_buffer_updates = _weight_update_utils.apply_buffer_updates
split_buffer_updates = _weight_update_utils.split_buffer_updates


def _load_vllm_rollout_utils():
    """Load vllm_rollout/utils.py with heavyweight deps stubbed.

    Injected ``sys.modules`` entries are restored afterwards so the fakes do not
    leak into other tests; the loaded module keeps working since it binds the
    names it needs at import time.
    """
    module_name = "verl.workers.rollout.vllm_rollout.utils"
    module_path = _REPO_ROOT / "verl/workers/rollout/vllm_rollout/utils.py"

    fake_outputs = types.ModuleType("vllm.outputs")

    class _FakeRequestOutput:
        pass

    fake_outputs.RequestOutput = _FakeRequestOutput
    fake_vllm = types.ModuleType("vllm")
    fake_vllm.outputs = fake_outputs

    fake_vllm_third_party = types.ModuleType("verl.third_party.vllm")
    fake_vllm_third_party.VLLM_SLEEP_LEVEL = 1
    fake_vllm_third_party.get_version = lambda pkg: "0.8.0"

    fake_vllm_utils = types.ModuleType("verl.utils.vllm")

    class _FakeTensorLoRARequest:
        pass

    class _FakeVLLMHijack:
        @staticmethod
        def hijack():
            return None

    fake_vllm_utils.TensorLoRARequest = _FakeTensorLoRARequest
    fake_vllm_utils.VLLMHijack = _FakeVLLMHijack

    fake_vllm_patch = types.ModuleType("verl.utils.vllm.patch")
    fake_vllm_patch.patch_vllm_moe_model_weight_loader = lambda model: None

    fake_vllm_fp8 = types.ModuleType("verl.utils.vllm.vllm_fp8_utils")
    fake_vllm_fp8.apply_vllm_fp8_patches = lambda: None
    fake_vllm_fp8.is_fp8_model = lambda config: False
    fake_vllm_fp8.load_quanted_weights = lambda weights, runner, is_drafter=False: weights

    fake_platform = types.ModuleType("verl.plugin.platform")
    fake_platform_instance = types.SimpleNamespace(
        communication_backend_name=lambda: "nccl",
        current_device=lambda: 0,
        device_module=torch.cuda,
        device_name="cuda",
        empty_cache=lambda: None,
        get_device_capability=lambda device_id=0: (None, None),
        get_device_uuid=lambda device_id=0: f"GPU-{device_id}",
        is_available=lambda: True,
        is_ipc_supported=lambda: True,
        manual_seed=lambda seed: None,
        manual_seed_all=lambda seed: None,
        ray_resource_name=lambda: "GPU",
        set_allocator_settings=lambda settings: None,
        vendor_name="nvidia",
        visible_devices_envvar=lambda: "CUDA_VISIBLE_DEVICES",
    )
    fake_platform.get_platform = lambda: fake_platform_instance

    fakes = {
        "vllm": fake_vllm,
        "vllm.outputs": fake_outputs,
        "verl.third_party.vllm": fake_vllm_third_party,
        "verl.utils.vllm": fake_vllm_utils,
        "verl.utils.vllm.patch": fake_vllm_patch,
        "verl.utils.vllm.vllm_fp8_utils": fake_vllm_fp8,
        "verl.plugin.platform": fake_platform,
        "verl.workers.rollout.vllm_rollout.weight_update_utils": _weight_update_utils,
    }

    saved = {name: sys.modules.get(name) for name in fakes}
    saved["verl.utils.device"] = sys.modules.get("verl.utils.device")
    try:
        sys.modules.update(fakes)
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        spec.loader.exec_module(module)
    finally:
        for name, prev in saved.items():
            if prev is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prev
    return module


_vllm_rollout_utils = _load_vllm_rollout_utils()
vLLMColocateWorkerExtension = _vllm_rollout_utils.vLLMColocateWorkerExtension


class _ToyBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4, bias=False)
        self.register_buffer("e_score_correction_bias", torch.zeros(4, dtype=torch.float32))


class _ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([_ToyBlock()])


class _FakeVllmConfig:
    def __init__(self, speculative_config=None):
        self.speculative_config = speculative_config
        self.model_config = object()


class _FakeModelRunner:
    def __init__(self, inner_model, speculative_config=None, model_state=None):
        self.model = inner_model
        self.vllm_config = _FakeVllmConfig(speculative_config=speculative_config)
        if model_state is not None:
            self.model_state = model_state


def test_split_buffer_updates_routes_registered_buffers():
    model = _ToyModel()
    weights = [
        ("model.layers.0.linear.weight", torch.ones(4, 4, dtype=torch.float32)),
        ("model.layers.0.e_score_correction_bias", torch.arange(4, dtype=torch.float32)),
    ]

    param_updates, buffer_updates, named_buffers = split_buffer_updates(model, weights)

    assert [name for name, _ in param_updates] == ["model.layers.0.linear.weight"]
    assert [name for name, _ in buffer_updates] == ["model.layers.0.e_score_correction_bias"]
    assert "model.layers.0.e_score_correction_bias" in named_buffers


def test_apply_buffer_updates_copies_buffer_values():
    model = _ToyModel()
    updates = [("model.layers.0.e_score_correction_bias", torch.arange(4, dtype=torch.float32) + 1)]

    loaded = apply_buffer_updates(model, updates)

    assert loaded == 1
    torch.testing.assert_close(
        model.model.layers[0].e_score_correction_bias, torch.tensor([1, 2, 3, 4], dtype=torch.float32)
    )


def test_apply_buffer_updates_ignores_non_buffer_weights():
    model = _ToyModel()
    weights = [("model.layers.0.linear.weight", torch.ones(4, 4, dtype=torch.float32))]

    loaded = apply_buffer_updates(model, weights)

    assert loaded == 0
    assert torch.count_nonzero(model.model.layers[0].e_score_correction_bias) == 0


def test_vllm_update_weights_loads_params_and_buffers():
    model = _ToyModel()
    loaded_param_names = []
    apply_named_buffers = []

    def _fake_load_weights(weights):
        loaded_param_names.extend(name for name, _ in weights)

    model.load_weights = _fake_load_weights

    original_apply_buffer_updates = _vllm_rollout_utils.apply_buffer_updates

    def _spy_apply_buffer_updates(inner_model, buffer_updates, named_buffers=None):
        apply_named_buffers.append(named_buffers)
        return original_apply_buffer_updates(inner_model, buffer_updates, named_buffers=named_buffers)

    _vllm_rollout_utils.apply_buffer_updates = _spy_apply_buffer_updates

    worker = object.__new__(vLLMColocateWorkerExtension)
    worker.model_runner = _FakeModelRunner(model)

    weights = [
        ("model.layers.0.linear.weight", torch.ones(4, 4, dtype=torch.float32)),
        ("model.layers.0.e_score_correction_bias", torch.arange(4, dtype=torch.float32) + 5),
    ]

    try:
        worker._update_weights(weights, peft_config=None, base_sync_done=False)
    finally:
        _vllm_rollout_utils.apply_buffer_updates = original_apply_buffer_updates

    assert loaded_param_names == ["model.layers.0.linear.weight"]
    assert apply_named_buffers and apply_named_buffers[0] is not None
    torch.testing.assert_close(
        model.model.layers[0].e_score_correction_bias, torch.tensor([5, 6, 7, 8], dtype=torch.float32)
    )


def test_vllm_update_weights_syncs_buffers_to_mtp_drafter():
    """When an MTP drafter is synced, its registered buffers must be updated too."""
    main_model = _ToyModel()
    drafter_model = _ToyModel()
    main_model.load_weights = lambda weights: None
    drafter_model.load_weights = lambda weights: None

    class _SpecConfig:
        method = "mtp"
        draft_model_config = object()

    class _Drafter:
        def __init__(self, m):
            self.model = m

    worker = object.__new__(vLLMColocateWorkerExtension)
    worker.model_runner = _FakeModelRunner(main_model, speculative_config=_SpecConfig())
    worker.model_runner.drafter = _Drafter(drafter_model)

    weights = [
        ("model.layers.0.linear.weight", torch.ones(4, 4, dtype=torch.float32)),
        ("model.layers.0.e_score_correction_bias", torch.arange(4, dtype=torch.float32) + 5),
    ]

    worker._update_weights(weights, peft_config=None, base_sync_done=False)

    expected = torch.tensor([5, 6, 7, 8], dtype=torch.float32)
    torch.testing.assert_close(main_model.model.layers[0].e_score_correction_bias, expected)
    torch.testing.assert_close(drafter_model.model.layers[0].e_score_correction_bias, expected)


def test_weight_update_cache_reset_preempts_running_requests_and_resets_aux_caches():
    calls = []

    class _FakeEngine:
        async def reset_prefix_cache(self, **kwargs):
            calls.append(("reset_prefix_cache", kwargs))
            return True

        async def reset_mm_cache(self):
            calls.append("reset_mm_cache")

        async def reset_encoder_cache(self):
            calls.append("reset_encoder_cache")

    asyncio.run(_vllm_rollout_utils.reset_vllm_weight_update_caches(_FakeEngine(), version.parse("0.16.0")))

    assert calls == [
        (
            "reset_prefix_cache",
            {"reset_connector": True, "reset_running_requests": True},
        ),
        "reset_mm_cache",
        "reset_encoder_cache",
    ]


def test_weight_update_cache_reset_fails_closed_when_prefix_cache_reset_fails():
    class _FakeEngine:
        async def reset_prefix_cache(self, **kwargs):
            return False

        async def reset_mm_cache(self):
            raise AssertionError("mm cache reset must not run after prefix cache reset failure")

    with pytest.raises(RuntimeError, match="Failed to reset vLLM prefix cache after weight update"):
        asyncio.run(_vllm_rollout_utils.reset_vllm_weight_update_caches(_FakeEngine(), version.parse("0.13.0")))


def test_vllm_update_weights_from_ipc_wraps_transactional_model_update():
    model = _ToyModel()
    events = []
    active = False

    def _start_weight_update():
        nonlocal active
        events.append("start")
        active = True
        return True

    def _load_weights(weights):
        assert active
        events.append(("load", [name for name, _ in weights]))

    def _finish_weight_update():
        nonlocal active
        assert active
        events.append("finish")
        active = False

    def _abort_weight_update():
        events.append("abort")

    model.start_weight_update = _start_weight_update
    model.load_weights = _load_weights
    model.finish_weight_update = _finish_weight_update
    model.abort_weight_update = _abort_weight_update

    worker = object.__new__(vLLMColocateWorkerExtension)
    model_state = types.SimpleNamespace(reset_after_weight_update=lambda: events.append("reset_state"))
    worker.model_runner = _FakeModelRunner(model, model_state=model_state)
    worker.device = torch.device("cpu")
    worker.local_rank = 0
    worker._is_qat_model = False
    worker._is_modelopt_qat = False

    fake_platforms = types.ModuleType("vllm.platforms")
    fake_platforms.current_platform = types.SimpleNamespace(device_type="cuda")

    fake_transfer = types.ModuleType("verl.workers.rollout.vllm_rollout.bucketed_weight_transfer")

    class _FakeBucketedWeightReceiver:
        def __init__(self, zmq_handle, device, use_shm):
            assert zmq_handle.startswith("ipc:///tmp/rl-colocate-zmq-")
            assert device == torch.device("cpu")
            assert use_shm is False

        def receive_weights(self, on_bucket_received):
            on_bucket_received(
                [
                    ("model.layers.0.linear.weight", torch.ones(4, 4, dtype=torch.float32)),
                    ("model.layers.0.e_score_correction_bias", torch.arange(4, dtype=torch.float32) + 7),
                ]
            )

    fake_transfer.BucketedWeightReceiver = _FakeBucketedWeightReceiver

    fake_vllm = types.ModuleType("vllm")
    fake_vllm.__path__ = []
    fake_model_executor = types.ModuleType("vllm.model_executor")
    fake_model_executor.__path__ = []
    fake_model_loader = types.ModuleType("vllm.model_executor.model_loader")
    fake_model_loader.__path__ = []
    fake_model_loader_utils = types.ModuleType("vllm.model_executor.model_loader.utils")
    fake_model_loader_utils.process_weights_after_loading = lambda *args: events.append("post_process")

    fakes = {
        "vllm": fake_vllm,
        "vllm.platforms": fake_platforms,
        "vllm.model_executor": fake_model_executor,
        "vllm.model_executor.model_loader": fake_model_loader,
        "vllm.model_executor.model_loader.utils": fake_model_loader_utils,
        "verl.workers.rollout.vllm_rollout.bucketed_weight_transfer": fake_transfer,
    }
    saved = {name: sys.modules.get(name) for name in fakes}

    try:
        sys.modules.update(fakes)
        worker.update_weights_from_ipc(peft_config=None, base_sync_done=False, use_shm=False)
    finally:
        for name, prev in saved.items():
            if prev is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prev

    assert events == [
        "start",
        ("load", ["model.layers.0.linear.weight"]),
        "finish",
        "reset_state",
        "post_process",
    ]
    torch.testing.assert_close(
        model.model.layers[0].e_score_correction_bias, torch.tensor([7, 8, 9, 10], dtype=torch.float32)
    )
