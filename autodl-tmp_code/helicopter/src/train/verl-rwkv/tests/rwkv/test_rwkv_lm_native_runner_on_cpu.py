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
from types import ModuleType, SimpleNamespace

import pytest


def _load_runner_module():
    package_name = "rwkv_lm_engine_test"
    package = types.ModuleType(package_name)
    package.__path__ = [str(Path("verl/workers/engine/rwkv_lm").resolve())]
    sys.modules[package_name] = package

    for name in ("args", "checkpoint", "env", "native_runner"):
        path = Path(f"verl/workers/engine/rwkv_lm/{name}.py")
        spec = importlib.util.spec_from_file_location(f"{package_name}.{name}", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[f"{package_name}.{name}"] = module
        spec.loader.exec_module(module)
    return sys.modules[f"{package_name}.native_runner"]


class FakeRWKV:
    def __init__(self, args):
        self.args = args
        self.loaded = None

    def load_state_dict(self, state):
        self.loaded = state


class FakeCallback:
    def __init__(self, args):
        self.args = args


def test_native_runner_imports_native_modules_with_native_env_and_path():
    runner_module = _load_runner_module()
    calls = []

    def importer(module_name, *, rwkv_lm_path=None, native_env=None):
        calls.append((module_name, rwkv_lm_path, native_env))
        if module_name == "src.model":
            return ModuleType("src.model")
        return ModuleType("src.trainer")

    engine_config = SimpleNamespace(rwkv_lm_path="/src/rwkv-lm", precision="bf16", native_env={"RWKV_KERNEL": ""})
    runner = runner_module.NativeRWKVLMRunner(engine_config=engine_config, importer=importer)

    runner.import_native_modules()

    assert [call[0] for call in calls] == ["src.model", "src.trainer"]
    assert calls[0][1] == "/src/rwkv-lm"
    assert calls[0][2]["RWKV_FLOAT_MODE"] == "bf16"
    assert calls[0][2]["RWKV_HEAD_SIZE"] == "64"


def test_native_runner_builds_upstream_model_and_loads_native_checkpoint():
    runner_module = _load_runner_module()
    model_module = ModuleType("src.model")
    model_module.RWKV = FakeRWKV
    trainer_module = ModuleType("src.trainer")
    trainer_module.train_callback = FakeCallback

    def importer(module_name, **kwargs):
        return model_module if module_name == "src.model" else trainer_module

    runner = runner_module.NativeRWKVLMRunner(
        model_config=SimpleNamespace(path="/models/rwkv.pth"),
        importer=importer,
        checkpoint_loader=lambda path: {"loaded_from": path},
    )

    model = runner.build_model()
    callback = runner.build_train_callback()

    assert isinstance(model, FakeRWKV)
    assert model.loaded == {"loaded_from": "/models/rwkv.pth"}
    assert isinstance(callback, FakeCallback)
    assert callback.args is runner.args
    assert runner.state().model is model


def test_default_import_rejects_wrong_pytorch_lightning_version(monkeypatch):
    runner_module = _load_runner_module()

    monkeypatch.setattr(
        runner_module.metadata,
        "version",
        lambda package: "2.6.4" if package == "pytorch-lightning" else "0.19.0",
    )

    with pytest.raises(RuntimeError) as raised:
        runner_module._default_import_rwkv_lm("src.model", rwkv_lm_path="/src/rwkv-lm")

    assert str(raised.value) == (
        "native rwkv-lm requires pytorch-lightning==1.9.5; found 2.6.4. "
        "Fix the uv environment with `uv pip install pytorch-lightning==1.9.5` "
        "before running RWKV training."
    )
