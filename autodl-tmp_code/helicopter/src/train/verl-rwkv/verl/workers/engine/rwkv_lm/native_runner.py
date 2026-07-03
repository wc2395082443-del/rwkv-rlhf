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
"""Native rwkv-lm runner wrapper.

This module wires adapter interfaces to upstream ``src.model.RWKV`` and
``src.trainer``. It does not reimplement rwkv-lm model, optimizer, callback,
or training-step logic.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from importlib import metadata
from types import ModuleType, SimpleNamespace
from typing import Any

from .args import build_rwkv_lm_args
from .checkpoint import load_rwkv_lm_checkpoint
from .env import build_rwkv_lm_env, rwkv_lm_env


NativeImporter = Callable[..., ModuleType]
CheckpointLoader = Callable[..., dict[str, Any]]


def _validate_rwkv_lm_runtime() -> None:
    """Validate upstream rwkv-lm runtime packages before importing native code."""

    try:
        lightning_version = metadata.version("pytorch-lightning")
    except metadata.PackageNotFoundError as exc:
        raise ModuleNotFoundError(
            "native rwkv-lm requires pytorch-lightning==1.9.5; install it with "
            "`uv pip install pytorch-lightning==1.9.5`."
        ) from exc
    if lightning_version != "1.9.5":
        raise RuntimeError(
            "native rwkv-lm requires pytorch-lightning==1.9.5; found "
            f"{lightning_version}. Fix the uv environment with "
            "`uv pip install pytorch-lightning==1.9.5` before running RWKV training."
        )

    try:
        metadata.version("deepspeed")
    except metadata.PackageNotFoundError as exc:
        raise ModuleNotFoundError(
            "native rwkv-lm requires deepspeed; install or upgrade it with "
            "`uv pip install --upgrade deepspeed`."
        ) from exc


def _default_import_rwkv_lm(
    module_name: str,
    *,
    rwkv_lm_path: str | None = None,
    native_env: Mapping[str, str] | None = None,
) -> ModuleType:
    _validate_rwkv_lm_runtime()
    from verl.models.rwkv.native_imports import import_rwkv_lm

    return import_rwkv_lm(module_name, rwkv_lm_path=rwkv_lm_path, native_env=native_env)


class NativeRWKVLMRunner:
    """Adapter-owned handle for native rwkv-lm construction."""

    def __init__(
        self,
        *,
        model_config: Any = None,
        engine_config: Any = None,
        optimizer_config: Any = None,
        overrides: dict[str, Any] | None = None,
        importer: NativeImporter = _default_import_rwkv_lm,
        checkpoint_loader: CheckpointLoader = load_rwkv_lm_checkpoint,
    ):
        self.model_config = model_config
        self.engine_config = engine_config
        self.optimizer_config = optimizer_config
        self.overrides = overrides
        self.importer = importer
        self.checkpoint_loader = checkpoint_loader
        self.args = build_rwkv_lm_args(
            model_config=model_config,
            engine_config=engine_config,
            optimizer_config=optimizer_config,
            overrides=overrides,
        )
        self.native_env = build_rwkv_lm_env(self.args, extra_env=self._engine_native_env())
        self.model: Any = None
        self.model_module: ModuleType | None = None
        self.trainer_module: ModuleType | None = None

    def _engine_native_env(self) -> Mapping[str, str]:
        if self.engine_config is None:
            return {}
        native_env = getattr(self.engine_config, "native_env", None)
        return native_env or {}

    def _rwkv_lm_path(self) -> str | None:
        return getattr(self.engine_config, "rwkv_lm_path", None) if self.engine_config is not None else None

    def import_native_modules(self) -> tuple[ModuleType, ModuleType]:
        """Import upstream ``src.model`` and ``src.trainer`` under native env."""

        rwkv_lm_path = self._rwkv_lm_path()
        self.model_module = self.importer("src.model", rwkv_lm_path=rwkv_lm_path, native_env=self.native_env)
        self.trainer_module = self.importer("src.trainer", rwkv_lm_path=rwkv_lm_path, native_env=self.native_env)
        return self.model_module, self.trainer_module

    def build_model(self) -> Any:
        """Instantiate upstream ``RWKV(args)`` and load native checkpoint if set."""

        model_module = self.model_module
        if model_module is None:
            model_module, _ = self.import_native_modules()
        with rwkv_lm_env(self.args, extra_env=self._engine_native_env()):
            model = model_module.RWKV(self.args)
        if self.args.load_model:
            model.load_state_dict(self.checkpoint_loader(self.args.load_model))
        self.model = model
        return model

    def build_train_callback(self) -> Any:
        """Instantiate upstream ``src.trainer.train_callback(args)``."""

        if self.trainer_module is None:
            _, self.trainer_module = self.import_native_modules()
        return self.trainer_module.train_callback(self.args)

    def state(self) -> SimpleNamespace:
        """Return runner state for engine integration."""

        return SimpleNamespace(args=self.args, native_env=self.native_env, model=self.model)
