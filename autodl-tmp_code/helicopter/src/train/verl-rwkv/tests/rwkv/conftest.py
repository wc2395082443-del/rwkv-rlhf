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

import sys

import pytest


_RESTORE_MODULE_PREFIXES = (
    "rwkv_lm_engine_test",
    "verl",
)


def _should_restore(name: str) -> bool:
    return any(name == prefix or name.startswith(f"{prefix}.") for prefix in _RESTORE_MODULE_PREFIXES)


@pytest.fixture(autouse=True)
def restore_rwkv_test_modules():
    saved = {name: sys.modules.get(name) for name in list(sys.modules) if _should_restore(name)}
    yield
    for name in list(sys.modules):
        if _should_restore(name):
            sys.modules.pop(name, None)
    for name, module in saved.items():
        if module is not None:
            sys.modules[name] = module
