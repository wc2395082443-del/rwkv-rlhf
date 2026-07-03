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

import os
import sys
from pathlib import Path

import pytest

from verl.models.rwkv import (
    import_rwkv_lm,
    resolve_rwkv_lm_paths,
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _clear_modules(*module_names: str) -> None:
    for module_name in module_names:
        sys.modules.pop(module_name, None)


def _write_required_train_files(train_dir: Path) -> None:
    _write(train_dir / "train.py", "")
    _write(train_dir / "src/model.py", "")
    _write(train_dir / "src/trainer.py", "")


def test_resolve_rwkv_lm_paths_accepts_flat_train_layout(tmp_path):
    _write_required_train_files(tmp_path)

    paths = resolve_rwkv_lm_paths(str(tmp_path))

    assert paths.repo_root == tmp_path.resolve()
    assert paths.train_dir == tmp_path.resolve()


def test_resolve_rwkv_lm_paths_rejects_missing_native_files(tmp_path):
    with pytest.raises(FileNotFoundError, match="rwkv-lm checkout is missing required flat-layout files"):
        resolve_rwkv_lm_paths(str(tmp_path))


def test_import_rwkv_lm_uses_flat_train_sys_path_and_cwd(tmp_path):
    _write_required_train_files(tmp_path)
    _write(
        tmp_path / "src/import_probe.py",
        "from pathlib import Path\n"
        "CWD = Path.cwd()\n"
        "VALUE = 'rwkv-lm'\n",
    )
    _clear_modules("src", "src.import_probe")

    module = import_rwkv_lm("src.import_probe", rwkv_lm_path=str(tmp_path))

    assert module.VALUE == "rwkv-lm"
    assert module.CWD == tmp_path.resolve()


def test_import_rwkv_lm_patches_native_env_only_during_import(tmp_path, monkeypatch):
    _write_required_train_files(tmp_path)
    _write(
        tmp_path / "src/env_probe.py",
        "import os\n"
        "VALUE = os.environ['RWKV_HEAD_SIZE']\n",
    )
    _clear_modules("src", "src.env_probe")
    monkeypatch.delenv("RWKV_HEAD_SIZE", raising=False)

    module = import_rwkv_lm("src.env_probe", rwkv_lm_path=str(tmp_path), native_env={"RWKV_HEAD_SIZE": "64"})

    assert module.VALUE == "64"
    assert "RWKV_HEAD_SIZE" not in os.environ
