from __future__ import annotations

import os
from pathlib import Path
from string import Template


ROOT_MARKERS = ("pyproject.toml", "src/infer/vllm-rwkv", "src/train/verl-rwkv")


def find_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if all((parent / marker).exists() for marker in ROOT_MARKERS):
            return parent
    return current.parents[3]


def resolve_path(value: str | Path, *, root: Path, env: dict[str, str]) -> Path:
    expanded = Path(os.path.expanduser(Template(str(value)).safe_substitute(env)))
    if expanded.is_absolute():
        return expanded
    return root / expanded
