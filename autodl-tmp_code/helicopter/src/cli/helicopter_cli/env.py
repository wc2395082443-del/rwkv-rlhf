from __future__ import annotations

import os
from pathlib import Path
from typing import Any


DEFAULT_ENV_FILE = ".env.local"
ENV_FALLBACKS = (".env.remote", ".env")


def load_dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values

    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if (
            len(value) >= 2
            and value[0] == value[-1]
            and value[0] in {"'", '"'}
        ):
            value = value[1:-1]
        values[key] = value
    return values


def load_env(root: Path, env_file: str) -> tuple[dict[str, str], Path | None]:
    env = dict(os.environ)
    candidates = [Path(env_file)]
    if env_file == DEFAULT_ENV_FILE:
        candidates.extend(Path(name) for name in ENV_FALLBACKS)

    for candidate in candidates:
        path = candidate if candidate.is_absolute() else root / candidate
        if path.exists():
            for key, value in load_dotenv(path).items():
                env.setdefault(key, value)
            return env, path
    return env, None


def env_value(env: dict[str, str], *names: str) -> str | None:
    for name in names:
        value = env.get(name)
        if value:
            return value
    return None


def pick(*values: Any, default: Any = None) -> Any:
    for value in values:
        if value is not None:
            return value
    return default
