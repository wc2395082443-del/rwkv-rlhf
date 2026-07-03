from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]

from .env import env_value, pick
from .paths import resolve_path


DEFAULT_LOCAL_CONFIG_DIR = Path("configs/local")
DEFAULT_EXAMPLE_CONFIG = Path("configs/example.toml")


def default_config_path(root: Path) -> Path:
    local_dir = root / DEFAULT_LOCAL_CONFIG_DIR
    if local_dir.exists():
        local_configs = sorted(path for path in local_dir.glob("*.toml") if path.is_file())
        if local_configs:
            return local_configs[-1]
    return root / DEFAULT_EXAMPLE_CONFIG


def load_config(
    root: Path,
    config_path: str | None,
) -> tuple[dict[str, Any], Path]:
    path = Path(config_path) if config_path else default_config_path(root)
    if not path.is_absolute():
        path = root / path
    if not path.exists():
        raise SystemExit(f"config file not found: {path}")
    with path.open("rb") as file:
        return tomllib.load(file), path


def table(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name, {})
    return value if isinstance(value, dict) else {}


def resolve_model_entry(config: dict[str, Any], model_name: str) -> dict[str, Any]:
    models = table(config, "models")
    seen: list[str] = []
    current_name = model_name

    while True:
        entry = models.get(current_name)
        if not isinstance(entry, dict):
            raise SystemExit(f"model alias not found in config: {model_name}")
        if current_name in seen:
            chain = " -> ".join([*seen, current_name])
            raise SystemExit(f"cyclic model alias in config: {chain}")
        seen.append(current_name)
        alias = entry.get("alias")
        if not alias:
            resolved = dict(entry)
            resolved.setdefault("name", current_name)
            resolved.setdefault("requested_name", model_name)
            return resolved
        current_name = str(alias)


def resolve_model_path(
    config: dict[str, Any],
    model_name: str,
    *,
    root: Path,
    env: dict[str, str],
) -> tuple[Path, dict[str, Any]]:
    entry = resolve_model_entry(config, model_name)
    paths = table(config, "paths")

    if "path" in entry:
        return resolve_path(str(entry["path"]), root=root, env=env), entry

    filename = entry.get("file")
    if not filename:
        raise SystemExit(f"model {model_name} needs either path or file in config")

    base_value = pick(
        entry.get("weight_path"),
        paths.get("weight_path"),
        env_value(env, "WEIGHT_PATH", "HELICOPTER_WEIGHT_PATH"),
    )
    if not base_value:
        raise SystemExit("WEIGHT_PATH is not set and config paths.weight_path is missing")

    base = resolve_path(str(base_value), root=root, env=env)
    base_dir = base.parent if base.suffix == ".pth" else base
    return base_dir / str(filename), entry


def dataset_root(
    config: dict[str, Any],
    dataset_name: str,
    *,
    root: Path,
    env: dict[str, str],
) -> Path:
    datasets = table(config, "datasets")
    entry = datasets.get(dataset_name)
    paths = table(config, "paths")

    if isinstance(entry, dict) and entry.get("root"):
        return resolve_path(str(entry["root"]), root=root, env=env)

    base_value = pick(
        paths.get("datasets_path"),
        env_value(env, "DATASETS_PATH", "HELICOPTER_DATASETS_PATH"),
        "/workspace/Datasets",
    )
    return resolve_path(str(base_value), root=root, env=env) / dataset_name
