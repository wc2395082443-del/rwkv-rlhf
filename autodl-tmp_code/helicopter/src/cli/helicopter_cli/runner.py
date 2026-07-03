from __future__ import annotations

import shlex
import subprocess
from pathlib import Path


def run_command(
    command: list[str],
    *,
    cwd: Path | None,
    env: dict[str, str],
    shown_env: dict[str, str],
    dry_run: bool,
) -> int:
    if dry_run:
        pieces: list[str] = []
        if cwd is not None:
            pieces.extend(["cd", shlex.quote(str(cwd)), "&&"])
        if shown_env:
            pieces.append("env")
            for key in sorted(shown_env):
                pieces.append(f"{key}={shlex.quote(shown_env[key])}")
        pieces.extend(shlex.quote(item) for item in command)
        print(" ".join(pieces))
        return 0
    return subprocess.call(command, cwd=str(cwd) if cwd else None, env=env)
