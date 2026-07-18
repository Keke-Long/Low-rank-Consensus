"""Portable paths for local development and public releases.

Set paths in the shell or in the repository-root ``.env`` file. The latter is
local-only and intentionally excluded from version control; see
``.env.example`` for the required variables.
"""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_local_env() -> None:
    """Load simple KEY=VALUE entries without requiring python-dotenv."""
    env_file = PROJECT_ROOT / ".env"
    if not env_file.exists():
        return
    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _required_path(variable: str) -> Path:
    _load_local_env()
    value = os.environ.get(variable)
    if not value:
        raise RuntimeError(
            f"{variable} is not configured. Set it in your shell or copy "
            ".env.example to .env and set the local path."
        )
    return Path(value).expanduser()


def nuscenes_dir() -> Path:
    return _required_path("NUSCENES_DIR")


def model_offload_dir() -> Path:
    _load_local_env()
    value = os.environ.get("MODEL_OFFLOAD_DIR")
    return Path(value).expanduser() if value else PROJECT_ROOT / ".cache" / "offload"


def project_path(*parts: str) -> Path:
    return PROJECT_ROOT.joinpath(*parts)
