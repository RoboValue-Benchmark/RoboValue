"""Private judge credentials, isolated from archived evaluation configs."""
from __future__ import annotations

import os
from pathlib import Path

import yaml

from vmbmk.errors import VMBMKError


def load_judge_config() -> dict[str, str]:
    """Read the adjacent config.yaml or the explicit VMBMK_SIA_CONFIG path."""
    configured_path = os.getenv("VMBMK_SIA_CONFIG")
    path = (
        Path(configured_path).expanduser()
        if configured_path else Path(__file__).with_name("config.yaml")
    )
    if not path.exists() and configured_path is None:
        return {}
    try:
        if os.name == "posix" and path.stat().st_mode & 0o077:
            raise VMBMKError(f"SIA private config requires chmod 600: {path}")
        row = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError):
        raise VMBMKError(f"Cannot read SIA private config: {path}") from None
    if not isinstance(row, dict) or set(row) - {"api_key", "base_url", "model"}:
        raise VMBMKError(
            f"SIA private config must contain only api_key, base_url and model: {path}"
        )
    settings = {}
    for name, value in row.items():
        if value is None:
            continue
        if not isinstance(value, str) or not value.strip():
            raise VMBMKError(f"SIA private config {name} must be a nonempty string: {path}")
        settings[name] = value.strip()
    return settings
