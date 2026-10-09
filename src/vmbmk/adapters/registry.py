"""Lazy adapter registration and explicit checkpoint identities."""
from __future__ import annotations

import re
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:
    from .base import Adapter


ADAPTERS = {
    "robometer": "RobometerAdapter",
    "robodopamine": "RoboDopamineAdapter",
    "procvlm": "ProcVLMAdapter",
    "roboreward": "RoboRewardAdapter",
    "vlac": "VLACAdapter",
    "topreward": "TOPRewardAdapter",
    "robofac": "RoboFACAdapter",
    "rynnvalue": "RynnValueAdapter",
    "failsafe": "FailSafeAdapter",
    "liv": "LIVAdapter",
    "mock_service": "MockServiceAdapter",
}


def adapter_class(name: str) -> type[Adapter]:
    return getattr(import_module(f".{name}", __package__), ADAPTERS[name])


def canonical_baseline(config: Mapping[str, Any], *, fallback: str | None = None) -> str:
    """Recognize known checkpoints; never guess the size of an unknown model."""
    model = str(config.get("model") or fallback or "unknown")
    checkpoint = str(config.get("checkpoint") or "").rstrip("/")
    name = Path(checkpoint).name.lower()
    if model == "topreward" and (config.get("model_options") or {}).get("backend", "qwen") == "molmo":
        safe_name = re.sub(r"[^a-z0-9._-]+", "-", name).strip("-") or "unknown"
        return f"topreward_molmo__{safe_name}"
    if model == "topreward" and re.search(r"(?<!\d)32b(?!\d)", name):
        return "topreward_qwen32b"
    if model == "robodopamine":
        match = re.search(r"(?<!\d)2\.0[-_]([48])b[-_]preview", name)
        if match:
            return f"robodopamine_2_0_{match.group(1)}b_preview"
        match = re.search(r"(?<!\d)(?:grm[-_])?([348])b$", name)
        if match:
            return f"robodopamine_{match.group(1)}b"
    elif model == "rynnvalue":
        match = re.search(r"(?<!\d)([48])b(?!\d)", name)
        if match:
            return f"rynnvalue_{match.group(1)}b"
    elif model == "vlac":
        match = re.search(r"(?<!\d)([28])b(?!\d)", name)
        mode = adapter_class("vlac").resolve_shot_mode(config)
        if match:
            return f"vlac_{match.group(1)}b_{mode}"
        safe_name = re.sub(r"[^a-z0-9._-]+", "-", name).strip("-") or "unknown"
        return f"vlac__{safe_name}_{mode}"
    elif model == "procvlm":
        options = config.get("model_options") or {}
        return "procvlm_one-shot" if adapter_class("procvlm").resolve_use_lora(options) else "procvlm"
    elif model == "roboreward":
        if re.search(r"(?<!\d)4b(?!\d)", name):
            return "roboreward_4b"
        if re.search(r"(?<!\d)8b(?!\d)", name) or name == "roboreward":
            return "roboreward_8b"
    if model in {"robodopamine", "rynnvalue", "roboreward", "topreward"} and name:
        safe_name = re.sub(r"[^a-z0-9._-]+", "-", name).strip("-")
        return f"{model}__{safe_name}"
    return model
