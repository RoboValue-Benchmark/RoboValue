from __future__ import annotations

from typing import Any, Mapping

from vmbmk.data.dataset import Dataset
from vmbmk.errors import ConfigurationError
from .registry import ADAPTERS, adapter_class
from .base import Adapter


def create_adapter(config: Mapping[str, Any], dataset: Dataset) -> Adapter:
    """Construct one adapter without importing other model implementations."""
    name = config.get("adapter")
    if name not in ADAPTERS:
        raise ConfigurationError(
            f"config adapter must be one of {sorted(ADAPTERS)}; got {name!r}"
        )
    return adapter_class(name).from_config(config, dataset)


__all__ = ["Adapter", "create_adapter"]
