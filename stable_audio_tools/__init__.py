"""Public stable-audio-tools API with lazy heavyweight imports.

Keeping package import lightweight lets config, manifest, and metadata tools run
without importing torch or probing CUDA. The public names remain compatible
with the historical eager imports.
"""

from __future__ import annotations

from typing import Any

__all__ = (
    "create_model_from_config",
    "create_model_from_config_path",
    "get_pretrained_model",
)


def __getattr__(name: str) -> Any:
    if name in {"create_model_from_config", "create_model_from_config_path"}:
        from .models.factory import (
            create_model_from_config,
            create_model_from_config_path,
        )

        value = {
            "create_model_from_config": create_model_from_config,
            "create_model_from_config_path": create_model_from_config_path,
        }[name]
    elif name == "get_pretrained_model":
        from .models.pretrained import get_pretrained_model

        value = get_pretrained_model
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *__all__))
