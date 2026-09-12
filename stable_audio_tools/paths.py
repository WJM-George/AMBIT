"""Portable path helpers for checkpoints, data, and caches.

Machine-specific mounts are not hard-coded. Set the environment variables
below, or pass explicit CLI/config paths.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def data_root() -> Path:
    return Path(os.environ.get("AMBIT_DATA_ROOT", "data")).expanduser()


def ckpt_root() -> Path:
    return Path(os.environ.get("AMBIT_CKPT_ROOT", "checkpoints")).expanduser()


def cache_root() -> Path:
    return Path(os.environ.get("AMBIT_CACHE_ROOT", "cache")).expanduser()


def data_path(*parts: str | os.PathLike[str]) -> Path:
    return data_root().joinpath(*parts)


def ckpt_path(*parts: str | os.PathLike[str]) -> Path:
    return ckpt_root().joinpath(*parts)


def cache_path(*parts: str | os.PathLike[str]) -> Path:
    return cache_root().joinpath(*parts)


def expand_path_template(value: str) -> str:
    """Expand ``${AMBIT_*_ROOT}`` placeholders, with portable defaults."""

    replacements = {
        "AMBIT_DATA_ROOT": str(data_root()),
        "AMBIT_CKPT_ROOT": str(ckpt_root()),
        "AMBIT_CACHE_ROOT": str(cache_root()),
        "AUDIO_DATASET_ROOT": os.environ.get("AUDIO_DATASET_ROOT", str(data_root())),
        "AUDIO_DATASET_CACHE_ROOT": os.environ.get(
            "AUDIO_DATASET_CACHE_ROOT", str(cache_root())
        ),
        "AUDIO_DATASET_TMP": os.environ.get(
            "AUDIO_DATASET_TMP", str(cache_root() / "tmp")
        ),
    }
    text = value
    for key, replacement in replacements.items():
        text = text.replace(f"${{{key}}}", replacement)
    return os.path.expandvars(text)


def expand_config_values(value: Any) -> Any:
    if isinstance(value, str):
        return expand_path_template(value)
    if isinstance(value, list):
        return [expand_config_values(item) for item in value]
    if isinstance(value, dict):
        return {key: expand_config_values(item) for key, item in value.items()}
    return value
