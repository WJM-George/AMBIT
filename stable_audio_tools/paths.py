"""Portable path helpers for checkpoints, data, and caches.

Machine-specific mounts are not hard-coded. Set the environment variables
below, or pass explicit CLI/config paths.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def repo_root() -> Path:
    raw = os.environ.get("AMBIT_REPO_ROOT")
    if raw:
        return Path(raw).expanduser()
    here = Path(__file__).resolve().parent
    for candidate in (here, *here.parents):
        if (candidate / "stable_audio_tools").is_dir() and (candidate / "train.py").is_file():
            return candidate
    return here.parent


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


def editing_bench() -> Path:
    raw = os.environ.get("AMBIT_EDITING_BENCH")
    if raw:
        return Path(raw).expanduser()
    return data_path("editing_bench")


def clap_task_root() -> Path:
    raw = os.environ.get("AMBIT_CLAP_TASK_ROOT")
    if raw:
        return Path(raw).expanduser()
    return data_path("editing_clap_task")


def whisper_model() -> Path:
    raw = os.environ.get("AMBIT_WHISPER_MODEL")
    if raw:
        return Path(raw).expanduser()
    return data_path("models", "faster-distil-whisper-large-v3")


def opsd_config_path() -> Path:
    raw = os.environ.get("AMBIT_OPSD_CONFIG")
    if raw:
        return Path(raw).expanduser()
    return ckpt_path("transfusion_opsd", "editing_v3", "config.json")


def generation_ar_root() -> Path:
    raw = os.environ.get("AMBIT_GENERATION_AR_ROOT")
    if raw:
        return Path(raw).expanduser()
    return ckpt_path("generation_ar")


def expand_path_template(value: str) -> str:
    """Expand ``${AMBIT_*_ROOT}`` placeholders, with portable defaults."""

    replacements = {
        "AMBIT_REPO_ROOT": str(repo_root()),
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
