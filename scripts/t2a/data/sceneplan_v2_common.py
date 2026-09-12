"""Shared deterministic helpers for the ScenePlan v2 revision-4 build."""

from __future__ import annotations

import hashlib
from collections import Counter
import io
import json
import math
import os
import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from scipy.signal import resample_poly


MODEL_SAMPLE_RATE = 44_100
MAX_MODEL_SAMPLES = 442_368
MAX_DURATION_SEC = MAX_MODEL_SAMPLES / MODEL_SAMPLE_RATE
VAE_HOP_SAMPLES = 1024
MAX_LATENT_FRAMES = 432
CONTRACT_REVISION = 4
DATASET_ROOT = Path(os.environ.get("AMBIT_DATA_ROOT", "data") + "/sceneplan_v2_1p124m")
VAE_CHECKPOINT_SHA256 = "0229e48729bb6cf138c277d37c598d659000cf78e0a16f498171f2f1f83e8a87"


_WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")
_SHARD_RE = re.compile(r"-\d{5}-of-\d{5}.*$")


def normalized_transcript(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return " ".join(_WORD_RE.findall(text))


def clean_text(value: Any) -> str:
    # Some HiFiTTS normalized strings wrap the utterance in quote characters
    # separated from the first/last word by whitespace. Stripping those quotes
    # can therefore expose a new boundary space, which is not canonical text.
    text = " ".join(str(value or "").split()).strip().strip('"“”')
    return text.strip()


def source_split_from_parquet(path: Path, dataset_root: Path) -> str:
    relative = path.relative_to(dataset_root)
    if len(relative.parts) >= 3 and relative.parts[0] == "data":
        return relative.parts[1]
    name = _SHARD_RE.sub("", path.name)
    return re.sub(r"\.parquet$", "", name)


def model_num_samples(native_num_samples: int, native_sample_rate_hz: int) -> int:
    if native_num_samples <= 0 or native_sample_rate_hz <= 0:
        return 0
    return int(math.ceil(native_num_samples * MODEL_SAMPLE_RATE / native_sample_rate_hz))


def deterministic_digest(*parts: Any, digest_size: int = 16) -> str:
    payload = "\x1f".join(str(value) for value in parts).encode("utf-8")
    return hashlib.blake2b(payload, digest_size=digest_size).hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def require_dataset_not_frozen() -> None:
    marker = DATASET_ROOT / "FROZEN_P9.json"
    if marker.is_file():
        raise RuntimeError(
            f"ScenePlan-v2 is frozen at P9; refusing a mutating build step: {marker}"
        )


@lru_cache(maxsize=2)
def _load_parquet_row_group(path_text: str, row_group: int):
    """Cache a small number of complete speech row groups per render process."""

    path = Path(path_text)
    parquet_file = pq.ParquetFile(path)
    columns = [
        name
        for name in parquet_file.schema_arrow.names
        if name
        in {
            "audio",
            "id",
            "file",
            "path",
            "speaker",
            "speaker_id",
            "chapter_id",
            "duration",
            "text",
            "text_original",
            "text_normalized",
            "text_no_preprocessing",
        }
    ]
    return parquet_file.read_row_group(row_group, columns=columns)


def load_parquet_source(locator: dict[str, Any]) -> tuple[bytes, dict[str, Any]]:
    path = Path(str(locator["parquet_path"]))
    row_group = int(locator["row_group"])
    row_in_group = int(locator["row_in_group"])
    table = _load_parquet_row_group(str(path), row_group)
    if not 0 <= row_in_group < table.num_rows:
        raise IndexError(f"row {row_in_group} outside row group {row_group}: {path}")
    row = table.slice(row_in_group, 1).to_pylist()[0]
    audio = row.pop("audio", None) or {}
    blob = audio.get("bytes")
    if not blob:
        raise ValueError(f"Parquet row has no embedded audio bytes: {path}:{row_group}:{row_in_group}")
    return bytes(blob), row


def decode_complete_mono(
    blob: bytes,
    *,
    max_model_samples: int = MAX_MODEL_SAMPLES,
) -> tuple[np.ndarray, int, int]:
    """Decode one complete mono source without cropping it.

    The default remains the frozen revision-4/5 432-frame ceiling.  Dataset
    revisions with a larger, explicitly audited envelope must pass that limit
    at the call site; keeping the default narrow prevents an old build from
    silently accepting longer sources.
    """

    if int(max_model_samples) <= 0:
        raise ValueError("max_model_samples must be positive")
    data, native_rate = sf.read(io.BytesIO(blob), dtype="float32", always_2d=True)
    if data.ndim != 2 or data.shape[1] != 1:
        raise ValueError(f"canonical source must be mono, got shape={data.shape}")
    native_frames = int(data.shape[0])
    mono = data[:, 0]
    if native_rate != MODEL_SAMPLE_RATE:
        divisor = math.gcd(int(native_rate), MODEL_SAMPLE_RATE)
        mono = resample_poly(
            mono,
            MODEL_SAMPLE_RATE // divisor,
            int(native_rate) // divisor,
        ).astype(np.float32, copy=False)
    else:
        mono = mono.astype(np.float32, copy=False)
    if len(mono) > int(max_model_samples):
        raise ValueError(
            "complete utterance exceeds model limit: "
            f"{len(mono)} > {int(max_model_samples)}"
        )
    if len(mono) <= 0 or not np.isfinite(mono).all():
        raise ValueError("source is empty or non-finite")
    return mono, int(native_rate), native_frames


def expected_quotas(config: dict[str, Any], mode: str) -> Counter[tuple[str, str, int]]:
    if mode == "pilot":
        return Counter(
            {
                ("train", family, count): value
                for family in ("speech", "no_speech")
                for count, value in {1: 700, 2: 700, 3: 400, 4: 200}.items()
            }
        )
    return Counter(
        {
            (split, family, int(count)): int(value)
            for split, split_spec in config["splits"].items()
            for family, cells in split_spec["joint_quotas"].items()
            for count, value in cells.items()
        }
    )

