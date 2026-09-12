#!/usr/bin/env python3
"""Shared, dependency-light I/O for the frozen P10 cross-system benchmark."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import soundfile as sf


DEFAULT_MANIFEST = Path(
    os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/baselines/p10_60k_15row_v1/generation_requests.jsonl"
)


def add_common_arguments(parser: argparse.ArgumentParser, baseline_id: str) -> None:
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--baseline-id", default=baseline_id, choices=(baseline_id,))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)


def load_requests(
    path: Path,
    baseline_id: str,
    *,
    shard_index: int = 0,
    num_shards: int = 1,
) -> list[dict[str, Any]]:
    manifest = path.expanduser().resolve(strict=True)
    rows = [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line
    ]
    selected = [row for row in rows if row["baseline_id"] == baseline_id]
    if not selected:
        raise RuntimeError(f"no requests for baseline {baseline_id!r} in {manifest}")
    panel_ids = [row["panel_id"] for row in selected]
    if len(panel_ids) != len(set(panel_ids)):
        raise RuntimeError(f"duplicate panel IDs for {baseline_id}")
    if num_shards <= 0 or not 0 <= shard_index < num_shards:
        raise ValueError(
            f"invalid request shard {shard_index}/{num_shards} for {baseline_id}"
        )
    return [
        row for index, row in enumerate(selected) if index % num_shards == shard_index
    ]


def seed_process(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def output_is_valid(row: dict[str, Any]) -> bool:
    for key, channels in (("native_output_path", None), ("quality_w_path", 1)):
        path = Path(row[key])
        if not path.is_file() or path.stat().st_size <= 44:
            return False
        info = sf.info(path)
        if channels is not None and info.channels != channels:
            return False
        expected = int(round(float(row["duration_sec"]) * info.samplerate))
        if abs(info.frames - expected) > 1:
            return False
    metadata_path = Path(row["native_output_path"]).with_name("generation.json")
    if not metadata_path.is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if (
        metadata.get("status") != "PASS"
        or metadata.get("baseline_id") != row["baseline_id"]
        or metadata.get("panel_id") != row["panel_id"]
        or metadata.get("domain") != row["domain"]
        or int(metadata.get("seed", -1)) != int(row["seed"])
    ):
        return False
    return True


def _as_channels_first(audio: Any) -> np.ndarray:
    try:
        import torch

        if torch.is_tensor(audio):
            audio = audio.detach().float().cpu().numpy()
    except ImportError:
        pass
    value = np.asarray(audio, dtype=np.float32)
    while value.ndim > 2 and value.shape[0] == 1:
        value = value[0]
    if value.ndim == 1:
        value = value[None, :]
    if value.ndim != 2:
        raise ValueError(f"expected [channels,samples] audio, got {value.shape}")
    if value.shape[0] > value.shape[1] and value.shape[1] <= 8:
        value = value.T
    if value.shape[0] not in (1, 2):
        raise ValueError(f"baseline output must be mono or stereo, got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError("baseline output contains NaN or Inf")
    return value


def _fit_duration(audio: np.ndarray, sample_rate: int, duration_sec: float) -> np.ndarray:
    target = int(round(float(duration_sec) * int(sample_rate)))
    if target <= 0:
        raise ValueError(f"invalid target duration: {duration_sec}")
    if audio.shape[1] < target:
        audio = np.pad(audio, ((0, 0), (0, target - audio.shape[1])))
    else:
        audio = audio[:, :target]
    return audio


def save_result(
    row: dict[str, Any],
    audio: Any,
    sample_rate: int,
    *,
    backend_metadata: dict[str, Any],
) -> dict[str, Any]:
    """Write native audio plus the frozen mono quality view.

    Mono is unchanged. Stereo uses an arithmetic mean. This is deliberately not
    an FOA encoder and must never be used for spatial metrics.
    """

    value = _fit_duration(
        _as_channels_first(audio), int(sample_rate), float(row["duration_sec"])
    )
    peak = float(np.max(np.abs(value)))
    if peak > 1.0:
        value = value / peak
    rms = float(np.sqrt(np.mean(np.square(value, dtype=np.float64))))
    if rms <= 1.0e-7:
        raise RuntimeError(f"near-silent baseline output for {row['panel_id']}: rms={rms}")

    if value.shape[0] == 1:
        quality_w = value
        reduction = "mono unchanged"
    else:
        quality_w = value.mean(axis=0, keepdims=True)
        reduction = "stereo arithmetic mean"

    native_path = Path(row["native_output_path"])
    quality_path = Path(row["quality_w_path"])
    native_path.parent.mkdir(parents=True, exist_ok=True)
    quality_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(native_path, value.T, int(sample_rate), format="WAV", subtype="FLOAT")
    sf.write(quality_path, quality_w.T, int(sample_rate), format="WAV", subtype="FLOAT")

    result = {
        "schema": "sceneplan_foa.baseline_generation_output",
        "schema_version": 1,
        "status": "PASS",
        "baseline_id": row["baseline_id"],
        "panel_id": row["panel_id"],
        "domain": row["domain"],
        "seed": int(row["seed"]),
        "duration_sec": float(row["duration_sec"]),
        "sample_rate_hz": int(sample_rate),
        "native_channels": int(value.shape[0]),
        "native_peak": float(np.max(np.abs(value))),
        "native_rms": rms,
        "quality_view": reduction,
        "native_output_path": str(native_path),
        "quality_w_path": str(quality_path),
        "backend": backend_metadata,
    }
    metadata_path = native_path.with_name("generation.json")
    temporary = metadata_path.with_name(f".{metadata_path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(metadata_path)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


def pending_rows(
    rows: Iterable[dict[str, Any]], *, force: bool
) -> list[dict[str, Any]]:
    return [row for row in rows if force or not output_is_valid(row)]
