#!/usr/bin/env python3
"""Build input-only frozen-ASR lexical evidence for P11 Understanding.

The builder decodes the indexed frozen FOA VAE latent, transcribes only the W
channel, and writes no ScenePlan/transcript target into the cache.  Confidence
is retained separately so the dataset can omit unreliable lexical tokens while
CLAP and FOA evidence remain available for every Understanding example.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sqlite3
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stable_audio_tools.data.sceneplan_p11_lexical_cache import (  # noqa: E402
    P11_LEXICAL_CACHE_SCHEMA,
    P11_LEXICAL_CACHE_VERSION,
    P11_LEXICAL_CONFIDENCE_CONTRACT,
    P11_LEXICAL_EVIDENCE_CONTRACT,
)
from stable_audio_tools.models.factory import create_model_from_config  # noqa: E402
from stable_audio_tools.models.utils import (  # noqa: E402
    copy_state_dict,
    load_ckpt_state_dict,
)


DEFAULT_VAE_CONFIG = REPO_ROOT / (
    "stable_audio_tools/configs/model_configs/autoencoders/"
    "stable_audio_4ch_vae_ds1024_z64_wdmix_scm.json"
)
DEFAULT_VAE_CHECKPOINT = Path(
    os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/compareVAE_ckpt/unwrapped_wdmix_1350000.ckpt"
)
DEFAULT_ASR = Path(
    os.environ.get("AMBIT_DATA_ROOT", "data") + "/sceneplan_v2_1p124m/models/"
    "faster-distil-whisper-large-v3"
)
DEFAULT_ASR_REVISION = "distil-whisper-large-v3-ct2-fp16-local-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _chunks(values: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(values), int(size)):
        yield values[start : start + int(size)]


def _selected_ordinals(
    manifest: Path,
    *,
    ordinal_inventory: Path | None,
    max_rows: int | None,
    shard_index: int,
    num_shards: int,
) -> tuple[list[int], dict[str, str] | None]:
    if ordinal_inventory is not None:
        connection = sqlite3.connect(
            f"file:{ordinal_inventory}?mode=ro&immutable=1", uri=True
        )
        try:
            metadata = dict(connection.execute("SELECT key,value FROM metadata"))
            required = {
                "schema": "stable_audio_tools.p11_cache_ordinal_inventory",
                "schema_version": "1",
                "selection_contract": (
                    "sorted_distinct_target_ordinal_then_strided_shard_v1"
                ),
                "source_manifest": str(manifest),
            }
            for key, expected in required.items():
                if metadata.get(key) != expected:
                    raise RuntimeError(
                        f"ordinal inventory {key}={metadata.get(key)!r}, "
                        f"expected {expected!r}"
                    )
            total = int(metadata.get("rows", -1))
            actual = int(
                connection.execute("SELECT COUNT(*) FROM ordinals").fetchone()[0]
            )
            if total <= 0 or actual != total:
                raise RuntimeError("ordinal inventory row metadata is stale")
            limit = total if max_rows is None else min(int(max_rows), total)
            values = [
                int(row[0])
                for row in connection.execute(
                    "SELECT target_ordinal FROM ordinals "
                    "WHERE position < ? AND (position % ?) = ? "
                    "ORDER BY position",
                    (limit, int(num_shards), int(shard_index)),
                )
            ]
        finally:
            connection.close()
        if not values:
            raise RuntimeError("P11 ordinal inventory selected no lexical ordinals")
        return values, metadata

    connection = sqlite3.connect(
        f"file:{manifest}?mode=ro&immutable=1", uri=True
    )
    try:
        values = [
            int(row[0])
            for row in connection.execute(
                "SELECT DISTINCT target_ordinal FROM rows ORDER BY target_ordinal"
            )
        ]
    finally:
        connection.close()
    if max_rows is not None:
        values = values[: int(max_rows)]
    values = values[int(shard_index) :: int(num_shards)]
    if not values:
        raise RuntimeError("P11 manifest selected no lexical-cache ordinals")
    return values, None


def _mono_for_asr(audio: torch.Tensor, *, valid_samples: int) -> np.ndarray:
    """Return peak-normalized FOA W at the Whisper 16 kHz sample rate."""

    mono = audio[0, : int(valid_samples)].float()
    if mono.numel() <= 0 or not bool(torch.isfinite(mono).all()):
        raise RuntimeError("VAE decoded empty or non-finite FOA")
    peak = mono.abs().amax()
    if float(peak) > 1.0e-8:
        mono = mono / peak * (10.0 ** (-1.0 / 20.0))
    mono = F.interpolate(
        mono[None, None],
        size=max(1, round(mono.numel() * 16_000 / 44_100)),
        mode="linear",
        align_corners=False,
    )[0, 0]
    return mono.clamp(-1.0, 1.0).cpu().numpy().astype(np.float32, copy=False)


def _weighted_mean(values: list[float], weights: list[float]) -> float:
    denominator = sum(weights)
    if denominator <= 0.0:
        return float(sum(values) / max(len(values), 1))
    return float(sum(value * weight for value, weight in zip(values, weights)) / denominator)


def _transcribe(model: Any, waveform: np.ndarray) -> dict[str, Any]:
    segments_iterator, info = model.transcribe(
        waveform,
        language="en",
        beam_size=5,
        condition_on_previous_text=False,
        vad_filter=True,
        vad_parameters={
            "min_silence_duration_ms": 300,
            "speech_pad_ms": 200,
        },
        word_timestamps=False,
    )
    segments = list(segments_iterator)
    text = " ".join(
        " ".join(str(segment.text).split())
        for segment in segments
        if str(segment.text).strip()
    ).strip()
    durations = [
        max(0.02, float(segment.end) - float(segment.start))
        for segment in segments
    ]
    language = str(getattr(info, "language", None) or "en")
    language_probability = float(
        min(1.0, max(0.0, getattr(info, "language_probability", 0.0)))
    )
    if segments:
        mean_log_probability = _weighted_mean(
            [float(segment.avg_logprob) for segment in segments], durations
        )
        mean_no_speech_probability = _weighted_mean(
            [float(segment.no_speech_prob) for segment in segments], durations
        )
        speech_seconds = float(
            sum(
                duration
                for duration, segment in zip(durations, segments)
                if str(segment.text).strip()
            )
        )
        speech_probability = min(
            1.0, max(0.0, 1.0 - mean_no_speech_probability)
        )
        token_probability = math.exp(min(0.0, mean_log_probability))
        confidence = float(
            max(
                0.0,
                speech_probability * token_probability * language_probability,
            )
            ** (1.0 / 3.0)
        )
    else:
        mean_log_probability = None
        mean_no_speech_probability = None
        speech_seconds = 0.0
        confidence = 0.0
    has_speech = bool(text) and speech_seconds >= 0.10
    if not has_speech:
        text = ""
        confidence = 0.0
    return {
        "text": text,
        "has_speech": int(has_speech),
        "confidence": min(1.0, max(0.0, confidence)),
        "language": language,
        "language_probability": language_probability,
        "mean_average_log_probability": mean_log_probability,
        "mean_no_speech_probability": mean_no_speech_probability,
        "speech_seconds": speech_seconds,
        "segment_count": len(segments),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ordinal-inventory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vae-config", type=Path, default=DEFAULT_VAE_CONFIG)
    parser.add_argument("--vae-checkpoint", type=Path, default=DEFAULT_VAE_CHECKPOINT)
    parser.add_argument("--asr-model", type=Path, default=DEFAULT_ASR)
    parser.add_argument("--encoder-revision", default=DEFAULT_ASR_REVISION)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--compute-type", default="float16")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.max_rows is not None and args.max_rows <= 0:
        raise ValueError("--max-rows must be positive")
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("lexical shard must satisfy 0 <= shard-index < num-shards")

    index = args.index.expanduser().resolve(strict=True)
    manifest = args.manifest.expanduser().resolve(strict=True)
    ordinal_inventory = (
        None
        if args.ordinal_inventory is None
        else args.ordinal_inventory.expanduser().resolve(strict=True)
    )
    vae_config_path = args.vae_config.expanduser().resolve(strict=True)
    vae_checkpoint = args.vae_checkpoint.expanduser().resolve(strict=True)
    asr_path = args.asr_model.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not args.overwrite:
        raise FileExistsError(output)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("P11 lexical-cache construction requires CUDA")
    device_index = 0 if device.index is None else int(device.index)
    ordinals, inventory_metadata = _selected_ordinals(
        manifest,
        ordinal_inventory=ordinal_inventory,
        max_rows=args.max_rows,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
    )

    from faster_whisper import WhisperModel

    vae_config = json.loads(vae_config_path.read_text(encoding="utf-8"))
    vae = create_model_from_config(vae_config)
    copy_state_dict(vae, load_ckpt_state_dict(str(vae_checkpoint)))
    vae.eval().requires_grad_(False).to(device)
    asr = WhisperModel(
        str(asr_path),
        device="cuda",
        device_index=device_index,
        compute_type=str(args.compute_type),
        local_files_only=True,
    )

    source = sqlite3.connect(f"file:{index}?mode=ro&immutable=1", uri=True)
    records: list[tuple[int, str, int, int, str]] = []
    for ordinal_chunk in _chunks(ordinals, 900):
        placeholders = ",".join("?" for _ in ordinal_chunk)
        records.extend(
            (
                int(ordinal),
                str(key),
                int(valid_frames),
                int(model_num_samples),
                str(path),
            )
            for ordinal, key, valid_frames, model_num_samples, path in source.execute(
                f"""
                SELECT samples.ordinal, samples.latent_key,
                       samples.latent_frames_valid, samples.model_num_samples,
                       latent_shards.path
                FROM samples
                JOIN latent_shards ON latent_shards.id = samples.latent_shard_id
                WHERE samples.ordinal IN ({placeholders})
                """,
                tuple(ordinal_chunk),
            )
        )
    if len(records) != len(ordinals):
        raise RuntimeError(
            f"P11 index resolved {len(records)} of {len(ordinals)} lexical rows"
        )
    by_shard: dict[str, list[tuple[int, str, int, int]]] = defaultdict(list)
    for ordinal, key, frames, samples, path in records:
        by_shard[path].append((ordinal, key, frames, samples))

    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".sqlite", dir=output.parent
    )
    os.close(fd)
    temporary = Path(temporary_name)
    destination = sqlite3.connect(temporary)
    started = time.perf_counter()
    written = 0
    completed = False
    try:
        destination.executescript(
            """
            PRAGMA journal_mode=OFF;
            PRAGMA synchronous=OFF;
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE hypotheses (
                ordinal INTEGER PRIMARY KEY,
                text TEXT NOT NULL,
                has_speech INTEGER NOT NULL,
                confidence REAL NOT NULL,
                language TEXT NOT NULL,
                language_probability REAL NOT NULL,
                mean_average_log_probability REAL,
                mean_no_speech_probability REAL,
                speech_seconds REAL NOT NULL,
                segment_count INTEGER NOT NULL
            );
            """
        )
        for shard_number, (shard_path, shard_records) in enumerate(
            sorted(by_shard.items())
        ):
            with safe_open(shard_path, framework="pt", device="cpu") as tensors:
                records_by_frames: dict[
                    int, list[tuple[int, str, int, int]]
                ] = defaultdict(list)
                for record in shard_records:
                    records_by_frames[int(record[2])].append(record)
                for latent_frames, same_length_records in sorted(
                    records_by_frames.items()
                ):
                    for batch_records in _chunks(
                        sorted(same_length_records), args.batch_size
                    ):
                        latent_rows = []
                        for _, key, recorded_frames, _ in batch_records:
                            latent = tensors.get_tensor(key)
                            expected_shape = (64, int(recorded_frames))
                            if tuple(latent.shape) != expected_shape:
                                raise RuntimeError(
                                    f"latent {key} shape {tuple(latent.shape)} != "
                                    f"index shape {expected_shape}"
                                )
                            if not bool(torch.isfinite(latent).all()):
                                raise RuntimeError(f"latent {key} is non-finite")
                            latent_rows.append(latent)
                        latents = torch.stack(latent_rows).to(
                            device=device, dtype=torch.float32
                        )
                        with torch.inference_mode():
                            decoded = vae.decode(latents).float()
                        rows = []
                        for audio, (ordinal, _, _, model_num_samples) in zip(
                            decoded, batch_records
                        ):
                            if int(model_num_samples) > int(audio.shape[-1]):
                                raise RuntimeError(
                                    "decoded audio is shorter than model_num_samples"
                                )
                            result = _transcribe(
                                asr,
                                _mono_for_asr(
                                    audio, valid_samples=int(model_num_samples)
                                ),
                            )
                            rows.append(
                                (
                                    int(ordinal),
                                    result["text"],
                                    result["has_speech"],
                                    result["confidence"],
                                    result["language"],
                                    result["language_probability"],
                                    result["mean_average_log_probability"],
                                    result["mean_no_speech_probability"],
                                    result["speech_seconds"],
                                    result["segment_count"],
                                )
                            )
                        destination.executemany(
                            "INSERT INTO hypotheses VALUES (?,?,?,?,?,?,?,?,?,?)",
                            rows,
                        )
                        written += len(rows)
            destination.commit()
            print(
                json.dumps(
                    {
                        "event": "progress",
                        "shards": shard_number + 1,
                        "shards_total": len(by_shard),
                        "rows": written,
                        "elapsed_sec": time.perf_counter() - started,
                    }
                ),
                flush=True,
            )
        if written != len(ordinals):
            raise RuntimeError(
                f"lexical cache wrote {written} of {len(ordinals)} rows"
            )
        metadata = {
            "schema": P11_LEXICAL_CACHE_SCHEMA,
            "schema_version": str(P11_LEXICAL_CACHE_VERSION),
            "builder": str(Path(__file__).resolve()),
            "builder_sha256": _sha256(Path(__file__).resolve()),
            "contract": P11_LEXICAL_EVIDENCE_CONTRACT,
            "confidence_contract": P11_LEXICAL_CONFIDENCE_CONTRACT,
            "source_manifest": str(manifest),
            "source_index": str(index),
            "source": "input_foa_only",
            "target_transcript_access": "forbidden",
            "rows": str(written),
            "encoder_revision": str(args.encoder_revision),
            "encoder_model": str(asr_path),
            "encoder_model_sha256": _sha256(asr_path / "model.bin"),
            "language": "en",
            "beam_size": "5",
            "vad_filter": "true",
            "vae_config": str(vae_config_path),
            "vae_checkpoint": str(vae_checkpoint),
            "vae_checkpoint_sha256": _sha256(vae_checkpoint),
            "foa_channel": "W",
            "normalization": "per_clip_peak_minus_1db",
            "sample_rate": "16000",
            "shard_index": str(args.shard_index),
            "num_shards": str(args.num_shards),
            "elapsed_sec": f"{time.perf_counter() - started:.6f}",
        }
        if ordinal_inventory is not None:
            assert inventory_metadata is not None
            metadata.update(
                {
                    "ordinal_inventory": str(ordinal_inventory),
                    "ordinal_inventory_sha256": _sha256(ordinal_inventory),
                    "ordinal_selection_contract": inventory_metadata[
                        "selection_contract"
                    ],
                    "source_manifest_sha256": inventory_metadata[
                        "source_manifest_sha256"
                    ],
                }
            )
        destination.executemany(
            "INSERT INTO metadata(key,value) VALUES (?,?)", metadata.items()
        )
        destination.commit()
        destination.execute("VACUUM")
        destination.commit()
        completed = True
    finally:
        destination.close()
        source.close()
        if not completed:
            temporary.unlink(missing_ok=True)
    os.replace(temporary, output)
    print(
        json.dumps(
            {
                "output": str(output),
                "rows": written,
                "encoder_revision": str(args.encoder_revision),
                "elapsed_sec": time.perf_counter() - started,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
