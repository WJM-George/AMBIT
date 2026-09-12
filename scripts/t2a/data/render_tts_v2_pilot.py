#!/usr/bin/env python3
"""Build the revision-4 2k complete-speech Pyroom FOA pilot on SDB."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import random
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
import soundfile as sf


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from sceneplan_v2_common import (  # noqa: E402
    CONTRACT_REVISION,
    DATASET_ROOT,
    MAX_LATENT_FRAMES,
    MAX_MODEL_SAMPLES,
    MODEL_SAMPLE_RATE,
    VAE_HOP_SAMPLES,
    atomic_write_json,
    decode_complete_mono,
    deterministic_digest,
    load_parquet_source,
    normalized_transcript,
)
from sceneplan_v2_renderer import (  # noqa: E402
    FOA_LAYOUT,
    peak_safe_master_gain,
    render_complete_mono_source,
    true_peak,
)


SEED = 20260814
LEDGER = DATASET_ROOT / "split_ledgers/speech_v2/speech_split_ledger.parquet"
DEFAULT_OUTPUT = DATASET_ROOT / "pilots/tts_2k"
VAE_CHECKPOINT = Path("/mnt/sdc/ckpts/compareVAE_ckpt/unwrapped_wdmix_1350000.ckpt")
DURATION_BINS = ((0.0, 2.0), (2.0, 4.0), (4.0, 6.0), (6.0, 8.0), (8.0, MAX_MODEL_SAMPLES / MODEL_SAMPLE_RATE + 1e-9))
ROOM_CLASSES = ("dry", "moderate", "reverberant", "outdoor")
MOTIONS = ("static", "dynamic")


def ensure_sdb(path: Path) -> None:
    resolved = path.expanduser().resolve(strict=False)
    try:
        resolved.relative_to("/mnt/sdb")
    except ValueError as error:
        raise ValueError(f"revision-4 output must be on SDB: {resolved}") from error


def atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(temporary, path)


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TypeError(f"non-object JSONL row in {path}")
                yield value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def duration_bin(duration: float) -> int:
    for index, (lower, upper) in enumerate(DURATION_BINS):
        if lower <= duration < upper:
            return index
    raise ValueError(f"duration outside pilot bins: {duration}")


def room_recipe(room_class: str, seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    ranges = {
        "dry": ((6.0, 8.0), (5.0, 7.0), (3.0, 4.0), (0.16, 0.28), (5, 7)),
        "moderate": ((8.0, 12.0), (7.0, 10.0), (3.2, 4.8), (0.38, 0.62), (6, 9)),
        "reverberant": ((12.0, 18.0), (10.0, 15.0), (4.5, 7.0), (0.78, 1.20), (6, 9)),
        "outdoor": ((10.0, 16.0), (10.0, 16.0), (5.0, 8.0), (0.30, 0.45), (0, 0)),
    }
    if room_class not in ranges:
        raise ValueError(room_class)
    lr, wr, hr, rr, order = ranges[room_class]
    dimensions = [rng.uniform(*lr), rng.uniform(*wr), rng.uniform(*hr)]
    microphone = [dimensions[0] / 2.0, dimensions[1] / 2.0, min(1.7, dimensions[2] / 2.0)]
    return {
        "room_id": f"pilot_{room_class}_{seed:016x}",
        "class": room_class,
        "dimensions_m": dimensions,
        "rt60_sec": rng.uniform(*rr),
        "max_order": rng.randint(*order),
        "microphone_xyz_m": microphone,
        "material_model": "pyroom_inverse_sabine_v1",
    }


def trajectory(motion: str, duration_sec: float, room: dict[str, Any], seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    start_azimuth = rng.uniform(-180.0, 180.0)
    start_elevation = rng.uniform(-18.0, 18.0)
    start_distance = rng.uniform(0.9, 2.2)

    def position(azimuth: float, elevation: float, distance: float) -> dict[str, float]:
        wrapped = ((azimuth + 180.0) % 360.0) - 180.0
        return {
            "azimuth_deg": float(wrapped),
            "elevation_deg": float(max(-25.0, min(25.0, elevation))),
            "distance_m": float(max(0.8, min(2.4, distance))),
        }

    if motion == "static":
        return [{"time_sec": 0.0, "position": position(start_azimuth, start_elevation, start_distance)}]
    if motion != "dynamic":
        raise ValueError(motion)
    sweep = rng.choice((-1.0, 1.0)) * rng.uniform(65.0, 125.0)
    elevation_delta = rng.uniform(-18.0, 18.0)
    distance_delta = rng.uniform(-0.7, 0.7)
    stop = max(1.0 / MODEL_SAMPLE_RATE, duration_sec)
    return [
        {"time_sec": 0.0, "position": position(start_azimuth, start_elevation, start_distance)},
        {
            "time_sec": stop * 0.5,
            "position": position(
                start_azimuth + sweep * 0.5,
                start_elevation + elevation_delta * 0.5,
                start_distance + distance_delta * 0.5,
            ),
        },
        {
            "time_sec": stop,
            "position": position(
                start_azimuth + sweep,
                start_elevation + elevation_delta,
                start_distance + distance_delta,
            ),
        },
    ]


def caption_regions(transcript: str, speaker_id: str) -> dict[str, Any]:
    # renderer_text is already the frozen, whitespace-normalized ledger text.
    # Preserve every character inside the explicit transcript span, including
    # nested double quotes; character offsets, rather than quote parsing, bind
    # the transcript to its source.
    spoken = " ".join(transcript.split())
    speaker = f"English audiobook speaker {speaker_id}"
    prefix = speaker + " says \""
    caption = prefix + spoken + "\"."
    transcript_start = len(prefix)
    transcript_end = transcript_start + len(spoken)
    return {
        "compiler": "sceneplan_renderer_caption",
        "compiler_version": 4,
        "text": caption,
        "source_regions": [
            {
                "source_id": "source_0",
                "source_slot": 0,
                "start": 0,
                "end": transcript_end,
                "role": "source_semantic",
            }
        ],
        "speaker_info_regions": [
            {
                "source_id": "source_0",
                "source_slot": 0,
                "start": 0,
                "end": len(speaker),
                "role": "speaker_info",
            }
        ],
        "transcript_regions": [
            {
                "source_id": "source_0",
                "source_slot": 0,
                "start": transcript_start,
                "end": transcript_end,
                "role": "transcript",
            }
        ],
    }


def select_pilot(ledger_path: Path) -> list[dict[str, Any]]:
    columns = [
        "asset_id",
        "source_dataset",
        "source_id",
        "pool",
        "speaker_id",
        "speaker_key",
        "chapter_id",
        "source_split",
        "source_text",
        "renderer_text",
        "normalized_transcript",
        "parquet_path",
        "row_group",
        "row_in_group",
        "source_audio_sha256",
        "native_sample_rate_hz",
        "native_num_samples",
        "native_channels",
        "model_sample_rate_hz",
        "model_num_samples",
        "duration_sec",
        "selection_rank",
    ]
    table = pq.read_table(ledger_path, columns=columns, filters=[("pool", "=", "train")])
    selected: list[dict[str, Any]] = []
    for dataset in ("libritts", "hifi_tts"):
        for bucket, (lower, upper) in enumerate(DURATION_BINS):
            mask = pc.equal(table["source_dataset"], dataset)
            mask = pc.and_(mask, pc.greater_equal(table["duration_sec"], lower))
            mask = pc.and_(mask, pc.less(table["duration_sec"], upper))
            candidates = table.filter(mask).sort_by([("selection_rank", "ascending")])
            if candidates.num_rows < 200:
                raise RuntimeError(f"not enough {dataset} duration-bin {bucket} candidates")
            rows = candidates.slice(0, 200).to_pylist()
            for index, row in enumerate(rows):
                motion = MOTIONS[index % 2]
                motion_index = index // 2
                room_class = ROOM_CLASSES[motion_index % len(ROOM_CLASSES)]
                digest = deterministic_digest(SEED, "tts_pilot", row["asset_id"], motion)
                recipe_seed = int(digest[:16], 16)
                room = room_recipe(room_class, recipe_seed)
                source_samples = int(row["model_num_samples"])
                target_tail = {
                    "dry": 40 + round(0.05 * MODEL_SAMPLE_RATE),
                    "moderate": 40 + round(0.12 * MODEL_SAMPLE_RATE),
                    "reverberant": 40 + round(0.25 * MODEL_SAMPLE_RATE),
                    "outdoor": 40,
                }[room_class]
                render_tail = min(MAX_MODEL_SAMPLES - source_samples, target_tail)
                if render_tail < 40:
                    raise RuntimeError(f"source {row['asset_id']} cannot preserve Pyroom delay")
                scene_samples = source_samples + render_tail
                row.update(
                    {
                        "sample_id": f"tts2k_{dataset}_b{bucket}_{motion}_{motion_index:03d}",
                        "partition": "train",
                        "duration_bin": bucket,
                        "motion": motion,
                        "room_class": room_class,
                        "recipe_seed": recipe_seed,
                        "room": room,
                        "trajectory": trajectory(
                            motion,
                            source_samples / MODEL_SAMPLE_RATE,
                            room,
                            recipe_seed ^ 0xA57D1C,
                        ),
                        "render_tail_samples": render_tail,
                        "scene_num_samples": scene_samples,
                    }
                )
                selected.append(row)
    selected.sort(key=lambda row: str(row["sample_id"]))
    if len(selected) != 2_000 or len({row["asset_id"] for row in selected}) != 2_000:
        raise RuntimeError("pilot selection is not 2k unique assets")
    return selected


def normalize_dry(mono: np.ndarray) -> tuple[np.ndarray, float]:
    value = np.asarray(mono, dtype=np.float32)
    value = value - float(np.mean(value, dtype=np.float64))
    rms = float(np.sqrt(np.mean(np.square(value, dtype=np.float64))))
    peak = float(np.max(np.abs(value)))
    if not math.isfinite(rms) or rms < 1e-7 or peak < 1e-7:
        raise ValueError("silent/non-finite dry source")
    gain = min(10.0 ** (-24.0 / 20.0) / rms, 0.95 / peak, 10.0 ** (30.0 / 20.0))
    return (value * gain).astype(np.float32), float(gain)


def write_pcm24(path: Path, audio: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=path.suffix, dir=path.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        sf.write(temporary, audio.T, MODEL_SAMPLE_RATE, subtype="PCM_24")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def render_one(row: dict[str, Any], output_root: str) -> dict[str, Any]:
    root = Path(output_root)
    sample_id = str(row["sample_id"])
    shard = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:2]
    scene_root = root / "scenes" / shard / sample_id
    result_path = scene_root / "result.json"
    if result_path.is_file():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        # A quarantine replacement deliberately keeps the stable sample_id but
        # changes its donor asset.  Never reuse the former donor's result just
        # because the scene path and status still look valid.
        if (
            result.get("status") == "ok"
            and result.get("asset_id") == row.get("asset_id")
            and result.get("source_audio", {}).get("source_audio_sha256")
            == row.get("source_audio_sha256")
            and Path(result["foa_path"]).is_file()
        ):
            # Recompile metadata even when the expensive FOA is reusable.  This
            # makes a resumed pilot pick up deterministic compiler fixes without
            # touching the already verified audio bytes.
            caption = caption_regions(str(row["renderer_text"]), str(row["speaker_id"]))
            if (
                result.get("transcript") != row.get("renderer_text")
                or result.get("renderer_caption") != caption
            ):
                sceneplan_path = Path(result["sceneplan_path"])
                scene_plan = json.loads(sceneplan_path.read_text(encoding="utf-8"))
                scene_plan["caption"] = caption
                scene_plan["source"]["transcript"] = row["renderer_text"]
                atomic_write_json(sceneplan_path, scene_plan)
                result["transcript"] = row["renderer_text"]
                result["renderer_caption"] = caption
                atomic_write_json(result_path, result)
            return result
    started = time.time()
    try:
        locator = {
            "parquet_path": row["parquet_path"],
            "row_group": int(row["row_group"]),
            "row_in_group": int(row["row_in_group"]),
        }
        blob, source_metadata = load_parquet_source(locator)
        if hashlib.sha256(blob).hexdigest() != row["source_audio_sha256"]:
            raise RuntimeError("source audio SHA256 changed")
        mono, native_rate, native_frames = decode_complete_mono(blob)
        if native_rate != int(row["native_sample_rate_hz"]):
            raise RuntimeError("native sample rate changed")
        if native_frames != int(row["native_num_samples"]):
            raise RuntimeError("native sample count changed")
        if len(mono) != int(row["model_num_samples"]):
            raise RuntimeError("resampled complete utterance length changed")
        parquet_text = str(
            source_metadata.get("text_normalized")
            or source_metadata.get("text_no_preprocessing")
            or source_metadata.get("text_original")
            or source_metadata.get("text")
            or ""
        )
        if normalized_transcript(parquet_text) != row["normalized_transcript"]:
            raise RuntimeError("Parquet transcript lineage changed")
        dry, dry_gain = normalize_dry(mono)
        track, renderer_qc = render_complete_mono_source(
            dry,
            sample_rate=MODEL_SAMPLE_RATE,
            scene_num_samples=int(row["scene_num_samples"]),
            onset_sample=0,
            room=row["room"],
            keyframes=row["trajectory"],
        )
        master_gain, gain_qc = peak_safe_master_gain(track)
        foa = (track * master_gain).astype(np.float32, copy=False)
        measured_true_peak = true_peak(foa)
        ceiling = 10.0 ** (-1.0 / 20.0)
        if measured_true_peak > ceiling + 1e-5:
            raise RuntimeError(f"true-peak ceiling exceeded: {measured_true_peak}")
        if foa.shape != (4, int(row["scene_num_samples"])) or not np.isfinite(foa).all():
            raise RuntimeError(f"invalid rendered FOA: {foa.shape}")
        foa_path = scene_root / "foa_WYZX_SN3D.flac"
        write_pcm24(foa_path, foa)
        info = sf.info(foa_path)
        if (info.frames, info.samplerate, info.channels) != (
            int(row["scene_num_samples"]),
            MODEL_SAMPLE_RATE,
            4,
        ):
            raise RuntimeError("stored FOA format changed")
        decoded, _ = sf.read(foa_path, always_2d=True, dtype="float32")
        stored_true_peak = true_peak(decoded.T)
        if stored_true_peak > ceiling + 2e-5:
            raise RuntimeError(f"stored true-peak ceiling exceeded: {stored_true_peak}")
        caption = caption_regions(str(row["renderer_text"]), str(row["speaker_id"]))
        scene_plan = {
            "schema": "stable_audio_tools.tts_v2_pilot_sceneplan",
            "schema_version": 1,
            "contract_revision": CONTRACT_REVISION,
            "sample_id": sample_id,
            "caption": caption,
            "audio": {
                "sample_rate_hz": MODEL_SAMPLE_RATE,
                "source_num_samples": len(mono),
                "render_num_samples": int(row["scene_num_samples"]),
                "render_tail_samples": int(row["render_tail_samples"]),
                "latent_frames_valid": math.ceil(int(row["scene_num_samples"]) / VAE_HOP_SAMPLES),
                "random_crop": False,
            },
            "room": row["room"],
            "source": {
                "source_id": "source_0",
                "kind": "speech",
                "asset_id": row["asset_id"],
                "speaker_id": row["speaker_id"],
                "transcript": row["renderer_text"],
                "activity": {
                    "model_onset_sample": 0,
                    "model_offset_sample": len(mono),
                    "complete_native_utterance": True,
                },
                "motion": row["motion"],
                "trajectory": row["trajectory"],
                "dry_gain_db": 20.0 * math.log10(dry_gain),
            },
            "renderer": {
                "backend": "pyroomacoustics_single_pass_v2",
                "foa_layout": FOA_LAYOUT,
                "master_gain_db": 20.0 * math.log10(master_gain),
            },
        }
        scene_root.mkdir(parents=True, exist_ok=True)
        atomic_write_json(scene_root / "sceneplan.json", scene_plan)
        result = {
            "schema": "stable_audio_tools.tts_v2_pilot_sample",
            "schema_version": 2,
            "contract_revision": CONTRACT_REVISION,
            "sample_id": sample_id,
            "status": "ok",
            "partition": row["partition"],
            "source_dataset": row["source_dataset"],
            "source_id": row["source_id"],
            "asset_id": row["asset_id"],
            "speaker_id": row["speaker_id"],
            "source_split": row["source_split"],
            "transcript": row["renderer_text"],
            "parquet_transcript": parquet_text,
            "renderer_caption": caption,
            "sceneplan_path": str(scene_root / "sceneplan.json"),
            "foa_path": str(foa_path),
            "foa_sha256": file_sha256(foa_path),
            "audio": {
                "sample_rate_hz": MODEL_SAMPLE_RATE,
                "source_num_samples": len(mono),
                "num_samples": int(row["scene_num_samples"]),
                "render_tail_samples": int(row["render_tail_samples"]),
                "duration_sec": int(row["scene_num_samples"]) / MODEL_SAMPLE_RATE,
                "latent_frames_valid": math.ceil(int(row["scene_num_samples"]) / VAE_HOP_SAMPLES),
                "channels": 4,
                "channel_layout": FOA_LAYOUT,
            },
            "source_audio": {
                "native_sample_rate_hz": native_rate,
                "native_num_samples": native_frames,
                "model_num_samples": len(mono),
                "source_audio_sha256": row["source_audio_sha256"],
                "consumed_native_start_sample": 0,
                "consumed_native_end_sample": native_frames,
                "coverage_fraction": 1.0,
                "random_crop": False,
                "parquet_path": row["parquet_path"],
                "row_group": int(row["row_group"]),
                "row_in_group": int(row["row_in_group"]),
                "parquet_speaker_id": str(row["speaker_id"]),
            },
            "spatial": {
                "motion": row["motion"],
                "trajectory": row["trajectory"],
                "room": row["room"],
                "renderer_qc": renderer_qc,
            },
            "signal": {
                "dry_normalization_gain": dry_gain,
                "master_gain": master_gain,
                "render_gain_diagnostics": gain_qc,
                "true_peak": measured_true_peak,
                "stored_true_peak": stored_true_peak,
                "dc_abs_max": float(np.max(np.abs(np.mean(decoded, axis=0, dtype=np.float64)))),
            },
            "strata": {
                "duration_bin": int(row["duration_bin"]),
                "motion": row["motion"],
                "room_class": row["room_class"],
                "source_dataset": row["source_dataset"],
            },
            "elapsed_sec": round(time.time() - started, 4),
        }
        atomic_write_json(result_path, result)
        return result
    except Exception as exc:  # noqa: BLE001
        return {
            "schema": "stable_audio_tools.tts_v2_pilot_sample",
            "schema_version": 2,
            "contract_revision": CONTRACT_REVISION,
            "sample_id": sample_id,
            "status": "error",
            "source_dataset": row.get("source_dataset"),
            "source_id": row.get("source_id"),
            "asset_id": row.get("asset_id"),
            "error": repr(exc),
            "elapsed_sec": round(time.time() - started, 4),
        }


def validate_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    status = Counter(str(row["status"]) for row in results)
    strata = Counter(
        (
            str(row.get("source_dataset")),
            int((row.get("strata") or {}).get("duration_bin", -1)),
            str((row.get("strata") or {}).get("motion")),
            str((row.get("strata") or {}).get("room_class")),
        )
        for row in results
        if row.get("status") == "ok"
    )
    expected = {
        (dataset, bucket, motion, room): 25
        for dataset in ("libritts", "hifi_tts")
        for bucket in range(5)
        for motion in MOTIONS
        for room in ROOM_CLASSES
    }
    mismatches = {
        "|".join(map(str, key)): {"actual": strata[key], "expected": value}
        for key, value in expected.items()
        if strata[key] != value
    }
    max_samples = max((int(row["audio"]["num_samples"]) for row in results if row.get("status") == "ok"), default=0)
    max_frames = max((int(row["audio"]["latent_frames_valid"]) for row in results if row.get("status") == "ok"), default=0)
    exact_transcript_spans = 0
    for row in results:
        if row.get("status") != "ok":
            continue
        caption = row.get("renderer_caption") or {}
        regions = caption.get("transcript_regions") or []
        if len(regions) != 1:
            continue
        region = regions[0]
        start, end = int(region["start"]), int(region["end"])
        if str(caption.get("text") or "")[start:end] == str(row.get("transcript") or ""):
            exact_transcript_spans += 1
    ok = (
        status["ok"] == 2_000
        and exact_transcript_spans == 2_000
        and not mismatches
        and max_samples <= MAX_MODEL_SAMPLES
        and max_frames <= MAX_LATENT_FRAMES
    )
    return {
        "ok": ok,
        "rows": len(results),
        "status_counts": dict(status),
        "strata_mismatches": mismatches,
        "exact_transcript_spans": exact_transcript_spans,
        "max_render_num_samples": max_samples,
        "max_latent_frames_valid": max_frames,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=LEDGER)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--jobs", type=int, default=min(24, os.cpu_count() or 1))
    parser.add_argument("--selection-only", action="store_true")
    args = parser.parse_args()
    output = args.output_root.expanduser().resolve(strict=False)
    ensure_sdb(output)
    output.mkdir(parents=True, exist_ok=True)
    ready = output / "READY"
    selection_path = output / "selection.jsonl"
    if selection_path.is_file():
        selected = list(iter_jsonl(selection_path))
    else:
        selected = select_pilot(args.ledger.expanduser().resolve(strict=True))
        atomic_write_jsonl(selection_path, selected)
    if len(selected) != 2_000:
        raise RuntimeError(f"frozen pilot selection has {len(selected)} rows")
    if args.selection_only:
        print(json.dumps({"selection": str(selection_path), "rows": len(selected)}, indent=2))
        return 0

    started = time.time()
    results: list[dict[str, Any]] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.jobs) as executor:
        future_map = {
            executor.submit(render_one, row, str(output)): row["sample_id"] for row in selected
        }
        for index, future in enumerate(concurrent.futures.as_completed(future_map), start=1):
            results.append(future.result())
            if index % 50 == 0 or index == len(selected):
                print(
                    json.dumps(
                        {
                            "rendered": index,
                            "total": len(selected),
                            "ok": sum(row["status"] == "ok" for row in results),
                            "errors": sum(row["status"] != "ok" for row in results),
                            "elapsed_sec": round(time.time() - started, 1),
                        }
                    ),
                    flush=True,
                )
    results.sort(key=lambda row: str(row["sample_id"]))
    manifest = output / "pilot_manifest.jsonl"
    atomic_write_jsonl(manifest, results)
    validation = validate_results(results)
    summary = {
        "schema": "stable_audio_tools.tts_v2_pilot_render_summary",
        "schema_version": 2,
        "contract_revision": CONTRACT_REVISION,
        "selection": str(selection_path),
        "manifest": str(manifest),
        "output_root": str(output),
        "renderer": "pyroomacoustics_single_pass_v2",
        "source_policy": "complete_parquet_backed_dry_mono_no_crop",
        "vae_checkpoint_sha256": file_sha256(VAE_CHECKPOINT),
        "validation": validation,
        "elapsed_sec": round(time.time() - started, 3),
    }
    atomic_write_json(output / "render_summary.json", summary)
    if validation["ok"]:
        atomic_write_json(
            ready,
            {
                "schema": "stable_audio_tools.tts_v2_pilot_render_ready",
                "schema_version": 2,
                "contract_revision": CONTRACT_REVISION,
                "rows": 2_000,
                "manifest": str(manifest),
                "summary": str(output / "render_summary.json"),
            },
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if validation["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
