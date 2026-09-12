#!/usr/bin/env python3
"""Derive weak single-source activity boundaries into ScenePlan v1.3.

Per-source activity cannot be recovered from a multi-source mixture.  For
single-source rendered FOA (notably the 200K TTS route), conservative waveform
energy boundaries are honest weak labels and prevent several seconds of
renderer padding from being described as an active event.  Input audio and VAE
latents are never changed.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import copy
import functools
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterator, Optional

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.t2a.data.build_scene_plan_store import (  # noqa: E402
    SCHEMA,
    _build_sqlite,
    _iter_index_parts,
    _write_indexed_shard,
)
from scripts.t2a.data.upgrade_scene_plan_v1_1 import upgrade_record  # noqa: E402
from stable_audio_tools.data.audio_activity import measure_audio_file  # noqa: E402
from stable_audio_tools.data.scene_plan import DSL_VERSION, serialize_spatial_dsl  # noqa: E402
from stable_audio_tools.data.t2a_artifacts import (  # noqa: E402
    atomic_write_json,
    atomic_write_jsonl,
)


TARGET_VERSION = "1.3"
DEFAULT_DATASETS = {
    "spatial_speech_tts_sdb",
    "spatial_speech_tts_sdc",
}


def _iter_records(
    root: Path,
    *,
    filter_datasets: Optional[set[str]],
    limit: Optional[int],
) -> Iterator[dict[str, Any]]:
    emitted = 0
    for shard in sorted((root / "shards").glob("*.jsonl")):
        with shard.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                if filter_datasets and record.get("dataset_id") not in filter_datasets:
                    continue
                yield record
                emitted += 1
                if limit is not None and emitted >= limit:
                    return


def annotate_record(
    record: dict[str, Any],
    *,
    eligible_datasets: set[str],
    frame_ms: float,
    absolute_dbfs: float,
    relative_db: float,
) -> tuple[dict[str, Any], str]:
    upgraded = upgrade_record(record)
    upgraded["schema"] = SCHEMA
    upgraded["schema_version"] = TARGET_VERSION
    dataset_id = str(upgraded.get("dataset_id") or "unknown")
    sources = ((upgraded.get("scene") or {}).get("sources") or [])
    if dataset_id not in eligible_datasets:
        return upgraded, "dataset_not_selected"
    if len(sources) != 1:
        return upgraded, "not_single_source"
    audio_path = (upgraded.get("audio") or {}).get("path")
    if not audio_path:
        return upgraded, "missing_audio_path"
    try:
        measurement = measure_audio_file(
            audio_path,
            frame_ms=frame_ms,
            absolute_dbfs=absolute_dbfs,
            relative_db=relative_db,
        )
    except (OSError, RuntimeError, ValueError):
        return upgraded, "measurement_error"
    onset = measurement.get("activity_onset_sec")
    offset = measurement.get("activity_offset_sec")
    if onset is None or offset is None or float(offset) <= float(onset):
        return upgraded, "all_silent"
    activity = sources[0].setdefault("activity", {})
    measured_duration = float(measurement["audio_duration_sec"])
    audio = upgraded.setdefault("audio", {})
    audio["duration_sec"] = measured_duration
    activity.update(
        {
            "onset_sec": float(onset),
            "offset_sec": float(offset),
            "quality": "foa_energy_derived_single_source",
            "measurement": {
                "frame_ms": frame_ms,
                "absolute_dbfs": absolute_dbfs,
                "relative_db": relative_db,
                "active_ratio": measurement["active_ratio"],
                "leading_silence_sec": measurement["leading_silence_sec"],
                "trailing_silence_sec": measurement["trailing_silence_sec"],
            },
        }
    )
    if activity.get("render_window_sec") is not None:
        activity["render_window_sec"] = [0.0, measured_duration]
    supervision = upgraded.setdefault("supervision", {})
    supervision["source_activity_time"] = True
    supervision["source_activity_time_quality"] = "weak_single_source_energy"
    provenance = upgraded.setdefault("provenance", {})
    provenance["activity_annotation"] = "scene_plan_v1.3_single_source_energy"
    provenance["duration_normalization"] = "exact_audio_frames_over_sample_rate"
    return upgraded, "annotated"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset-id", action="append", default=None)
    parser.add_argument(
        "--filter-dataset",
        action="append",
        default=None,
        help="emit only these datasets (smoke/debug only; omit for production)",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--shard-size", type=int, default=50_000)
    parser.add_argument("--frame-ms", type=float, default=50.0)
    parser.add_argument("--absolute-dbfs", type=float, default=-55.0)
    parser.add_argument("--relative-db", type=float, default=-40.0)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Ordered process workers for independent per-audio measurements.",
    )
    parser.add_argument("--worker-chunksize", type=int, default=64)
    args = parser.parse_args()

    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive")
    if min(args.shard_size, args.frame_ms, args.workers, args.worker_chunksize) <= 0:
        raise SystemExit("shard-size, frame-ms and worker values must be positive")
    input_root = args.input_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if not (input_root / "READY").is_file():
        raise SystemExit(f"input ScenePlan store is not READY: {input_root}")
    if (output_root / "READY").exists():
        raise SystemExit(f"output ScenePlan store is already READY: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    dsl_root = output_root / "views" / DSL_VERSION
    eligible = set(args.dataset_id or DEFAULT_DATASETS)
    filters = set(args.filter_dataset or []) or None

    stats: dict[str, Any] = {
        "samples": 0,
        "dataset_counts": Counter(),
        "activity_status": Counter(),
        "trailing_silence_sec_sum": 0.0,
    }
    shard_count = 0
    buffer: list[dict[str, Any]] = []

    def flush() -> None:
        nonlocal buffer, shard_count
        if not buffer:
            return
        canonical_relative = Path("shards") / f"scene-plan-{shard_count:05d}.jsonl"
        canonical_index = _write_indexed_shard(
            output_root, canonical_relative, buffer
        )
        atomic_write_jsonl(
            output_root / "shard_indexes" / f"index-{shard_count:05d}.jsonl",
            canonical_index,
        )
        dsl_records = [
            {
                "schema": "stable_audio_tools.spatial_cot_target",
                "schema_version": 1,
                "sample_id": record["sample_id"],
                "audio_path": record["audio"]["path"],
                "caption": record["caption"],
                "target": serialize_spatial_dsl(record),
                "field_mask": record["supervision"],
            }
            for record in buffer
        ]
        dsl_relative = Path("shards") / f"targets-{shard_count:05d}.jsonl"
        dsl_index = _write_indexed_shard(dsl_root, dsl_relative, dsl_records)
        atomic_write_jsonl(
            dsl_root / "shard_indexes" / f"index-{shard_count:05d}.jsonl",
            dsl_index,
        )
        print(
            f"[scene-plan-v1.3] shard={shard_count} samples={len(buffer)}",
            flush=True,
        )
        buffer = []
        shard_count += 1

    records = _iter_records(
        input_root, filter_datasets=filters, limit=args.limit
    )
    annotate = functools.partial(
        annotate_record,
        eligible_datasets=eligible,
        frame_ms=args.frame_ms,
        absolute_dbfs=args.absolute_dbfs,
        relative_db=args.relative_db,
    )
    executor = None
    if args.workers == 1:
        annotated_records = map(annotate, records)
    else:
        executor = concurrent.futures.ProcessPoolExecutor(max_workers=args.workers)
        annotated_records = executor.map(
            annotate,
            records,
            chunksize=args.worker_chunksize,
        )
    try:
        for annotated, status in annotated_records:
            buffer.append(annotated)
            stats["samples"] += 1
            stats["dataset_counts"][annotated.get("dataset_id", "unknown")] += 1
            stats["activity_status"][status] += 1
            if status == "annotated":
                measurement = annotated["scene"]["sources"][0]["activity"]["measurement"]
                stats["trailing_silence_sec_sum"] += float(
                    measurement["trailing_silence_sec"]
                )
            if len(buffer) >= args.shard_size:
                flush()
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    flush()
    if not stats["samples"]:
        raise SystemExit("no ScenePlan records selected")

    canonical_count, canonical_sha = atomic_write_jsonl(
        output_root / "index.jsonl", _iter_index_parts(output_root, shard_count)
    )
    dsl_count, dsl_sha = atomic_write_jsonl(
        dsl_root / "index.jsonl", _iter_index_parts(dsl_root, shard_count)
    )
    if canonical_count != stats["samples"] or dsl_count != stats["samples"]:
        raise RuntimeError("ScenePlan v1.3 index count mismatch")
    if _build_sqlite(
        output_root, _iter_index_parts(output_root, shard_count)
    ) != stats["samples"]:
        raise RuntimeError("ScenePlan v1.3 SQLite count mismatch")
    if _build_sqlite(
        dsl_root, _iter_index_parts(dsl_root, shard_count)
    ) != stats["samples"]:
        raise RuntimeError("ScenePlan v1.3 DSL SQLite count mismatch")

    serializable_stats = {
        key: dict(value) if isinstance(value, Counter) else value
        for key, value in stats.items()
    }
    atomic_write_json(output_root / "stats.json", serializable_stats)
    atomic_write_json(
        output_root / "schema.json",
        {
            "schema": SCHEMA,
            "schema_version": TARGET_VERSION,
            "activity_timing": (
                "strong annotations are preserved; selected single-source FOA "
                "receives conservative waveform-energy weak boundaries; "
                "multi-source activity remains unknown"
            ),
            "eligible_dataset_ids": sorted(eligible),
        },
    )
    atomic_write_json(
        output_root / "details.json",
        {
            "schema": SCHEMA,
            "schema_version": TARGET_VERSION,
            "input_root": str(input_root),
            "filter_datasets": sorted(filters) if filters else None,
            "frame_ms": args.frame_ms,
            "absolute_dbfs": args.absolute_dbfs,
            "relative_db": args.relative_db,
            "samples": stats["samples"],
        },
    )
    atomic_write_json(
        dsl_root / "READY",
        {
            "schema": "stable_audio_tools.spatial_cot_target",
            "schema_version": 1,
            "samples": stats["samples"],
            "index_sha256": dsl_sha,
            "canonical_store": str(output_root),
        },
    )
    atomic_write_json(
        output_root / "READY",
        {
            "schema": SCHEMA,
            "schema_version": TARGET_VERSION,
            "samples": stats["samples"],
            "shards": shard_count,
            "index_sha256": canonical_sha,
            "dsl_view": str(dsl_root),
        },
    )
    print(
        f"[scene-plan-v1.3] READY samples={stats['samples']} "
        f"annotated={stats['activity_status'].get('annotated', 0)} "
        f"root={output_root}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
