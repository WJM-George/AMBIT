#!/usr/bin/env python3
"""Recompile ScenePlan v1 into v1.1 without touching audio or VAE latents.

The migration is deliberately JSONL-to-JSONL. It fixes duplicated TTS speech
semantics, separates weak renderer extent from true activity annotations, and
upgrades motion metadata for arbitrary keyframes. Byte-offset JSONL and SQLite
indexes are rebuilt atomically for the new store.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
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
from stable_audio_tools.data.scene_plan import (  # noqa: E402
    DSL_VERSION,
    serialize_spatial_dsl,
)
from stable_audio_tools.data.t2a_artifacts import (  # noqa: E402
    atomic_write_json,
    atomic_write_jsonl,
)


TARGET_SCHEMA_VERSION = "1.1"
TTS_DATASETS = {
    "spatial_speech_tts_sdb",
    "spatial_speech_tts_sdc",
}


def upgrade_record(record: dict[str, Any]) -> dict[str, Any]:
    upgraded = copy.deepcopy(record)
    upgraded["schema"] = SCHEMA
    upgraded["schema_version"] = TARGET_SCHEMA_VERSION
    duration = upgraded.get("audio", {}).get("duration_sec")
    dataset_id = upgraded.get("dataset_id")
    has_activity = False
    for source in upgraded.get("scene", {}).get("sources", []):
        event = source.setdefault("event", {})
        content = source.setdefault("content", {})
        if dataset_id in TTS_DATASETS:
            event["label"] = "speech"
            event["category"] = event.get("category") or "speech"
        activity = source.setdefault("activity", {})
        onset = activity.get("onset_sec")
        offset = activity.get("offset_sec")
        annotated = onset is not None and offset is not None and float(offset) > float(onset)
        activity["quality"] = "source_annotation" if annotated else "not_annotated"
        if not annotated:
            activity["onset_sec"] = None
            activity["offset_sec"] = None
        has_activity |= annotated
        if duration is not None:
            activity["render_window_sec"] = [0.0, float(duration)]
            activity["render_window_quality"] = "renderer_defined_extent"

        motion = source.setdefault("motion", {})
        keyframes = motion.get("keyframes") or []
        keyframes.sort(key=lambda item: float(item.get("t_norm", 0.0)))
        motion["keyframes"] = keyframes
        if len(keyframes) > 2:
            motion["type"] = "keyframed"
        elif len(keyframes) == 2:
            motion["type"] = "linear"
        else:
            motion["type"] = "static"
        motion.setdefault("time_basis", "full_clip")
        motion.setdefault(
            "timing_quality",
            "renderer_defined" if len(keyframes) > 1 else "static",
        )
        motion.setdefault("interpolation", "linear_shortest_azimuth_arc")

        # v1 TTS duplicated the transcript into event.label. The transcript is
        # retained exactly once; source paths/provenance are intentionally not
        # part of the generative compact codec.
        if dataset_id in TTS_DATASETS and content.get("transcript"):
            event["label"] = "speech"

    supervision = upgraded.setdefault("supervision", {})
    supervision["source_activity_time"] = bool(has_activity)
    provenance = upgraded.setdefault("provenance", {})
    provenance["scene_plan_migration"] = "v1_to_v1.1"
    return upgraded


def _iter_input_records(root: Path, limit: Optional[int]) -> Iterator[dict[str, Any]]:
    emitted = 0
    for shard in sorted((root / "shards").glob("*.jsonl")):
        with shard.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                yield json.loads(line)
                emitted += 1
                if limit is not None and emitted >= limit:
                    return


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--shard-size", type=int, default=50_000)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    if args.shard_size <= 0 or (args.limit is not None and args.limit <= 0):
        raise SystemExit("--shard-size/--limit must be positive")

    input_root = args.input_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if not (input_root / "READY").is_file():
        raise SystemExit(f"input ScenePlan store is not READY: {input_root}")
    if (output_root / "READY").exists():
        raise SystemExit(f"output ScenePlan store is already READY: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    dsl_root = output_root / "views" / DSL_VERSION

    stats = {
        "samples": 0,
        "shards": 0,
        "dataset_counts": {},
        "tts_speech_labels_normalized": 0,
        "activity_annotated_samples": 0,
        "dynamic_samples": 0,
    }
    def write_shard(upgraded: list[dict[str, Any]], shard_id: int) -> None:
        canonical_relative = Path("shards") / f"scene-plan-{shard_id:05d}.jsonl"
        canonical_index = _write_indexed_shard(
            output_root, canonical_relative, upgraded
        )
        atomic_write_jsonl(
            output_root / "shard_indexes" / f"index-{shard_id:05d}.jsonl",
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
            for record in upgraded
        ]
        dsl_relative = Path("shards") / f"targets-{shard_id:05d}.jsonl"
        dsl_index = _write_indexed_shard(dsl_root, dsl_relative, dsl_records)
        atomic_write_jsonl(
            dsl_root / "shard_indexes" / f"index-{shard_id:05d}.jsonl",
            dsl_index,
        )
        for record in upgraded:
            dataset_id = record.get("dataset_id", "unknown")
            stats["dataset_counts"][dataset_id] = stats["dataset_counts"].get(dataset_id, 0) + 1
            stats["tts_speech_labels_normalized"] += int(dataset_id in TTS_DATASETS)
            stats["activity_annotated_samples"] += int(
                bool(record.get("supervision", {}).get("source_activity_time"))
            )
            stats["dynamic_samples"] += int(
                bool(record.get("supervision", {}).get("motion"))
            )
        stats["samples"] += len(upgraded)
        stats["shards"] += 1
        print(
            f"[scene-plan-v1.1] shard={shard_id + 1} samples={len(upgraded)}",
            flush=True,
        )

    buffer: list[dict[str, Any]] = []
    num_shards = 0
    for record in _iter_input_records(input_root, args.limit):
        buffer.append(upgrade_record(record))
        if len(buffer) < args.shard_size:
            continue
        write_shard(buffer, num_shards)
        num_shards += 1
        buffer = []
    if buffer:
        write_shard(buffer, num_shards)
        num_shards += 1
    sample_count = int(stats["samples"])
    if sample_count == 0:
        raise SystemExit("input ScenePlan store contains no records")

    canonical_count, canonical_sha = atomic_write_jsonl(
        output_root / "index.jsonl", _iter_index_parts(output_root, num_shards)
    )
    dsl_count, dsl_sha = atomic_write_jsonl(
        dsl_root / "index.jsonl", _iter_index_parts(dsl_root, num_shards)
    )
    if canonical_count != sample_count or dsl_count != sample_count:
        raise RuntimeError("migrated ScenePlan index count mismatch")
    if _build_sqlite(output_root, _iter_index_parts(output_root, num_shards)) != sample_count:
        raise RuntimeError("migrated canonical SQLite count mismatch")
    if _build_sqlite(dsl_root, _iter_index_parts(dsl_root, num_shards)) != sample_count:
        raise RuntimeError("migrated DSL SQLite count mismatch")

    atomic_write_json(
        output_root / "details.json",
        {
            "schema": SCHEMA,
            "schema_version": TARGET_SCHEMA_VERSION,
            "source_store": str(input_root),
            "migration": "v1_to_v1.1",
            "samples": sample_count,
            "shard_size": args.shard_size,
        },
    )
    atomic_write_json(
        output_root / "schema.json",
        {
            "schema": SCHEMA,
            "schema_version": TARGET_SCHEMA_VERSION,
            "activity_timing": "strong onset/offset remains unknown unless annotated; renderer extent is separate",
            "tts_event_normalization": "event.label=speech; content.transcript retained once",
            "motion": "static, linear, or arbitrary keyframed paths",
        },
    )
    atomic_write_json(output_root / "stats.json", stats)
    atomic_write_json(
        dsl_root / "READY",
        {
            "schema": "stable_audio_tools.spatial_cot_target",
            "schema_version": 1,
            "samples": sample_count,
            "index_sha256": dsl_sha,
            "canonical_store": str(output_root),
        },
    )
    atomic_write_json(
        output_root / "READY",
        {
            "schema": SCHEMA,
            "schema_version": TARGET_SCHEMA_VERSION,
            "samples": sample_count,
            "shards": num_shards,
            "index_sha256": canonical_sha,
            "dsl_view": str(dsl_root),
        },
    )
    print(
        f"[scene-plan-v1.1] READY samples={sample_count} root={output_root}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
