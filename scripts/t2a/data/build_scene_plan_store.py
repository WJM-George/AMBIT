#!/usr/bin/env python3
"""Compile the finalized T2A mix into canonical ScenePlan v1.1 + SpatialDSL v1.

The finalized latent manifest is the authoritative sample selection and caption
view. Rich geometry is joined from the five source manifests by normalized FOA
path. Canonical JSONL shards and the derived DSL target shards both receive a
portable JSONL index and a read-only SQLite byte-offset index.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stable_audio_tools.data.scene_plan import (  # noqa: E402
    DSL_VERSION,
    serialize_spatial_dsl,
)
from stable_audio_tools.data.t2a_artifacts import (  # noqa: E402
    atomic_write_json,
    atomic_write_jsonl,
    normalize_audio_path,
    sha256_file,
    stable_sample_id,
)


SCHEMA = "stable_audio_tools.spatial_scene_plan"
SCHEMA_VERSION = "1.1"
DEFAULT_REQUIRED_MOUNTS = (os.environ.get("AMBIT_DATA_ROOT", "data"), os.environ.get("AMBIT_CKPT_ROOT", "checkpoints"), os.environ.get("AMBIT_DATA_ROOT", "data"))

SOURCE_SPECS = (
    {
        "dataset_id": "spatial_foa_existing",
        "kind": "synthetic_old",
        "manifest": os.environ.get("AMBIT_DATA_ROOT", "data") + "/spatial_foa/captions.jsonl",
    },
    {
        "dataset_id": "spatial_foa_v2_expansion",
        "kind": "synthetic_v2",
        "manifest": os.environ.get("AMBIT_DATA_ROOT", "data") + "/spatial_foa_v2/manifest_train_expansion.jsonl",
    },
    {
        "dataset_id": "spatial_librispeech",
        "kind": "sls",
        "manifest": os.environ.get("AMBIT_DATA_ROOT", "data") + "/spatial_foa/caption_jsonl/sls_train_218957.jsonl",
    },
    {
        "dataset_id": "spatial_speech_tts_sdb",
        "kind": "tts",
        "manifest": (
            os.environ.get("AMBIT_DATA_ROOT", "data") + "/datasets/spatial_speech_foa_tts_v1_part_sdb/"
            "manifests/render_manifest_qc_clean.jsonl"
        ),
        "source_plan_manifest": (
            os.environ.get("AMBIT_DATA_ROOT", "data") + "/datasets/spatial_speech_foa_tts_v1_part_sdb/"
            "manifests/source_plan_sdb.jsonl"
        ),
    },
    {
        "dataset_id": "spatial_speech_tts_sdc",
        "kind": "tts",
        "manifest": (
            os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/speech_dataset/spatial_speech_foa_tts_v1_part_sdc/"
            "manifests/render_manifest_qc_clean.jsonl"
        ),
        "source_plan_manifest": (
            os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/speech_dataset/spatial_speech_foa_tts_v1_part_sdc/"
            "manifests/source_plan_sdc.jsonl"
        ),
    },
)


def _caption(metadata: dict[str, Any]) -> str:
    for key in ("prompt", "text", "caption"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise KeyError("latent metadata has no prompt/text/caption")


def _load_selection(
    latent_root: Path, filelist: Path, limit: Optional[int]
) -> dict[str, dict[str, Any]]:
    selection: dict[str, dict[str, Any]] = {}
    with filelist.open("r", encoding="utf-8") as handle:
        for line in handle:
            relative = line.strip()
            if not relative:
                continue
            metadata_path = (latent_root / relative).with_suffix(".json")
            with metadata_path.open("r", encoding="utf-8") as metadata_handle:
                metadata = json.load(metadata_handle)
            audio_path = normalize_audio_path(metadata["path"])
            if audio_path in selection:
                raise RuntimeError(f"duplicate finalized audio path: {audio_path}")
            selection[audio_path] = {
                "latent_relpath": Path(relative).as_posix(),
                "caption": _caption(metadata),
                "duration_sec": float(metadata["seconds_total"]),
                "sample_rate": int(metadata.get("sample_rate", 44_100)),
                "timestamps": list(metadata.get("timestamps", [0.0, 1.0])),
            }
            if limit is not None and len(selection) >= limit:
                break
    return selection


def _row_audio_path(row: dict[str, Any]) -> str:
    for key in ("foa_path", "audio_path", "path"):
        value = row.get(key)
        if isinstance(value, str) and value:
            return normalize_audio_path(value)
    raise KeyError("source manifest row has no foa_path/audio_path/path")


def _position(raw: dict[str, Any], geometry_quality: str) -> dict[str, Any]:
    return {
        "azimuth_deg": raw.get("az_deg", raw.get("azimuth_deg")),
        "elevation_deg": raw.get("el_deg", raw.get("elevation_deg")),
        "distance_m": raw.get("dist_m", raw.get("distance_m")),
        "direction": raw.get("dir", raw.get("direction")),
        "elevation": raw.get("elev", raw.get("elevation_word")),
        "distance_label": raw.get("dist_word"),
        "geometry_quality": geometry_quality,
    }


def _room_standard(room: dict[str, Any], quality: str) -> dict[str, Any]:
    return {
        "type": room.get("type"),
        "description": room.get("desc", room.get("description")),
        "dimensions_m": room.get("dim", room.get("dimensions_m")),
        "rt60_s": room.get("rt60", room.get("rt60_s")),
        "reverb_label": room.get("reverb"),
        "free_field": room.get("free_field"),
        "quality": quality,
    }


def _source_standard(
    raw: dict[str, Any],
    source_index: int,
    *,
    geometry_quality: str,
    transcript: Optional[str] = None,
    event_label: Optional[str] = None,
) -> dict[str, Any]:
    raw_motion = raw.get("motion", "static")
    motion_config = raw_motion if isinstance(raw_motion, dict) else {}
    motion_name = str(motion_config.get("type", raw_motion))
    start = raw.get("start") or {}
    end = raw.get("end")
    raw_keyframes = raw.get("keyframes") or motion_config.get("keyframes") or []
    keyframes = []
    for keyframe in raw_keyframes:
        if not isinstance(keyframe, dict):
            continue
        position = keyframe.get("position") or keyframe
        t_norm = keyframe.get("t_norm", keyframe.get("time_norm"))
        if t_norm is None:
            continue
        keyframes.append(
            {
                "t_norm": min(1.0, max(0.0, float(t_norm))),
                "position": _position(position, geometry_quality),
            }
        )
    keyframes.sort(key=lambda item: item["t_norm"])
    if not keyframes:
        keyframes = [{"t_norm": 0.0, "position": _position(start, geometry_quality)}]
    motion_type = "static"
    if len(keyframes) > 1:
        motion_type = "linear" if len(keyframes) == 2 else "keyframed"
    elif motion_name == "dynamic" and isinstance(end, dict):
        motion_type = "linear"
        keyframes.append({"t_norm": 1.0, "position": _position(end, geometry_quality)})

    raw_activity = raw.get("activity") if isinstance(raw.get("activity"), dict) else {}
    onset = raw_activity.get("onset_sec", raw.get("onset_sec", raw.get("onset")))
    offset = raw_activity.get("offset_sec", raw.get("offset_sec", raw.get("offset")))
    has_activity = onset is not None and offset is not None and float(offset) > float(onset)
    return {
        "source_id": f"source_{source_index}",
        "event": {
            "label": event_label if event_label is not None else raw.get("label"),
            "category": raw.get("category"),
            "source_dataset": raw.get("dataset"),
        },
        "content": {
            "transcript": transcript,
            "speaker_id": raw.get("speaker"),
            "source_audio_path": raw.get("path"),
        },
        "activity": {
            "onset_sec": float(onset) if has_activity else None,
            "offset_sec": float(offset) if has_activity else None,
            "quality": "source_annotation" if has_activity else "not_annotated",
        },
        "motion": {
            "type": motion_type,
            "time_basis": "full_clip",
            "timing_quality": (
                "renderer_defined"
                if motion_type in {"linear", "keyframed"}
                else "static"
            ),
            "interpolation": "linear_shortest_azimuth_arc",
            "keyframes": keyframes,
        },
    }


def _normalize_synthetic(
    row: dict[str, Any], spec: dict[str, str], manifest_path: Path
) -> dict[str, Any]:
    if spec["kind"] == "synthetic_old":
        simulation = row.get("sim_params") or {}
        room = simulation.get("room") or {}
        sources = simulation.get("sources") or []
        sample_rate = simulation.get("sample_rate")
        channel_layout = simulation.get("channel_layout")
        duration = None
    else:
        room = row.get("room") or {}
        sources = row.get("sources") or []
        sample_rate = row.get("sample_rate")
        channel_layout = row.get("channel_layout")
        duration = row.get("duration_sec")
    normalized_sources = [
        _source_standard(source, index, geometry_quality="exact")
        for index, source in enumerate(sources)
    ]
    return {
        "dataset_id": spec["dataset_id"],
        "manifest_id": str(row["id"]),
        "audio": {
            "duration_sec": duration,
            "source_sample_rate": sample_rate,
            "spatial_format": "foa",
            "channel_layout": channel_layout or "WYZX_ACN_SN3D",
        },
        "mix": {
            "type": row.get("mix_type", "single"),
            "num_sources": int(row.get("n_sources", len(normalized_sources))),
        },
        "scene": {
            "room": _room_standard(room, "exact"),
            "sources": normalized_sources,
        },
        "supervision": {
            "event_semantic": True,
            "room_categorical": True,
            "room_metric": True,
            "position_categorical": True,
            "position_metric": True,
            "motion": any(s["motion"]["type"] == "linear" for s in normalized_sources),
            "source_activity_time": any(
                source["activity"]["onset_sec"] is not None
                for source in normalized_sources
            ),
        },
        "provenance": {"manifest_path": str(manifest_path)},
    }


def _normalize_sls(
    row: dict[str, Any], spec: dict[str, str], manifest_path: Path
) -> dict[str, Any]:
    transcript = row.get("transcription") or row.get("transcript")
    source = {
        "dataset": "spatial_librispeech",
        "category": "speech",
        "label": "speech",
        "speaker": row.get("reader_id"),
        "motion": "static",
        "start": {
            "azimuth_deg": row.get("azimuth_deg"),
            "elevation_deg": row.get("elevation_deg"),
            "distance_m": row.get("distance_m"),
            "direction": row.get("direction"),
            "elevation_word": row.get("elevation_word"),
        },
    }
    room = {
        "type": row.get("room_size"),
        "description": f"Spatial LibriSpeech room {row.get('room_id')}",
        "reverb": row.get("reverb"),
    }
    normalized_source = _source_standard(
        source, 0, geometry_quality="exact", transcript=transcript
    )
    normalized_room = _room_standard(room, "measured")
    normalized_room["metrics"] = {
        "room_id": row.get("room_id"),
        "t30_s": row.get("t30_s"),
        "c50_db": row.get("c50_db"),
        "drr_db": row.get("drr_db"),
        "noise_snr_db": row.get("noise_snr_db"),
        "volume_m3": row.get("room_volume_m3"),
        "floor_area_m2": row.get("room_floor_area_m2"),
        "surface_area_m2": row.get("room_surface_area_m2"),
    }
    return {
        "dataset_id": spec["dataset_id"],
        "manifest_id": str(row["id"]),
        "audio": {
            "duration_sec": None,
            "source_sample_rate": None,
            "spatial_format": "foa",
            "channel_layout": "WYZX_ACN_SN3D",
        },
        "mix": {"type": "single", "num_sources": 1},
        "scene": {"room": normalized_room, "sources": [normalized_source]},
        "supervision": {
            "event_semantic": True,
            "room_categorical": True,
            "room_metric": True,
            "position_categorical": True,
            "position_metric": True,
            "motion": False,
            "source_activity_time": normalized_source["activity"]["onset_sec"] is not None,
        },
        "provenance": {"manifest_path": str(manifest_path)},
    }


def _normalize_tts(
    row: dict[str, Any], spec: dict[str, str], manifest_path: Path
) -> dict[str, Any]:
    transcript = row.get("normalized_text") or row.get("text")
    sources = [
        _source_standard(
            source,
            index,
            geometry_quality="categorical_angles_exact_distance",
            transcript=transcript,
            # v1 stored the transcript both here and under content.transcript.
            # The event is semantic (speech); the words have one canonical home.
            event_label="speech",
        )
        for index, source in enumerate(row.get("sources") or [])
    ]
    if sources:
        # The render manifest contains the waveform and duration facts, while
        # source_plan_*.jsonl contains the stable dry-source locator.  Preserve
        # both: the old absolute cache path may be stale after a mount/cache
        # move, but (source_dataset, source_audio_id) remains resolvable from
        # the parquet source index used by spatial_speech_foa_tts_v1_render.py.
        content = sources[0].setdefault("content", {})
        legacy_path = row.get("source_audio_path")
        resolved_path = None
        if isinstance(legacy_path, str):
            candidate = Path(legacy_path).expanduser()
            try:
                if candidate.is_file():
                    resolved_path = str(candidate.resolve())
            except OSError:
                # A stale path below another user's private cache can raise
                # PermissionError rather than simply returning False.
                resolved_path = None
        content["source_audio_id"] = row.get("source_id")
        content["source_audio_path"] = resolved_path
        content["source_audio_legacy_path"] = legacy_path
        content["source_locator"] = {
            "type": "parquet_source_id",
            "source_dataset": row.get("source_dataset"),
            "source_id": row.get("source_id"),
        }
    return {
        "dataset_id": spec["dataset_id"],
        "manifest_id": str(row["id"]),
        "audio": {
            "duration_sec": row.get("duration_sec"),
            "source_sample_rate": row.get("sample_rate"),
            "spatial_format": "foa",
            "channel_layout": row.get("channel_layout", "WYZX_ACN_SN3D"),
        },
        "mix": {
            "type": row.get("mix_type", "single"),
            "num_sources": int(row.get("n_sources", len(sources))),
        },
        "scene": {
            "room": _room_standard(row.get("room") or {}, "categorical"),
            "sources": sources,
        },
        "supervision": {
            "event_semantic": True,
            "room_categorical": True,
            "room_metric": False,
            "position_categorical": True,
            "position_metric": False,
            "motion": any(s["motion"]["type"] == "linear" for s in sources),
            "source_activity_time": any(
                source["activity"]["onset_sec"] is not None for source in sources
            ),
        },
        "provenance": {"manifest_path": str(manifest_path)},
    }


def _normalize_source_row(
    row: dict[str, Any], spec: dict[str, str], manifest_path: Path
) -> dict[str, Any]:
    if spec["kind"].startswith("synthetic"):
        plan = _normalize_synthetic(row, spec, manifest_path)
    elif spec["kind"] == "sls":
        plan = _normalize_sls(row, spec, manifest_path)
    elif spec["kind"] == "tts":
        plan = _normalize_tts(row, spec, manifest_path)
    else:
        raise ValueError(f"unsupported source kind: {spec['kind']}")
    if plan["mix"]["num_sources"] != len(plan["scene"]["sources"]):
        raise RuntimeError(
            f"source-count mismatch for {spec['dataset_id']}:{plan['manifest_id']}"
        )
    return plan


def _join_source_plans(
    wanted: set[str], *, smoke: bool
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    remaining = set(wanted)
    plans: dict[str, dict[str, Any]] = {}
    used_manifests: list[str] = []
    for spec in SOURCE_SPECS:
        if not remaining:
            break
        manifest_path = Path(spec["manifest"])
        if not manifest_path.is_file():
            if smoke:
                continue
            raise FileNotFoundError(f"required source manifest is unavailable: {manifest_path}")
        used_manifests.append(str(manifest_path))
        source_plan_rows: dict[str, dict[str, Any]] = {}
        source_plan_value = spec.get("source_plan_manifest")
        if source_plan_value:
            source_plan_path = Path(source_plan_value)
            if not source_plan_path.is_file():
                if not smoke:
                    raise FileNotFoundError(
                        f"required TTS source-plan manifest is unavailable: {source_plan_path}"
                    )
            else:
                used_manifests.append(str(source_plan_path))
                with source_plan_path.open("r", encoding="utf-8") as source_handle:
                    for source_line in source_handle:
                        if not source_line.strip():
                            continue
                        source_row = json.loads(source_line)
                        try:
                            source_audio_path = _row_audio_path(source_row)
                        except KeyError:
                            continue
                        if source_audio_path in remaining:
                            source_plan_rows[source_audio_path] = source_row
        with manifest_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                try:
                    audio_path = _row_audio_path(row)
                except KeyError:
                    continue
                if audio_path not in remaining:
                    continue
                if source_plan_rows:
                    # The executed render row wins for duration/status/output;
                    # the source-plan row contributes stable dry-source fields.
                    merged = dict(source_plan_rows.get(audio_path) or {})
                    merged.update(row)
                    row = merged
                plan = _normalize_source_row(row, spec, manifest_path)
                plans[audio_path] = plan
                remaining.remove(audio_path)
                if not remaining:
                    break
    if remaining:
        examples = "\n".join(sorted(remaining)[:20])
        raise RuntimeError(
            f"ScenePlan source join missed {len(remaining)} finalized audio paths; examples:\n{examples}"
        )
    return plans, used_manifests


def _final_record(
    audio_path: str,
    selection: dict[str, Any],
    source_plan: dict[str, Any],
) -> dict[str, Any]:
    record = dict(source_plan)
    record["schema"] = SCHEMA
    record["schema_version"] = SCHEMA_VERSION
    record["sample_id"] = stable_sample_id(audio_path)
    record["audio"] = dict(record["audio"])
    record["audio"]["path"] = audio_path
    record["audio"]["latent_sample_rate"] = selection["sample_rate"]
    if record["audio"].get("duration_sec") is None:
        record["audio"]["duration_sec"] = selection["duration_sec"]
    duration = record["audio"].get("duration_sec")
    if duration is not None:
        # All source tracks were rendered over this clip extent, but that does
        # not prove acoustic event activity at every instant (speech may contain
        # leading/trailing/internal silence). Keep onset/offset unknown unless an
        # annotation exists, and expose the weaker renderer fact separately.
        for source in record.get("scene", {}).get("sources", []):
            activity = source.setdefault("activity", {})
            activity["render_window_sec"] = [0.0, float(duration)]
            activity["render_window_quality"] = "renderer_defined_extent"
    record["caption"] = selection["caption"]
    record["provenance"] = dict(record["provenance"])
    record["provenance"]["latent_relpath"] = selection["latent_relpath"]
    record["provenance"]["latent_source_timestamps"] = selection["timestamps"]
    return record


def _write_indexed_shard(
    root: Path,
    shard_relative: Path,
    records: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    output = root / shard_relative
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    temporary = Path(temporary_name)
    index_rows: list[dict[str, Any]] = []
    try:
        with os.fdopen(fd, "wb") as handle:
            for record in records:
                payload = json.dumps(
                    record, ensure_ascii=False, separators=(",", ":"), sort_keys=True
                ).encode("utf-8")
                offset = handle.tell()
                handle.write(payload)
                handle.write(b"\n")
                index_rows.append(
                    {
                        "sample_id": record["sample_id"],
                        "audio_path": record["audio_path"]
                        if "audio_path" in record
                        else record["audio"]["path"],
                        "shard": shard_relative.as_posix(),
                        "byte_offset": offset,
                        "byte_length": len(payload),
                    }
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return index_rows


def _indexed_shard_is_complete(
    root: Path,
    shard_relative: Path,
    index_path: Path,
    expected_audio_paths: list[str],
) -> bool:
    """Return whether a prior deterministic shard can be reused safely."""

    shard_path = root / shard_relative
    if not shard_path.is_file() or not index_path.is_file() or shard_path.stat().st_size == 0:
        return False
    try:
        rows = [
            json.loads(line)
            for line in index_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    if len(rows) != len(expected_audio_paths):
        return False
    expected_shard = shard_relative.as_posix()
    return all(
        row.get("audio_path") == audio_path
        and row.get("sample_id") == stable_sample_id(audio_path)
        and row.get("shard") == expected_shard
        and isinstance(row.get("byte_offset"), int)
        and isinstance(row.get("byte_length"), int)
        and row["byte_offset"] >= 0
        and row["byte_length"] > 0
        for row, audio_path in zip(rows, expected_audio_paths)
    )


def _iter_index_parts(root: Path, count: int) -> Iterator[dict[str, Any]]:
    for shard_id in range(count):
        path = root / "shard_indexes" / f"index-{shard_id:05d}.jsonl"
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def _build_sqlite(root: Path, records: Iterable[dict[str, Any]]) -> int:
    output = root / "index.sqlite"
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    temporary.unlink(missing_ok=True)
    connection = sqlite3.connect(temporary)
    count = 0
    try:
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute(
            "CREATE TABLE samples ("
            "sample_id TEXT PRIMARY KEY, audio_path TEXT NOT NULL UNIQUE, "
            "shard TEXT NOT NULL, byte_offset INTEGER NOT NULL, "
            "byte_length INTEGER NOT NULL) WITHOUT ROWID"
        )
        batch = []
        for row in records:
            batch.append(
                (
                    row["sample_id"], row["audio_path"], row["shard"],
                    int(row["byte_offset"]), int(row["byte_length"]),
                )
            )
            if len(batch) >= 10_000:
                connection.executemany("INSERT INTO samples VALUES (?, ?, ?, ?, ?)", batch)
                count += len(batch)
                batch.clear()
        if batch:
            connection.executemany("INSERT INTO samples VALUES (?, ?, ?, ?, ?)", batch)
            count += len(batch)
        connection.commit()
        if connection.execute("SELECT COUNT(*) FROM samples").fetchone()[0] != count:
            raise RuntimeError("ScenePlan SQLite count mismatch")
    finally:
        connection.close()
    os.replace(temporary, output)
    return count


def _mount_preflight(paths: list[str]) -> None:
    missing = [path for path in paths if not os.path.ismount(path)]
    if missing:
        raise SystemExit(
            "full ScenePlan build requires all source volumes mounted; missing: "
            + ", ".join(missing)
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--filelist", type=Path, default=None)
    parser.add_argument("--expected-count", type=int, default=1_018_957)
    parser.add_argument("--shard-size", type=int, default=50_000)
    parser.add_argument("--limit", type=int, default=None, help="Smoke-test sample cap")
    parser.add_argument("--required-mount", action="append", default=None)
    args = parser.parse_args()

    latent_root = args.latent_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    filelist = (
        args.filelist.expanduser().resolve()
        if args.filelist
        else latent_root / "filelist.txt"
    )
    if not latent_root.is_dir() or not filelist.is_file():
        raise SystemExit(f"finalized latent root/filelist unavailable: {latent_root}, {filelist}")
    if args.shard_size <= 0:
        raise SystemExit("--shard-size must be positive")
    if args.limit is None:
        _mount_preflight(list(args.required_mount or DEFAULT_REQUIRED_MOUNTS))

    selection = _load_selection(latent_root, filelist, args.limit)
    expected = args.limit if args.limit is not None else args.expected_count
    if len(selection) != expected:
        raise SystemExit(f"selection count mismatch: found={len(selection)} expected={expected}")
    source_plans, used_manifests = _join_source_plans(
        set(selection), smoke=args.limit is not None
    )

    sorted_paths = sorted(selection)
    num_shards = math.ceil(len(sorted_paths) / args.shard_size)
    canonical_root = output_root
    dsl_root = output_root / "views" / DSL_VERSION
    details = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "dsl_version": DSL_VERSION,
        "latent_root": str(latent_root),
        "filelist": str(filelist),
        "filelist_sha256": sha256_file(filelist),
        "target_count": len(sorted_paths),
        "shard_size": args.shard_size,
        "source_manifests": used_manifests,
    }
    canonical_root.mkdir(parents=True, exist_ok=True)
    if (canonical_root / "details.json").is_file():
        existing = json.loads((canonical_root / "details.json").read_text())
        if existing != details:
            raise SystemExit("existing ScenePlan store has a different build configuration")
    else:
        atomic_write_json(canonical_root / "details.json", details)
    atomic_write_json(
        canonical_root / "schema.json",
        {
            "schema": SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "identity": "sample_id = blake2b-96(normalized absolute audio_path)",
            "activity_timing": (
                "onset/offset are supervised only when explicitly present; "
                "render_window_sec records the weaker renderer-defined clip extent"
            ),
            "tts_event_normalization": (
                "event.label=speech; transcript appears only in content.transcript"
            ),
            "motion": (
                "arbitrary normalized keyframes are supported; current generated "
                "linear motion remains represented by its two renderer-defined endpoints"
            ),
            "geometry_quality_values": [
                "exact", "categorical_angles_exact_distance"
            ],
        },
    )
    atomic_write_json(
        dsl_root / "details.json",
        {
            "schema": "stable_audio_tools.spatial_cot_target",
            "schema_version": 1,
            "dsl_version": DSL_VERSION,
            "canonical_store": str(canonical_root),
            "tokenization": "crop-aware tokenization is performed by ScenePlanMetadata",
        },
    )

    for shard_id in range(num_shards):
        paths = sorted_paths[
            shard_id * args.shard_size : min((shard_id + 1) * args.shard_size, len(sorted_paths))
        ]
        canonical_relative = Path("shards") / f"scene-plan-{shard_id:05d}.jsonl"
        canonical_index_path = (
            canonical_root / "shard_indexes" / f"index-{shard_id:05d}.jsonl"
        )
        dsl_relative = Path("shards") / f"targets-{shard_id:05d}.jsonl"
        dsl_index_path = dsl_root / "shard_indexes" / f"index-{shard_id:05d}.jsonl"
        if _indexed_shard_is_complete(
            canonical_root, canonical_relative, canonical_index_path, paths
        ) and _indexed_shard_is_complete(dsl_root, dsl_relative, dsl_index_path, paths):
            print(
                f"[scene-plan] shard={shard_id + 1}/{num_shards} "
                f"samples={len(paths)} cached",
                flush=True,
            )
            continue

        canonical_records = [
            _final_record(path, selection[path], source_plans[path]) for path in paths
        ]
        canonical_index = _write_indexed_shard(
            canonical_root, canonical_relative, canonical_records
        )
        atomic_write_jsonl(
            canonical_index_path,
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
            for record in canonical_records
        ]
        dsl_index = _write_indexed_shard(dsl_root, dsl_relative, dsl_records)
        atomic_write_jsonl(
            dsl_index_path,
            dsl_index,
        )
        print(
            f"[scene-plan] shard={shard_id + 1}/{num_shards} samples={len(paths)}",
            flush=True,
        )

    canonical_count, canonical_sha = atomic_write_jsonl(
        canonical_root / "index.jsonl", _iter_index_parts(canonical_root, num_shards)
    )
    dsl_count, dsl_sha = atomic_write_jsonl(
        dsl_root / "index.jsonl", _iter_index_parts(dsl_root, num_shards)
    )
    if canonical_count != len(sorted_paths) or dsl_count != len(sorted_paths):
        raise RuntimeError("ScenePlan merged index count mismatch")
    if _build_sqlite(canonical_root, _iter_index_parts(canonical_root, num_shards)) != len(sorted_paths):
        raise RuntimeError("ScenePlan SQLite count mismatch")
    if _build_sqlite(dsl_root, _iter_index_parts(dsl_root, num_shards)) != len(sorted_paths):
        raise RuntimeError("DSL SQLite count mismatch")

    stats = {
        "samples": len(sorted_paths),
        "shards": num_shards,
        "dataset_counts": {},
        "metric_position_samples": 0,
        "categorical_only_position_samples": 0,
        "dynamic_samples": 0,
        "activity_annotated_samples": 0,
    }
    for path in sorted_paths:
        plan = source_plans[path]
        dataset_id = plan["dataset_id"]
        stats["dataset_counts"][dataset_id] = stats["dataset_counts"].get(dataset_id, 0) + 1
        if plan["supervision"]["position_metric"]:
            stats["metric_position_samples"] += 1
        else:
            stats["categorical_only_position_samples"] += 1
        if plan["supervision"]["motion"]:
            stats["dynamic_samples"] += 1
        if plan["supervision"]["source_activity_time"]:
            stats["activity_annotated_samples"] += 1
    atomic_write_json(canonical_root / "stats.json", stats)
    atomic_write_json(
        dsl_root / "READY",
        {
            "schema": "stable_audio_tools.spatial_cot_target",
            "schema_version": 1,
            "samples": len(sorted_paths),
            "index_sha256": dsl_sha,
            "canonical_store": str(canonical_root),
        },
    )
    atomic_write_json(
        canonical_root / "READY",
        {
            "schema": SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "samples": len(sorted_paths),
            "shards": num_shards,
            "index_sha256": canonical_sha,
            "dsl_view": str(dsl_root),
        },
    )
    print(
        f"[scene-plan] READY samples={len(sorted_paths)} root={canonical_root}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
