#!/usr/bin/env python3
"""Audit the validated original dry sound/music pools into an SDB catalog.

No waveform is copied and no derived FOA is accepted.  Every eligible row is
an on-disk mono asset whose complete resampled waveform plus the fixed Pyroom
delay fits the 442368-sample model ceiling.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf


SCRIPT_DIR = Path(__file__).resolve().parent
import sys

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from sceneplan_v2_common import (  # noqa: E402
    DATASET_ROOT,
    MAX_MODEL_SAMPLES,
    MODEL_SAMPLE_RATE,
    atomic_write_json,
    deterministic_digest,
)


DEFAULT_SOURCE_ROOT = Path(
    "/mnt/sdd/audio_dataset/spatial_foa_v2/spatial_sources_v2_final"
)
DEFAULT_OUTPUT = DATASET_ROOT / "source_catalog/nonspeech"
MIN_MODEL_SAMPLES = round(0.20 * MODEL_SAMPLE_RATE)
REQUIRED_TAIL_SAMPLES = 40

# The legacy pool already applied its strict speech/vocal filter.  This second
# fail-closed pass covers inflections and caption constructions that the former
# literal term list missed (for example "a man is narrating").
SPEECH_TEXT = re.compile(
    r"\b(?:speech|speak(?:s|ing|er)?|spoken|talk(?:s|ing|er)?|conversation|"
    r"dialogue|monologue|narrat(?:e|es|ed|ing|ion|or)|say(?:s|ing)?|said|"
    r"whisper(?:s|ing)?|shout(?:s|ing)?|scream(?:s|ing)?|yell(?:s|ing)?|"
    r"voice(?:s)?|vocal(?:s)?|sing(?:s|ing|er)?|choir|chant(?:s|ing)?|lyrics?|"
    r"exchange(?:s|d|ing)?\s+words?|reads?\s+aloud)\b",
    flags=re.IGNORECASE,
)


SCHEMA = pa.schema(
    [
        ("asset_id", pa.string()),
        ("source_id", pa.string()),
        ("source_dataset", pa.string()),
        ("kind", pa.string()),
        ("description", pa.string()),
        ("dry_audio_path", pa.string()),
        ("identity_hash", pa.string()),
        ("native_sample_rate_hz", pa.int32()),
        ("native_num_samples", pa.int64()),
        ("native_channels", pa.int16()),
        ("model_sample_rate_hz", pa.int32()),
        ("model_num_samples", pa.int64()),
        ("duration_sec", pa.float64()),
        ("file_num_bytes", pa.int64()),
        ("file_mtime_ns", pa.int64()),
        ("eligible", pa.bool_()),
        ("rejection_reason", pa.string()),
        ("selection_rank", pa.string()),
        ("lineage_policy", pa.string()),
    ]
)


def iter_jsonl(path: Path, kind: str) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            yield {
                "source_id": str(row["id"]),
                "source_dataset": str(row.get("dataset") or "unknown"),
                "kind": kind,
                "description": " ".join(str(row.get("label") or kind).split()),
                "dry_audio_path": str(row["path"]),
            }


def inspect(row: dict[str, Any]) -> dict[str, Any]:
    path = Path(row["dry_audio_path"])
    reason = None
    rate = frames = channels = model_frames = file_size = mtime_ns = 0
    try:
        stat = path.stat()
        file_size, mtime_ns = int(stat.st_size), int(stat.st_mtime_ns)
        info = sf.info(str(path))
        rate, frames, channels = int(info.samplerate), int(info.frames), int(info.channels)
        model_frames = math.ceil(frames * MODEL_SAMPLE_RATE / rate) if rate > 0 else 0
        if SPEECH_TEXT.search(row["description"]):
            reason = "speech_or_vocal_text"
        elif channels != 1:
            reason = "not_canonical_mono"
        elif rate < 8_000 or frames <= 0:
            reason = "invalid_audio_geometry"
        elif model_frames < MIN_MODEL_SAMPLES:
            reason = "shorter_than_200ms"
        elif model_frames + REQUIRED_TAIL_SAMPLES > MAX_MODEL_SAMPLES:
            reason = "complete_source_plus_pyroom_delay_over_limit"
        elif file_size <= 0:
            reason = "empty_file"
    except FileNotFoundError:
        reason = "missing_path"
    except Exception as exc:  # noqa: BLE001
        reason = f"audio_info_error:{type(exc).__name__}"
    identity = hashlib.sha256(
        "\0".join(
            [
                row["source_dataset"],
                row["source_id"],
                row["kind"],
                str(path),
                str(file_size),
                str(mtime_ns),
                str(rate),
                str(frames),
            ]
        ).encode("utf-8")
    ).hexdigest()
    asset_id = f"{row['kind']}:{row['source_dataset']}:{row['source_id']}"
    return {
        "asset_id": asset_id,
        **row,
        "identity_hash": identity,
        "native_sample_rate_hz": rate,
        "native_num_samples": frames,
        "native_channels": channels,
        "model_sample_rate_hz": MODEL_SAMPLE_RATE,
        "model_num_samples": model_frames,
        "duration_sec": model_frames / MODEL_SAMPLE_RATE if model_frames else 0.0,
        "file_num_bytes": file_size,
        "file_mtime_ns": mtime_ns,
        "eligible": reason is None,
        "rejection_reason": reason,
        "selection_rank": deterministic_digest(20260814, "nonspeech", asset_id),
        "lineage_policy": "validated_original_dry_mono_stat_fingerprint_v2",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--jobs", type=int, default=min(64, os.cpu_count() or 1))
    parser.add_argument("--batch-rows", type=int, default=20_000)
    args = parser.parse_args()
    source_root = args.source_root.expanduser().resolve(strict=True)
    output = args.output_root.expanduser().resolve(strict=False)
    try:
        output.relative_to("/mnt/sdb")
    except ValueError as error:
        raise ValueError(f"new catalog must be on SDB: {output}") from error
    output.mkdir(parents=True, exist_ok=True)
    catalog = output / "nonspeech_catalog.parquet"
    temporary = catalog.with_name(catalog.name + f".tmp.{os.getpid()}")
    rows = (
        row
        for kind, name in (
            ("sound", "sources_sound.jsonl"),
            ("music", "sources_music.jsonl"),
        )
        for row in iter_jsonl(source_root / name, kind)
    )
    counts: Counter[str] = Counter()
    by_kind: defaultdict[str, Counter[str]] = defaultdict(Counter)
    seen_assets: set[str] = set()
    seen_paths: dict[str, str] = {}
    duplicate_paths = []
    pending = []
    started = time.time()
    writer = pq.ParquetWriter(temporary, SCHEMA, compression="zstd")
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.jobs) as pool:
            for index, result in enumerate(pool.map(inspect, rows, chunksize=64), start=1):
                if result["asset_id"] in seen_assets:
                    raise RuntimeError(f"duplicate nonspeech asset_id: {result['asset_id']}")
                seen_assets.add(result["asset_id"])
                prior = seen_paths.setdefault(result["dry_audio_path"], result["asset_id"])
                if prior != result["asset_id"] and len(duplicate_paths) < 100:
                    duplicate_paths.append(
                        {
                            "dry_audio_path": result["dry_audio_path"],
                            "first_asset_id": prior,
                            "second_asset_id": result["asset_id"],
                        }
                    )
                status = "eligible" if result["eligible"] else str(result["rejection_reason"])
                counts[status] += 1
                by_kind[result["kind"]][status] += 1
                pending.append(result)
                if len(pending) >= args.batch_rows:
                    writer.write_table(pa.Table.from_pylist(pending, schema=SCHEMA))
                    pending.clear()
                if index % 50_000 == 0:
                    print(
                        json.dumps(
                            {
                                "inspected": index,
                                "eligible": counts["eligible"],
                                "elapsed_sec": round(time.time() - started, 1),
                            }
                        ),
                        flush=True,
                    )
            if pending:
                writer.write_table(pa.Table.from_pylist(pending, schema=SCHEMA))
                pending.clear()
    finally:
        writer.close()
    os.replace(temporary, catalog)
    eligible_table = pq.read_table(catalog, columns=["asset_id", "identity_hash", "eligible"])
    eligible_mask = eligible_table["eligible"].to_numpy(zero_copy_only=False)
    if len(set(eligible_table["asset_id"].filter(pa.array(eligible_mask)).to_pylist())) != int(
        eligible_mask.sum()
    ):
        raise RuntimeError("eligible asset IDs are not unique after Parquet reopen")
    summary = {
        "schema": "stable_audio_tools.sceneplan_nonspeech_catalog_summary",
        "schema_version": 2,
        "source_root": str(source_root),
        "catalog": str(catalog),
        "source_policy": "validated_original_complete_dry_mono_no_crop",
        "rows": len(seen_assets),
        "eligible": counts["eligible"],
        "rejected": len(seen_assets) - counts["eligible"],
        "status_counts": dict(counts),
        "by_kind": {kind: dict(values) for kind, values in sorted(by_kind.items())},
        "duplicate_path_count": len(seen_paths) - len(seen_assets),
        "duplicate_path_examples": duplicate_paths,
        "content_checksum_policy": "computed_and_verified_on_first_P6_or_P8_render_read",
        "elapsed_sec": round(time.time() - started, 3),
    }
    # Correct sign for the accounting expression above and fail if any path is
    # shared across logical assets: one physical source must have one identity.
    summary["duplicate_path_count"] = len(seen_assets) - len(seen_paths)
    atomic_write_json(output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["eligible"] > 0 and summary["duplicate_path_count"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
