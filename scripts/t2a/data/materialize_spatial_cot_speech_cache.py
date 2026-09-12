#!/usr/bin/env python3
"""Sequential-row-group, resumable materialization of the TTS source cache.

The lazy resolver reads a complete Parquet row group for one requested row.
That is correct for sparse access but wasteful when the 1M build will consume
most of the 400K indexed speech assets. This command groups all locators by
Parquet file and row group, reads each group once, and atomically publishes the
same verified cache files/sidecars used by the lazy resolver.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stable_audio_tools.data.source_assets import (  # noqa: E402
    ParquetAudioSourceIndex,
    SourceAssetError,
)
from stable_audio_tools.data.t2a_artifacts import atomic_write_json  # noqa: E402


SCHEMA = "stable_audio_tools.spatial_cot_speech_cache"
VERSION = 1
DEFAULT_INDEX = Path(
    "/mnt/sdb/audio_dataset/spatial_cot_v1/source_index/tts/index.sqlite"
)
DEFAULT_CACHE = Path(
    "/mnt/sdb/audio_dataset/spatial_cot_v1/source_cache/speech"
)
SOURCE_AUDIO_SUFFIXES = (".flac", ".wav", ".ogg")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _cache_inventory(cache_root: Path) -> dict[str, int]:
    """Prove a one-to-one sidecar/audio inventory and reject old orphans."""

    sidecars = 0
    missing_audio: list[str] = []
    for metadata_path in cache_root.rglob("*.source.json"):
        sidecars += 1
        audio_path = Path(str(metadata_path)[: -len(".source.json")])
        if audio_path.suffix.lower() not in SOURCE_AUDIO_SUFFIXES or not audio_path.is_file():
            missing_audio.append(str(audio_path))
            if len(missing_audio) >= 10:
                break
    if missing_audio:
        raise RuntimeError(
            "speech cache has sidecars without audio: " + ", ".join(missing_audio)
        )
    audio_files = sum(
        1
        for suffix in SOURCE_AUDIO_SUFFIXES
        for _ in cache_root.rglob(f"*{suffix}")
    )
    if audio_files != sidecars:
        raise RuntimeError(
            f"speech cache has orphan audio/sidecars: "
            f"audio_files={audio_files} sidecars={sidecars}"
        )
    return {"asset_files": audio_files, "asset_sidecars": sidecars}


def _locators(index_path: Path) -> list[dict[str, Any]]:
    connection = sqlite3.connect(
        f"file:{index_path.as_posix()}?mode=ro&immutable=1",
        uri=True,
    )
    try:
        rows = connection.execute(
            "SELECT locator_json FROM sources ORDER BY source_dataset, source_id"
        ).fetchall()
    finally:
        connection.close()
    return [json.loads(row[0]) for row in rows]


def _process_file(
    payload: tuple[str, list[dict[str, Any]], str, str],
) -> dict[str, Any]:
    parquet_path, locators, index_path, cache_root = payload
    import pyarrow.parquet as pq

    resolver = ParquetAudioSourceIndex(index_path)
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for locator in locators:
        grouped[int(locator["row_group"])].append(locator)
    cached = 0
    materialized = 0
    bytes_written = 0
    groups_read = 0
    parquet = pq.ParquetFile(parquet_path)
    try:
        for row_group in sorted(grouped):
            missing: list[dict[str, Any]] = []
            for locator in grouped[row_group]:
                hit = resolver.cached_asset(
                    source_dataset=str(locator["source_dataset"]),
                    source_id=str(locator["source_id"]),
                    cache_root=cache_root,
                    locator=locator,
                )
                if hit is None:
                    missing.append(locator)
                else:
                    cached += 1
            if not missing:
                continue
            table = parquet.read_row_group(
                row_group,
                columns=["audio"],
                use_threads=False,
            )
            groups_read += 1
            column = table.column("audio")
            for locator in missing:
                row = int(locator["row_in_group"])
                if not 0 <= row < table.num_rows:
                    raise SourceAssetError(
                        f"row {row} outside {parquet_path} row-group {row_group}"
                    )
                audio = column[row].as_py()
                if not isinstance(audio, dict) or not isinstance(
                    audio.get("bytes"), bytes
                ):
                    raise SourceAssetError(
                        f"missing embedded audio: {parquet_path}:{row_group}:{row}"
                    )
                raw_path = str(
                    audio.get("path")
                    or locator.get("audio_path_in_parquet")
                    or "audio.flac"
                )
                suffix = Path(raw_path).suffix.lower()
                if suffix not in {".wav", ".flac", ".ogg"}:
                    suffix = ".flac"
                asset = resolver.materialize_payload(
                    source_dataset=str(locator["source_dataset"]),
                    source_id=str(locator["source_id"]),
                    cache_root=cache_root,
                    locator=locator,
                    payload=audio["bytes"],
                    payload_suffix=suffix,
                )
                materialized += 1
                bytes_written += int(asset["size_bytes"])
    finally:
        close = getattr(parquet, "close", None)
        if callable(close):
            close()
        resolver.close()
    return {
        "parquet_path": parquet_path,
        "sources": len(locators),
        "cached": cached,
        "materialized": materialized,
        "groups_total": len(grouped),
        "groups_read": groups_read,
        "bytes_written": bytes_written,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Debug/smoke boundary; a limited run never publishes READY.",
    )
    args = parser.parse_args()
    if args.workers <= 0 or (args.max_files is not None and args.max_files <= 0):
        raise SystemExit("workers and max-files must be positive")
    index_path = args.index.expanduser().resolve()
    cache_root = args.cache_root.expanduser().resolve()
    ready_path = cache_root / "READY"
    locators = _locators(index_path)
    expected = len(locators)
    if args.max_files is None and ready_path.is_file():
        ready = json.loads(ready_path.read_text(encoding="utf-8"))
        if (
            ready.get("schema") == SCHEMA
            and int(ready.get("sources", -1)) == expected
            and ready.get("index_sha256") == _sha256(index_path)
        ):
            inventory = _cache_inventory(cache_root)
            if inventory["asset_files"] != expected:
                raise SystemExit(
                    f"speech-cache READY inventory is incomplete: "
                    f"{inventory['asset_files']} != {expected}"
                )
            for key, value in inventory.items():
                recorded = ready.get(key)
                if recorded is not None and int(recorded) != value:
                    raise SystemExit(
                        f"speech-cache READY {key} mismatch: {recorded} != {value}"
                    )
            if any(ready.get(key) is None for key in inventory):
                ready.update(inventory)
                ready["inventory_verified"] = True
                atomic_write_json(ready_path, ready)
            print(
                json.dumps(
                    {
                        "status": "ALREADY_READY",
                        "sources": expected,
                        "cache_root": str(cache_root),
                        **inventory,
                    },
                    indent=2,
                )
            )
            return 0
        raise SystemExit(f"invalid speech-cache READY marker: {ready_path}")

    by_file: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for locator in locators:
        by_file[str(Path(locator["parquet_path"]).expanduser().resolve())].append(
            locator
        )
    selected = sorted(by_file)
    if args.max_files is not None:
        selected = selected[: args.max_files]
    tasks = [
        (path, by_file[path], str(index_path), str(cache_root)) for path in selected
    ]
    totals = {
        "sources": 0,
        "cached": 0,
        "materialized": 0,
        "groups_total": 0,
        "groups_read": 0,
        "bytes_written": 0,
    }
    completed = 0
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=min(args.workers, len(tasks))
    ) as executor:
        futures = [executor.submit(_process_file, task) for task in tasks]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            completed += 1
            for key in totals:
                totals[key] += int(result[key])
            if completed % 10 == 0 or completed == len(tasks):
                print(
                    f"[speech-cache] files={completed}/{len(tasks)} "
                    f"sources={totals['sources']:,} "
                    f"new={totals['materialized']:,}",
                    flush=True,
                )
    if totals["sources"] != sum(len(by_file[path]) for path in selected):
        raise RuntimeError("speech-cache source accounting mismatch")
    report = {
        "schema": SCHEMA,
        "schema_version": VERSION,
        "status": "READY" if args.max_files is None else "LIMITED_PASS",
        "index": str(index_path),
        "index_sha256": _sha256(index_path),
        "cache_root": str(cache_root),
        "parquet_files": len(selected),
        **totals,
        "workers": args.workers,
        "pid": os.getpid(),
    }
    if args.max_files is None:
        if totals["sources"] != expected:
            raise RuntimeError(
                f"materialized {totals['sources']} sources, expected {expected}"
            )
        inventory = _cache_inventory(cache_root)
        if inventory["asset_files"] != expected:
            raise RuntimeError(
                f"speech-cache inventory has {inventory['asset_files']} assets, "
                f"expected {expected}"
            )
        report.update(inventory)
        report["inventory_verified"] = True
        atomic_write_json(ready_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
