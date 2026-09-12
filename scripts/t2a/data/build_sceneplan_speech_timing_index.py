#!/usr/bin/env python3
"""Compile audited Qwen-token speech timing into an immutable SQLite sidecar."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sqlite3
import zlib
from pathlib import Path
from typing import Any


SCHEMA = "stable_audio_tools.sceneplan_speech_timing_index"
SCHEMA_VERSION = 1


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _token_weight(item: dict[str, Any]) -> int:
    text = str(item.get("text") or "")
    alnum = sum(character.isalnum() for character in text)
    if alnum:
        return alnum
    return max(
        int(item["transcript_char_end"]) - int(item["transcript_char_start"]),
        1,
    )


def _refined_centers(
    items: list[dict[str, Any]], *, activity_onset: float, activity_offset: float
) -> list[float]:
    """Split one forced-aligned word across any Qwen subword tokens."""

    centers = [0.0] * len(items)
    groups: list[tuple[int, int, float, float]] = []
    cursor = 0
    while cursor < len(items):
        first = items[cursor]
        key = (
            float(first["absolute_start_sec"]),
            float(first["absolute_end_sec"]),
        )
        stop = cursor + 1
        while stop < len(items):
            candidate = items[stop]
            candidate_key = (
                float(candidate["absolute_start_sec"]),
                float(candidate["absolute_end_sec"]),
            )
            if candidate_key != key:
                break
            stop += 1
        group = items[cursor:stop]
        groups.append((cursor, stop, key[0], key[1]))
        weights = [_token_weight(item) for item in group]
        total = float(sum(weights))
        start_sec, end_sec = key
        width = end_sec - start_sec
        if width > 0.0:
            cumulative = 0.0
            for offset, weight in enumerate(weights):
                left = cumulative / total
                cumulative += float(weight)
                right = cumulative / total
                centers[cursor + offset] = (
                    start_sec + 0.5 * (left + right) * width
                )
        cursor = stop

    # Qwen's 40-ms alignment grid occasionally emits a zero-duration short
    # word.  Preserve that raw fact in the immutable source registry, but give
    # every affected training token positive support between the adjacent
    # *token* centers.  Using the next word's midpoint is not sufficient: when
    # that word splits into several Qwen subtokens, its first token center can
    # precede the midpoint and invert the order.  Allocate a whole consecutive
    # zero-width run at once, bounded by the previous and next already-refined
    # non-zero token centers.  Character weights retain deterministic subtoken
    # proportions while guaranteeing strict monotonicity.
    group_index = 0
    while group_index < len(groups):
        start, stop, left_sec, right_sec = groups[group_index]
        if right_sec > left_sec:
            group_index += 1
            continue
        run_stop_group = group_index + 1
        while (
            run_stop_group < len(groups)
            and groups[run_stop_group][3] <= groups[run_stop_group][2]
        ):
            run_stop_group += 1
        run_start = start
        run_stop = groups[run_stop_group - 1][1]
        previous_center = (
            centers[run_start - 1]
            if run_start > 0
            else float(activity_onset)
        )
        next_center = (
            centers[run_stop]
            if run_stop < len(items)
            else float(activity_offset)
        )
        if not next_center > previous_center:
            raise ValueError(
                "a zero-duration lexical run has no positive neighbour interval"
            )
        weights = [_token_weight(item) for item in items[run_start:run_stop]]
        total = float(sum(weights))
        width = next_center - previous_center
        cumulative = 0.0
        for offset, weight in enumerate(weights):
            left = cumulative / total
            cumulative += float(weight)
            right = cumulative / total
            centers[run_start + offset] = previous_center + (
                0.5 * (left + right) * width
            )
        group_index = run_stop_group
    if any(right <= left for left, right in zip(centers, centers[1:])):
        raise ValueError("refined Qwen speech-token centers are not strictly monotonic")
    return centers


def _compile_row(row: dict[str, Any]) -> dict[str, Any]:
    all_timing = list(row.get("full_caption_token_timing") or ())
    lexical = [item for item in all_timing if bool(item.get("semantic"))]
    if not lexical:
        raise ValueError(f"{row.get('sample_id')}: no lexical speech tokens")
    if any(
        item.get("absolute_start_sec") is None
        or item.get("absolute_end_sec") is None
        for item in lexical
    ):
        raise ValueError(f"{row.get('sample_id')}: lexical timing is incomplete")
    token_indices = [int(item["token_index"]) for item in lexical]
    if len(set(token_indices)) != len(token_indices) or token_indices != sorted(token_indices):
        raise ValueError(f"{row.get('sample_id')}: token indices are invalid")

    onset = float(row["activity_onset_sec"])
    offset = float(row["activity_offset_sec"])
    if not math.isfinite(onset) or not math.isfinite(offset) or offset <= onset:
        raise ValueError(f"{row.get('sample_id')}: speech activity is invalid")
    centers = _refined_centers(
        lexical, activity_onset=onset, activity_offset=offset
    )
    if centers[0] < onset - 1.0e-6 or centers[-1] > offset + 1.0e-6:
        raise ValueError(f"{row.get('sample_id')}: token center is outside activity")
    boundaries = [onset]
    boundaries.extend(
        0.5 * (left + right) for left, right in zip(centers, centers[1:])
    )
    boundaries.append(offset)
    durations = [right - left for left, right in zip(boundaries, boundaries[1:])]
    if any(not math.isfinite(value) or value <= 0.0 for value in durations):
        raise ValueError(f"{row.get('sample_id')}: duration partition is invalid")
    total = sum(durations)
    fractions = [value / total for value in durations]
    # Correct the final floating-point residue so the serialized teacher sums
    # exactly to one within the loader's strict tolerance.
    fractions[-1] += 1.0 - sum(fractions)
    targets = [
        {
            "token_index": int(item["token_index"]),
            "token_id": int(item["token_id"]),
            "fraction": float(fraction),
            "center_sec": float(center),
        }
        for item, fraction, center in zip(lexical, fractions, centers)
    ]
    return {
        "sample_id": str(row["sample_id"]),
        "source_audio_sha256": str(row["source_audio_sha256"]),
        "source_label": int(row["source_label"]),
        "caption_text_sha256": _sha256_text(str(row["caption_text"])),
        "caption_valid_tokens": int(row["caption_valid_tokens"]),
        "speech_role_tokens": int(row["speech_role_tokens"]),
        "activity_onset_sec": onset,
        "activity_offset_sec": offset,
        "duration_targets": targets,
    }


def _build_database(
    source: Path,
    output: Path,
    *,
    expected_rows: int,
) -> dict[str, Any]:
    rows = [_compile_row(row) for row in _read_jsonl(source)]
    rows.sort(key=lambda row: row["sample_id"])
    if len(rows) != int(expected_rows):
        raise RuntimeError(f"timing input has {len(rows)} rows, expected {expected_rows}")
    if len({row["sample_id"] for row in rows}) != len(rows):
        raise RuntimeError("speech timing input contains duplicate sample IDs")
    if len({row["source_audio_sha256"] for row in rows}) != len(rows):
        raise RuntimeError("speech timing input contains duplicate source audio")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    if temporary.exists():
        temporary.unlink()
    connection = sqlite3.connect(temporary)
    try:
        connection.execute("PRAGMA page_size=4096")
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        connection.execute(
            """
            CREATE TABLE speech_timing (
                sample_id TEXT PRIMARY KEY,
                source_audio_sha256 TEXT NOT NULL UNIQUE,
                payload_zlib BLOB NOT NULL
            ) WITHOUT ROWID
            """
        )
        source_sha = _sha256_file(source)
        metadata = {
            "schema": SCHEMA,
            "schema_version": str(SCHEMA_VERSION),
            "rows": str(len(rows)),
            "source_jsonl": str(source),
            "source_jsonl_sha256": source_sha,
            "caption_compiler": "sceneplan_semantic_caption_v1",
            "caption_max_tokens": "512",
            "token_roles": "event_and_speech_-1_to_4",
            "teacher": "relative_lexical_duration_partition_v1",
        }
        connection.executemany(
            "INSERT INTO metadata(key,value) VALUES (?,?)", sorted(metadata.items())
        )
        connection.executemany(
            "INSERT INTO speech_timing(sample_id,source_audio_sha256,payload_zlib) VALUES (?,?,?)",
            [
                (
                    row["sample_id"],
                    row["source_audio_sha256"],
                    zlib.compress(
                        json.dumps(
                            row, ensure_ascii=False, separators=(",", ":")
                        ).encode("utf-8"),
                        level=9,
                    ),
                )
                for row in rows
            ],
        )
        connection.commit()
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            raise RuntimeError(f"SQLite integrity check failed: {integrity}")
        count = int(connection.execute("SELECT COUNT(*) FROM speech_timing").fetchone()[0])
        if count != len(rows):
            raise RuntimeError("speech timing SQLite row count changed")
    finally:
        connection.close()
    temporary.replace(output)
    return {
        "schema": "stable_audio_tools.sceneplan_speech_timing_index_receipt",
        "schema_version": 1,
        "status": "PASS",
        "rows": len(rows),
        "unique_sample_ids": len({row["sample_id"] for row in rows}),
        "unique_source_audio_sha256": len(
            {row["source_audio_sha256"] for row in rows}
        ),
        "source_jsonl": str(source),
        "source_jsonl_sha256": _sha256_file(source),
        "index": str(output),
        "index_sha256": _sha256_file(output),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--caption-timing-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, required=True)
    args = parser.parse_args()
    source = args.caption_timing_jsonl.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    if args.expected_rows <= 0:
        raise ValueError("expected-rows must be positive")
    receipt = _build_database(
        source, output, expected_rows=int(args.expected_rows)
    )
    receipt_path = output.with_suffix(output.suffix + ".receipt.json")
    _atomic_text(
        receipt_path,
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
