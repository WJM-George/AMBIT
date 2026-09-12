#!/usr/bin/env python3
"""Atomically align completed A2T annotation split metadata to frozen input.

The audio description and all model-generation provenance remain unchanged.
This migration exists because split assignment is dataset metadata, while the
expensive Qwen3-Omni description is a source-audio annotation that must not be
regenerated merely to move an eligible source from internal test to train or
validation.
"""

from __future__ import annotations

import argparse
from collections import Counter
import glob
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def canonical_json(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, value: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--annotations-glob", required=True)
    parser.add_argument("--audit-json", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be positive")
    input_path = args.input_jsonl.expanduser().resolve(strict=True)
    annotation_paths = [Path(value).resolve(strict=True) for value in sorted(glob.glob(args.annotations_glob))]
    if len(annotation_paths) != args.num_shards:
        raise RuntimeError(
            f"expected {args.num_shards} annotation shards, found {len(annotation_paths)}"
        )

    ordinal_by_id: dict[str, int] = {}
    split_by_id: dict[str, str] = {}
    split_counts: Counter[str] = Counter()
    with input_path.open(encoding="utf-8") as handle:
        for ordinal, line in enumerate(line for line in handle if line.strip()):
            row = json.loads(line)
            annotation_id = str(row.get("id") or "")
            split = str(row.get("split") or "")
            if not annotation_id or annotation_id in ordinal_by_id:
                raise RuntimeError(f"invalid or duplicate frozen input id: {annotation_id}")
            if split not in {"train", "validation"}:
                raise RuntimeError(f"new Sound input contains forbidden split {split!r}")
            ordinal_by_id[annotation_id] = ordinal
            split_by_id[annotation_id] = split
            split_counts[split] += 1

    before_hashes = {str(path): sha256_file(path) for path in annotation_paths}
    seen: set[str] = set()
    transitions: Counter[str] = Counter()
    changed_rows = 0
    description_hashes_checked = 0
    for shard, path in enumerate(annotation_paths):
        raw_bytes = path.read_bytes()
        if raw_bytes and not raw_bytes.endswith(b"\n"):
            raise RuntimeError(f"annotation shard has an incomplete tail: {path}")
        output_lines: list[str] = []
        changed_file = False
        for line_number, line in enumerate(raw_bytes.decode("utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            annotation_id = str(row.get("id") or "")
            if annotation_id not in ordinal_by_id or annotation_id in seen:
                raise RuntimeError(f"unexpected/duplicate annotation id at {path}:{line_number}")
            expected_shard = ordinal_by_id[annotation_id] % args.num_shards
            if expected_shard != shard or int(row.get("shard", -1)) != shard:
                raise RuntimeError(f"annotation ordinal/shard mismatch for {annotation_id}")
            if int(row.get("num_shards", -1)) != args.num_shards:
                raise RuntimeError(f"annotation shard-count mismatch for {annotation_id}")
            description = str(row.get("source_description") or "")
            expected_description_hash = hashlib.sha256(description.encode("utf-8")).hexdigest()
            if expected_description_hash != str(row.get("source_description_sha256") or ""):
                raise RuntimeError(f"description hash mismatch for {annotation_id}")
            description_hashes_checked += 1
            old_split = str(row.get("split") or "")
            new_split = split_by_id[annotation_id]
            if old_split != new_split:
                transitions[f"{old_split}->{new_split}"] += 1
                row["split"] = new_split
                changed_rows += 1
                changed_file = True
            output_lines.append(canonical_json(row))
            seen.add(annotation_id)
        if changed_file:
            atomic_text(path, "".join(value + "\n" for value in output_lines))

    missing = set(ordinal_by_id) - seen
    if missing or len(seen) != len(ordinal_by_id):
        raise RuntimeError(f"annotation coverage changed; missing={len(missing)}")
    after_hashes = {str(path): sha256_file(path) for path in annotation_paths}
    audit = {
        "schema": "stable_audio_tools.sceneplan_annotation_split_repartition",
        "schema_version": 1,
        "status": "PASS",
        "policy": "base_test_unchanged_train_validation_only_v1",
        "input_jsonl": str(input_path),
        "input_jsonl_sha256": sha256_file(input_path),
        "rows": len(ordinal_by_id),
        "split_counts": {
            "train": split_counts["train"],
            "validation": split_counts["validation"],
            "test": 0,
        },
        "changed_rows": changed_rows,
        "transitions": dict(sorted(transitions.items())),
        "description_hashes_checked": description_hashes_checked,
        "description_rewrite": False,
        "annotation_ordinal_shard_contract_preserved": True,
        "annotation_files_before_sha256": before_hashes,
        "annotation_files_after_sha256": after_hashes,
    }
    atomic_json(args.audit_json.expanduser().resolve(strict=False), audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
