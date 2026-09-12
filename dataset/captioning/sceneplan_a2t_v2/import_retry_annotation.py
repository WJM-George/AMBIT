#!/usr/bin/env python3
"""Import one successful direct batch-1 retry into its production shard."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from description_contract import validate_source_description


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_single(path: Path) -> tuple[dict[str, Any], str]:
    lines = [line.rstrip("\n") for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) != 1:
        raise RuntimeError(f"retry artifact must contain exactly one row: {path}")
    row = json.loads(lines[0])
    if lines[0] != stable_json(row):
        raise RuntimeError(f"retry artifact is not canonical JSON: {path}")
    return row, lines[0]


def find_input(path: Path, annotation_id: str) -> tuple[int, dict[str, Any]]:
    ordinal = 0
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row.get("id")) == annotation_id:
                return ordinal, row
            ordinal += 1
    raise RuntimeError(f"retry id is absent from frozen input: {annotation_id}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retry-jsonl", type=Path, required=True)
    parser.add_argument("--target-jsonl", type=Path, required=True)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--audit-json", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument(
        "--replace-existing",
        action="store_true",
        help="Atomically replace the one existing row instead of appending a missing row.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    retry_path = args.retry_jsonl.expanduser().resolve(strict=True)
    target_path = args.target_jsonl.expanduser().resolve(strict=True)
    input_path = args.input_jsonl.expanduser().resolve(strict=True)
    audit_path = args.audit_json.expanduser().resolve(strict=False)
    retry, retry_text = load_single(retry_path)
    annotation_id = str(retry.get("id") or "")
    ordinal, source = find_input(input_path, annotation_id)
    if ordinal % args.num_shards != args.shard:
        raise RuntimeError("retry id does not belong to the requested production shard")
    if int(retry.get("shard", -1)) != args.shard or int(
        retry.get("num_shards", -1)
    ) != args.num_shards:
        raise RuntimeError("retry output does not carry production shard provenance")
    if retry.get("finish_reason") != "stop" or bool(retry.get("generation_capped")):
        raise RuntimeError("retry did not end with a complete EOS response")
    description = str(retry.get("source_description") or "")
    if (
        not description
        or description != str(retry.get("decoder_text") or "")
        or hashlib.sha256(description.encode("utf-8")).hexdigest()
        != str(retry.get("source_description_sha256") or "")
    ):
        raise RuntimeError("retry description/hash contract failed")
    qc = validate_source_description(description)
    if qc.hard_flags:
        raise RuntimeError(f"retry description hard QC failed: {qc.hard_flags}")
    for output_key, input_key in (
        ("audio_path", "audio_path"),
        ("asset_id", "asset_id"),
        ("kind", "kind"),
        ("split", "split"),
        ("source_audio_sha256", "source_audio_sha256"),
    ):
        if str(retry.get(output_key)) != str(source.get(input_key)):
            raise RuntimeError(f"retry lineage mismatch: {output_key}")

    before_size = target_path.stat().st_size
    before_rows = 0
    matching_rows = 0
    annotation_needle = f'"id":"{annotation_id}"'.encode("utf-8")
    with target_path.open("rb") as existing:
        last_byte = b"\n"
        for raw in existing:
            last_byte = raw[-1:]
            if raw.strip():
                before_rows += 1
            if annotation_needle in raw:
                candidate = json.loads(raw)
                if str(candidate.get("id")) == annotation_id:
                    matching_rows += 1
    if before_size and last_byte != b"\n":
        raise RuntimeError(f"production shard has an incomplete tail: {target_path}")
    if matching_rows > 1:
        raise RuntimeError(f"production shard contains duplicate retry id: {annotation_id}")
    already_present = matching_rows == 1
    replaced_existing = False
    if args.replace_existing:
        if not already_present:
            raise RuntimeError(f"replace target is absent: {annotation_id}")
        temporary = target_path.with_name(target_path.name + f".tmp.{os.getpid()}")
        with target_path.open("rb") as existing, temporary.open("wb") as sink:
            for raw in existing:
                if annotation_needle in raw:
                    candidate = json.loads(raw)
                    if str(candidate.get("id")) == annotation_id:
                        sink.write(retry_text.encode("utf-8") + b"\n")
                        replaced_existing = True
                        continue
                sink.write(raw)
            sink.flush()
            os.fsync(sink.fileno())
        if not replaced_existing:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(f"replace target disappeared while rewriting: {annotation_id}")
        os.replace(temporary, target_path)
    elif not already_present:
        with target_path.open("ab") as sink:
            sink.write(retry_text.encode("utf-8") + b"\n")
            sink.flush()
            os.fsync(sink.fileno())
    after_size = target_path.stat().st_size
    audit = {
        "schema": "stable_audio_tools.sceneplan_retry_annotation_import",
        "schema_version": 1,
        "annotation_id": annotation_id,
        "frozen_input_ordinal": ordinal,
        "production_shard": args.shard,
        "num_shards": args.num_shards,
        "direct_batch_1_retry": True,
        "description_rewrite": False,
        "finish_reason": retry["finish_reason"],
        "generation_capped": retry["generation_capped"],
        "generated_tokens": retry["generated_tokens"],
        "description_sha256": retry["source_description_sha256"],
        "retry_jsonl": str(retry_path),
        "retry_jsonl_sha256": hashlib.sha256(retry_path.read_bytes()).hexdigest(),
        "target_jsonl": str(target_path),
        "already_present": already_present,
        "replace_existing_requested": bool(args.replace_existing),
        "replaced_existing": replaced_existing,
        "rows_before": before_rows,
        "rows_after": before_rows + int(not already_present),
        "bytes_before": before_size,
        "bytes_after": after_size,
        "p8_started": False,
        "p9_started": False,
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(audit_path, audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
