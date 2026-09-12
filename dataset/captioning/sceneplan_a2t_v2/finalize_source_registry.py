#!/usr/bin/env python3
"""Audit resumable A2T shards and finalize the hash-keyed source registry.

Partial runs produce a progress/resume audit only.  The --finalize flag is a
hard gate: every frozen universe row and every spoken-language label must be
present exactly once before JSONL/Parquet registry artifacts and READY are
written.
"""

from __future__ import annotations

import argparse
from collections import Counter
import glob
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterator

import pyarrow as pa
import pyarrow.parquet as pq

from description_contract import compact_whitespace, validate_source_description


DATASET_ROOT = Path(os.environ.get("AMBIT_DATA_ROOT", "data") + "/sceneplan_v2_1p124m")
DEFAULT_SOURCE_ROOT = DATASET_ROOT / "source_annotations/nonspeech_instruct_v2"
DEFAULT_UNIVERSE = DEFAULT_SOURCE_ROOT / "source_universe.parquet"
DEFAULT_INPUT = DEFAULT_SOURCE_ROOT / "instruct_input.jsonl"
DEFAULT_ANNOTATIONS_GLOB = str(
    DEFAULT_SOURCE_ROOT
    / (
        "annotations/source_descriptions_instruct."
        "shard[0-9][0-9][0-9]-of-[0-9][0-9][0-9].jsonl"
    )
)
DEFAULT_OUTPUT = DEFAULT_SOURCE_ROOT / "registry"
MODEL_REVISION = "26291f793822fb6be9555850f06dfe95f2d7e695"
ANNOTATION_SCHEMA = "stable_audio_tools.sceneplan_source_description_annotation"
REGISTRY_SCHEMA_NAME = "stable_audio_tools.sceneplan_source_description_registry_entry"
REGISTRY_SCHEMA_VERSION = 1
SPOKEN_POLICY_VERSION = "spoken_language_background_v1"
REGISTRY_JSON_SCHEMA = (
    Path(__file__).with_name("schemas")
    / "source_description_registry_v1.schema.json"
)

PARQUET_SCHEMA = pa.schema(
    [
        ("schema", pa.string()),
        ("schema_version", pa.int16()),
        ("annotation_id", pa.string()),
        ("source_audio_sha256", pa.string()),
        ("primary_asset_id", pa.string()),
        ("alias_asset_ids", pa.list_(pa.string())),
        ("split", pa.string()),
        ("kind", pa.string()),
        ("audio_path", pa.string()),
        ("source_description", pa.string()),
        ("description_word_count", pa.int32()),
        ("spoken_language_background", pa.bool_()),
        (
            "source_description_provenance",
            pa.struct(
                [
                    ("model_revision", pa.string()),
                    ("engine", pa.string()),
                    ("engine_version", pa.string()),
                    ("prompt_sha256", pa.string()),
                    ("decoder_text_sha256", pa.string()),
                    ("finish_reason", pa.string()),
                ]
            ),
        ),
        (
            "spoken_language_provenance",
            pa.struct(
                [
                    ("policy_version", pa.string()),
                    ("method", pa.string()),
                    ("classifier_revision", pa.string()),
                    ("raw_response", pa.string()),
                ]
            ),
        ),
    ]
)


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ends_with_newline(path: Path) -> bool:
    if not path.stat().st_size:
        return True
    with path.open("rb") as source:
        source.seek(-1, os.SEEK_END)
        return source.read(1) == b"\n"


def iter_jsonl(path: Path, *, tolerate_incomplete_tail: bool) -> Iterator[dict[str, Any]]:
    size = path.stat().st_size
    offset = 0
    with path.open("rb") as source:
        for line_number, raw_line in enumerate(source, 1):
            offset += len(raw_line)
            if not raw_line.strip():
                continue
            try:
                line = raw_line.decode("utf-8")
                row = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                incomplete_tail = (
                    tolerate_incomplete_tail
                    and offset == size
                    and not raw_line.endswith(b"\n")
                )
                if incomplete_tail:
                    return
                raise ValueError(f"invalid JSON at {path}:{line_number}") from error
            if not isinstance(row, dict):
                raise ValueError(f"non-object JSON at {path}:{line_number}")
            yield row


def load_universe(path: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    columns = [
        "annotation_id",
        "source_audio_sha256",
        "primary_asset_id",
        "alias_asset_ids",
        "split",
        "kind",
        "dry_audio_path",
    ]
    rows = pq.read_table(path, columns=columns).to_pylist()
    ordinal_by_id: dict[str, int] = {}
    for ordinal, row in enumerate(rows):
        annotation_id = str(row["annotation_id"])
        source_hash = str(row["source_audio_sha256"])
        if annotation_id != f"sha256:{source_hash}":
            raise RuntimeError(f"universe annotation/hash mismatch: {annotation_id}")
        if annotation_id in ordinal_by_id:
            raise RuntimeError(f"duplicate universe annotation id: {annotation_id}")
        ordinal_by_id[annotation_id] = ordinal
    if not rows:
        raise RuntimeError(f"empty universe: {path}")
    return rows, ordinal_by_id


def audit_annotations(
    paths: list[Path],
    universe_rows: list[dict[str, Any]],
    ordinal_by_id: dict[str, int],
    *,
    num_shards: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    consistency: dict[str, str] = {}
    soft_flags: Counter[str] = Counter()
    hard_flags: Counter[str] = Counter()
    rows_by_shard: Counter[int] = Counter()
    incomplete_tail_files = 0
    for path in paths:
        raw_size = path.stat().st_size
        if raw_size and not ends_with_newline(path):
            incomplete_tail_files += 1
        for raw in iter_jsonl(path, tolerate_incomplete_tail=True):
            annotation_id = str(raw.get("id") or "")
            if annotation_id not in ordinal_by_id:
                raise RuntimeError(f"annotation id outside frozen universe: {annotation_id}")
            if annotation_id in by_id:
                raise RuntimeError(f"duplicate annotation output id: {annotation_id}")
            universe = universe_rows[ordinal_by_id[annotation_id]]
            expected_shard = ordinal_by_id[annotation_id] % num_shards
            actual_shard = int(raw.get("shard", -1))
            actual_num_shards = int(raw.get("num_shards", -1))
            if actual_shard != expected_shard or actual_num_shards != num_shards:
                raise RuntimeError(
                    f"annotation shard mismatch for {annotation_id}: "
                    f"{actual_shard}/{actual_num_shards} != {expected_shard}/{num_shards}"
                )
            for output_key, universe_key in (
                ("source_audio_sha256", "source_audio_sha256"),
                ("asset_id", "primary_asset_id"),
                ("kind", "kind"),
                ("split", "split"),
                ("audio_path", "dry_audio_path"),
            ):
                if str(raw.get(output_key)) != str(universe[universe_key]):
                    raise RuntimeError(
                        f"{output_key} lineage mismatch for {annotation_id}"
                    )
            if str(raw.get("schema")) != ANNOTATION_SCHEMA:
                raise RuntimeError(f"annotation schema mismatch for {annotation_id}")
            if str(raw.get("model_revision")) != MODEL_REVISION:
                raise RuntimeError(f"model revision mismatch for {annotation_id}")
            if str(raw.get("engine")) != "transformers":
                raise RuntimeError(f"unapproved engine for {annotation_id}")
            if str(raw.get("finish_reason")) != "stop" or bool(
                raw.get("generation_capped")
            ):
                raise RuntimeError(f"truncated generation for {annotation_id}")
            description = compact_whitespace(str(raw.get("source_description") or ""))
            decoder_text = compact_whitespace(str(raw.get("decoder_text") or ""))
            if description != decoder_text:
                raise RuntimeError(f"description was rewritten for {annotation_id}")
            description_hash = hashlib.sha256(description.encode("utf-8")).hexdigest()
            if description_hash != str(raw.get("source_description_sha256")):
                raise RuntimeError(f"description hash mismatch for {annotation_id}")
            if description_hash != str(raw.get("decoder_text_sha256")):
                raise RuntimeError(f"decoder text hash mismatch for {annotation_id}")
            qc = validate_source_description(description)
            if qc.hard_flags:
                raise RuntimeError(
                    f"hard description QC failure for {annotation_id}: {qc.hard_flags}"
                )
            hard_flags.update(str(value) for value in raw.get("description_hard_qc_flags", []))
            soft_flags.update(qc.soft_flags)
            for key in (
                "model_config_sha256",
                "prompt_file_sha256",
                "prompt_sha256",
                "prompt_template_version",
                "description_cleanup",
            ):
                value = str(raw.get(key) or "")
                previous = consistency.setdefault(key, value)
                if not value or value != previous:
                    raise RuntimeError(
                        f"inconsistent annotation provenance {key} for {annotation_id}"
                    )
            by_id[annotation_id] = raw
            rows_by_shard[actual_shard] += 1
    if hard_flags:
        raise RuntimeError(f"runner emitted hard QC flags: {dict(hard_flags)}")
    return by_id, {
        "annotation_files": [str(path) for path in paths],
        "annotation_rows": len(by_id),
        "rows_by_shard": {
            str(shard): rows_by_shard[shard] for shard in range(num_shards)
        },
        "incomplete_tail_files_ignored": incomplete_tail_files,
        "soft_qc_flags": dict(sorted(soft_flags.items())),
        "consistent_provenance": consistency,
    }


def load_spoken_labels(
    path: Path | None,
    ordinal_by_id: dict[str, int],
) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    labels: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(path, tolerate_incomplete_tail=False):
        annotation_id = str(row.get("id") or row.get("annotation_id") or "")
        if annotation_id not in ordinal_by_id:
            raise RuntimeError(f"spoken label outside frozen universe: {annotation_id}")
        if annotation_id in labels:
            raise RuntimeError(f"duplicate spoken label: {annotation_id}")
        flag = row.get("spoken_language_background")
        if not isinstance(flag, bool):
            raise RuntimeError(f"non-boolean spoken label: {annotation_id}")
        if str(row.get("policy_version")) != SPOKEN_POLICY_VERSION:
            raise RuntimeError(f"spoken policy mismatch: {annotation_id}")
        method = str(row.get("method") or "")
        if method not in {"text_classifier", "manual_override"}:
            raise RuntimeError(f"spoken-label method mismatch: {annotation_id}")
        if not str(row.get("classifier_revision") or ""):
            raise RuntimeError(f"missing spoken classifier revision: {annotation_id}")
        labels[annotation_id] = row
    return labels


def registry_row(
    universe: dict[str, Any],
    annotation: dict[str, Any],
    spoken: dict[str, Any],
) -> dict[str, Any]:
    description = compact_whitespace(str(annotation["source_description"]))
    return {
        "schema": REGISTRY_SCHEMA_NAME,
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "annotation_id": str(universe["annotation_id"]),
        "source_audio_sha256": str(universe["source_audio_sha256"]),
        "primary_asset_id": str(universe["primary_asset_id"]),
        "alias_asset_ids": [str(value) for value in universe["alias_asset_ids"]],
        "split": str(universe["split"]),
        "kind": str(universe["kind"]),
        "audio_path": str(universe["dry_audio_path"]),
        "source_description": description,
        "description_word_count": int(validate_source_description(description).word_count),
        "spoken_language_background": bool(spoken["spoken_language_background"]),
        "source_description_provenance": {
            "model_revision": str(annotation["model_revision"]),
            "engine": str(annotation["engine"]),
            "engine_version": str(annotation["engine_version"]),
            "prompt_sha256": str(annotation["prompt_sha256"]),
            "decoder_text_sha256": str(annotation["decoder_text_sha256"]),
            "finish_reason": str(annotation["finish_reason"]),
        },
        "spoken_language_provenance": {
            "policy_version": str(spoken["policy_version"]),
            "method": str(spoken["method"]),
            "classifier_revision": str(spoken["classifier_revision"]),
            "raw_response": str(spoken.get("raw_response") or ""),
        },
    }


def write_registry(
    output_root: Path,
    universe_rows: list[dict[str, Any]],
    annotations: dict[str, dict[str, Any]],
    spoken_labels: dict[str, dict[str, Any]],
) -> tuple[Path, Path]:
    jsonl_path = output_root / "source_description_registry.jsonl"
    parquet_path = output_root / "source_description_registry.parquet"
    jsonl_temporary = jsonl_path.with_name(jsonl_path.name + f".tmp.{os.getpid()}")
    parquet_temporary = parquet_path.with_name(parquet_path.name + f".tmp.{os.getpid()}")
    output_root.mkdir(parents=True, exist_ok=True)
    writer: pq.ParquetWriter | None = None
    chunk: list[dict[str, Any]] = []
    try:
        writer = pq.ParquetWriter(parquet_temporary, PARQUET_SCHEMA, compression="zstd")
        with jsonl_temporary.open("w", encoding="utf-8") as sink:
            for universe in universe_rows:
                annotation_id = str(universe["annotation_id"])
                row = registry_row(
                    universe,
                    annotations[annotation_id],
                    spoken_labels[annotation_id],
                )
                sink.write(stable_json(row) + "\n")
                chunk.append(row)
                if len(chunk) >= 25000:
                    writer.write_table(pa.Table.from_pylist(chunk, schema=PARQUET_SCHEMA))
                    chunk.clear()
            if chunk:
                writer.write_table(pa.Table.from_pylist(chunk, schema=PARQUET_SCHEMA))
            sink.flush()
            os.fsync(sink.fileno())
        writer.close()
        writer = None
        os.replace(jsonl_temporary, jsonl_path)
        os.replace(parquet_temporary, parquet_path)
    finally:
        if writer is not None:
            writer.close()
        for temporary in (jsonl_temporary, parquet_temporary):
            if temporary.exists():
                temporary.unlink()
    return jsonl_path, parquet_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe-parquet", type=Path, default=DEFAULT_UNIVERSE)
    parser.add_argument("--input-jsonl", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--annotations-glob", default=DEFAULT_ANNOTATIONS_GLOB)
    parser.add_argument("--spoken-labels-jsonl", type=Path)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--finalize", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be positive")
    universe_path = args.universe_parquet.expanduser().resolve(strict=True)
    input_path = args.input_jsonl.expanduser().resolve(strict=True)
    output_root = args.output_root.expanduser().resolve(strict=False)
    registry_schema_path = REGISTRY_JSON_SCHEMA.resolve(strict=True)
    paths = [
        Path(value).resolve(strict=True)
        for value in sorted(glob.glob(str(Path(args.annotations_glob).expanduser())))
    ]
    spoken_path = (
        args.spoken_labels_jsonl.expanduser().resolve(strict=True)
        if args.spoken_labels_jsonl
        else None
    )
    universe_rows, ordinal_by_id = load_universe(universe_path)
    annotations, annotation_audit = audit_annotations(
        paths,
        universe_rows,
        ordinal_by_id,
        num_shards=args.num_shards,
    )
    spoken_labels = load_spoken_labels(spoken_path, ordinal_by_id)
    for annotation_id in set(annotations).intersection(spoken_labels):
        expected_description_sha256 = hashlib.sha256(
            compact_whitespace(
                str(annotations[annotation_id]["source_description"])
            ).encode("utf-8")
        ).hexdigest()
        if str(
            spoken_labels[annotation_id].get("source_description_sha256") or ""
        ) != expected_description_sha256:
            raise RuntimeError(
                f"spoken label/description checksum mismatch: {annotation_id}"
            )
    production_annotation_root = (DEFAULT_SOURCE_ROOT / "annotations").resolve(
        strict=False
    )
    production_annotation_files = [
        path
        for path in paths
        if path.is_relative_to(production_annotation_root)
    ]
    missing_annotations = len(universe_rows) - len(annotations)
    missing_spoken_labels = len(universe_rows) - len(spoken_labels)
    missing_by_shard = Counter(
        ordinal % args.num_shards
        for annotation_id, ordinal in ordinal_by_id.items()
        if annotation_id not in annotations
    )
    complete = missing_annotations == 0 and missing_spoken_labels == 0
    summary: dict[str, Any] = {
        "schema": "stable_audio_tools.sceneplan_source_registry_finalizer_audit",
        "schema_version": 1,
        "universe_parquet": str(universe_path),
        "universe_parquet_sha256": file_sha256(universe_path),
        "input_jsonl": str(input_path),
        "input_jsonl_sha256": file_sha256(input_path),
        "registry_json_schema": str(registry_schema_path),
        "registry_json_schema_sha256": file_sha256(registry_schema_path),
        "expected_rows": len(universe_rows),
        **annotation_audit,
        "spoken_labels_jsonl": str(spoken_path) if spoken_path else None,
        "spoken_label_rows": len(spoken_labels),
        "spoken_language_background_true": sum(
            bool(row["spoken_language_background"]) for row in spoken_labels.values()
        ),
        "missing_annotation_rows": missing_annotations,
        "missing_spoken_label_rows": missing_spoken_labels,
        "missing_annotation_rows_by_shard": {
            str(shard): missing_by_shard[shard] for shard in range(args.num_shards)
        },
        "resume_contract": (
            "rerun the same frozen input, num_shards, shard id, and output path; "
            "the runner skips durable completed ids"
        ),
        "complete": complete,
        "finalize_requested": bool(args.finalize),
        "annotation_scope": (
            "production" if production_annotation_files else "preflight_or_empty"
        ),
        "full_annotation_started": bool(production_annotation_files),
        "registry_finalized": False,
        "p8_started": False,
        "p9_started": False,
    }
    if args.finalize:
        if not complete:
            raise RuntimeError(
                f"refusing finalization: missing annotations={missing_annotations}, "
                f"missing spoken labels={missing_spoken_labels}"
            )
        jsonl_path, parquet_path = write_registry(
            output_root,
            universe_rows,
            annotations,
            spoken_labels,
        )
        summary.update(
            registry_finalized=True,
            registry_jsonl=str(jsonl_path),
            registry_jsonl_sha256=file_sha256(jsonl_path),
            registry_parquet=str(parquet_path),
            registry_parquet_sha256=file_sha256(parquet_path),
        )
    summary_path = output_root / "finalizer_audit.json"
    atomic_text(summary_path, json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    if summary["registry_finalized"]:
        atomic_text(output_root / "READY", file_sha256(summary_path) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
