#!/usr/bin/env python3
"""Freeze the accepted 100-source pilot into the production registry schema."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from description_contract import compact_whitespace, validate_source_description
from finalize_source_registry import (
    PARQUET_SCHEMA,
    REGISTRY_SCHEMA_NAME,
    REGISTRY_SCHEMA_VERSION,
    SPOKEN_POLICY_VERSION,
    stable_json,
)


DATASET_ROOT = Path("/mnt/sdb/audio_dataset/sceneplan_v2_1p124m")
ROOT = DATASET_ROOT / "audit/a2t_pilot_100"
ANNOTATIONS = ROOT / "source_descriptions_instruct_100.jsonl"
LABELS = ROOT / "spoken_language_background_v2.jsonl"
UNIVERSE = (
    DATASET_ROOT
    / "source_annotations/nonspeech_instruct_v2/source_universe.parquet"
)
JSON_SCHEMA = (
    Path(__file__).with_name("schemas")
    / "source_description_registry_v1.schema.json"
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path, id_key: str) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                annotation_id = str(row[id_key])
            except (json.JSONDecodeError, KeyError) as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from error
            if annotation_id in output:
                raise ValueError(f"duplicate id in {path}: {annotation_id}")
            output[annotation_id] = row
    return output


def atomic_text(path: Path, value: str) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    root = ROOT.resolve(strict=True)
    annotations_path = ANNOTATIONS.resolve(strict=True)
    labels_path = LABELS.resolve(strict=True)
    universe_path = UNIVERSE.resolve(strict=True)
    schema_path = JSON_SCHEMA.resolve(strict=True)
    annotations = load_jsonl(annotations_path, "annotation_id")
    labels = load_jsonl(labels_path, "id")
    if len(annotations) != 100 or set(annotations) != set(labels):
        raise RuntimeError(
            f"pilot registry requires the same 100 annotations/labels: "
            f"{len(annotations)}/{len(labels)}"
        )
    table = pq.read_table(
        universe_path,
        columns=[
            "annotation_id",
            "source_audio_sha256",
            "primary_asset_id",
            "alias_asset_ids",
            "split",
            "kind",
            "dry_audio_path",
        ],
        filters=pc.field("annotation_id").isin(list(annotations)),
    )
    universe_rows = table.to_pylist()
    universe_by_id = {str(row["annotation_id"]): row for row in universe_rows}
    if len(universe_by_id) != 100 or set(universe_by_id) != set(annotations):
        raise RuntimeError("pilot annotations do not map one-to-one to the frozen universe")

    rows: list[dict[str, Any]] = []
    for annotation_id in sorted(
        annotations,
        key=lambda value: (
            str(universe_by_id[value]["split"]),
            str(universe_by_id[value]["kind"]),
            value,
        ),
    ):
        annotation = annotations[annotation_id]
        label = labels[annotation_id]
        universe = universe_by_id[annotation_id]
        description = compact_whitespace(annotation["source_description"])
        description_sha256 = hashlib.sha256(description.encode("utf-8")).hexdigest()
        if description_sha256 != str(label.get("source_description_sha256")):
            raise RuntimeError(f"description/label hash mismatch: {annotation_id}")
        qc = validate_source_description(description)
        if qc.hard_flags or annotation.get("hard_qc_flags"):
            raise RuntimeError(f"hard description QC failure: {annotation_id}")
        if str(label.get("policy_version")) != SPOKEN_POLICY_VERSION:
            raise RuntimeError(f"spoken policy mismatch: {annotation_id}")
        if str(annotation["source_audio_sha256"]) != str(
            universe["source_audio_sha256"]
        ):
            raise RuntimeError(f"source hash mismatch: {annotation_id}")
        provenance = annotation["provenance"]
        rows.append(
            {
                "schema": REGISTRY_SCHEMA_NAME,
                "schema_version": REGISTRY_SCHEMA_VERSION,
                "annotation_id": annotation_id,
                "source_audio_sha256": str(universe["source_audio_sha256"]),
                "primary_asset_id": str(universe["primary_asset_id"]),
                "alias_asset_ids": [
                    str(value) for value in universe["alias_asset_ids"]
                ],
                "split": str(universe["split"]),
                "kind": str(universe["kind"]),
                "audio_path": str(universe["dry_audio_path"]),
                "source_description": description,
                "description_word_count": qc.word_count,
                "spoken_language_background": bool(
                    label["spoken_language_background"]
                ),
                "source_description_provenance": {
                    "model_revision": str(provenance["model_revision"]),
                    "engine": str(provenance["engine"]),
                    "engine_version": str(provenance["engine_version"]),
                    "prompt_sha256": str(provenance["prompt_sha256"]),
                    "decoder_text_sha256": str(
                        provenance["decoder_text_sha256"]
                    ),
                    "finish_reason": str(provenance["finish_reason"]),
                },
                "spoken_language_provenance": {
                    "policy_version": str(label["policy_version"]),
                    "method": str(label["method"]),
                    "classifier_revision": str(label["classifier_revision"]),
                    "raw_response": str(label["raw_response"]),
                },
            }
        )

    jsonl_path = root / "source_description_registry_100.jsonl"
    parquet_path = root / "source_description_registry_100.parquet"
    atomic_text(jsonl_path, "".join(stable_json(row) + "\n" for row in rows))
    temporary = parquet_path.with_name(parquet_path.name + f".tmp.{os.getpid()}")
    pq.write_table(pa.Table.from_pylist(rows, schema=PARQUET_SCHEMA), temporary, compression="zstd")
    os.replace(temporary, parquet_path)
    summary = {
        "schema": "stable_audio_tools.sceneplan_source_description_pilot_registry",
        "schema_version": 1,
        "rows": len(rows),
        "kind_counts": {
            kind: sum(row["kind"] == kind for row in rows)
            for kind in ("music", "sound")
        },
        "spoken_language_background_true": sum(
            row["spoken_language_background"] for row in rows
        ),
        "json_schema": str(schema_path),
        "json_schema_sha256": file_sha256(schema_path),
        "registry_jsonl": str(jsonl_path),
        "registry_jsonl_sha256": file_sha256(jsonl_path),
        "registry_parquet": str(parquet_path),
        "registry_parquet_sha256": file_sha256(parquet_path),
        "description_rewrite": False,
    }
    summary_path = root / "source_description_registry_100_summary.json"
    atomic_text(summary_path, json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
