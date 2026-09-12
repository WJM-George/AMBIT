#!/usr/bin/env python3
"""Build and audit 100 ScenePlans from the frozen source registry."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.dataset as ds
from transformers import AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_SCRIPT_ROOT = REPO_ROOT / "scripts/t2a/data"
for path in (REPO_ROOT, DATA_SCRIPT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from build_sceneplan_manifests_v2 import (  # noqa: E402
    AssetCycler,
    atomic_parquet,
    canonical_json,
    clean_description,
    load_speech,
    plan_scene,
)
from sceneplan_v2_common import atomic_write_json  # noqa: E402
from stable_audio_tools.data.sceneplan_v2 import (  # noqa: E402
    compile_structured_source_controls,
    renderer_semantic_fragment,
    tokenize_renderer_caption,
)


DATASET_ROOT = Path(os.environ.get("AMBIT_DATA_ROOT", "data") + "/sceneplan_v2_1p124m")
REGISTRY = (
    DATASET_ROOT / "audit/a2t_pilot_100/source_description_registry_100.parquet"
)
CATALOG = DATASET_ROOT / "source_catalog/nonspeech/nonspeech_signal_catalog.parquet"
SPEECH_LEDGER = DATASET_ROOT / "split_ledgers/speech_v2/speech_split_ledger.parquet"
OUTPUT_ROOT = DATASET_ROOT / "pilots/revised_sceneplan_100_registry_v1"
TOKENIZER = Path(os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/pretrained/Qwen/Qwen3.5-0.8B")
CAPTION_MAX_TOKENS = 512
CAPTION_P99_TARGET = 384
QUOTAS = {1: 18, 2: 18, 3: 10, 4: 4}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")
    os.replace(temporary, path)


def percentile(values: list[int], value: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), value))


def load_registry(path: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows = ds.dataset(path, format="parquet").to_table().to_pylist()
    by_asset = {str(row["primary_asset_id"]): row for row in rows}
    if len(rows) != 100 or len(by_asset) != 100:
        raise RuntimeError(f"expected 100 unique pilot registry rows, got {len(rows)}/{len(by_asset)}")
    counts = Counter(str(row["kind"]) for row in rows)
    if counts != {"music": 50, "sound": 50}:
        raise RuntimeError(f"registry kind counts changed: {dict(counts)}")
    for row in rows:
        if (
            row.get("schema")
            != "stable_audio_tools.sceneplan_source_description_registry_entry"
            or int(row.get("schema_version", -1)) != 1
        ):
            raise RuntimeError(f"invalid registry schema: {row['primary_asset_id']}")
        if not str(row.get("source_description") or "").strip():
            raise RuntimeError(f"registry has an empty description: {row['primary_asset_id']}")
    return rows, by_asset


def load_registry_catalog(
    catalog_path: Path,
    registry_rows: list[dict[str, Any]],
    by_asset: dict[str, dict[str, Any]],
) -> dict[str, AssetCycler]:
    asset_ids = list(by_asset)
    columns = [
        "asset_id",
        "source_dataset",
        "kind",
        "description",
        "dry_audio_path",
        "source_audio_sha256",
        "native_sample_rate_hz",
        "native_num_samples",
        "model_num_samples",
        "selection_rank",
        "eligible",
    ]
    table = ds.dataset(catalog_path, format="parquet").to_table(
        columns=columns,
        filter=ds.field("asset_id").isin(asset_ids) & (ds.field("eligible") == True),  # noqa: E712
    )
    catalog_by_asset = {str(row["asset_id"]): row for row in table.to_pylist()}
    missing = sorted(set(asset_ids) - set(catalog_by_asset))
    if missing:
        raise RuntimeError(f"approved annotations missing from eligible catalog: {missing[:3]}")

    ordered = {"music": [], "sound": []}
    for annotation in registry_rows:
        asset_id = str(annotation["primary_asset_id"])
        row = dict(catalog_by_asset[asset_id])
        if str(row["kind"]) != str(annotation["kind"]):
            raise RuntimeError(f"kind mismatch for {asset_id}")
        if str(row["source_audio_sha256"]) != str(annotation["source_audio_sha256"]):
            raise RuntimeError(f"audio SHA256 mismatch for {asset_id}")
        row["description"] = str(annotation["source_description"])
        row["spoken_language_background"] = bool(
            annotation["spoken_language_background"]
        )
        ordered[str(row["kind"])].append(row)
    return {kind: AssetCycler(rows) for kind, rows in ordered.items()}


def verify_caption_regions(record: dict[str, Any]) -> None:
    caption = record["renderer_caption"]
    text = str(caption["text"])
    sources = record["scene_plan"]["sources"]
    for region in caption["source_semantic_regions"]:
        source = sources[int(region["source_slot"])]
        expected = (
            source["speech"]["speaker_description"]
            if source["kind"] == "speech"
            else source["description"]
        )
        expected = renderer_semantic_fragment(expected)
        if text[int(region["start"]) : int(region["end"])] != expected:
            raise RuntimeError(f"semantic character span mismatch: {record['sample_id']}")
    for region in caption["transcript_regions"]:
        source = sources[int(region["source_slot"])]
        if text[int(region["start"]) : int(region["end"])] != source["speech"]["transcript"]:
            raise RuntimeError(f"transcript character span mismatch: {record['sample_id']}")


def audit_conditioning(
    row: dict[str, Any],
    tokenizer: Any,
    annotations: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    record = json.loads(row["record_json"])
    verify_caption_regions(record)
    caption = record["renderer_caption"]
    scene_plan = record["scene_plan"]
    tokenized = tokenize_renderer_caption(
        caption, tokenizer, max_length=CAPTION_MAX_TOKENS
    )
    controls = compile_structured_source_controls(scene_plan)
    token_count = int(len(tokenized["input_ids"]))
    row["qwen_token_count"] = token_count

    semantic_counts = tokenized["source_semantic_token_masks"].sum(axis=1).astype(int).tolist()
    motion_counts = tokenized["source_motion_activity_token_masks"].sum(axis=1).astype(int).tolist()
    speaker_count = int(tokenized["speaker_info_token_mask"].sum())
    transcript_count = int(tokenized["quoted_transcript_token_mask"].sum())
    present_sources = [source for source in scene_plan["sources"] if source["present"]]
    for source in scene_plan["sources"]:
        slot = int(source["slot"])
        if bool(source["present"]) != bool(semantic_counts[slot]):
            raise RuntimeError(f"semantic mask/presence mismatch: {row['sample_id']} slot={slot}")
        if bool(source["present"]) != bool(motion_counts[slot]):
            raise RuntimeError(f"motion mask/presence mismatch: {row['sample_id']} slot={slot}")
    has_speech = any(source["kind"] == "speech" for source in present_sources)
    if has_speech != bool(speaker_count) or has_speech != bool(transcript_count):
        raise RuntimeError(f"speech mask mismatch: {row['sample_id']}")
    if tuple(controls["source_present_mask"].shape) != (4,):
        raise RuntimeError(f"source-present shape mismatch: {row['sample_id']}")
    if tuple(controls["source_activity_frame_masks"].shape) != (
        4,
        int(scene_plan["audio"]["latent_frames_valid"]),
    ):
        raise RuntimeError(f"activity control shape mismatch: {row['sample_id']}")
    if tuple(controls["source_position_activity_features"].shape) != (
        4,
        int(scene_plan["audio"]["latent_frames_valid"]),
        8,
    ):
        raise RuntimeError(f"position control shape mismatch: {row['sample_id']}")
    if int(controls["source_present_mask"].sum()) != int(row["source_count"]):
        raise RuntimeError(f"structured source count mismatch: {row['sample_id']}")

    annotation_assets = [
        source["asset_ref"]["asset_id"]
        for source in present_sources
        if source["kind"] in {"music", "sound"}
    ]
    if any(asset_id not in annotations for asset_id in annotation_assets):
        raise RuntimeError(f"unapproved nonspeech description entered {row['sample_id']}")
    for source in present_sources:
        if source["kind"] not in {"music", "sound"}:
            continue
        asset_id = source["asset_ref"]["asset_id"]
        expected = clean_description(
            annotations[asset_id]["source_description"], source["kind"]
        )
        if source["description"] != expected:
            raise RuntimeError(f"source description injection mismatch: {row['sample_id']}")
    flagged_background_assets = [
        asset_id
        for asset_id in annotation_assets
        if bool(annotations[asset_id]["spoken_language_background"])
    ]
    if has_speech and flagged_background_assets:
        raise RuntimeError(
            f"formal TTS scene contains spoken-language background: "
            f"{row['sample_id']} {flagged_background_assets}"
        )
    return {
        "sample_id": row["sample_id"],
        "family": row["family"],
        "source_count": int(row["source_count"]),
        "source_kinds": row["source_kinds"],
        "annotation_asset_ids": annotation_assets,
        "annotation_ids": [annotations[asset_id]["annotation_id"] for asset_id in annotation_assets],
        "spoken_language_background_asset_ids": flagged_background_assets,
        "caption": row["caption"],
        "qwen_token_count": token_count,
        "sequence_length": token_count,
        "source_semantic_token_counts": semantic_counts,
        "source_motion_activity_token_counts": motion_counts,
        "speaker_info_token_count": speaker_count,
        "quoted_transcript_token_count": transcript_count,
        "structured_controls": {
            "source_present_mask": controls["source_present_mask"].astype(int).tolist(),
            "source_kind_ids": controls["source_kind_ids"].astype(int).tolist(),
            "source_slot_ids": controls["source_slot_ids"].astype(int).tolist(),
            "activity_shape": list(controls["source_activity_frame_masks"].shape),
            "position_activity_shape": list(controls["source_position_activity_features"].shape),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=REGISTRY)
    parser.add_argument("--catalog", type=Path, default=CATALOG)
    parser.add_argument("--speech-ledger", type=Path, default=SPEECH_LEDGER)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()

    registry_path = args.registry.expanduser().resolve(strict=True)
    output_root = args.output_root.expanduser().resolve(strict=False)
    try:
        output_root.relative_to(os.environ.get("AMBIT_DATA_ROOT", "data"))
    except ValueError as error:
        raise ValueError(f"pilot output must be on SDB: {output_root}") from error
    output_root.mkdir(parents=True, exist_ok=True)

    registry_rows, registry_by_asset = load_registry(registry_path)
    cyclers = load_registry_catalog(args.catalog, registry_rows, registry_by_asset)
    speech_rows = load_speech(args.speech_ledger, "train", True)[:50]
    if len(speech_rows) != 50:
        raise RuntimeError("speech reserve did not provide 50 pilot donors")

    rows: list[dict[str, Any]] = []
    global_index = 0
    speech_cursor = 0
    for family in ("speech", "no_speech"):
        for source_count, count in QUOTAS.items():
            for cell_index in range(count):
                speech_row = None
                if family == "speech":
                    speech_row = speech_rows[speech_cursor]
                    speech_cursor += 1
                rows.append(
                    plan_scene(
                        sample_id=(
                            f"revised100_{family}_{source_count}_{cell_index:03d}"
                        ),
                        split="train",
                        family=family,
                        source_count=source_count,
                        cell_index=global_index,
                        speech_row=speech_row,
                        cyclers=cyclers,
                    )
                )
                global_index += 1
    if len(rows) != 100 or speech_cursor != 50:
        raise RuntimeError(f"pilot row count changed: rows={len(rows)} speech={speech_cursor}")

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)
    approved_assets = set(registry_by_asset)
    conditioning_rows = [
        audit_conditioning(row, tokenizer, registry_by_asset) for row in rows
    ]

    token_counts = [int(row["qwen_token_count"]) for row in rows]
    annotation_references = [
        asset_id
        for audit in conditioning_rows
        for asset_id in audit["annotation_asset_ids"]
    ]
    annotation_unique = set(annotation_references)
    source_kind_counts = Counter(
        kind for row in rows for kind in row["source_kinds"]
    )
    family_counts = Counter(str(row["family"]) for row in rows)
    source_count_counts = Counter(int(row["source_count"]) for row in rows)
    room_counts = Counter(str(row["room_class"]) for row in rows)
    max_tokens = max(token_counts)
    p99_tokens = percentile(token_counts, 99)
    all_annotations_covered = annotation_unique == approved_assets
    hard_token_gate_ok = max_tokens <= CAPTION_MAX_TOKENS
    p99_target_ok = p99_tokens <= CAPTION_P99_TARGET

    parquet_path = output_root / "sceneplans/revised-sceneplans-00000.parquet"
    atomic_parquet(parquet_path, rows)
    sceneplan_jsonl_path = output_root / "revised_sceneplans_100.jsonl"
    write_jsonl(
        sceneplan_jsonl_path,
        [json.loads(row["record_json"]) for row in rows],
    )
    conditioning_path = output_root / "conditioning_audit.jsonl"
    write_jsonl(conditioning_path, conditioning_rows)

    sample_order: list[dict[str, Any]] = []
    for family in ("speech", "no_speech"):
        for source_count in (1, 2, 3, 4):
            candidates = [
                row
                for row in conditioning_rows
                if row["family"] == family and row["source_count"] == source_count
            ]
            sample_order.append(max(candidates, key=lambda row: row["qwen_token_count"]))
    sample_order.extend(
        sorted(conditioning_rows, key=lambda row: row["qwen_token_count"], reverse=True)[:2]
    )
    deduplicated_samples: list[dict[str, Any]] = []
    seen_samples: set[str] = set()
    for row in sample_order:
        if row["sample_id"] not in seen_samples:
            deduplicated_samples.append(row)
            seen_samples.add(row["sample_id"])
    for row in sorted(
        conditioning_rows, key=lambda value: value["qwen_token_count"], reverse=True
    ):
        if len(deduplicated_samples) >= 10:
            break
        if row["sample_id"] not in seen_samples:
            deduplicated_samples.append(row)
            seen_samples.add(row["sample_id"])
    samples_path = output_root / "caption_442_samples.jsonl"
    write_jsonl(samples_path, deduplicated_samples)

    summary = {
        "schema": "stable_audio_tools.revised_sceneplan_description_pilot_audit",
        "schema_version": 1,
        "status": "awaiting_user_review",
        "ok": all_annotations_covered and hard_token_gate_ok and p99_target_ok,
        "rows": len(rows),
        "family_counts": dict(sorted(family_counts.items())),
        "source_count_counts": {str(k): v for k, v in sorted(source_count_counts.items())},
        "room_counts": dict(sorted(room_counts.items())),
        "source_kind_counts": dict(sorted(source_kind_counts.items())),
        "source_description_registry": {
            "input": str(registry_path),
            "input_sha256": sha256_file(registry_path),
            "approved_unique": len(approved_assets),
            "referenced_unique": len(annotation_unique),
            "total_references": len(annotation_references),
            "all_approved_annotations_covered": all_annotations_covered,
            "spoken_language_background_registry_rows": sum(
                bool(row["spoken_language_background"]) for row in registry_rows
            ),
            "formal_tts_scenes_with_spoken_language_background": sum(
                row["family"] == "speech"
                and bool(row["spoken_language_background_asset_ids"])
                for row in conditioning_rows
            ),
        },
        "caption_tokens": {
            "tokenizer": str(TOKENIZER),
            "min": min(token_counts),
            "p50": percentile(token_counts, 50),
            "p90": percentile(token_counts, 90),
            "p99": p99_tokens,
            "max": max_tokens,
            "p99_target": CAPTION_P99_TARGET,
            "p99_target_ok": p99_target_ok,
            "hard_ceiling": CAPTION_MAX_TOKENS,
            "hard_ceiling_ok": hard_token_gate_ok,
            "truncated_rows": 0,
        },
        "conditioning": {
            "caption_character_spans_exact": True,
            "source_semantic_masks": 4,
            "source_motion_activity_masks": 4,
            "speaker_info_masks": 1,
            "quoted_transcript_masks": 1,
            "mask_audit_rows_passed": len(conditioning_rows),
            "structured_control_rows_passed": len(conditioning_rows),
            "structured_position_feature_dim": 8,
        },
        "outputs": {
            "sceneplan_jsonl": str(sceneplan_jsonl_path),
            "sceneplan_jsonl_sha256": sha256_file(sceneplan_jsonl_path),
            "sceneplan_parquet": str(parquet_path),
            "sceneplan_parquet_sha256": sha256_file(parquet_path),
            "conditioning_audit_jsonl": str(conditioning_path),
            "conditioning_audit_sha256": sha256_file(conditioning_path),
            "caption_442_samples_jsonl": str(samples_path),
            "caption_442_samples_sha256": sha256_file(samples_path),
        },
        "full_scale_annotation_started": False,
        "full_sceneplan_revision_started": False,
        "p8_started": False,
        "p9_started": False,
    }
    summary_path = output_root / "summary.json"
    atomic_write_json(summary_path, summary)
    marker_path = output_root / "REVISED_SCENEPLAN_100_AWAITING_USER_REVIEW"
    atomic_write_json(marker_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
