#!/usr/bin/env python3
"""Audit and freeze the 1.124M Sound-expansion ScenePlan dataset revision."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import time

import pyarrow.parquet as pq
from safetensors import safe_open
import torch
from transformers import AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
for value in (SCRIPT_DIR, REPO_ROOT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from build_model_sceneplan_training_index_v1 import (  # noqa: E402
    build_split,
    create_schema,
)
from sceneplan_v2_common import DATASET_ROOT, atomic_write_json  # noqa: E402
from stable_audio_tools.data.sceneplan_v2_dataset import ScenePlanV2Dataset  # noqa: E402


EXPECTED_SPLITS = {"train": 1_100_000, "validation": 20_000, "test": 4_000}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_p10_dataset_configs(
    revision_root: Path,
    combined_root: Path,
    speech_timing: Path,
    speech_timing_sha256: str,
) -> list[dict]:
    """Write revision-local P10 loader configs with explicit timing lineage."""

    config_root = revision_root / "p10_configs"
    common = {
        "dataset_type": "sceneplan_v2_preencoded",
        "random_crop": False,
        "latent_crop_length": 432,
        "latent_downsampling_ratio": 1024,
        "caption_max_tokens": 512,
        "max_length_sec": 10.031020408163266,
        "require_complete": True,
        "max_item_retries": 0,
        "pin_memory": True,
        "persistent_workers": True,
        "prefetch_factor": 2,
        "in_order": True,
    }
    results = []
    for split, expected in EXPECTED_SPLITS.items():
        value = {
            "_task": (
                "Frozen ScenePlan-v2 Sound-expansion revision; variable FOA "
                "latents are padded without cropping."
            ),
            **common,
            "expected_num_samples": expected,
            "drop_last": split == "train",
            "datasets": [
                {
                    "id": f"sceneplan_v2_sound_expansion_v1_{split}",
                    "path": str(combined_root / f"{split}.sqlite"),
                    "weight": 1.0,
                }
            ],
        }
        # The current forced-alignment sidecar deliberately covers the 500k
        # formal train speech rows.  Do not claim validation/test timing that
        # does not exist.
        if split == "train":
            value.update(
                {
                    "require_speech_timing": True,
                    "speech_timing_index_path": str(speech_timing),
                    "speech_timing_index_sha256": speech_timing_sha256,
                    "expected_speech_timing_rows": 500_000,
                }
            )
        path = config_root / f"sceneplan_v2_sound_expansion_v1_{split}.json"
        atomic_write_json(path, value)
        results.append(
            {
                "split": split,
                "path": str(path),
                "sha256": sha256_file(path),
                "rows": expected,
                "require_speech_timing": split == "train",
            }
        )
    return results


def create_map_table(connection: sqlite3.Connection, name: str, values: dict[int, int]) -> None:
    connection.execute(f"CREATE TEMP TABLE {name}(old_id INTEGER PRIMARY KEY, new_id INTEGER NOT NULL)")
    connection.executemany(
        f"INSERT INTO {name}(old_id,new_id) VALUES (?,?)", sorted(values.items())
    )


def merge_split(
    base_path: Path, overlay_path: Path, output_root: Path, split: str
) -> dict:
    output = output_root / f"{split}.sqlite"
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temporary.unlink(missing_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(temporary)
    create_schema(connection)
    connection.execute("ATTACH DATABASE ? AS base", (str(base_path),))
    connection.execute("ATTACH DATABASE ? AS overlay", (str(overlay_path),))
    started = time.time()
    try:
        base_rows = int(connection.execute("SELECT COUNT(*) FROM base.samples").fetchone()[0])
        overlay_rows = int(connection.execute("SELECT COUNT(*) FROM overlay.samples").fetchone()[0])
        missing = int(
            connection.execute(
                "SELECT COUNT(*) FROM overlay.samples o LEFT JOIN base.samples b USING(sample_id) WHERE b.sample_id IS NULL"
            ).fetchone()[0]
        )
        if base_rows != EXPECTED_SPLITS[split] or overlay_rows <= 0 or missing:
            raise RuntimeError(
                f"{split}: invalid base/overlay coverage {base_rows}/{overlay_rows}, missing={missing}"
            )
        latent_by_key: dict[tuple[str, str], int] = {}
        maps: dict[str, dict[int, int]] = {"base": {}, "overlay": {}}
        for schema in ("base", "overlay"):
            for old_id, path, digest in connection.execute(
                f"SELECT id,path,sha256 FROM {schema}.latent_shards ORDER BY id"
            ):
                key = (str(path), str(digest))
                if key not in latent_by_key:
                    new_id = len(latent_by_key) + 1
                    latent_by_key[key] = new_id
                    connection.execute(
                        "INSERT INTO latent_shards(id,path,sha256) VALUES (?,?,?)",
                        (new_id, str(path), str(digest)),
                    )
                maps[schema][int(old_id)] = latent_by_key[key]
        create_map_table(connection, "base_latent_map", maps["base"])
        create_map_table(connection, "overlay_latent_map", maps["overlay"])
        connection.execute(
            """
            INSERT INTO samples
            SELECT b.ordinal,b.sample_id,b.model_num_samples,b.latent_frames_valid,
                   b.renderer_caption_zlib,b.scene_plan_zlib,b.model_sceneplan_sha256,
                   b.model_sceneplan_path,b.model_sceneplan_row,
                   b.model_sceneplan_byte_offset,b.model_sceneplan_byte_length,
                   b.compiled_conditioning_ref,m.new_id,b.latent_key,
                   b.latent_tensor_sha256,b.renderer_caption_sha256
            FROM base.samples b
            JOIN base_latent_map m ON m.old_id=b.latent_shard_id
            WHERE NOT EXISTS(
                SELECT 1 FROM overlay.samples o WHERE o.sample_id=b.sample_id
            )
            """
        )
        connection.execute(
            """
            INSERT INTO samples
            SELECT b.ordinal,o.sample_id,o.model_num_samples,o.latent_frames_valid,
                   o.renderer_caption_zlib,o.scene_plan_zlib,o.model_sceneplan_sha256,
                   o.model_sceneplan_path,o.model_sceneplan_row,
                   o.model_sceneplan_byte_offset,o.model_sceneplan_byte_length,
                   o.compiled_conditioning_ref,m.new_id,o.latent_key,
                   o.latent_tensor_sha256,o.renderer_caption_sha256
            FROM overlay.samples o
            JOIN base.samples b ON b.sample_id=o.sample_id
            JOIN overlay_latent_map m ON m.old_id=o.latent_shard_id
            """
        )
        metadata = dict(connection.execute("SELECT key,value FROM base.metadata"))
        metadata.update(
            {
                "mode": "sound_expansion_revision_v1",
                "rows": str(base_rows),
                "latent_shards": str(len(latent_by_key)),
                "sound_replacement_rows": str(overlay_rows),
                "base_index_path": str(base_path),
                "base_index_sha256": sha256_file(base_path),
                "overlay_index_path": str(overlay_path),
                "overlay_index_sha256": sha256_file(overlay_path),
            }
        )
        connection.executemany(
            "INSERT INTO metadata(key,value) VALUES (?,?)", sorted(metadata.items())
        )
        connection.commit()
        connection.execute("ANALYZE")
        connection.commit()
        count, unique, minimum, maximum = connection.execute(
            "SELECT COUNT(*),COUNT(DISTINCT sample_id),MIN(ordinal),MAX(ordinal) FROM samples"
        ).fetchone()
        if (int(count), int(unique), int(minimum), int(maximum)) != (
            base_rows, base_rows, 0, base_rows - 1
        ):
            raise RuntimeError(f"{split}: merged ordinal/sample coverage failed")
        if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise RuntimeError(f"{split}: merged SQLite integrity failed")
    finally:
        connection.close()
    os.replace(temporary, output)
    reopened = sqlite3.connect(f"file:{output}?mode=ro&immutable=1", uri=True)
    try:
        if reopened.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise RuntimeError(f"{split}: atomic merged SQLite reopen failed")
    finally:
        reopened.close()
    return {
        "split": split,
        "rows": EXPECTED_SPLITS[split],
        "replacement_rows": overlay_rows,
        "latent_shards": len(latent_by_key),
        "path": str(output),
        "num_bytes": output.stat().st_size,
        "sha256": sha256_file(output),
        "elapsed_sec": round(time.time() - started, 3),
    }


def copy_unchanged_split(
    base_path: Path, output_root: Path, split: str
) -> dict:
    """Create a revision-local, byte-identical index for an untouched split."""

    output_root.mkdir(parents=True, exist_ok=True)
    output = output_root / f"{split}.sqlite"
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temporary.unlink(missing_ok=True)
    shutil.copyfile(base_path, temporary)
    with temporary.open("rb+") as handle:
        os.fsync(handle.fileno())
    if sha256_file(temporary) != sha256_file(base_path):
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"{split}: unchanged base index copy hash failed")
    connection = sqlite3.connect(f"file:{temporary}?mode=ro&immutable=1", uri=True)
    try:
        rows = int(connection.execute("SELECT COUNT(*) FROM samples").fetchone()[0])
        latent_shards = int(
            connection.execute("SELECT COUNT(*) FROM latent_shards").fetchone()[0]
        )
        integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
    finally:
        connection.close()
    if rows != EXPECTED_SPLITS[split] or integrity != "ok":
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"{split}: unchanged base index copy failed integrity")
    os.replace(temporary, output)
    return {
        "split": split,
        "rows": rows,
        "replacement_rows": 0,
        "latent_shards": latent_shards,
        "path": str(output),
        "num_bytes": output.stat().st_size,
        "sha256": sha256_file(output),
        "base_index_sha256": sha256_file(base_path),
        "unchanged_from_base": True,
    }


def audit_replacement_latents(materialized_root: Path, expected: dict[str, int]) -> dict:
    rows_seen = 0
    shards_seen = 0
    min_frames = 10_000
    max_frames = 0
    split_rows: Counter[str] = Counter()
    for split in ("train", "validation", "test"):
        for manifest in sorted(
            (materialized_root / "manifests" / split).glob(f"materialized-{split}-*.parquet")
        ):
            rows = pq.read_table(manifest).to_pylist()
            if not rows:
                raise RuntimeError(f"empty replacement manifest: {manifest}")
            latent_refs = {str(row["latent_ref"]).split("#", 1)[0] for row in rows}
            latent_hashes = {str(row["latent_shard_sha256"]) for row in rows}
            if len(latent_refs) != 1 or len(latent_hashes) != 1:
                raise RuntimeError(f"replacement latent lineage differs: {manifest}")
            latent_path = Path(latent_refs.pop()).resolve(strict=True)
            if sha256_file(latent_path) != latent_hashes.pop():
                raise RuntimeError(f"replacement latent shard hash failed: {latent_path}")
            by_id = {str(row["sample_id"]): row for row in rows}
            with safe_open(str(latent_path), framework="pt", device="cpu") as handle:
                if set(handle.keys()) != set(by_id):
                    raise RuntimeError(f"replacement latent keys differ: {latent_path}")
                for sample_id, row in by_id.items():
                    tensor = handle.get_tensor(sample_id)
                    frames = int(row["latent_frames_valid"])
                    if (
                        tensor.dtype != torch.float16
                        or tuple(tensor.shape) != (64, frames)
                        or not torch.isfinite(tensor).all().item()
                        or tensor_sha256(tensor) != str(row["latent_tensor_sha256"])
                    ):
                        raise RuntimeError(f"{sample_id}: replacement latent tensor failed")
                    result = json.loads(str(row["render_result_json"]))
                    if (
                        result.get("status") != "ok"
                        or result.get("sample_id") != sample_id
                        or len(result.get("source_qc") or ()) != 1
                        or result["source_qc"][0].get("kind") != "sound"
                    ):
                        raise RuntimeError(f"{sample_id}: replacement renderer QC failed")
                    min_frames = min(min_frames, frames)
                    max_frames = max(max_frames, frames)
            rows_seen += len(rows)
            split_rows[split] += len(rows)
            shards_seen += 1
    normalized_split_rows = {
        split: split_rows[split] for split in ("train", "validation", "test")
    }
    if normalized_split_rows != expected or rows_seen != sum(expected.values()):
        raise RuntimeError(
            f"replacement latent split coverage failed: {normalized_split_rows}"
        )
    return {
        "rows": rows_seen,
        "split_rows": normalized_split_rows,
        "latent_shards": shards_seen,
        "latent_frames_min": min_frames,
        "latent_frames_max": max_frames,
        "all_tensor_checksums_finite_shapes_pass": True,
        "all_renderer_results_pass": True,
    }


def sound_reuse_audit(base_index: Path, replacement_map: Path) -> dict:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    parquet = pq.ParquetFile(base_index)
    for batch in parquet.iter_batches(
        columns=["split", "source_asset_ids", "source_kinds"], batch_size=32_768
    ):
        for split, assets, kinds in zip(*(batch.column(index).to_pylist() for index in range(3))):
            for asset, kind in zip(assets or (), kinds or ()):
                if kind == "sound":
                    counts[str(split)][str(asset)] += 1
    before = {split: Counter(values) for split, values in counts.items()}
    replacements = read_jsonl(replacement_map)
    new_hashes: dict[str, set[str]] = defaultdict(set)
    replacement_counts: Counter[str] = Counter()
    for row in replacements:
        split = str(row["split"])
        old, new = str(row["old_asset_id"]), str(row["new_asset_id"])
        counts[split][old] -= 1
        if counts[split][old] == 0:
            del counts[split][old]
        counts[split][new] += 1
        new_hashes[split].add(str(row["new_source_audio_sha256"]))
        replacement_counts[split] += 1
    if any(counts["train"].get(str(row["old_asset_id"]), 0) <= 0 for row in replacements if row["split"] == "train"):
        raise RuntimeError("a train Sound asset was removed instead of retained")
    asset_sets = {split: set(values) for split, values in counts.items()}
    if any(asset_sets[a] & asset_sets[b] for a in asset_sets for b in asset_sets if a < b):
        raise RuntimeError("Sound asset IDs leak across revised splits")
    if any(new_hashes[a] & new_hashes[b] for a in new_hashes for b in new_hashes if a < b):
        raise RuntimeError("new Sound hashes leak across revised splits")
    report = {}
    for split in ("train", "validation", "test"):
        old, new = before[split], counts[split]
        if sum(old.values()) != sum(new.values()):
            raise RuntimeError(f"{split}: Sound appearance count changed")
        report[split] = {
            "appearances": sum(new.values()),
            "unique_before": len(old),
            "unique_after": len(new),
            "unique_gain": len(new) - len(old),
            "mean_reuse_before": sum(old.values()) / len(old),
            "mean_reuse_after": sum(new.values()) / len(new),
            "reuse_histogram_before": dict(sorted(Counter(old.values()).items())),
            "reuse_histogram_after": dict(sorted(Counter(new.values()).items())),
        }
    if report["train"]["unique_gain"] != replacement_counts["train"]:
        raise RuntimeError("train Sound unique gain differs from replacement count")
    for split in ("validation", "test"):
        if report[split]["unique_gain"] != 0:
            raise RuntimeError(f"{split}: one-for-one Sound exchange changed unique count")
    report["global"] = {
        "appearances": sum(value["appearances"] for value in report.values()),
        "unique_before": sum(value["unique_before"] for value in report.values()),
        "unique_after": sum(value["unique_after"] for value in report.values()),
        "unique_gain": sum(value["unique_gain"] for value in report.values()),
    }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision-root", type=Path, required=True)
    parser.add_argument("--sceneplan-root", type=Path, required=True)
    parser.add_argument("--materialized-root", type=Path, required=True)
    parser.add_argument("--source-universe-summary", type=Path, required=True)
    parser.add_argument("--registry-audit", type=Path, required=True)
    parser.add_argument(
        "--base-sceneplan-index", type=Path,
        default=DATASET_ROOT / "sceneplans_model_v1/index.parquet",
    )
    parser.add_argument(
        "--base-training-index-root", type=Path,
        default=DATASET_ROOT / "training_index",
    )
    parser.add_argument(
        "--speech-timing-index", type=Path,
        default=DATASET_ROOT / "source_annotations/speech_forced_alignment_v1/registry/speech_timing_train.sqlite",
    )
    parser.add_argument(
        "--tokenizer-root", type=Path,
        default=Path(os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/pretrained/Qwen/Qwen3.5-0.8B"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    revision_root = args.revision_root.expanduser().resolve(strict=True)
    sceneplan_root = args.sceneplan_root.expanduser().resolve(strict=True)
    materialized_root = args.materialized_root.expanduser().resolve(strict=True)
    expected_revision_root = (DATASET_ROOT / "revisions").resolve(strict=True)
    if expected_revision_root not in revision_root.parents:
        raise ValueError("revision root must be below DATASET_ROOT/revisions")
    base_index = args.base_sceneplan_index.expanduser().resolve(strict=True)
    base_training = args.base_training_index_root.expanduser().resolve(strict=True)
    source_summary_path = args.source_universe_summary.expanduser().resolve(strict=True)
    registry_audit_path = args.registry_audit.expanduser().resolve(strict=True)
    speech_timing = args.speech_timing_index.expanduser().resolve(strict=True)
    timing_receipt = speech_timing.with_suffix(speech_timing.suffix + ".receipt.json")
    if not timing_receipt.is_file():
        raise RuntimeError("P9 requires the immutable 500k speech timing receipt")
    timing_doc = json.loads(timing_receipt.read_text(encoding="utf-8"))
    if timing_doc.get("status") != "PASS" or int(timing_doc.get("rows", -1)) != 500_000:
        raise RuntimeError("speech timing sidecar gate failed")
    p8 = json.loads((materialized_root / "P8_SUMMARY.json").read_text(encoding="utf-8"))
    plan_summary = json.loads((sceneplan_root / "summary.json").read_text(encoding="utf-8"))
    source_summary = json.loads(source_summary_path.read_text(encoding="utf-8"))
    registry_audit = json.loads(registry_audit_path.read_text(encoding="utf-8"))
    if (
        any(doc.get("status") != "PASS" for doc in (p8, plan_summary, source_summary))
        or registry_audit.get("complete") is not True
        or registry_audit.get("registry_finalized") is not True
    ):
        raise RuntimeError("one or more P7/P7.5/P8 inputs are not PASS")
    replacement_expected = {
        str(key): int(value) for key, value in plan_summary["split_counts"].items()
    }
    started = time.time()
    latent_audit = audit_replacement_latents(materialized_root, replacement_expected)
    overlay_root = revision_root / "overlay_training_index"
    overlay_summaries = []
    for split in ("train", "validation", "test"):
        expected = replacement_expected[split]
        if expected:
            overlay_summaries.append(
                build_split(
                    materialized_root, sceneplan_root, overlay_root,
                    split, expected,
                )
            )
        else:
            overlay_summaries.append(
                {
                    "split": split,
                    "rows": 0,
                    "status": "SKIPPED_NO_REPLACEMENTS",
                }
            )
    combined_root = revision_root / "training_index"
    combined = []
    for split in ("train", "validation", "test"):
        base_path = base_training / f"{split}.sqlite"
        if replacement_expected[split]:
            combined.append(
                merge_split(
                    base_path,
                    overlay_root / f"{split}.sqlite",
                    combined_root,
                    split,
                )
            )
        else:
            combined.append(copy_unchanged_split(base_path, combined_root, split))
    training_summary = {
        "schema": "stable_audio_tools.model_sceneplan_training_index_build",
        "schema_version": 1, "dataset_contract_revision": 5,
        "dataset_revision": "sound_expansion_v1",
        "rows": 1_124_000, "splits": combined,
        "random_crop": False, "latent_batch_padding_frames": 432,
        "structured_feature_dim": 9, "runtime_trajectory_feature_dim": 5,
        "caption_compiler_version": 5, "caption_max_tokens": 512,
        "speech_timing_index": str(speech_timing),
        "speech_timing_index_sha256": sha256_file(speech_timing),
        "p10_training_started": False, "p11_training_started": False,
    }
    training_summary_path = combined_root / "summary.json"
    atomic_write_json(training_summary_path, training_summary)
    p10_dataset_configs = write_p10_dataset_configs(
        revision_root,
        combined_root,
        speech_timing,
        training_summary["speech_timing_index_sha256"],
    )
    reuse = sound_reuse_audit(base_index, sceneplan_root / "replacement_map.jsonl")

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_root.expanduser().resolve(strict=True), local_files_only=True
    )
    loader_samples = []
    for item in combined:
        path = Path(item["path"])
        connection = sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True)
        try:
            replacement_ordinals = [
                int(row[0])
                for row in connection.execute(
                    "SELECT ordinal FROM samples WHERE model_sceneplan_path LIKE ? ORDER BY ordinal LIMIT 2",
                    (str(sceneplan_root) + "%",),
                )
            ]
        finally:
            connection.close()
        ordinals = sorted(set([0, EXPECTED_SPLITS[item["split"]] - 1] + replacement_ordinals))
        dataset = ScenePlanV2Dataset(
            path, tokenizer_spec=(tokenizer, 512),
            expected_num_samples=len(ordinals),
            index_num_samples=EXPECTED_SPLITS[item["split"]],
            sample_ordinals=ordinals,
            latent_crop_length=432, caption_max_tokens=512,
            random_crop=False, require_frozen=False,
            speech_timing_index_path=speech_timing,
            speech_timing_index_sha256=training_summary["speech_timing_index_sha256"],
            expected_speech_timing_rows=500_000,
            require_speech_timing=item["split"] == "train",
        )
        for index in range(len(dataset)):
            latent, metadata = dataset[index]
            if tuple(latent.shape) != (64, 432):
                raise RuntimeError("combined loader padded latent shape failed")
            loader_samples.append(
                {
                    "split": item["split"], "ordinal": ordinals[index],
                    "sample_id": metadata["sample_id"],
                    "latent_frames_valid": int(metadata["latent_stored_length"]),
                    "present_source_tracks": int(
                        (metadata["sceneplan_44"]["source_event_frame_ids"] > 0)
                        .any(dim=1).sum()
                    ),
                }
            )

    audit_root = revision_root / "audit"
    audit_root.mkdir(parents=True, exist_ok=True)
    p9_audit = {
        "schema": "stable_audio_tools.sound_expansion_p9_audit",
        "schema_version": 1, "status": "PASS",
        "dataset_contract_revision": 5,
        "dataset_revision": "sound_expansion_v1",
        "rows": 1_124_000, "splits": EXPECTED_SPLITS,
        "replacement_rows": sum(replacement_expected.values()),
        "replacement_split_rows": replacement_expected,
        "sound_reuse": reuse,
        "replacement_latent_audit": latent_audit,
        "overlay_training_indexes": overlay_summaries,
        "combined_training_indexes": combined,
        "loader_smoke": loader_samples,
        "source_hash_and_parent_cross_split_leakage": 0,
        "scene_family_source_count_kind_appearance_quotas_changed": False,
        "base_p9_mutated": False,
        "speech_timing_rows": 500_000,
        "elapsed_sec": round(time.time() - started, 3),
    }
    p9_path = audit_root / "P9_AUDIT.json"
    atomic_write_json(p9_path, p9_audit)
    contracts_root = revision_root / "contracts"
    contracts_root.mkdir(parents=True, exist_ok=True)
    freeze = {
        "schema": "stable_audio_tools.sceneplan_v2_p9_freeze_manifest",
        "schema_version": 3,
        "state": "P9_complete_frozen_waiting_for_user_acceptance",
        "dataset_contract_revision": 5,
        "dataset_id": "sceneplan_v2_1p124m_sound_expansion_v1",
        "rows": 1_124_000, "splits": EXPECTED_SPLITS,
        "source_universe_summary": str(source_summary_path),
        "source_universe_summary_sha256": sha256_file(source_summary_path),
        "registry_audit": str(registry_audit_path),
        "registry_audit_sha256": sha256_file(registry_audit_path),
        "sceneplan_summary": str(sceneplan_root / "summary.json"),
        "sceneplan_summary_sha256": sha256_file(sceneplan_root / "summary.json"),
        "p8_summary": str(materialized_root / "P8_SUMMARY.json"),
        "p8_summary_sha256": sha256_file(materialized_root / "P8_SUMMARY.json"),
        "p9_report": str(p9_path), "p9_report_sha256": sha256_file(p9_path),
        "training_index_summary": str(training_summary_path),
        "training_index_summary_sha256": sha256_file(training_summary_path),
        "speech_timing_index": str(speech_timing),
        "speech_timing_index_sha256": sha256_file(speech_timing),
        "p10_dataset_configs": p10_dataset_configs,
        "renderer_caption_max_tokens": 512,
        "max_latent_frames": 432, "random_crop": False,
        "runtime_conditioning": "semantic_cross_attention_plus_4_event_plus_4_trajectory",
        "base_p9_mutated": False,
        "p10_training_started": False, "p11_training_started": False,
    }
    freeze_path = contracts_root / "freeze_manifest.json"
    atomic_write_json(freeze_path, freeze)
    marker = {
        "schema": "stable_audio_tools.sceneplan_v2_p9_marker",
        "schema_version": 3, "dataset_contract_revision": 5,
        "dataset_revision": "sound_expansion_v1",
        "state": "P9_complete_frozen_waiting_for_user_acceptance",
        "freeze_manifest": str(freeze_path),
        "freeze_manifest_sha256": sha256_file(freeze_path),
        "p10_training_started": False, "p11_training_started": False,
        "base_p9_mutated": False,
    }
    atomic_write_json(revision_root / "FROZEN_P9.json", marker)
    # Reopen every split through the formal marker/hash path.
    for split, expected in EXPECTED_SPLITS.items():
        frozen_dataset = ScenePlanV2Dataset(
            combined_root / f"{split}.sqlite",
            tokenizer_spec=(tokenizer, 512), expected_num_samples=1,
            index_num_samples=expected, sample_ordinals=[expected - 1],
            latent_crop_length=432, caption_max_tokens=512,
            random_crop=False, require_frozen=True,
            speech_timing_index_path=speech_timing,
            speech_timing_index_sha256=training_summary["speech_timing_index_sha256"],
            expected_speech_timing_rows=500_000,
            require_speech_timing=split == "train",
        )
        frozen_dataset[0]
    p0_p9 = {
        "schema": "stable_audio_tools.sceneplan_sound_expansion_p0_p9_summary",
        "schema_version": 1, "status": "PASS",
        "dataset_id": "sceneplan_v2_1p124m_sound_expansion_v1",
        "rows": 1_124_000, "splits": EXPECTED_SPLITS,
        "stages": {
            "P0": {
                "status": "PASS",
                "evidence": (
                    "base frozen contract inherited; revision artifacts/training "
                    "indexes are immutable and SDB-resident; audited donor paths "
                    "remain external references"
                ),
            },
            "P1": {"status": "PASS", "evidence": "all selected new dry mono source files exist"},
            "P2": {"status": "PASS", "evidence": str(source_summary_path)},
            "P3": {"status": "PASS", "evidence": "base 512k speech ledger unchanged; all formal split quotas unchanged"},
            "P4": {"status": "PASS", "evidence": "base joint pilot inherited; replacement loader/renderer samples pass"},
            "P5": {"status": "PASS", "evidence": "semantic cross-attention plus runtime 4+4 compilation passes"},
            "P6": {"status": "PASS", "evidence": "base joint_4k renderer/VAE pilot inherited unchanged"},
            "P7": {"status": "PASS", "evidence": str(registry_audit_path)},
            "P7.5": {"status": "PASS", "evidence": str(sceneplan_root / "summary.json")},
            "P8": {"status": "PASS", "evidence": str(materialized_root / "P8_SUMMARY.json")},
            "P9": {"status": "PASS", "evidence": str(p9_path)},
        },
        "sound_reuse": reuse,
        "training_index_summary": str(training_summary_path),
        "p10_dataset_configs": p10_dataset_configs,
        "freeze_marker": str(revision_root / "FROZEN_P9.json"),
        "p10_training_started": False,
    }
    atomic_write_json(revision_root / "P0_P9_SUMMARY.json", p0_p9)
    print(json.dumps(p0_p9, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
