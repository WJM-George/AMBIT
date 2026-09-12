#!/usr/bin/env python3
"""Build frozen SQLite split indexes for revision-5 ScenePlan DiT loading."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import time
import zlib
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


SCRIPT_DIR = Path(__file__).resolve().parent
import sys

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from materialize_model_sceneplan_v1_shard import canonical_json, sha256_text  # noqa: E402
from sceneplan_v2_common import DATASET_ROOT, atomic_write_json, require_dataset_not_frozen  # noqa: E402
from stable_audio_tools.data.model_sceneplan import (  # noqa: E402
    compile_model_renderer_caption,
    validate_model_sceneplan,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl_with_offsets(path: Path) -> list[tuple[dict[str, Any], str, int, int]]:
    rows: list[tuple[dict[str, Any], str, int, int]] = []
    offset = 0
    with path.open("rb") as handle:
        for row_index, blob in enumerate(handle):
            length = len(blob)
            if not blob.endswith(b"\n"):
                raise RuntimeError(f"{path}:{row_index + 1}: missing JSONL newline")
            text = blob[:-1].decode("utf-8")
            value = json.loads(text)
            if canonical_json(value) != text:
                raise RuntimeError(f"{path}:{row_index + 1}: JSON is not canonical")
            rows.append((value, text, offset, length))
            offset += length
    return rows


def create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA page_size=65536;
        PRAGMA journal_mode=OFF;
        PRAGMA synchronous=OFF;
        PRAGMA temp_store=MEMORY;
        PRAGMA locking_mode=EXCLUSIVE;
        CREATE TABLE metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        ) WITHOUT ROWID;
        CREATE TABLE latent_shards (
            id INTEGER PRIMARY KEY,
            path TEXT NOT NULL UNIQUE,
            sha256 TEXT NOT NULL
        );
        CREATE TABLE samples (
            ordinal INTEGER PRIMARY KEY,
            sample_id TEXT NOT NULL UNIQUE,
            model_num_samples INTEGER NOT NULL,
            latent_frames_valid INTEGER NOT NULL,
            renderer_caption_zlib BLOB NOT NULL,
            scene_plan_zlib BLOB NOT NULL,
            model_sceneplan_sha256 TEXT NOT NULL,
            model_sceneplan_path TEXT NOT NULL,
            model_sceneplan_row INTEGER NOT NULL,
            model_sceneplan_byte_offset INTEGER NOT NULL,
            model_sceneplan_byte_length INTEGER NOT NULL,
            compiled_conditioning_ref TEXT NOT NULL,
            latent_shard_id INTEGER NOT NULL,
            latent_key TEXT NOT NULL,
            latent_tensor_sha256 TEXT NOT NULL,
            renderer_caption_sha256 TEXT NOT NULL,
            FOREIGN KEY(latent_shard_id) REFERENCES latent_shards(id)
        );
        """
    )


def build_split(
    materialized_root: Path,
    sceneplan_root: Path,
    output_root: Path,
    split: str,
    expected_rows: int,
    *,
    contract_revision: int = 5,
    model_sceneplan_schema_version: int = 1,
    max_latent_frames: int = 432,
    mode: str = "full",
) -> dict[str, Any]:
    manifests = sorted(
        (materialized_root / "manifests" / split).glob(
            f"materialized-{split}-*.parquet"
        )
    )
    if not manifests:
        raise RuntimeError(f"no materialized manifests for split {split}")
    output = output_root / f"{split}.sqlite"
    temporary = output.with_name(output.name + f".tmp.{os.getpid()}")
    temporary.unlink(missing_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(temporary)
    create_schema(connection)
    latent_ids: dict[str, tuple[int, str]] = {}
    ordinal = 0
    started = time.time()
    try:
        for manifest_index, manifest in enumerate(manifests, start=1):
            rows = pq.read_table(manifest).to_pylist()
            if not rows:
                raise RuntimeError(f"empty materialized manifest: {manifest}")
            shard = int(rows[0]["work_shard"])
            model_path = (
                sceneplan_root
                / split
                / f"model-sceneplans-{split}-{shard:05d}.jsonl"
            )
            recipe_path = (
                sceneplan_root / split / f"render-recipes-{split}-{shard:05d}.jsonl"
            )
            conditioning_path = (
                sceneplan_root / split / f"conditioning-{split}-{shard:05d}.jsonl"
            )
            model_rows = read_jsonl_with_offsets(model_path)
            recipe_rows = read_jsonl_with_offsets(recipe_path)
            conditioning_rows = read_jsonl_with_offsets(conditioning_path)
            if not (
                len(rows)
                == len(model_rows)
                == len(recipe_rows)
                == len(conditioning_rows)
            ):
                raise RuntimeError(f"{manifest}: P7.5/P8 shard row count mismatch")
            rows.sort(key=lambda row: int(row["row_in_shard"]))
            for expected_row_index, row in enumerate(rows):
                if int(row["row_in_shard"]) != expected_row_index:
                    raise RuntimeError(f"{manifest}: row_in_shard is not contiguous")
                model, model_text, model_offset, model_length = model_rows[expected_row_index]
                recipe, recipe_text, _, _ = recipe_rows[expected_row_index]
                conditioning, conditioning_text, conditioning_offset, conditioning_length = (
                    conditioning_rows[expected_row_index]
                )
                sample_id = str(row["sample_id"])
                if not (
                    model.get("sample_id")
                    == recipe.get("sample_id")
                    == conditioning.get("sample_id")
                    == sample_id
                ):
                    raise RuntimeError(f"{sample_id}: three-view/materialized id mismatch")
                validate_model_sceneplan(model)
                caption = conditioning["renderer_caption"]
                if caption != compile_model_renderer_caption(model):
                    raise RuntimeError(f"{sample_id}: conditioning compiler drift")
                model_sha = sha256_text(model_text)
                recipe_sha = sha256_text(recipe_text)
                caption_sha = sha256_text(canonical_json(caption))
                if (
                    row["model_sceneplan_sha256"] != model_sha
                    or row["render_recipe_sha256"] != recipe_sha
                    or row["renderer_caption_sha256"] != caption_sha
                    or recipe["model_sceneplan_sha256"] != model_sha
                ):
                    raise RuntimeError(f"{sample_id}: frozen three-view hash mismatch")
                latent_path, separator, latent_key = str(row["latent_ref"]).partition("#")
                if not separator or latent_key != sample_id:
                    raise RuntimeError(f"{sample_id}: invalid latent reference")
                shard_sha = str(row["latent_shard_sha256"])
                existing = latent_ids.get(latent_path)
                if existing is None:
                    shard_id = len(latent_ids) + 1
                    latent_ids[latent_path] = (shard_id, shard_sha)
                    connection.execute(
                        "INSERT INTO latent_shards(id, path, sha256) VALUES (?, ?, ?)",
                        (shard_id, latent_path, shard_sha),
                    )
                else:
                    shard_id, previous_sha = existing
                    if previous_sha != shard_sha:
                        raise RuntimeError(f"latent shard checksum conflict: {latent_path}")
                conditioning_ref = (
                    f"{conditioning_path}#{conditioning_offset}:{conditioning_length}"
                )
                # ``scene_plan_zlib`` retains its historical column name so a
                # single fail-closed loader can read revision 4 and revision 5;
                # its revision-5 payload is the compact model ScenePlan only.
                connection.execute(
                    """
                    INSERT INTO samples(
                        ordinal, sample_id, model_num_samples,
                        latent_frames_valid, renderer_caption_zlib,
                        scene_plan_zlib, model_sceneplan_sha256,
                        model_sceneplan_path, model_sceneplan_row,
                        model_sceneplan_byte_offset,
                        model_sceneplan_byte_length, compiled_conditioning_ref,
                        latent_shard_id, latent_key, latent_tensor_sha256,
                        renderer_caption_sha256
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        ordinal,
                        sample_id,
                        int(row["model_num_samples"]),
                        int(row["latent_frames_valid"]),
                        sqlite3.Binary(
                            zlib.compress(canonical_json(caption).encode("utf-8"), 3)
                        ),
                        sqlite3.Binary(
                            zlib.compress(canonical_json(model).encode("utf-8"), 3)
                        ),
                        model_sha,
                        str(model_path),
                        expected_row_index,
                        model_offset,
                        model_length,
                        conditioning_ref,
                        shard_id,
                        latent_key,
                        str(row["latent_tensor_sha256"]),
                        caption_sha,
                    ),
                )
                ordinal += 1
            if manifest_index % 16 == 0:
                connection.commit()
            if manifest_index % 50 == 0 or manifest_index == len(manifests):
                print(
                    json.dumps(
                        {
                            "split": split,
                            "indexed_manifests": manifest_index,
                            "total_manifests": len(manifests),
                            "rows": ordinal,
                            "elapsed_sec": round(time.time() - started, 1),
                        }
                    ),
                    flush=True,
                )
        if ordinal != int(expected_rows):
            raise RuntimeError(f"{split} training-index rows {ordinal} != {expected_rows}")
        metadata = {
            "schema": "stable_audio_tools.sceneplan_v2_training_index",
            "schema_version": "3",
            "contract_revision": str(int(contract_revision)),
            "model_sceneplan_schema_version": str(
                int(model_sceneplan_schema_version)
            ),
            "conditioning_contract_revision": "2",
            "caption_compiler_version": "5",
            "mode": str(mode),
            "split": split,
            "rows": str(ordinal),
            "latent_shards": str(len(latent_ids)),
            "latent_channels": "64",
            "max_latent_frames": str(int(max_latent_frames)),
            "caption_max_tokens": "512",
            "structured_feature_dim": "9",
            "random_crop": "false",
            "frozen": "true",
            "materialized_root": str(materialized_root),
            "sceneplan_root": str(sceneplan_root),
        }
        connection.executemany(
            "INSERT INTO metadata(key, value) VALUES (?, ?)", metadata.items()
        )
        connection.commit()
        connection.execute("ANALYZE")
        connection.commit()
        if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise RuntimeError(f"SQLite integrity check failed for {split}")
        count, unique = connection.execute(
            "SELECT COUNT(*), COUNT(DISTINCT sample_id) FROM samples"
        ).fetchone()
        if int(count) != ordinal or int(unique) != ordinal:
            raise RuntimeError(f"SQLite row/uniqueness check failed for {split}")
    finally:
        connection.close()
    os.replace(temporary, output)
    reopened = sqlite3.connect(f"file:{output}?mode=ro&immutable=1", uri=True)
    try:
        if reopened.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise RuntimeError(f"atomic SQLite reopen failed for {split}")
        if int(reopened.execute("SELECT COUNT(*) FROM samples").fetchone()[0]) != ordinal:
            raise RuntimeError(f"atomic SQLite row count changed for {split}")
    finally:
        reopened.close()
    return {
        "split": split,
        "rows": ordinal,
        "manifests": len(manifests),
        "latent_shards": len(latent_ids),
        "path": str(output),
        "num_bytes": output.stat().st_size,
        "sha256": sha256_file(output),
        "elapsed_sec": round(time.time() - started, 3),
    }


def main() -> int:
    require_dataset_not_frozen()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--materialized-root", type=Path, default=DATASET_ROOT / "materialized")
    parser.add_argument(
        "--sceneplan-root", type=Path, default=DATASET_ROOT / "sceneplans_model_v1"
    )
    parser.add_argument("--output-root", type=Path, default=DATASET_ROOT / "training_index")
    args = parser.parse_args()
    materialized_root = args.materialized_root.expanduser().resolve(strict=True)
    sceneplan_root = args.sceneplan_root.expanduser().resolve(strict=True)
    output_root = args.output_root.expanduser().resolve(strict=False)
    try:
        materialized_root.relative_to(os.environ.get("AMBIT_DATA_ROOT", "data"))
        sceneplan_root.relative_to(os.environ.get("AMBIT_DATA_ROOT", "data"))
        output_root.relative_to(os.environ.get("AMBIT_DATA_ROOT", "data"))
    except ValueError as error:
        raise ValueError("P9 inputs and training indexes must be on SDB") from error
    expected = {"train": 1_100_000, "validation": 20_000, "test": 4_000}
    summaries = [
        build_split(materialized_root, sceneplan_root, output_root, split, rows)
        for split, rows in expected.items()
    ]
    summary = {
        "schema": "stable_audio_tools.model_sceneplan_training_index_build",
        "schema_version": 1,
        "dataset_contract_revision": 5,
        "rows": sum(item["rows"] for item in summaries),
        "splits": summaries,
        "random_crop": False,
        "latent_batch_padding_frames": 432,
        "structured_feature_dim": 9,
        "caption_compiler_version": 5,
        "caption_max_tokens": 512,
        "p10_training_started": False,
        "p11_training_started": False,
    }
    atomic_write_json(output_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
