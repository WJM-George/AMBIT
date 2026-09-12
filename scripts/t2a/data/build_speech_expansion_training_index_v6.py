#!/usr/bin/env python3
"""Build the revision-6 1.640M P10 SQLite view without duplicating latents."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.t2a.data.build_model_sceneplan_training_index_v1 import (  # noqa: E402
    build_split,
    create_schema,
)


DATASET_ROOT = Path("/mnt/sdb/audio_dataset/sceneplan_v2_1p124m")
REVISION_ROOT = DATASET_ROOT / "revisions/speech_expansion_noalign_15s_v1"
BASE_INDEX = DATASET_ROOT / "revisions/sound_expansion_v1/training_index"
DEFAULT_MATERIALIZED = REVISION_ROOT / "materialized_delta"
DEFAULT_SCENEPLANS = REVISION_ROOT / "sceneplans_model_v2_delta"
DEFAULT_EVAL_MATERIALIZED = REVISION_ROOT / "materialized_eval_delta"
DEFAULT_EVAL_SCENEPLANS = REVISION_ROOT / "sceneplans_model_v2_eval_delta"
DEFAULT_DELTA_INDEX = REVISION_ROOT / "training_index_delta"
DEFAULT_EVAL_DELTA_INDEX = REVISION_ROOT / "training_index_eval_delta"
DEFAULT_OUTPUT = REVISION_ROOT / "training_index"
EXPECTED = {"train": 1_600_000, "validation": 32_000, "test": 8_000}
EXPECTED_BUCKETS = {
    "train": (1_200_000, 400_000),
    "validation": (24_000, 8_000),
    "test": (6_000, 2_000),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def source_metadata(path: Path) -> dict[str, str]:
    connection = sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True)
    try:
        if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise RuntimeError(f"source SQLite integrity failed: {path}")
        return dict(connection.execute("SELECT key,value FROM metadata"))
    finally:
        connection.close()


def merge_split(
    *,
    split: str,
    sources: list[Path],
    output_root: Path,
    expected_rows: int,
) -> dict[str, Any]:
    started = time.monotonic()
    output = output_root / f"{split}.sqlite"
    temporary = output.with_name(output.name + f".tmp.{os.getpid()}")
    temporary.unlink(missing_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(temporary)
    create_schema(connection)
    offset = 0
    source_receipts = []
    try:
        for source_index, source in enumerate(sources):
            source = source.expanduser().resolve(strict=True)
            metadata = source_metadata(source)
            if metadata.get("split") != split:
                raise RuntimeError(f"source index split mismatch: {source}")
            alias = f"source_{source_index}"
            connection.execute(f"ATTACH DATABASE ? AS {alias}", (str(source),))
            connection.execute(
                f"""
                INSERT OR IGNORE INTO latent_shards(path,sha256)
                SELECT path,sha256 FROM {alias}.latent_shards
                """
            )
            conflicts = connection.execute(
                f"""
                SELECT COUNT(*)
                FROM {alias}.latent_shards AS source
                JOIN main.latent_shards AS target ON target.path=source.path
                WHERE target.sha256 != source.sha256
                """
            ).fetchone()[0]
            if int(conflicts):
                raise RuntimeError(f"latent shard checksum conflict in {source}")
            source_rows = int(
                connection.execute(
                    f"SELECT COUNT(*) FROM {alias}.samples"
                ).fetchone()[0]
            )
            connection.execute(
                f"""
                INSERT INTO main.samples(
                    ordinal,sample_id,model_num_samples,latent_frames_valid,
                    renderer_caption_zlib,scene_plan_zlib,model_sceneplan_sha256,
                    model_sceneplan_path,model_sceneplan_row,
                    model_sceneplan_byte_offset,model_sceneplan_byte_length,
                    compiled_conditioning_ref,latent_shard_id,latent_key,
                    latent_tensor_sha256,renderer_caption_sha256
                )
                SELECT
                    source.ordinal + ?, source.sample_id,
                    source.model_num_samples,source.latent_frames_valid,
                    source.renderer_caption_zlib,source.scene_plan_zlib,
                    source.model_sceneplan_sha256,source.model_sceneplan_path,
                    source.model_sceneplan_row,source.model_sceneplan_byte_offset,
                    source.model_sceneplan_byte_length,
                    source.compiled_conditioning_ref,target_shard.id,
                    source.latent_key,source.latent_tensor_sha256,
                    source.renderer_caption_sha256
                FROM {alias}.samples AS source
                JOIN {alias}.latent_shards AS source_shard
                  ON source_shard.id=source.latent_shard_id
                JOIN main.latent_shards AS target_shard
                  ON target_shard.path=source_shard.path
                ORDER BY source.ordinal
                """,
                (offset,),
            )
            connection.commit()
            connection.execute(f"DETACH DATABASE {alias}")
            source_receipts.append(
                {
                    "path": str(source),
                    "sha256": sha256_file(source),
                    "rows": source_rows,
                    "contract_revision": metadata.get("contract_revision"),
                }
            )
            offset += source_rows
        if offset != int(expected_rows):
            raise RuntimeError(f"{split}: merged {offset}, expected {expected_rows}")
        counts = connection.execute(
            """
            SELECT COUNT(*),COUNT(DISTINCT sample_id),
                   SUM(latent_frames_valid<=432),SUM(latent_frames_valid>432),
                   MIN(latent_frames_valid),MAX(latent_frames_valid)
            FROM samples
            """
        ).fetchone()
        if int(counts[0]) != offset or int(counts[1]) != offset:
            raise RuntimeError(f"{split}: merged IDs are not unique")
        if (
            int(expected_rows) == EXPECTED[split]
            and (int(counts[2]), int(counts[3])) != EXPECTED_BUCKETS[split]
        ):
            raise RuntimeError(f"{split} length distribution changed: {counts[2:4]}")
        metadata = {
            "schema": "stable_audio_tools.sceneplan_v2_training_index",
            "schema_version": "3",
            "contract_revision": "6",
            "model_sceneplan_schema_version": "2",
            "conditioning_contract_revision": "2",
            # Revision 6 extends the ScenePlan duration envelope and schema;
            # it does not change the deterministic caption compiler.  Keep
            # this at 5 so copied revision-5 validation/test payloads and the
            # newly built delta are governed by one truthful compiler ABI.
            "caption_compiler_version": "5",
            "mode": "speech_expansion_noalign_15s_v1",
            "split": split,
            "rows": str(offset),
            "latent_shards": str(
                connection.execute("SELECT COUNT(*) FROM latent_shards").fetchone()[0]
            ),
            "latent_channels": "64",
            "max_latent_frames": "648",
            "caption_max_tokens": "512",
            "structured_feature_dim": "9",
            "runtime_trajectory_feature_dim": "5",
            "random_crop": "false",
            "speech_timing_sidecar": "none",
            "word_level_timestamp_teacher": "false",
            "frozen": "true",
            "base_revision": "sceneplan_v2_1p124m_sound_expansion_v1",
            "source_indexes_json": json.dumps(source_receipts, sort_keys=True),
        }
        connection.executemany(
            "INSERT INTO metadata(key,value) VALUES (?,?)", metadata.items()
        )
        connection.commit()
        connection.execute("ANALYZE")
        connection.commit()
        if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise RuntimeError(f"{split}: merged SQLite integrity failed")
    finally:
        connection.close()
    os.replace(temporary, output)
    reopened = sqlite3.connect(f"file:{output}?mode=ro&immutable=1", uri=True)
    try:
        if int(reopened.execute("SELECT COUNT(*) FROM samples").fetchone()[0]) != offset:
            raise RuntimeError(f"{split}: atomic reopen count changed")
        low, high, min_frames, max_frames = reopened.execute(
            """
            SELECT SUM(latent_frames_valid<=432),SUM(latent_frames_valid>432),
                   MIN(latent_frames_valid),MAX(latent_frames_valid)
            FROM samples
            """
        ).fetchone()
        latent_shards = int(
            reopened.execute("SELECT COUNT(*) FROM latent_shards").fetchone()[0]
        )
    finally:
        reopened.close()
    return {
        "split": split,
        "rows": offset,
        "length_buckets": {"432": int(low), "648": int(high)},
        "latent_frames_valid_range": [int(min_frames), int(max_frames)],
        "latent_shards": latent_shards,
        "source_indexes": source_receipts,
        "path": str(output),
        "sha256": sha256_file(output),
        "num_bytes": output.stat().st_size,
        "elapsed_sec": round(time.monotonic() - started, 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--materialized-root", type=Path, default=DEFAULT_MATERIALIZED)
    parser.add_argument("--sceneplan-root", type=Path, default=DEFAULT_SCENEPLANS)
    parser.add_argument(
        "--eval-materialized-root", type=Path, default=DEFAULT_EVAL_MATERIALIZED
    )
    parser.add_argument(
        "--eval-sceneplan-root", type=Path, default=DEFAULT_EVAL_SCENEPLANS
    )
    parser.add_argument("--delta-index-root", type=Path, default=DEFAULT_DELTA_INDEX)
    parser.add_argument(
        "--eval-delta-index-root", type=Path, default=DEFAULT_EVAL_DELTA_INDEX
    )
    parser.add_argument("--base-index-root", type=Path, default=BASE_INDEX)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    materialized = args.materialized_root.expanduser().resolve(strict=True)
    sceneplans = args.sceneplan_root.expanduser().resolve(strict=True)
    ready = REVISION_ROOT / "EVAL_READY_FOR_MERGE.json"
    wait_started = time.monotonic()
    while not ready.is_file():
        failures = sorted((REVISION_ROOT / "eval_stage_state").glob("*.failed"))
        if failures:
            raise RuntimeError(
                "validation/test expansion failed before index merge: "
                + ", ".join(str(path) for path in failures)
            )
        print(
            json.dumps(
                {
                    "state": "waiting_for_validation_test_expansion",
                    "required": str(ready),
                    "elapsed_sec": round(time.monotonic() - wait_started, 1),
                }
            ),
            flush=True,
        )
        time.sleep(30)
    eval_ready = json.loads(ready.read_text(encoding="utf-8"))
    if (
        eval_ready.get("ok") is not True
        or eval_ready.get("ready_for_training_index_merge") is not True
        or eval_ready.get("split_counts") != {"validation": 12_000, "test": 4_000}
    ):
        raise RuntimeError("validation/test expansion marker is not an all-pass gate")
    eval_materialized = args.eval_materialized_root.expanduser().resolve(strict=True)
    eval_sceneplans = args.eval_sceneplan_root.expanduser().resolve(strict=True)
    delta_root = args.delta_index_root.expanduser().resolve(strict=False)
    eval_delta_root = args.eval_delta_index_root.expanduser().resolve(strict=False)
    base_root = args.base_index_root.expanduser().resolve(strict=True)
    output = args.output_root.expanduser().resolve(strict=False)
    for path in (
        materialized,
        sceneplans,
        eval_materialized,
        eval_sceneplans,
        delta_root,
        eval_delta_root,
        base_root,
        output,
    ):
        if not str(path).startswith("/mnt/sdb/audio_dataset/"):
            raise ValueError(f"training-index artifact must remain on SDB: {path}")

    delta = build_split(
        materialized,
        sceneplans,
        delta_root,
        "train",
        500_000,
        contract_revision=6,
        model_sceneplan_schema_version=2,
        max_latent_frames=648,
        mode="speech_expansion_delta",
    )
    eval_deltas = {
        split: build_split(
            eval_materialized,
            eval_sceneplans,
            eval_delta_root,
            split,
            rows,
            contract_revision=6,
            model_sceneplan_schema_version=2,
            max_latent_frames=648,
            mode="speech_expansion_eval_delta",
        )
        for split, rows in {"validation": 12_000, "test": 4_000}.items()
    }
    summaries = [
        merge_split(
            split="train",
            sources=[base_root / "train.sqlite", Path(delta["path"])],
            output_root=output,
            expected_rows=EXPECTED["train"],
        ),
        merge_split(
            split="validation",
            sources=[
                base_root / "validation.sqlite",
                Path(eval_deltas["validation"]["path"]),
            ],
            output_root=output,
            expected_rows=EXPECTED["validation"],
        ),
        merge_split(
            split="test",
            sources=[
                base_root / "test.sqlite",
                Path(eval_deltas["test"]["path"]),
            ],
            output_root=output,
            expected_rows=EXPECTED["test"],
        ),
    ]
    summary = {
        "schema": "stable_audio_tools.sceneplan_speech_expansion_training_index",
        "schema_version": 1,
        "dataset_contract_revision": 6,
        "state": "complete",
        "rows": sum(row["rows"] for row in summaries),
        "splits": summaries,
        "train_length_distribution": {"432": 1_200_000, "648": 400_000},
        "validation_length_distribution": {"432": 24_000, "648": 8_000},
        "test_length_distribution": {"432": 6_000, "648": 2_000},
        "latent_batch_padding_frames": [432, 648],
        "random_crop": False,
        "speech_timing_sidecar": None,
        "word_level_timestamp_teacher": False,
        "p10_training_started": False,
    }
    atomic_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
