#!/usr/bin/env python3
"""Merge and freeze the five Editing M2D-CLAP cache shards."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stable_audio_tools.data.sceneplan_transfusion_editing_index import (  # noqa: E402
    sha256_file,
)
from stable_audio_tools.data.sceneplan_transfusion_editing_m2d_clap import (  # noqa: E402
    EDITING_M2D_AUDIO_PREPROCESS,
    EDITING_M2D_CACHE_SCHEMA,
    EDITING_M2D_CACHE_SCHEMA_VERSION,
    EDITING_M2D_CACHE_STATE,
    EDITING_M2D_CAPTION_TARGET,
    EDITING_M2D_VAE_CHECKPOINT_SHA256,
    EDITING_M2D_VAE_CONFIG_SHA256,
    ScenePlanTransfusionEditingM2DCLAPCache,
    validate_editing_m2d_temporal_pilot,
)
from stable_audio_tools.models.sceneplan_transfusion_editing_m2d_clap import (  # noqa: E402
    EDITING_M2D_CLAP_CONTRACT,
    EDITING_M2D_CLAP_EMBED_DIM,
)
from stable_audio_tools.models.sceneplan_transfusion_editing_m2d_runtime import (  # noqa: E402
    M2D_CLAP_SOURCE_AUDIO_VIEW,
    M2D_CLAP_TEMPORAL_POLICY,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs=5, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-index", type=Path, required=True)
    parser.add_argument("--source-index-sha256", required=True)
    parser.add_argument("--expected-index-rows", type=int, required=True)
    parser.add_argument("--expected-cache-rows", type=int, required=True)
    parser.add_argument(
        "--split", choices=("train", "validation", "test"), required=True
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def _shard_summary(
    path: Path,
    *,
    source_index: Path,
    source_index_sha256: str,
    expected_index_rows: int,
    expected_cache_rows: int,
    split: str,
) -> tuple[dict[str, str], dict[str, Any]]:
    path = path.expanduser().resolve(strict=True)
    marker_path = path.with_suffix(path.suffix + ".shard.json")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    observed_sha = sha256_file(path)
    connection = sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True)
    try:
        metadata = dict(connection.execute("SELECT key,value FROM metadata"))
        count, distinct = connection.execute(
            "SELECT COUNT(*),COUNT(DISTINCT pair_id) FROM features"
        ).fetchone()
    finally:
        connection.close()
    validate_editing_m2d_temporal_pilot(
        metadata.get("temporal_pilot", ""),
        expected_sha256=metadata.get("temporal_pilot_sha256"),
    )
    expected_metadata = {
        "schema": EDITING_M2D_CACHE_SCHEMA,
        "schema_version": EDITING_M2D_CACHE_SCHEMA_VERSION,
        "state": "shard_complete",
        "split": str(split),
        "full_selection_rows": str(int(expected_cache_rows)),
        "source_index": str(source_index),
        "source_index_sha256": str(source_index_sha256),
        "source_index_rows": str(int(expected_index_rows)),
        "dimension": str(EDITING_M2D_CLAP_EMBED_DIM),
        "dtype": "float16",
        "semantic_contract": EDITING_M2D_CLAP_CONTRACT,
        "audio_preprocess": EDITING_M2D_AUDIO_PREPROCESS,
        "source_audio_view": M2D_CLAP_SOURCE_AUDIO_VIEW,
        "temporal_policy": M2D_CLAP_TEMPORAL_POLICY,
        "caption_target": EDITING_M2D_CAPTION_TARGET,
        "vae_config_sha256": EDITING_M2D_VAE_CONFIG_SHA256,
        "vae_checkpoint_sha256": EDITING_M2D_VAE_CHECKPOINT_SHA256,
        "old_sceneplan_model_input": "false",
        "caption_model_input": "false",
        "target_information_used": "false",
        "num_shards": "5",
        "selection": "pair_ordinal_lt_N_then_modulo_5_v1",
    }
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            raise RuntimeError(f"M2D shard {path.name} metadata {key} changed")
    shard_index = int(metadata.get("shard_index", -1))
    expected_rows = len(range(shard_index, int(expected_cache_rows), 5))
    if (
        shard_index not in range(5)
        or int(count) != expected_rows
        or int(distinct) != expected_rows
        or marker.get("schema")
        != "sceneplan_transfusion_editing_m2d_clap_cache_shard"
        or str(marker.get("schema_version", ""))
        != EDITING_M2D_CACHE_SCHEMA_VERSION
        or marker.get("state") != "shard_complete"
        or Path(str(marker.get("path") or "")).resolve() != path
        or marker.get("sha256") != observed_sha
        or int(marker.get("rows", -1)) != expected_rows
        or int(marker.get("shard_index", -1)) != shard_index
        or int(marker.get("physical_gpu", -1)) != 3 + shard_index
    ):
        raise RuntimeError(f"M2D shard {path.name} identity/count changed")
    return metadata, {
        "path": str(path),
        "sha256": observed_sha,
        "marker": str(marker_path.resolve()),
        "marker_sha256": sha256_file(marker_path),
        "rows": expected_rows,
        "shard_index": shard_index,
        "physical_gpu": 3 + shard_index,
    }


def main() -> int:
    args = _parse_args()
    torch.set_float32_matmul_precision("high")
    if (
        int(args.expected_index_rows) <= 0
        or not 2 <= int(args.expected_cache_rows) <= int(args.expected_index_rows)
    ):
        raise ValueError("Editing M2D merge row counts are invalid")
    source_index = args.source_index.expanduser().resolve(strict=True)
    if sha256_file(source_index) != str(args.source_index_sha256):
        raise RuntimeError("Editing M2D merge source-index SHA256 changed")
    summaries = []
    shard_metadata = []
    for path in args.inputs:
        metadata, summary = _shard_summary(
            path,
            source_index=source_index,
            source_index_sha256=str(args.source_index_sha256),
            expected_index_rows=int(args.expected_index_rows),
            expected_cache_rows=int(args.expected_cache_rows),
            split=args.split,
        )
        shard_metadata.append(metadata)
        summaries.append(summary)
    summaries.sort(key=lambda value: int(value["shard_index"]))
    if [int(value["shard_index"]) for value in summaries] != list(range(5)):
        raise RuntimeError("Editing M2D merge requires exactly shards 0--4")
    immutable_keys = {
        "vae_config",
        "vae_config_sha256",
        "vae_checkpoint",
        "vae_checkpoint_sha256",
        "temporal_pilot",
        "temporal_pilot_sha256",
        "m2d_assets_json",
        "implementation_sha256_json",
        "numeric_runtime_fingerprint_json",
        "builder",
        "builder_sha256",
        "vae_batch_size",
        "audio_m2d_batch_size",
    }
    reference = shard_metadata[0]
    if any(
        any(metadata.get(key) != reference.get(key) for key in immutable_keys)
        for metadata in shard_metadata[1:]
    ):
        raise RuntimeError("Editing M2D shards used different frozen encoders")

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not args.overwrite:
        raise FileExistsError(output)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".sqlite", dir=output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    destination = sqlite3.connect(temporary)
    completed = False
    try:
        destination.executescript(
            """
            PRAGMA journal_mode=OFF;
            PRAGMA synchronous=OFF;
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE features (
                pair_ordinal INTEGER PRIMARY KEY,
                pair_id TEXT NOT NULL UNIQUE,
                source_sample_id TEXT NOT NULL,
                source_latent_tensor_sha256 TEXT NOT NULL,
                source_caption_sha256 TEXT NOT NULL,
                audio_embedding BLOB NOT NULL,
                text_embedding BLOB NOT NULL,
                audio_embedding_sha256 TEXT NOT NULL,
                text_embedding_sha256 TEXT NOT NULL,
                audio_norm_before_l2 REAL NOT NULL,
                text_norm_before_l2 REAL NOT NULL,
                record_sha256 TEXT NOT NULL
            );
            """
        )
        columns = (
            "pair_ordinal,pair_id,source_sample_id,"
            "source_latent_tensor_sha256,source_caption_sha256,"
            "audio_embedding,text_embedding,audio_embedding_sha256,"
            "text_embedding_sha256,audio_norm_before_l2,"
            "text_norm_before_l2,record_sha256"
        )
        for summary in summaries:
            source = sqlite3.connect(
                f"file:{summary['path']}?mode=ro&immutable=1", uri=True
            )
            try:
                cursor = source.execute(
                    f"SELECT {columns} FROM features ORDER BY pair_ordinal"
                )
                while True:
                    rows = cursor.fetchmany(1000)
                    if not rows:
                        break
                    destination.executemany(
                        "INSERT INTO features VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                        rows,
                    )
            finally:
                source.close()
            destination.commit()
        count, minimum, maximum, distinct_ordinals, distinct_ids = destination.execute(
            "SELECT COUNT(*),MIN(pair_ordinal),MAX(pair_ordinal),"
            "COUNT(DISTINCT pair_ordinal),COUNT(DISTINCT pair_id) FROM features"
        ).fetchone()
        expected_rows = int(args.expected_cache_rows)
        if (
            int(count) != expected_rows
            or int(minimum) != 0
            or int(maximum) != expected_rows - 1
            or int(distinct_ordinals) != expected_rows
            or int(distinct_ids) != expected_rows
        ):
            raise RuntimeError("merged Editing M2D cache is not dense and complete")
        metadata = {
            "schema": EDITING_M2D_CACHE_SCHEMA,
            "schema_version": EDITING_M2D_CACHE_SCHEMA_VERSION,
            "state": EDITING_M2D_CACHE_STATE,
            "split": str(args.split),
            "rows": str(expected_rows),
            "source_index": str(source_index),
            "source_index_sha256": str(args.source_index_sha256),
            "source_index_rows": str(int(args.expected_index_rows)),
            "dimension": str(EDITING_M2D_CLAP_EMBED_DIM),
            "dtype": "float16",
            "semantic_contract": EDITING_M2D_CLAP_CONTRACT,
            "audio_preprocess": EDITING_M2D_AUDIO_PREPROCESS,
            "source_audio_view": M2D_CLAP_SOURCE_AUDIO_VIEW,
            "temporal_policy": M2D_CLAP_TEMPORAL_POLICY,
            "temporal_pilot": reference["temporal_pilot"],
            "temporal_pilot_sha256": reference["temporal_pilot_sha256"],
            "caption_target": EDITING_M2D_CAPTION_TARGET,
            "old_sceneplan_model_input": "false",
            "caption_model_input": "false",
            "target_information_used": "false",
            "offline_source_plan_used_for_caption_label": "true",
            "vae_config": reference["vae_config"],
            "vae_config_sha256": reference["vae_config_sha256"],
            "vae_checkpoint": reference["vae_checkpoint"],
            "vae_checkpoint_sha256": reference["vae_checkpoint_sha256"],
            "m2d_assets_json": reference["m2d_assets_json"],
            "implementation_sha256_json": reference[
                "implementation_sha256_json"
            ],
            "numeric_runtime_fingerprint_json": reference[
                "numeric_runtime_fingerprint_json"
            ],
            "shard_builder": reference["builder"],
            "shard_builder_sha256": reference["builder_sha256"],
            "merger": str(Path(__file__).resolve()),
            "merger_sha256": sha256_file(Path(__file__).resolve()),
            "selection": "pair_ordinal_lt_N_then_modulo_5_v1",
            "num_shards": "5",
            "shards_json": json.dumps(summaries, sort_keys=True),
        }
        destination.executemany(
            "INSERT INTO metadata(key,value) VALUES (?,?)", metadata.items()
        )
        destination.commit()
        destination.execute("VACUUM")
        destination.commit()
        completed = True
    finally:
        destination.close()
        if not completed:
            temporary.unlink(missing_ok=True)
    if output.exists():
        output.unlink()
    os.replace(temporary, output)
    cache_sha = sha256_file(output)
    marker = {
        "schema": EDITING_M2D_CACHE_SCHEMA,
        "schema_version": int(EDITING_M2D_CACHE_SCHEMA_VERSION),
        "state": EDITING_M2D_CACHE_STATE,
        "cache_path": str(output),
        "cache_sha256": cache_sha,
        "rows": int(args.expected_cache_rows),
        "split": str(args.split),
        "source_index": str(source_index),
        "source_index_sha256": str(args.source_index_sha256),
        "physical_gpus": [3, 4, 5, 6, 7],
        "shards": summaries,
    }
    marker_path = output.with_suffix(output.suffix + ".frozen.json")
    _atomic_json(marker_path, marker)
    # Reopen representative boundaries and validate every cache-level contract.
    try:
        cache = ScenePlanTransfusionEditingM2DCLAPCache(
            output,
            source_index=source_index,
            source_index_sha256=str(args.source_index_sha256),
            expected_rows=int(args.expected_cache_rows),
            expected_split=args.split,
            expected_cache_sha256=cache_sha,
            verify_cache_file_hash=True,
        )
    except Exception:
        # Never leave a terminal-looking frozen marker for a publication that
        # failed its independent reopen gate.
        marker_path.unlink(missing_ok=True)
        raise
    if len(cache) != int(args.expected_cache_rows):
        raise RuntimeError("frozen Editing M2D cache reopen changed row count")
    result = {**marker, "marker": str(marker_path.resolve())}
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
