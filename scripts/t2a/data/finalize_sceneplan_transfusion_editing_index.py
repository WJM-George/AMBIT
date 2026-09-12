#!/usr/bin/env python3
"""Freeze planned Editing pairs plus target manifests into a training index."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import sys
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import torch
from safetensors import safe_open


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stable_audio_tools.data.sceneplan_transfusion_editing import (  # noqa: E402
    canonical_json,
    sha256_json,
)
from stable_audio_tools.data.sceneplan_transfusion_editing_index import (  # noqa: E402
    sha256_file,
)


FINAL_INDEX_SCHEMA = "sceneplan_transfusion_editing_training_index"
FINAL_INDEX_SCHEMA_VERSION = 1
MATERIALIZATION_CONTRACT = "sceneplan_transfusion_editing_materialized_target_v1"


def _tensor_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()


def _manifest_path(target_root: Path, split: str, shard: int) -> Path:
    return (
        target_root
        / "materialized/manifests"
        / split
        / f"materialized-{split}-{shard:05d}.parquet"
    )


def _load_planned_metadata(path: Path) -> tuple[dict[str, str], int, str]:
    uri = f"file:{path}?mode=ro&immutable=1"
    connection = sqlite3.connect(uri, uri=True)
    try:
        metadata = dict(connection.execute("SELECT key,value FROM metadata"))
        rows = int(connection.execute("SELECT COUNT(*) FROM pairs").fetchone()[0])
        split_values = connection.execute(
            "SELECT DISTINCT split FROM pairs"
        ).fetchall()
    finally:
        connection.close()
    if (
        metadata.get("schema") != "sceneplan_transfusion_editing_pair_index"
        or metadata.get("state") != "planned_targets_not_materialized"
        or int(metadata.get("rows", -1)) != rows
    ):
        raise RuntimeError("input is not a complete planned Editing pair index")
    if len(split_values) != 1 or str(split_values[0][0]) != metadata.get("split"):
        raise RuntimeError("planned Editing index split metadata changed")
    return metadata, rows, str(split_values[0][0])


def finalize(planned_path: Path, output_path: Path, *, replace: bool) -> dict[str, Any]:
    planned_path = planned_path.expanduser().resolve(strict=True)
    metadata, expected_rows, split = _load_planned_metadata(planned_path)
    target_root = Path(metadata["target_root"]).resolve(strict=True)
    output_path = output_path.expanduser().resolve()
    if output_path.exists() and not replace:
        raise FileExistsError(f"refusing to replace frozen index: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + f".tmp.{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(temporary)
    shutil.copyfile(planned_path, temporary)
    connection = sqlite3.connect(temporary)
    manifest_inventory = []
    try:
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.executescript(
            """
            ALTER TABLE pairs ADD COLUMN target_materialized_manifest_path TEXT;
            ALTER TABLE pairs ADD COLUMN target_materialized_manifest_sha256 TEXT;
            ALTER TABLE pairs ADD COLUMN materialized_record_sha256 TEXT;
            CREATE TABLE target_shards(
                work_shard INTEGER PRIMARY KEY,
                rows INTEGER NOT NULL,
                latent_path TEXT NOT NULL UNIQUE,
                latent_shard_sha256 TEXT NOT NULL,
                manifest_path TEXT NOT NULL UNIQUE,
                manifest_sha256 TEXT NOT NULL
            );
            """
        )
        planned_by_shard = {
            int(shard): int(count)
            for shard, count in connection.execute(
                "SELECT work_shard,COUNT(*) FROM pairs GROUP BY work_shard"
            )
        }
        updated = 0
        for shard, expected_shard_rows in sorted(planned_by_shard.items()):
            manifest_path = _manifest_path(target_root, split, shard).resolve(
                strict=True
            )
            manifest_sha = sha256_file(manifest_path)
            table = pq.read_table(manifest_path)
            rows = table.to_pylist()
            if len(rows) != expected_shard_rows:
                raise RuntimeError(
                    f"work shard {shard} manifest rows differ: "
                    f"{len(rows)} != {expected_shard_rows}"
                )
            by_pair = {str(row["pair_id"]): row for row in rows}
            if len(by_pair) != len(rows):
                raise RuntimeError(f"work shard {shard} repeats a pair id")
            planned_rows = list(
                connection.execute(
                    "SELECT * FROM pairs WHERE work_shard=? ORDER BY row_in_shard",
                    (shard,),
                )
            )
            columns = [item[1] for item in connection.execute("PRAGMA table_info(pairs)")]
            latent_paths = {str(row["target_latent_ref"]).rsplit("#", 1)[0] for row in rows}
            latent_shas = {str(row["target_latent_shard_sha256"]) for row in rows}
            if len(latent_paths) != 1 or len(latent_shas) != 1:
                raise RuntimeError(f"work shard {shard} has multiple latent artifacts")
            latent_path = Path(next(iter(latent_paths))).resolve(strict=True)
            latent_sha = next(iter(latent_shas))
            if sha256_file(latent_path) != latent_sha:
                raise RuntimeError(f"work shard {shard} latent-shard SHA256 changed")
            expected_keys = {str(dict(zip(columns, row))["target_sample_id"]) for row in planned_rows}
            with safe_open(str(latent_path), framework="pt", device="cpu") as handle:
                if set(handle.keys()) != expected_keys:
                    raise RuntimeError(f"work shard {shard} target latent keys differ")
                for raw in planned_rows:
                    pair = dict(zip(columns, raw))
                    target_id = str(pair["target_sample_id"])
                    manifest = by_pair.get(str(pair["pair_id"]))
                    if manifest is None:
                        raise RuntimeError(f"{pair['pair_id']}: target manifest row absent")
                    required_equal = {
                        "pair_ordinal": int(pair["pair_ordinal"]),
                        "pair_id": str(pair["pair_id"]),
                        "split": split,
                        "work_shard": shard,
                        "row_in_shard": int(pair["row_in_shard"]),
                        "source_sample_id": str(pair["source_sample_id"]),
                        "target_sample_id": target_id,
                        "operation_family": str(pair["operation_family"]),
                        "operation": str(pair["operation"]),
                        "model_num_samples": int(pair["model_num_samples"]),
                        "latent_frames_valid": int(pair["latent_frames_valid"]),
                        "source_latent_ref": str(pair["source_latent_ref"]),
                        "source_latent_tensor_sha256": str(
                            pair["source_latent_tensor_sha256"]
                        ),
                        "source_foa_sha256": str(pair["source_foa_sha256"]),
                        "pair_record_sha256": str(pair["pair_record_sha256"]),
                    }
                    for key, expected in required_equal.items():
                        observed = manifest[key]
                        if observed != expected:
                            raise RuntimeError(
                                f"{pair['pair_id']}: manifest {key} differs: "
                                f"{observed!r} != {expected!r}"
                            )
                    target_ref = str(manifest["target_latent_ref"])
                    if target_ref != str(pair["target_latent_ref"]):
                        raise RuntimeError(f"{pair['pair_id']}: planned target ref changed")
                    target = handle.get_tensor(target_id).clone()
                    if target.dtype != torch.float16 or tuple(target.shape) != (
                        64,
                        int(pair["latent_frames_valid"]),
                    ):
                        raise RuntimeError(f"{pair['pair_id']}: target latent geometry changed")
                    if not torch.isfinite(target).all():
                        raise RuntimeError(f"{pair['pair_id']}: target latent is non-finite")
                    target_tensor_sha = _tensor_sha256(target)
                    if target_tensor_sha != str(
                        manifest["target_latent_tensor_sha256"]
                    ):
                        raise RuntimeError(f"{pair['pair_id']}: target tensor SHA changed")
                    render_text = str(manifest["target_render_result_json"])
                    render_result = json.loads(render_text)
                    if (
                        render_result.get("schema") != MATERIALIZATION_CONTRACT
                        or render_result.get("status") != "ok"
                        or sha256_json(render_result)
                        != str(manifest["target_render_result_sha256"])
                    ):
                        raise RuntimeError(f"{pair['pair_id']}: target render result invalid")
                    target_foa_path = manifest.get("target_foa_path")
                    if target_foa_path is not None:
                        if sha256_file(target_foa_path) != str(
                            manifest["target_foa_sha256"]
                        ):
                            raise RuntimeError(f"{pair['pair_id']}: target FOA SHA changed")
                    materialized_record_sha = sha256_json(
                        {
                            "pair_record_sha256": pair["pair_record_sha256"],
                            "target_foa_sha256": manifest["target_foa_sha256"],
                            "target_latent_ref": target_ref,
                            "target_latent_tensor_sha256": target_tensor_sha,
                            "target_latent_shard_sha256": latent_sha,
                            "target_render_result_sha256": manifest[
                                "target_render_result_sha256"
                            ],
                            "target_materialized_manifest_sha256": manifest_sha,
                        }
                    )
                    connection.execute(
                        """
                        UPDATE pairs SET
                            target_latent_tensor_sha256=?,
                            target_latent_shard_sha256=?,
                            target_foa_path=?,
                            target_foa_sha256=?,
                            target_render_result_sha256=?,
                            pair_gain_policy=pair_gain_policy,
                            materialization_status='encoded',
                            target_materialized_manifest_path=?,
                            target_materialized_manifest_sha256=?,
                            materialized_record_sha256=?
                        WHERE pair_id=?
                        """,
                        (
                            target_tensor_sha,
                            latent_sha,
                            target_foa_path,
                            str(manifest["target_foa_sha256"]),
                            str(manifest["target_render_result_sha256"]),
                            str(manifest_path),
                            manifest_sha,
                            materialized_record_sha,
                            str(pair["pair_id"]),
                        ),
                    )
                    updated += 1
            connection.execute(
                "INSERT INTO target_shards VALUES(?,?,?,?,?,?)",
                (
                    shard,
                    len(rows),
                    str(latent_path),
                    latent_sha,
                    str(manifest_path),
                    manifest_sha,
                ),
            )
            manifest_inventory.append(
                {
                    "work_shard": shard,
                    "rows": len(rows),
                    "latent_path": str(latent_path),
                    "latent_shard_sha256": latent_sha,
                    "manifest_path": str(manifest_path),
                    "manifest_sha256": manifest_sha,
                }
            )
            connection.commit()
        if updated != expected_rows:
            raise RuntimeError(f"finalizer updated {updated} rows, expected {expected_rows}")
        incomplete = int(
            connection.execute(
                """
                SELECT COUNT(*) FROM pairs
                WHERE materialization_status!='encoded'
                   OR target_latent_tensor_sha256 IS NULL
                   OR target_latent_shard_sha256 IS NULL
                   OR target_foa_sha256 IS NULL
                   OR target_render_result_sha256 IS NULL
                   OR materialized_record_sha256 IS NULL
                """
            ).fetchone()[0]
        )
        if incomplete:
            raise RuntimeError(f"finalized index retains {incomplete} incomplete pairs")
        inventory_json = canonical_json(manifest_inventory)
        updates = {
            "schema": FINAL_INDEX_SCHEMA,
            "schema_version": str(FINAL_INDEX_SCHEMA_VERSION),
            "state": "materialized_complete_frozen",
            "source_planned_pair_index_path": str(planned_path),
            "source_planned_pair_index_sha256": sha256_file(planned_path),
            "materialized_inventory_json": inventory_json,
            "materialized_inventory_sha256": hashlib.sha256(
                inventory_json.encode("utf-8")
            ).hexdigest(),
            "finalizer_path": str(Path(__file__).resolve()),
            "finalizer_sha256": sha256_file(Path(__file__).resolve()),
            "target_latents_exhaustively_reopened": "true",
            "target_tensor_hashes_exhaustively_verified": "true",
        }
        connection.executemany(
            "INSERT OR REPLACE INTO metadata(key,value) VALUES(?,?)",
            sorted(updates.items()),
        )
        connection.commit()
        connection.execute("PRAGMA optimize")
        connection.commit()
    except Exception:
        connection.close()
        temporary.unlink(missing_ok=True)
        raise
    connection.close()
    os.replace(temporary, output_path)
    final_sha = sha256_file(output_path)
    marker = output_path.with_suffix(output_path.suffix + ".frozen.json")
    marker_value = {
        "schema": FINAL_INDEX_SCHEMA,
        "schema_version": FINAL_INDEX_SCHEMA_VERSION,
        "state": "materialized_complete_frozen",
        "split": split,
        "rows": expected_rows,
        "index_path": str(output_path),
        "index_sha256": final_sha,
    }
    marker_tmp = marker.with_name(marker.name + f".tmp.{os.getpid()}")
    marker_tmp.write_text(
        json.dumps(marker_value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(marker_tmp, marker)
    return {**marker_value, "marker": str(marker), "marker_sha256": sha256_file(marker)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--planned-pair-index", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    planned = args.planned_pair_index.expanduser().resolve(strict=True)
    if args.output is None:
        metadata, _, split = _load_planned_metadata(planned)
        output = Path(metadata["target_root"]) / "training_index" / f"{split}.sqlite"
    else:
        output = args.output
    report = finalize(planned, output, replace=bool(args.replace))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
