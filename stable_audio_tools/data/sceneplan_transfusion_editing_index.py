"""Authoritative P10 source resolver for paired Transfusion Editing data.

The Editing inventory references immutable P10 source latents instead of
copying them.  This module joins one frozen training-index row to its exact
three-view render recipe and materialization record, checks every content
digest, and exposes the renderer gain history needed for a locality-preserving
target render.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import zlib
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pyarrow.parquet as pq

from .sceneplan_transfusion_editing import (
    canonical_json,
    sha256_json,
    validate_render_recipe_binding,
)


SOURCE_RESOLUTION_CONTRACT = "sceneplan_transfusion_editing_p10_source_v1"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scene_domain(sceneplan: Mapping[str, Any]) -> str:
    kinds = {str(source["kind"]) for source in sceneplan["sources"]}
    if kinds == {"speech"}:
        return "speech_only"
    if "speech" in kinds:
        return "speech_mixed"
    if kinds == {"music"}:
        return "music_only"
    if kinds == {"sound"}:
        return "sound_only"
    if kinds == {"music", "sound"}:
        return "music_sound"
    raise ValueError(f"unsupported ScenePlan source-domain set: {sorted(kinds)}")


def latent_bucket(latent_frames_valid: int) -> int:
    frames = int(latent_frames_valid)
    if not 0 < frames <= 648:
        raise ValueError(f"invalid P10 latent frame count: {frames}")
    return 432 if frames <= 432 else 648


def companion_recipe_path(model_sceneplan_path: str | Path) -> Path:
    path = Path(model_sceneplan_path)
    prefix = "model-sceneplans-"
    if path.suffix != ".jsonl" or not path.name.startswith(prefix):
        raise ValueError(f"not a P10 model-ScenePlan shard: {path}")
    return path.with_name("render-recipes-" + path.name[len(prefix) :])


def materialized_manifest_path(latent_path: str | Path) -> Path:
    path = Path(latent_path)
    parts = list(path.parts)
    positions = [index for index, value in enumerate(parts) if value == "latents"]
    if len(positions) != 1:
        raise ValueError(f"cannot derive materialized manifest from {path}")
    parts[positions[0]] = "manifests"
    name = path.name
    if not name.startswith("latents-") or path.suffix != ".safetensors":
        raise ValueError(f"unexpected P10 latent shard name: {path}")
    parts[-1] = "materialized-" + name[len("latents-") : -len(path.suffix)] + ".parquet"
    return Path(*parts)


@dataclass(frozen=True)
class P10ResolvedSource:
    split: str
    source_ordinal: int
    source_sample_id: str
    model_num_samples: int
    latent_frames_valid: int
    latent_bucket_frames: int
    source_count: int
    domain: str
    sceneplan: dict[str, Any]
    model_sceneplan_sha256: str
    model_sceneplan_path: str
    model_sceneplan_row: int
    render_recipe: dict[str, Any]
    render_recipe_path: str
    render_recipe_sha256: str
    render_result: dict[str, Any]
    source_manifest_path: str
    source_manifest_sha256: str
    source_foa_sha256: str
    source_latent_path: str
    source_latent_key: str
    source_latent_tensor_sha256: str
    source_latent_shard_sha256: str

    @property
    def stratum(self) -> tuple[int, str, int]:
        return self.source_count, self.domain, self.latent_bucket_frames


class _LRU:
    def __init__(self, maximum: int) -> None:
        self.maximum = max(1, int(maximum))
        self.values: OrderedDict[Path, Any] = OrderedDict()

    def get(self, path: Path, loader) -> Any:
        value = self.values.pop(path, None)
        if value is None:
            value = loader(path)
        self.values[path] = value
        while len(self.values) > self.maximum:
            self.values.popitem(last=False)
        return value


class P10EditingSourceResolver:
    """Resolve and validate rows from one immutable P10 split index."""

    def __init__(
        self,
        index_path: str | Path,
        *,
        expected_split: str,
        jsonl_cache_size: int = 8,
        parquet_cache_size: int = 8,
    ) -> None:
        if expected_split not in {"train", "validation", "test"}:
            raise ValueError(f"unknown P10 split: {expected_split!r}")
        self.index_path = Path(index_path).expanduser().resolve(strict=True)
        self.expected_split = expected_split
        self._connection: sqlite3.Connection | None = None
        self._jsonl = _LRU(jsonl_cache_size)
        self._parquet = _LRU(parquet_cache_size)
        connection = self._open()
        metadata = dict(connection.execute("SELECT key,value FROM metadata"))
        if metadata.get("schema") != "stable_audio_tools.sceneplan_v2_training_index":
            raise RuntimeError("source index is not a frozen P10 ScenePlan index")
        if metadata.get("frozen") != "true" or metadata.get("latent_channels") != "64":
            raise RuntimeError("source P10 index is not frozen 64-channel latent truth")
        self.index_metadata = metadata
        self.row_count = int(
            connection.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
        )
        if int(metadata.get("rows", -1)) != self.row_count:
            raise RuntimeError("source P10 index metadata row count changed")

    def _open(self) -> sqlite3.Connection:
        if self._connection is None:
            uri = f"file:{self.index_path}?mode=ro&immutable=1"
            self._connection = sqlite3.connect(
                uri, uri=True, check_same_thread=False
            )
            self._connection.row_factory = sqlite3.Row
            self._connection.execute("PRAGMA query_only=ON")
        return self._connection

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def __enter__(self) -> "P10EditingSourceResolver":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    @staticmethod
    def _load_jsonl(path: Path) -> tuple[tuple[dict[str, Any], str], ...]:
        rows = []
        with path.resolve(strict=True).open("r", encoding="utf-8") as handle:
            for row_number, raw in enumerate(handle):
                text = raw.rstrip("\n")
                if not text or "\r" in text:
                    raise RuntimeError(f"invalid canonical JSONL row {path}:{row_number + 1}")
                value = json.loads(text)
                if canonical_json(value) != text:
                    raise RuntimeError(f"non-canonical JSONL row {path}:{row_number + 1}")
                rows.append((value, text))
        return tuple(rows)

    @staticmethod
    def _load_parquet(path: Path) -> tuple[dict[str, Any], str]:
        path = path.resolve(strict=True)
        table = pq.read_table(path)
        rows = table.to_pylist()
        by_id = {str(row["sample_id"]): row for row in rows}
        if len(by_id) != len(rows):
            raise RuntimeError(f"duplicate sample ids in materialized manifest: {path}")
        return by_id, sha256_file(path)

    def iter_selection_fields(self):
        """Yield compact fields needed for deterministic split selection."""

        rows = self._open().execute(
            """
            SELECT ordinal,sample_id,latent_frames_valid,scene_plan_zlib
            FROM samples ORDER BY ordinal
            """
        )
        for row in rows:
            plan = json.loads(zlib.decompress(row["scene_plan_zlib"]))
            yield {
                "ordinal": int(row["ordinal"]),
                "sample_id": str(row["sample_id"]),
                "source_count": len(plan["sources"]),
                "domain": scene_domain(plan),
                "latent_bucket_frames": latent_bucket(row["latent_frames_valid"]),
                "has_static": any(
                    source["trajectory"]["type"] == "static"
                    for source in plan["sources"]
                ),
                "has_linear": any(
                    source["trajectory"]["type"] == "linear"
                    for source in plan["sources"]
                ),
                "sceneplan": plan,
            }

    def resolve(self, ordinal: int) -> P10ResolvedSource:
        row = self._open().execute(
            """
            SELECT s.*,l.path AS latent_path,l.sha256 AS latent_shard_sha256
            FROM samples AS s
            JOIN latent_shards AS l ON l.id=s.latent_shard_id
            WHERE s.ordinal=?
            """,
            (int(ordinal),),
        ).fetchone()
        if row is None:
            raise IndexError(f"P10 split has no ordinal {ordinal}")
        sample_id = str(row["sample_id"])
        if self.expected_split == "train":
            valid_prefix = "_train_"
        else:
            valid_prefix = f"_{self.expected_split}_"
        if valid_prefix not in sample_id:
            raise RuntimeError(f"{sample_id}: source row crosses the requested split")

        plan = json.loads(zlib.decompress(row["scene_plan_zlib"]))
        model_sha = sha256_json(plan)
        if model_sha != str(row["model_sceneplan_sha256"]):
            raise RuntimeError(f"{sample_id}: compact ScenePlan checksum changed")
        if plan.get("sample_id") != sample_id:
            raise RuntimeError(f"{sample_id}: compact ScenePlan sample id changed")

        recipe_path = companion_recipe_path(row["model_sceneplan_path"])
        recipe_rows = self._jsonl.get(recipe_path, self._load_jsonl)
        recipe_row = int(row["model_sceneplan_row"])
        if not 0 <= recipe_row < len(recipe_rows):
            raise RuntimeError(f"{sample_id}: render-recipe row is outside its shard")
        recipe, recipe_text = recipe_rows[recipe_row]
        validate_render_recipe_binding(plan, recipe)
        recipe_sha = hashlib.sha256(recipe_text.encode("utf-8")).hexdigest()

        latent_path = Path(row["latent_path"]).resolve(strict=True)
        manifest_path = materialized_manifest_path(latent_path)
        manifest_by_id, manifest_sha = self._parquet.get(
            manifest_path, self._load_parquet
        )
        materialized = manifest_by_id.get(sample_id)
        if materialized is None:
            raise RuntimeError(f"{sample_id}: source materialization row is absent")
        if str(materialized["render_recipe_sha256"]) != recipe_sha:
            raise RuntimeError(f"{sample_id}: render-recipe checksum/materialization drift")
        if str(materialized["model_sceneplan_sha256"]) != model_sha:
            raise RuntimeError(f"{sample_id}: ScenePlan/materialization drift")
        expected_ref = f"{latent_path}#{row['latent_key']}"
        if str(materialized["latent_ref"]) != expected_ref:
            raise RuntimeError(f"{sample_id}: source latent reference changed")
        for observed, expected, label in (
            (
                materialized["latent_tensor_sha256"],
                row["latent_tensor_sha256"],
                "latent tensor",
            ),
            (
                materialized["latent_shard_sha256"],
                row["latent_shard_sha256"],
                "latent shard",
            ),
        ):
            if str(observed) != str(expected):
                raise RuntimeError(f"{sample_id}: {label} checksum changed")
        if (
            int(materialized["model_num_samples"]) != int(row["model_num_samples"])
            or int(materialized["latent_frames_valid"])
            != int(row["latent_frames_valid"])
        ):
            raise RuntimeError(f"{sample_id}: source materialization geometry changed")
        render_result = json.loads(str(materialized["render_result_json"]))
        if (
            render_result.get("status") != "ok"
            or render_result.get("sample_id") != sample_id
            or not math.isfinite(float(render_result.get("master_gain", math.nan)))
            or float(render_result["master_gain"]) <= 0.0
        ):
            raise RuntimeError(f"{sample_id}: invalid source render result")
        recipe_by_id = {
            str(source["source_id"]): source for source in recipe["sources"]
        }
        qc_by_id = {
            str(source["source_id"]): source
            for source in render_result.get("source_qc") or ()
        }
        if set(qc_by_id) != set(recipe_by_id):
            raise RuntimeError(f"{sample_id}: source QC membership changed")
        for source_id, qc in qc_by_id.items():
            recipe_source = recipe_by_id[source_id]
            if (
                qc.get("asset_id") != recipe_source["asset_ref"]["asset_id"]
                or str(qc.get("content_lineage", {}).get("source_audio_sha256"))
                != str(recipe_source["asset_ref"]["identity_hash"])
            ):
                raise RuntimeError(f"{sample_id}/{source_id}: dry identity/QC drift")
            for key in (
                "dry_normalization_gain",
                "source_rms_normalization_gain",
                "calibrated_gain_correction_db",
                "actual_gain_db",
            ):
                if not math.isfinite(float(qc.get(key, math.nan))):
                    raise RuntimeError(f"{sample_id}/{source_id}: invalid source QC {key}")

        return P10ResolvedSource(
            split=self.expected_split,
            source_ordinal=int(row["ordinal"]),
            source_sample_id=sample_id,
            model_num_samples=int(row["model_num_samples"]),
            latent_frames_valid=int(row["latent_frames_valid"]),
            latent_bucket_frames=latent_bucket(row["latent_frames_valid"]),
            source_count=len(plan["sources"]),
            domain=scene_domain(plan),
            sceneplan=plan,
            model_sceneplan_sha256=model_sha,
            model_sceneplan_path=str(row["model_sceneplan_path"]),
            model_sceneplan_row=recipe_row,
            render_recipe=recipe,
            render_recipe_path=str(recipe_path),
            render_recipe_sha256=recipe_sha,
            render_result=render_result,
            source_manifest_path=str(manifest_path),
            source_manifest_sha256=manifest_sha,
            source_foa_sha256=str(materialized["foa_sha256"]),
            source_latent_path=str(latent_path),
            source_latent_key=str(row["latent_key"]),
            source_latent_tensor_sha256=str(row["latent_tensor_sha256"]),
            source_latent_shard_sha256=str(row["latent_shard_sha256"]),
        )


__all__ = [
    "P10EditingSourceResolver",
    "P10ResolvedSource",
    "SOURCE_RESOLUTION_CONTRACT",
    "companion_recipe_path",
    "latent_bucket",
    "materialized_manifest_path",
    "scene_domain",
    "sha256_file",
]
