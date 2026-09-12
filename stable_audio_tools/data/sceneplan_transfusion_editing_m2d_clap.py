"""Frozen M2D-CLAP cache reader for the Editing-AR source side.

The cache stores only derived vectors and immutable identities.  A source
caption is used offline to create the frozen text target, but neither caption
text nor a source/old ScenePlan is exposed by this reader.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any

import numpy as np
import torch

from .sceneplan_transfusion_editing import sha256_json
from .sceneplan_transfusion_editing_index import sha256_file
from stable_audio_tools.models.sceneplan_transfusion_editing_m2d_clap import (
    EDITING_M2D_CLAP_CONTRACT,
    EDITING_M2D_CLAP_EMBED_DIM,
    sha256_group_id,
)
from stable_audio_tools.models.sceneplan_transfusion_editing_m2d_runtime import (
    M2D_CLAP_SOURCE_AUDIO_VIEW,
    M2D_CLAP_TEMPORAL_POLICY,
    editing_m2d_numeric_runtime_fingerprint,
    editing_m2d_software_runtime_fingerprint,
    expected_editing_m2d_clap_asset_report,
)


EDITING_M2D_CACHE_SCHEMA = "sceneplan_transfusion_editing_m2d_clap_cache"
EDITING_M2D_CACHE_SCHEMA_VERSION = "2"
EDITING_M2D_CACHE_STATE = "complete_frozen"
EDITING_M2D_VAE_CONFIG_SHA256 = (
    "0f0373c9b32deb3d9ea875a3a0f98a898fa1d3a3aa0d82f6cade6dc2ab97b179"
)
EDITING_M2D_VAE_CHECKPOINT_SHA256 = (
    "0229e48729bb6cf138c277d37c598d659000cf78e0a16f498171f2f1f83e8a87"
)
EDITING_M2D_AUDIO_PREPROCESS = (
    "source_latent_fp16_frozen_foa_vae_decode_exact_valid_frames_"
    "W_peak_minus_1db_sinc_antialias_resample_16khz_"
    "aligned_992mel_full_coverage_m2d_cpu_l2norm_fp16_model_boundary_v6"
)
EDITING_M2D_CAPTION_TARGET = (
    "source_sceneplan_semantic_caption_v2_offline_training_label_only"
)
EDITING_M2D_TEMPORAL_PILOT_CHECKS = frozenset(
    {
        "source_audio_view_is_decoded_W",
        "all_waveforms_cover_at_least_15_seconds",
        "full_projector_has_more_tokens_than_10s",
        "tail_intervention_preserves_full_token_geometry",
        "synthetic_15s_probe_uses_long_sequence",
        "all_embeddings_are_finite",
        "full_embeddings_are_l2_normalized",
        "real_active_tail_rows_present",
        "every_real_active_tail_changes_embedding",
        "early_middle_late_probes_all_reach_embedding",
        "boundary_narrow_probe_reaches_embedding",
    }
)
_REPO_ROOT = Path(__file__).resolve().parents[2]
EDITING_M2D_TEMPORAL_VERIFIER = _REPO_ROOT / (
    "scripts/t2a/test/validate_sceneplan_transfusion_editing_m2d_temporal_policy.py"
)
EDITING_M2D_SHARD_BUILDER = _REPO_ROOT / (
    "scripts/t2a/data/build_sceneplan_transfusion_editing_m2d_clap_cache.py"
)
EDITING_M2D_CACHE_MERGER = _REPO_ROOT / (
    "scripts/t2a/data/merge_sceneplan_transfusion_editing_m2d_clap_cache.py"
)
EDITING_M2D_CACHE_ONLINE_PARITY_VERIFIER = _REPO_ROOT / (
    "scripts/t2a/test/validate_sceneplan_transfusion_editing_m2d_cache_online_parity.py"
)
EDITING_M2D_CACHE_ONLINE_PARITY_SCHEMA = (
    "sceneplan_transfusion_editing_m2d_cache_online_parity"
)
EDITING_M2D_CACHE_ONLINE_PARITY_CHECKS = frozenset(
    {
        "selected_rows_cover_both_buckets_and_all_five_cache_shards",
        "source_latent_tensor_hashes_verified",
        "cache_online_batch1_fp16_exact",
        "cache_online_batch2_fp16_exact",
        "cache_online_batch4_fp16_exact",
        "decoded_frozen_vae_W_view_used",
        "old_sceneplan_model_input_false",
        "target_information_used_false",
        "runtime_and_assets_match_cache_contract",
    }
)
EDITING_M2D_CACHE_IMPLEMENTATION_PATHS = (
    "scripts/t2a/data/build_sceneplan_transfusion_editing_m2d_clap_cache.py",
    "scripts/t2a/data/merge_sceneplan_transfusion_editing_m2d_clap_cache.py",
    "scripts/t2a/test/validate_sceneplan_transfusion_editing_m2d_cache_online_parity.py",
    "scripts/t2a/test/validate_sceneplan_transfusion_editing_m2d_temporal_policy.py",
    "stable_audio_tools/configuration.py",
    "stable_audio_tools/data/model_sceneplan.py",
    "stable_audio_tools/data/sceneplan_transfusion_editing.py",
    "stable_audio_tools/data/sceneplan_transfusion_editing_index.py",
    "stable_audio_tools/data/sceneplan_transfusion_editing_m2d_clap.py",
    "stable_audio_tools/models/autoencoders.py",
    "stable_audio_tools/models/blocks.py",
    "stable_audio_tools/models/bottleneck.py",
    "stable_audio_tools/models/factory.py",
    "stable_audio_tools/models/sceneplan_transfusion_editing_m2d_clap.py",
    "stable_audio_tools/models/sceneplan_transfusion_editing_m2d_runtime.py",
    "stable_audio_tools/models/sceneplan_transfusion_editing_pipeline.py",
    "stable_audio_tools/models/utils.py",
)


def editing_m2d_cache_implementation_sha256() -> dict[str, str]:
    """Hash every local implementation file that affects cache/online parity."""

    return {
        relative: sha256_file((_REPO_ROOT / relative).resolve(strict=True))
        for relative in EDITING_M2D_CACHE_IMPLEMENTATION_PATHS
    }


def _readonly_connection(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(
        f"file:{path}?mode=ro&immutable=1", uri=True, check_same_thread=False
    )
    connection.execute("PRAGMA query_only=ON")
    return connection


def _embedding_from_blob(value: bytes, *, field: str, pair_id: str) -> torch.Tensor:
    expected_bytes = EDITING_M2D_CLAP_EMBED_DIM * np.dtype(np.float16).itemsize
    if not isinstance(value, bytes) or len(value) != expected_bytes:
        raise RuntimeError(f"{pair_id}: {field} M2D blob has invalid length")
    array = np.frombuffer(value, dtype=np.float16).copy()
    tensor = torch.from_numpy(array)
    if tuple(tensor.shape) != (EDITING_M2D_CLAP_EMBED_DIM,) or not bool(
        torch.isfinite(tensor).all()
    ):
        raise RuntimeError(f"{pair_id}: {field} M2D embedding is invalid")
    norm = float(tensor.float().norm().item())
    if not 0.99 <= norm <= 1.01:
        raise RuntimeError(
            f"{pair_id}: {field} M2D embedding lost L2 normalization ({norm})"
        )
    return tensor


def validate_editing_m2d_temporal_pilot(
    path: str | Path, *, expected_sha256: str | None = None
) -> dict[str, Any]:
    """Reopen the source-only 15-second M2D coverage gate."""

    if not str(path):
        raise RuntimeError("Editing M2D temporal-pilot path is absent")
    resolved = Path(path).expanduser().resolve(strict=True)
    observed_sha = sha256_file(resolved)
    if expected_sha256 is not None and observed_sha != str(expected_sha256):
        raise RuntimeError("Editing M2D temporal-pilot SHA256 changed")
    value = json.loads(resolved.read_text(encoding="utf-8"))
    checks = value.get("checks")
    verifier_raw = str(value.get("verifier") or "")
    if not verifier_raw:
        raise RuntimeError("Editing M2D temporal-pilot verifier is absent")
    verifier = Path(verifier_raw).resolve(strict=True)
    expected_verifier = EDITING_M2D_TEMPORAL_VERIFIER.resolve(strict=True)
    source_index_raw = str(value.get("source_index") or "")
    if not source_index_raw:
        raise RuntimeError("Editing M2D temporal-pilot source index is absent")
    source_index = Path(source_index_raw).resolve(strict=True)
    vae = dict(value.get("vae") or {})
    assets = dict(value.get("m2d_assets") or {})
    numeric_runtime = dict(value.get("numeric_runtime_fingerprint") or {})
    software_runtime = editing_m2d_software_runtime_fingerprint()
    if not (
        value.get("schema")
        == "sceneplan_transfusion_editing_m2d_temporal_policy_pilot"
        and int(value.get("schema_version", -1)) == 2
        and value.get("status") == "PASS"
        and isinstance(checks, dict)
        and set(checks) == EDITING_M2D_TEMPORAL_PILOT_CHECKS
        and all(result is True for result in checks.values())
        and int(value.get("physical_gpu", -1)) == 3
        and int(value.get("rows", -1)) == 4
        and int(value.get("source_index_rows", -1)) == 20_000
        and value.get("source_index_sha256") == sha256_file(source_index)
        and value.get("old_sceneplan_model_input") is False
        and value.get("new_sceneplan_or_target_information_used") is False
        and value.get("source_audio_view") == M2D_CLAP_SOURCE_AUDIO_VIEW
        and value.get("temporal_policy") == M2D_CLAP_TEMPORAL_POLICY
        and value.get("audio_preprocess") == EDITING_M2D_AUDIO_PREPROCESS
        and vae.get("config_sha256") == EDITING_M2D_VAE_CONFIG_SHA256
        and vae.get("checkpoint_sha256") == EDITING_M2D_VAE_CHECKPOINT_SHA256
        and assets.get("source_audio_view") == M2D_CLAP_SOURCE_AUDIO_VIEW
        and assets.get("temporal_policy") == M2D_CLAP_TEMPORAL_POLICY
        and assets.get("license_scope") == "internal_noncommercial_evaluation_only"
        and assets == expected_editing_m2d_clap_asset_report(require_text=False)
        and value.get("implementation_sha256")
        == editing_m2d_cache_implementation_sha256()
        and all(
            numeric_runtime.get(key) == expected
            for key, expected in software_runtime.items()
        )
        and isinstance(numeric_runtime.get("device_name"), str)
        and bool(numeric_runtime.get("device_name"))
        and isinstance(numeric_runtime.get("device_capability"), list)
        and len(numeric_runtime.get("device_capability")) == 2
        and verifier == expected_verifier
        and value.get("verifier_sha256") == sha256_file(verifier)
    ):
        raise RuntimeError("Editing M2D temporal-pilot contract is stale or invalid")
    return {
        "path": str(resolved),
        "sha256": observed_sha,
        "physical_gpu": int(value["physical_gpu"]),
        "rows": int(value["rows"]),
        "source_index": str(source_index),
        "source_index_sha256": str(value["source_index_sha256"]),
        "source_audio_view": str(value["source_audio_view"]),
        "temporal_policy": str(value["temporal_policy"]),
        "projector_tokens": value["projector_tokens"],
    }


class ScenePlanTransfusionEditingM2DCLAPCache:
    """Read one complete, immutable pair-aligned M2D-CLAP cache."""

    def __init__(
        self,
        path: str | Path,
        *,
        source_index: str | Path,
        source_index_sha256: str,
        expected_rows: int,
        expected_split: str,
        expected_cache_sha256: str | None = None,
        verify_cache_file_hash: bool = False,
    ) -> None:
        self.path = Path(path).expanduser().resolve(strict=True)
        self.source_index = Path(source_index).expanduser().resolve(strict=True)
        self._connection: sqlite3.Connection | None = None
        marker_path = self.path.with_suffix(self.path.suffix + ".frozen.json")
        if not marker_path.is_file():
            raise RuntimeError("Editing M2D cache requires a frozen marker")
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        marker_sha = str(marker.get("cache_sha256") or "")
        if expected_cache_sha256 is not None and marker_sha != str(
            expected_cache_sha256
        ):
            raise RuntimeError("Editing M2D cache marker SHA256 changed")
        if verify_cache_file_hash:
            actual_sha = sha256_file(self.path)
            if marker_sha != actual_sha or (
                expected_cache_sha256 is not None
                and actual_sha != str(expected_cache_sha256)
            ):
                raise RuntimeError("Editing M2D cache file SHA256 changed")

        connection = _readonly_connection(self.path)
        try:
            metadata = dict(connection.execute("SELECT key,value FROM metadata"))
            count, minimum, maximum, distinct = connection.execute(
                "SELECT COUNT(*),MIN(pair_ordinal),MAX(pair_ordinal),"
                "COUNT(DISTINCT pair_id) FROM features"
            ).fetchone()
        finally:
            connection.close()
        expected_metadata = {
            "schema": EDITING_M2D_CACHE_SCHEMA,
            "schema_version": EDITING_M2D_CACHE_SCHEMA_VERSION,
            "state": EDITING_M2D_CACHE_STATE,
            "split": str(expected_split),
            "rows": str(int(expected_rows)),
            "source_index": str(self.source_index),
            "source_index_sha256": str(source_index_sha256),
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
            "m2d_assets_json": json.dumps(
                expected_editing_m2d_clap_asset_report(require_text=True),
                ensure_ascii=False,
                sort_keys=True,
            ),
            "shard_builder": str(EDITING_M2D_SHARD_BUILDER.resolve(strict=True)),
            "shard_builder_sha256": sha256_file(EDITING_M2D_SHARD_BUILDER),
            "merger": str(EDITING_M2D_CACHE_MERGER.resolve(strict=True)),
            "merger_sha256": sha256_file(EDITING_M2D_CACHE_MERGER),
            "implementation_sha256_json": json.dumps(
                editing_m2d_cache_implementation_sha256(), sort_keys=True
            ),
            "numeric_runtime_fingerprint_json": json.dumps(
                editing_m2d_numeric_runtime_fingerprint(), sort_keys=True
            ),
        }
        for key, expected in expected_metadata.items():
            if metadata.get(key) != expected:
                raise RuntimeError(
                    f"Editing M2D cache metadata {key} changed: "
                    f"{metadata.get(key)!r} != {expected!r}"
                )
        temporal_pilot = validate_editing_m2d_temporal_pilot(
            metadata.get("temporal_pilot", ""),
            expected_sha256=metadata.get("temporal_pilot_sha256"),
        )
        rows = int(count)
        shards = marker.get("shards")
        shard_layout_ok = (
            isinstance(shards, list)
            and len(shards) == 5
            and sorted(int(value.get("shard_index", -1)) for value in shards)
            == list(range(5))
            and sorted(int(value.get("physical_gpu", -1)) for value in shards)
            == [3, 4, 5, 6, 7]
            and sum(int(value.get("rows", -1)) for value in shards) == rows
        )
        if (
            rows != int(expected_rows)
            or int(minimum) != 0
            or int(maximum) != rows - 1
            or int(distinct) != rows
            or marker.get("schema") != EDITING_M2D_CACHE_SCHEMA
            or str(marker.get("schema_version", ""))
            != EDITING_M2D_CACHE_SCHEMA_VERSION
            or marker.get("state") != EDITING_M2D_CACHE_STATE
            or int(marker.get("rows", -1)) != rows
            or Path(str(marker.get("cache_path") or "")).resolve() != self.path
            or Path(str(marker.get("source_index") or "")).resolve()
            != self.source_index
            or marker.get("source_index_sha256") != str(source_index_sha256)
            or marker.get("split") != str(expected_split)
            or marker.get("physical_gpus") != [3, 4, 5, 6, 7]
            or not shard_layout_ok
        ):
            raise RuntimeError("Editing M2D cache is not dense and frozen")
        self.metadata = metadata
        self.marker_path = marker_path.resolve()
        self.cache_sha256 = marker_sha
        self.temporal_pilot = temporal_pilot
        self._length = rows

    def __len__(self) -> int:
        return self._length

    def _db(self) -> sqlite3.Connection:
        if self._connection is None:
            self._connection = _readonly_connection(self.path)
        return self._connection

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_connection"] = None
        return state

    def __del__(self) -> None:
        connection = getattr(self, "_connection", None)
        if connection is not None:
            connection.close()

    def get(
        self,
        pair_ordinal: int,
        *,
        pair_id: str,
        source_sample_id: str,
        source_latent_tensor_sha256: str,
    ) -> dict[str, Any]:
        row = self._db().execute(
            """
            SELECT pair_id,source_sample_id,source_latent_tensor_sha256,
                   source_caption_sha256,
                   audio_embedding,text_embedding,
                   audio_embedding_sha256,text_embedding_sha256,record_sha256
            FROM features WHERE pair_ordinal=?
            """,
            (int(pair_ordinal),),
        ).fetchone()
        if row is None:
            raise RuntimeError(f"Editing M2D cache is missing ordinal {pair_ordinal}")
        (
            cached_pair_id,
            cached_source_sample_id,
            cached_source_sha,
            caption_sha,
            audio_blob,
            text_blob,
            audio_sha,
            text_sha,
            record_sha,
        ) = row
        if (
            str(cached_pair_id) != str(pair_id)
            or str(cached_source_sample_id) != str(source_sample_id)
            or str(cached_source_sha) != str(source_latent_tensor_sha256)
        ):
            raise RuntimeError(f"{pair_id}: M2D cache/source index identity diverged")
        if hashlib.sha256(audio_blob).hexdigest() != str(audio_sha) or hashlib.sha256(
            text_blob
        ).hexdigest() != str(text_sha):
            raise RuntimeError(f"{pair_id}: M2D embedding checksum changed")
        expected_record = sha256_json(
            {
                "pair_ordinal": int(pair_ordinal),
                "pair_id": str(pair_id),
                "source_sample_id": str(source_sample_id),
                "source_latent_tensor_sha256": str(source_latent_tensor_sha256),
                "source_caption_sha256": str(caption_sha),
                "audio_embedding_sha256": str(audio_sha),
                "text_embedding_sha256": str(text_sha),
            }
        )
        if expected_record != str(record_sha):
            raise RuntimeError(f"{pair_id}: M2D cache record checksum changed")
        audio = _embedding_from_blob(audio_blob, field="audio", pair_id=pair_id)
        text = _embedding_from_blob(text_blob, field="caption", pair_id=pair_id)
        caption_group = sha256_group_id(str(caption_sha))
        source_group = sha256_group_id(str(source_latent_tensor_sha256))
        return {
            "source_m2d_audio_embedding": audio,
            "source_caption_m2d_embedding": text,
            "source_caption_sha256": str(caption_sha),
            "source_caption_group_ids": torch.tensor(
                caption_group, dtype=torch.int64
            ),
            "source_semantic_group_ids": torch.tensor(
                source_group, dtype=torch.int64
            ),
        }


def editing_m2d_cache_online_parity_path(cache_path: str | Path) -> Path:
    """Return the mandatory formal cache-to-online replay sidecar path."""

    resolved = Path(cache_path).expanduser().resolve()
    return Path(str(resolved) + ".online_parity.json")


def _expected_online_parity_rows(
    cache: ScenePlanTransfusionEditingM2DCLAPCache,
) -> list[dict[str, Any]]:
    connection = _readonly_connection(cache.source_index)
    try:
        rows = []
        for bucket in (432, 648):
            selected = []
            for shard_index in range(5):
                row = connection.execute(
                    """
                    SELECT pair_ordinal,pair_id,source_sample_id,model_num_samples,
                           latent_frames_valid,latent_bucket_frames,
                           source_latent_path,source_latent_key,
                           source_latent_tensor_sha256
                    FROM pairs
                    WHERE pair_ordinal < ? AND latent_bucket_frames=?
                          AND (pair_ordinal % 5)=?
                    ORDER BY pair_ordinal LIMIT 1
                    """,
                    (len(cache), bucket, shard_index),
                ).fetchone()
                if row is None:
                    raise RuntimeError(
                        "formal M2D cache parity requires every GPU3--7 shard "
                        "in both latent buckets"
                    )
                selected.append(row)
            rows.extend(
                {
                    "pair_ordinal": int(row[0]),
                    "pair_id": str(row[1]),
                    "source_sample_id": str(row[2]),
                    "model_num_samples": int(row[3]),
                    "latent_frames_valid": int(row[4]),
                    "latent_bucket_frames": int(row[5]),
                    "source_latent_path": str(Path(row[6]).resolve(strict=True)),
                    "source_latent_key": str(row[7]),
                    "source_latent_tensor_sha256": str(row[8]),
                }
                for row in selected
            )
    finally:
        connection.close()
    return rows


def validate_editing_m2d_cache_online_parity(
    path: str | Path,
    *,
    cache: ScenePlanTransfusionEditingM2DCLAPCache,
    expected_report_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate the sampled VAE->M2D cache/online FP16 replay gate."""

    resolved = Path(path).expanduser().resolve(strict=True)
    observed_sha = sha256_file(resolved)
    if expected_report_sha256 is not None and observed_sha != str(
        expected_report_sha256
    ):
        raise RuntimeError("Editing M2D online-parity report SHA256 changed")
    value = json.loads(resolved.read_text(encoding="utf-8"))
    checks = dict(value.get("checks") or {})
    expected_rows = _expected_online_parity_rows(cache)
    observed_rows = list(value.get("selected_rows") or [])
    if len(observed_rows) != len(expected_rows):
        raise RuntimeError("Editing M2D online-parity sampled row count changed")
    cache_connection = cache._db()
    for expected, observed in zip(expected_rows, observed_rows):
        cached = cache_connection.execute(
            "SELECT audio_embedding,audio_embedding_sha256 FROM features "
            "WHERE pair_ordinal=?",
            (expected["pair_ordinal"],),
        ).fetchone()
        if cached is None:
            raise RuntimeError("Editing M2D online-parity cache row disappeared")
        blob_sha = hashlib.sha256(cached[0]).hexdigest()
        identity = {
            key: observed.get(key)
            for key in (
                "pair_ordinal",
                "pair_id",
                "source_sample_id",
                "model_num_samples",
                "latent_frames_valid",
                "latent_bucket_frames",
                "source_latent_path",
                "source_latent_key",
                "source_latent_tensor_sha256",
            )
        }
        if identity != expected or not (
            blob_sha == str(cached[1])
            and observed.get("cache_audio_embedding_sha256") == blob_sha
            and all(
                observed.get(f"online_batch{batch}_sha256") == blob_sha
                for batch in (1, 2, 4)
            )
            and all(
                observed.get(f"online_batch{batch}_replica_sha256")
                == [blob_sha] * batch
                for batch in (1, 2, 4)
            )
            and all(
                int(observed.get(f"online_batch{batch}_mismatch_elements", -1))
                == 0
                for batch in (1, 2, 4)
            )
            and all(
                float(observed.get(f"online_batch{batch}_max_abs", -1.0)) == 0.0
                for batch in (1, 2, 4)
            )
        ):
            raise RuntimeError("Editing M2D cache/online FP16 replay changed")
    verifier = EDITING_M2D_CACHE_ONLINE_PARITY_VERIFIER.resolve(strict=True)
    numeric_runtime = json.loads(cache.metadata["numeric_runtime_fingerprint_json"])
    implementation = editing_m2d_cache_implementation_sha256()
    if not (
        value.get("schema") == EDITING_M2D_CACHE_ONLINE_PARITY_SCHEMA
        and int(value.get("schema_version", -1)) == 1
        and value.get("status") == "PASS"
        and int(value.get("physical_gpu", -1)) == 3
        and int(value.get("rows", -1)) == 10
        and int(value.get("rows_per_bucket", -1)) == 5
        and value.get("batch_sizes") == [1, 2, 4]
        and value.get("batch_shape_contract")
        == "repeat_same_exact_geometry_row_v1"
        and set(checks) == EDITING_M2D_CACHE_ONLINE_PARITY_CHECKS
        and all(result is True for result in checks.values())
        and Path(value.get("cache", "")).resolve() == cache.path
        and value.get("cache_sha256") == cache.cache_sha256
        and Path(value.get("cache_marker", "")).resolve() == cache.marker_path
        and value.get("cache_marker_sha256") == sha256_file(cache.marker_path)
        and Path(value.get("source_index", "")).resolve() == cache.source_index
        and value.get("source_index_sha256")
        == cache.metadata["source_index_sha256"]
        and value.get("split") == cache.metadata["split"]
        and int(value.get("cache_rows", -1)) == len(cache)
        and value.get("source_audio_view") == M2D_CLAP_SOURCE_AUDIO_VIEW
        and value.get("temporal_policy") == M2D_CLAP_TEMPORAL_POLICY
        and value.get("audio_preprocess") == EDITING_M2D_AUDIO_PREPROCESS
        and value.get("vae_config_sha256") == EDITING_M2D_VAE_CONFIG_SHA256
        and value.get("vae_checkpoint_sha256")
        == EDITING_M2D_VAE_CHECKPOINT_SHA256
        and value.get("m2d_audio_assets")
        == expected_editing_m2d_clap_asset_report(require_text=False)
        and value.get("numeric_runtime_fingerprint") == numeric_runtime
        and value.get("implementation_sha256") == implementation
        and Path(value.get("verifier", "")).resolve() == verifier
        and value.get("verifier_sha256") == sha256_file(verifier)
    ):
        raise RuntimeError("Editing M2D online-parity report is stale or invalid")
    return {
        "path": str(resolved),
        "sha256": observed_sha,
        "rows": 10,
        "batch_sizes": [1, 2, 4],
        "fp16_exact": True,
    }


__all__ = [
    "EDITING_M2D_AUDIO_PREPROCESS",
    "EDITING_M2D_CACHE_IMPLEMENTATION_PATHS",
    "EDITING_M2D_CACHE_MERGER",
    "EDITING_M2D_CACHE_ONLINE_PARITY_CHECKS",
    "EDITING_M2D_CACHE_ONLINE_PARITY_SCHEMA",
    "EDITING_M2D_CACHE_ONLINE_PARITY_VERIFIER",
    "EDITING_M2D_CACHE_SCHEMA",
    "EDITING_M2D_CACHE_SCHEMA_VERSION",
    "EDITING_M2D_CACHE_STATE",
    "EDITING_M2D_CAPTION_TARGET",
    "EDITING_M2D_SHARD_BUILDER",
    "EDITING_M2D_VAE_CHECKPOINT_SHA256",
    "EDITING_M2D_VAE_CONFIG_SHA256",
    "ScenePlanTransfusionEditingM2DCLAPCache",
    "editing_m2d_cache_implementation_sha256",
    "editing_m2d_cache_online_parity_path",
    "validate_editing_m2d_cache_online_parity",
    "validate_editing_m2d_temporal_pilot",
]
