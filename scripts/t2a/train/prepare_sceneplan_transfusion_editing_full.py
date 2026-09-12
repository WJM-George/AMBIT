#!/usr/bin/env python3
"""Verify frozen full Editing indices and emit pinned dataset configs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import sys
from typing import Any
import zlib


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stable_audio_tools.configuration import load_config, validate_training_configs  # noqa: E402
from stable_audio_tools.data.sceneplan_transfusion_editing_dataset import (  # noqa: E402
    verify_editing_source_latent_shards,
    verify_editing_target_latent_shards,
)
from stable_audio_tools.data.sceneplan_transfusion_editing_index import sha256_file  # noqa: E402
from stable_audio_tools.models.sceneplan_transfusion_editing_provenance import (  # noqa: E402
    verify_frozen_qwen_runtime,
)


LATEST_AR_INPUT_CONTRACT = "source_foa_latent_plus_raw_edit_request_v2"
DEFAULT_SHORT_BATCH_SIZE_PER_GPU = 72
DEFAULT_LONG_BATCH_SIZE_PER_GPU = 48
DEFAULT_NUM_WORKERS_PER_RANK = 12
EDITING_DIT_TRAINABLE_PARAMETERS = 319_314_304
EDITING_AR_BASE_SPECIFIC_TRAINABLE_PARAMETERS = 8_458_240
EDITING_M2D_FULL_BRIDGE_TRAINABLE_PARAMETERS = 1_575_936
EDITING_AR_M2D_SPECIFIC_TRAINABLE_PARAMETERS = (
    EDITING_AR_BASE_SPECIFIC_TRAINABLE_PARAMETERS
    + EDITING_M2D_FULL_BRIDGE_TRAINABLE_PARAMETERS
)
EDITING_JOINT_M2D_UNIQUE_TRAINABLE_PARAMETERS = (
    EDITING_DIT_TRAINABLE_PARAMETERS
    + EDITING_AR_M2D_SPECIFIC_TRAINABLE_PARAMETERS
)
FROZEN_QWEN_PARAMETER_ELEMENTS = 752_393_024
FROZEN_FOA_VAE_PARAMETER_ELEMENTS = 155_853_956
FROZEN_M2D_AUDIO_PARAMETER_ELEMENTS = 89_041_922
# The utilization artifact predates the M2D bridge.  Preserve and validate its
# historical projection as provenance, but never expose it as current joint
# accounting in a newly generated full-training preflight.
LEGACY_PRE_M2D_SHARED_JOINT_PARAMETER_ACCOUNTING = {
    "editing_ar_specific_trainable": 8_458_240,
    "unique_trainable": 327_772_544,
    "frozen_qwen35_0p8b_text_backbone": 752_393_024,
    "registered_frozen_after_vae_removed": 0,
    "runtime_parameter_elements_without_vae": 1_080_165_568,
    "audio_e2e_runtime_parameter_elements_with_separate_vae": 1_236_019_524,
    "shared_transformer_counted_once": True,
}
CURRENT_M2D_SHARED_JOINT_PARAMETER_ACCOUNTING = {
    "editing_ar_base_specific_trainable": (
        EDITING_AR_BASE_SPECIFIC_TRAINABLE_PARAMETERS
    ),
    "editing_m2d_full_bridge_trainable": (
        EDITING_M2D_FULL_BRIDGE_TRAINABLE_PARAMETERS
    ),
    "editing_ar_specific_trainable": (
        EDITING_AR_M2D_SPECIFIC_TRAINABLE_PARAMETERS
    ),
    "unique_trainable": EDITING_JOINT_M2D_UNIQUE_TRAINABLE_PARAMETERS,
    "frozen_qwen35_0p8b_text_backbone": FROZEN_QWEN_PARAMETER_ELEMENTS,
    "registered_frozen_after_vae_removed": 0,
    "joint_training_parameter_elements_without_vae_or_external_m2d": (
        EDITING_JOINT_M2D_UNIQUE_TRAINABLE_PARAMETERS
        + FROZEN_QWEN_PARAMETER_ELEMENTS
    ),
    "separate_frozen_foa_vae_parameter_elements": (
        FROZEN_FOA_VAE_PARAMETER_ELEMENTS
    ),
    "separate_frozen_m2d_audio_parameter_elements": (
        FROZEN_M2D_AUDIO_PARAMETER_ELEMENTS
    ),
    "audio_e2e_parameter_elements_with_separate_vae_and_m2d": (
        EDITING_JOINT_M2D_UNIQUE_TRAINABLE_PARAMETERS
        + FROZEN_QWEN_PARAMETER_ELEMENTS
        + FROZEN_FOA_VAE_PARAMETER_ELEMENTS
        + FROZEN_M2D_AUDIO_PARAMETER_ELEMENTS
    ),
    "shared_transformer_counted_once": True,
}
DEFAULT_UTILIZATION_AUDIT = Path(
    "/mnt/sdb/model_archives/transfusion_editing/benchmarks/"
    "EDITING_DIT_UTILIZATION_SELECTION_20260904.json"
)
EXPECTED_CODEC_V4_FINGERPRINT = (
    "b512c31b96c6775af6e46e9a5cdf3d513658dcab61b6d357853a019ac16e3874"
)
CANONICAL_P10_SHA256 = (
    "be8c90cd1434bd71f73951531175c3674ff0f3173d5db591e2e1c476152ff59e"
)
DEFAULT_ROOT = Path("/mnt/sdb/audio_dataset/sceneplan_transfusion_editing_v1")
DEFAULT_MODEL_CONFIG = REPO_ROOT / (
    "stable_audio_tools/configs/model_configs/txt2audio/t2a/dit/"
    "qwen35_0p8b_300m_sceneplan_transfusion_editing_dit_full_v1.json"
)
DEFAULT_P10 = Path(
    "/mnt/sdc/ckpts/dit/sceneplan_dit_v11_semantic_v2_protected_resume_150k/"
    "checkpoints/epoch=48-step=150000.ckpt"
)
CURRENT_SHARED_CONTRACT = REPO_ROOT / (
    "docs/sceneplan_v2/p11_p10v11_shared_transfusion_v1_contract_20260904.md"
)
# The three pair indices were planned against this immutable historical
# document identity.  The active document was subsequently amended with the
# M2D/E2E details; that expected update must not masquerade as corrupted data.
INDEX_BUILD_SHARED_CONTRACT_SHA256 = (
    "cadd21f8caef93a27dadcb485b7aadc13e8dc3390f4ff482f58b34fb1a65d53d"
)


def _utilization_audit_report(
    path: Path, *, model_config: Path
) -> dict[str, Any]:
    path = path.expanduser().resolve(strict=True)
    model_config = model_config.expanduser().resolve(strict=True)
    value = json.loads(path.read_text(encoding="utf-8"))
    selected = dict(value.get("selected") or {})
    comparisons = list(value.get("comparisons") or [])
    expected_settings = {
        "short_batch_size_per_gpu": DEFAULT_SHORT_BATCH_SIZE_PER_GPU,
        "long_batch_size_per_gpu": DEFAULT_LONG_BATCH_SIZE_PER_GPU,
        "num_workers_per_rank": DEFAULT_NUM_WORKERS_PER_RANK,
    }
    selected_throughput = float(
        selected.get("global_sequence_positions_per_second", -1.0)
    )
    benchmark_root = path.parent.resolve()
    source_logs = [selected, *comparisons]
    source_logs_valid = True
    for record in source_logs:
        try:
            log_path = Path(record.get("source_log", "")).resolve(strict=True)
            log_path.relative_to(benchmark_root)
            source_logs_valid = source_logs_valid and (
                record.get("source_log_sha256") == sha256_file(log_path)
            )
        except (OSError, TypeError, ValueError):
            source_logs_valid = False
    if not (
        value.get("schema")
        == "sceneplan_transfusion_editing_dit_utilization_selection"
        and int(value.get("schema_version", -1)) == 1
        and value.get("status") == "PASS"
        and value.get("physical_gpus") == [3, 4, 5, 6, 7]
        and int(value.get("world_size", -1)) == 5
        and value.get("model_facing_frame_input")
        == [
            "noisy_target_64",
            "complete_new_sceneplan_control_256",
            "clean_source_foa_latent_64",
        ]
        and int(value.get("model_facing_frame_channels", -1)) == 384
        and Path(value.get("model_config", "")).resolve() == model_config
        and value.get("model_config_sha256") == sha256_file(model_config)
        and int(value.get("editing_dit_trainable_parameters", -1))
        == EDITING_DIT_TRAINABLE_PARAMETERS
        and value.get("editing_dit_trainable_parameter_breakdown")
        == {
            "diffusion_route": 318_515_200,
            "conditioner": 799_104,
            "total": 319_314_304,
        }
        and value.get("frozen_runtime_parameter_breakdown")
        == {
            "qwen35_0p8b_text_backbone_unregistered": 752_393_024,
            "foa_vae_registered": 155_853_956,
            "total": 908_246_980,
        }
        and int(value.get("standalone_dit_instantiated_parameter_elements", -1))
        == 1_227_561_284
        and value.get("qwen35_0p8b_text_backbone")
        == {
            "class": "Qwen3_5TextModel",
            "parameter_elements": 752_393_024,
            "unique_tensors": 320,
            "dtype": "bfloat16",
            "storage_bytes": 1_504_786_048,
            "registered_under_parent_module": False,
            "frozen": True,
        }
        and value.get("shared_joint_parameter_accounting")
        == LEGACY_PRE_M2D_SHARED_JOINT_PARAMETER_ACCOUNTING
        and all(selected.get(key) == item for key, item in expected_settings.items())
        and int(selected.get("short_global_batch_size", -1)) == 360
        and int(selected.get("long_global_batch_size", -1)) == 240
        and math.isfinite(selected_throughput)
        and selected_throughput > 0.0
        and len(comparisons) == 3
        and all(
            math.isfinite(float(record.get("global_sequence_positions_per_second", -1.0)))
            and 0.0
            < float(record["global_sequence_positions_per_second"])
            < selected_throughput
            for record in comparisons
        )
        and Path(value.get("formal_launcher", "")).resolve()
        == REPO_ROOT
        / "scripts/t2a/train/run_sceneplan_transfusion_editing_dit_full_5gpu.sh"
        and value.get("formal_fail_closed_settings")
        == {
            **expected_settings,
            "cuda_device_order": "PCI_BUS_ID",
            "cuda_visible_devices": "3,4,5,6,7",
        }
        and source_logs_valid
    ):
        raise RuntimeError("Editing DiT utilization benchmark is stale or invalid")
    return {
        "status": "PASS",
        "path": str(path),
        "sha256": sha256_file(path),
        "physical_gpus": [3, 4, 5, 6, 7],
        "world_size": 5,
        "editing_dit_trainable_parameters": EDITING_DIT_TRAINABLE_PARAMETERS,
        "editing_dit_trainable_parameter_breakdown": value[
            "editing_dit_trainable_parameter_breakdown"
        ],
        "frozen_runtime_parameter_breakdown": value[
            "frozen_runtime_parameter_breakdown"
        ],
        "standalone_dit_instantiated_parameter_elements": value[
            "standalone_dit_instantiated_parameter_elements"
        ],
        "qwen35_0p8b_text_backbone": value["qwen35_0p8b_text_backbone"],
        "legacy_pre_m2d_shared_joint_parameter_accounting": value[
            "shared_joint_parameter_accounting"
        ],
        "shared_joint_parameter_accounting": (
            CURRENT_M2D_SHARED_JOINT_PARAMETER_ACCOUNTING
        ),
        "model_config": str(model_config),
        "model_config_sha256": sha256_file(model_config),
        "selected": selected,
        "comparisons": comparisons,
    }


def _plan_token_audit_report(
    path: Path, *, train_report: dict[str, Any]
) -> dict[str, Any]:
    path = path.expanduser().resolve(strict=True)
    value = json.loads(path.read_text(encoding="utf-8"))
    planned = train_report["provenance"]["planned_pair_index"]
    histogram_rows = sum(
        int(count) for count in value.get("token_length_histogram", {}).values()
    )
    operation_rows = sum(
        int(record.get("rows", -1))
        for record in value.get("by_operation", {}).values()
    )
    bucket_rows = sum(
        int(record.get("rows", -1))
        for record in value.get("by_latent_bucket", {}).values()
    )
    maximum = float(value.get("token_length", {}).get("max", -1))
    auditor = Path(value.get("auditor_path", "")).expanduser().resolve(strict=True)
    expected = {
        "schema": "sceneplan_transfusion_editing_plan_token_audit",
        "schema_version": 1,
        "status": "PASS",
        "rows": int(train_report["rows"]),
        "codec_fingerprint": EXPECTED_CODEC_V4_FINGERPRINT,
        "editing_ar_inputs": ["source_foa_latent", "raw_edit_request"],
        "old_sceneplan_input": False,
        "required_max_tokens": 1024,
        "required_budget_covers_all_rows": True,
        "evaluation_max_tokens": 512,
        "evaluation_budget_covers_all_rows": True,
        "rows_over_evaluation_budget": 0,
    }
    changed = {
        key: {"expected": expected_value, "observed": value.get(key)}
        for key, expected_value in expected.items()
        if value.get(key) != expected_value
    }
    if changed:
        raise RuntimeError(f"train ScenePlan token audit changed: {changed}")
    if not (
        Path(value["index_path"]).expanduser().resolve()
        == Path(planned["path"]).expanduser().resolve()
        and value.get("index_sha256") == planned["sha256"]
        and value.get("index_state") == "planned_targets_not_materialized"
        and histogram_rows
        == operation_rows
        == bucket_rows
        == int(train_report["rows"])
        and 0 < maximum <= 512
        and value.get("longest_rows")
        and float(value["longest_rows"][0]["tokens"]) == maximum
        and sha256_file(auditor) == value.get("auditor_sha256")
    ):
        raise RuntimeError("train ScenePlan token audit is incomplete or stale")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "auditor_path": str(auditor),
        "auditor_sha256": value["auditor_sha256"],
        "index_path": value["index_path"],
        "index_sha256": value["index_sha256"],
        "rows": int(value["rows"]),
        "codec_fingerprint": value["codec_fingerprint"],
        "required_max_tokens": int(value["required_max_tokens"]),
        "evaluation_max_tokens": int(value["evaluation_max_tokens"]),
        "evaluation_budget_covers_all_rows": True,
        "rows_over_evaluation_budget": 0,
        "token_length": value["token_length"],
        "by_operation": value["by_operation"],
        "by_latent_bucket": value["by_latent_bucket"],
        "longest_rows": value["longest_rows"],
    }


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _index_build_shared_contract_provenance(
    metadata: dict[str, str], *, split: str
) -> dict[str, str]:
    """Separate immutable pair-build history from the current run contract."""

    historical_contract = Path(metadata["shared_contract_path"]).expanduser().resolve(
        strict=True
    )
    current_contract = CURRENT_SHARED_CONTRACT.resolve(strict=True)
    if not (
        historical_contract == current_contract
        and metadata.get("shared_contract_sha256")
        == INDEX_BUILD_SHARED_CONTRACT_SHA256
    ):
        raise RuntimeError(f"{split} historical shared-contract identity changed")
    return {
        "path": str(historical_contract),
        "index_build_sha256": str(metadata["shared_contract_sha256"]),
        "current_sha256": sha256_file(current_contract),
        "role": "historical_pair_build_provenance_plus_current_execution_contract",
    }


def _index_report(path: Path, *, split: str, expected_rows: int) -> dict[str, Any]:
    path = path.resolve(strict=True)
    marker_path = path.with_suffix(path.suffix + ".frozen.json")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    index_sha = sha256_file(path)
    connection = sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True)
    try:
        metadata = dict(connection.execute("SELECT key,value FROM metadata"))
        row = connection.execute(
            "SELECT COUNT(*),COUNT(DISTINCT pair_id),"
            "COUNT(DISTINCT source_ordinal),COUNT(DISTINCT source_sample_id),"
            "COUNT(DISTINCT target_sample_id),"
            "SUM(latent_bucket_frames=432),SUM(latent_bucket_frames=648),"
            "SUM(materialization_status='encoded'),"
            "SUM(raw_edit_request='' OR target_latent_tensor_sha256 IS NULL "
            "OR target_latent_shard_sha256 IS NULL "
            "OR target_render_result_sha256 IS NULL "
            "OR materialized_record_sha256 IS NULL) FROM pairs"
        ).fetchone()
        operation_counts = {
            str(name): int(count)
            for name, count in connection.execute(
                "SELECT operation,COUNT(*) FROM pairs GROUP BY operation"
            )
        }
        family_counts = {
            str(name): int(count)
            for name, count in connection.execute(
                "SELECT operation_family,COUNT(*) FROM pairs "
                "GROUP BY operation_family"
            )
        }
    finally:
        connection.close()
    (
        rows,
        distinct_pairs,
        distinct_source_ordinals,
        distinct_source_ids,
        distinct_target_ids,
        short_rows,
        long_rows,
        encoded,
        invalid_rows,
    ) = (
        int(value or 0) for value in row
    )
    required = {
        "schema": "sceneplan_transfusion_editing_training_index",
        "schema_version": "1",
        "state": "materialized_complete_frozen",
        "split": split,
        "rows": str(expected_rows),
        "editing_ar_input_contract": LATEST_AR_INPUT_CONTRACT,
        "editing_ar_old_sceneplan_input": "false",
        "target_latents_exhaustively_reopened": "true",
        "target_tensor_hashes_exhaustively_verified": "true",
    }
    changed = {
        key: {"expected": expected, "observed": metadata.get(key)}
        for key, expected in required.items()
        if metadata.get(key) != expected
    }
    if changed:
        raise RuntimeError(f"{split} index contract changed: {changed}")
    expected_operation_counts = json.loads(metadata["operation_counts_json"])
    expected_family_counts = json.loads(metadata["operation_family_counts_json"])
    if not (
        rows
        == distinct_pairs
        == distinct_source_ordinals
        == distinct_source_ids
        == distinct_target_ids
        == encoded
        == int(expected_rows)
        and short_rows + long_rows == rows
        and invalid_rows == 0
        and operation_counts == expected_operation_counts
        and family_counts == expected_family_counts
        and sum(operation_counts.values()) == rows
        and sum(family_counts.values()) == rows
        and marker.get("index_sha256") == index_sha
        and int(marker.get("rows", -1)) == rows
        and marker.get("split") == split
    ):
        raise RuntimeError(f"{split} frozen index completeness changed")

    provenance = {}
    for label, path_key, sha_key in (
        ("planned_pair_index", "source_planned_pair_index_path", "source_planned_pair_index_sha256"),
        ("builder", "builder_path", "builder_sha256"),
        ("finalizer", "finalizer_path", "finalizer_sha256"),
    ):
        artifact = Path(metadata[path_key]).expanduser().resolve(strict=True)
        observed_sha = sha256_file(artifact)
        if observed_sha != metadata[sha_key]:
            raise RuntimeError(f"{split} {label} provenance SHA256 changed")
        provenance[label] = {"path": str(artifact), "sha256": observed_sha}
    provenance["shared_contract"] = _index_build_shared_contract_provenance(
        metadata, split=split
    )
    for label, sha_key, module_path in (
        (
            "mutation_module",
            "mutation_module_sha256",
            REPO_ROOT / "stable_audio_tools/data/sceneplan_transfusion_editing.py",
        ),
        (
            "resolver_module",
            "resolver_module_sha256",
            REPO_ROOT / "stable_audio_tools/data/sceneplan_transfusion_editing_index.py",
        ),
    ):
        observed_sha = sha256_file(module_path)
        if observed_sha != metadata[sha_key]:
            raise RuntimeError(f"{split} {label} provenance SHA256 changed")
        provenance[label] = {
            "path": str(module_path.resolve()),
            "sha256": observed_sha,
        }
    # SQLite binds declared tensor/shard digests, but both latent roles remain
    # external files. Hash each distinct source and target shard once so a
    # mutable/stale artifact cannot enter DiT, the shared joint run, or E2E.
    source_shard_audit = verify_editing_source_latent_shards(path)
    target_shard_audit = verify_editing_target_latent_shards(path)
    if not (
        source_shard_audit.get("source_latent_shards_exhaustively_verified")
        is True
        and int(source_shard_audit.get("source_pair_rows", -1)) == rows
        and int(source_shard_audit.get("source_latent_shards", -1)) > 0
        and len(
            str(
                source_shard_audit.get(
                    "source_latent_shard_inventory_sha256", ""
                )
            )
        )
        == 64
    ):
        raise RuntimeError(f"{split} source-latent shard audit is incomplete")
    if not (
        target_shard_audit.get("target_latent_shards_exhaustively_verified")
        is True
        and int(target_shard_audit.get("target_pair_rows", -1)) == rows
        and int(target_shard_audit.get("target_latent_shards", -1)) > 0
        and len(
            str(
                target_shard_audit.get(
                    "target_latent_shard_inventory_sha256", ""
                )
            )
        )
        == 64
    ):
        raise RuntimeError(f"{split} target-latent shard audit is incomplete")
    return {
        "path": str(path),
        "sha256": index_sha,
        "marker_path": str(marker_path.resolve()),
        "marker_sha256": sha256_file(marker_path),
        "rows": rows,
        "distinct_pair_ids": distinct_pairs,
        "distinct_source_ordinals": distinct_source_ordinals,
        "distinct_source_sample_ids": distinct_source_ids,
        "distinct_target_sample_ids": distinct_target_ids,
        "short_rows": short_rows,
        "long_rows": long_rows,
        "operation_counts": operation_counts,
        "operation_family_counts": family_counts,
        "provenance": provenance,
        **source_shard_audit,
        **target_shard_audit,
    }


def _dataset_config(report: dict[str, Any], *, split: str) -> dict[str, Any]:
    train = split == "train"
    value = {
        "_task": (
            "Transfusion Editing DiT: complete new ScenePlan and aligned clean "
            "source FOA latent condition the edited target latent."
        ),
        "_editing_ar_input_contract": LATEST_AR_INPUT_CONTRACT,
        "_editing_ar_old_sceneplan_input": False,
        "dataset_type": "sceneplan_transfusion_editing_preencoded",
        "datasets": [
            {
                "id": f"sceneplan_transfusion_editing_v1_{split}",
                "path": report["path"],
                "weight": 1.0,
            }
        ],
        "expected_num_samples": report["rows"],
        "index_num_samples": report["rows"],
        "index_sha256": report["sha256"],
        "latent_crop_length": 648,
        "latent_downsampling_ratio": 1024,
        "caption_max_tokens": 512,
        "max_length_sec": 15.047619047619047,
        "random_crop": False,
        "require_complete": True,
        "verify_tensor_hashes_on_access": False,
        "max_item_retries": 0,
        "pin_memory": True,
        "persistent_workers": train,
        "drop_last": train,
    }
    if train:
        value["length_bucket_batching"] = {
            "enabled": True,
            # Keep the proven P10-v11 per-rank operating point.  Editing adds
            # source-reference columns only before the fixed-width Transformer;
            # it does not lengthen the token sequence or hidden state.
            "long_batch_size": DEFAULT_LONG_BATCH_SIZE_PER_GPU,
            "seed": 42,
        }
    return value


_SPLIT_IDENTITY_COLUMNS = (
    "source_sample_id",
    "source_foa_sha256",
    "source_latent_tensor_sha256",
    "old_sceneplan_sha256",
    "source_members_sha256",
)


def _member_identities(*blobs: bytes) -> tuple[set[str], set[str]]:
    identities: set[str] = set()
    parents: set[str] = set()
    for blob in blobs:
        members = json.loads(zlib.decompress(blob))
        if not isinstance(members, list):
            raise RuntimeError("Editing member provenance is not a list")
        for member in members:
            identity = str(member.get("identity_hash") or "")
            asset = dict(member.get("asset_ref") or {})
            parent = str(asset.get("parent_asset_id") or asset.get("asset_id") or "")
            if len(identity) != 64 or not parent:
                raise RuntimeError("Editing member identity provenance is incomplete")
            identities.add(identity)
            parents.add(parent)
    return identities, parents


def _heldout_identity_inventory(path: Path) -> dict[str, set[str]]:
    connection = sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True)
    inventory = {column: set() for column in _SPLIT_IDENTITY_COLUMNS}
    inventory.update(
        {
            "constituent_identity_hash": set(),
            "constituent_parent_asset_id": set(),
            "raw_edit_request": set(),
        }
    )
    columns = ",".join(
        [*_SPLIT_IDENTITY_COLUMNS, "raw_edit_request", "source_members_zlib", "target_members_zlib"]
    )
    try:
        for row in connection.execute(f"SELECT {columns} FROM pairs"):
            for index, column in enumerate(_SPLIT_IDENTITY_COLUMNS):
                inventory[column].add(str(row[index]))
            inventory["raw_edit_request"].add(
                str(row[len(_SPLIT_IDENTITY_COLUMNS)])
            )
            identities, parents = _member_identities(row[-2], row[-1])
            inventory["constituent_identity_hash"].update(identities)
            inventory["constituent_parent_asset_id"].update(parents)
    finally:
        connection.close()
    return inventory


def _split_disjointness_audit(reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Prove that evaluation audio/assets cannot occur in training or calibration."""

    heldout = {
        split: _heldout_identity_inventory(Path(reports[split]["path"]))
        for split in ("validation", "test")
    }
    hard_keys = (
        *_SPLIT_IDENTITY_COLUMNS,
        "constituent_identity_hash",
        "constituent_parent_asset_id",
    )
    pair_reports: dict[str, Any] = {}
    direct = {
        key: len(heldout["validation"][key] & heldout["test"][key])
        for key in (*hard_keys, "raw_edit_request")
    }
    pair_reports["validation_vs_test"] = {
        "overlap_counts": direct,
        "hard_identity_disjoint": all(direct[key] == 0 for key in hard_keys),
        "exact_instruction_overlap": direct["raw_edit_request"],
    }

    counters = {
        split: {key: 0 for key in (*hard_keys, "raw_edit_request")}
        for split in ("validation", "test")
    }
    train_path = Path(reports["train"]["path"])
    connection = sqlite3.connect(f"file:{train_path}?mode=ro&immutable=1", uri=True)
    columns = ",".join(
        [*_SPLIT_IDENTITY_COLUMNS, "raw_edit_request", "source_members_zlib", "target_members_zlib"]
    )
    try:
        for row in connection.execute(f"SELECT {columns} FROM pairs"):
            scalar = {
                column: str(row[index])
                for index, column in enumerate(_SPLIT_IDENTITY_COLUMNS)
            }
            scalar["raw_edit_request"] = str(row[len(_SPLIT_IDENTITY_COLUMNS)])
            identities, parents = _member_identities(row[-2], row[-1])
            for split in ("validation", "test"):
                for key, value in scalar.items():
                    counters[split][key] += int(value in heldout[split][key])
                counters[split]["constituent_identity_hash"] += len(
                    identities & heldout[split]["constituent_identity_hash"]
                )
                counters[split]["constituent_parent_asset_id"] += len(
                    parents & heldout[split]["constituent_parent_asset_id"]
                )
    finally:
        connection.close()
    for split in ("validation", "test"):
        overlaps = counters[split]
        pair_reports[f"train_vs_{split}"] = {
            "overlap_counts": overlaps,
            "hard_identity_disjoint": all(overlaps[key] == 0 for key in hard_keys),
            # Reusing generic language is not an audio-identity leak; report it
            # transparently without turning instruction paraphrases into a gate.
            "exact_instruction_overlap": overlaps["raw_edit_request"],
        }
    checks = {
        name: bool(record["hard_identity_disjoint"])
        for name, record in pair_reports.items()
    }
    if not all(checks.values()):
        raise RuntimeError(f"Editing split identity leakage detected: {pair_reports}")
    return {
        "contract": "audio_and_constituent_identity_disjoint_across_splits_v1",
        "status": "PASS",
        "hard_identity_keys": list(hard_keys),
        "instruction_overlap_is_diagnostic_only": True,
        "checks": checks,
        "pairs": pair_reports,
        "heldout_unique_counts": {
            split: {key: len(values) for key, values in inventory.items()}
            for split, inventory in heldout.items()
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--p10-checkpoint", type=Path, default=DEFAULT_P10)
    parser.add_argument("--train-plan-token-audit", type=Path)
    parser.add_argument(
        "--utilization-audit", type=Path, default=DEFAULT_UTILIZATION_AUDIT
    )
    args = parser.parse_args()
    root = args.root.expanduser().resolve(strict=True)
    model_path = args.model_config.expanduser().resolve(strict=True)
    checkpoint = args.p10_checkpoint.expanduser().resolve(strict=True)
    if sha256_file(checkpoint) != CANONICAL_P10_SHA256:
        raise RuntimeError("canonical P10-v11 checkpoint SHA256 changed")
    model = load_config(model_path)
    diffusion = model["model"]["diffusion"]
    training = model["training"]
    prompt_configs = [
        value
        for value in model["model"]["conditioning"]["configs"]
        if value.get("id") == "prompt"
    ]
    if not (
        diffusion["input_concat_ids"] == ["sceneplan_44", "source_foa_latent"]
        and int(diffusion["config"]["input_concat_dim"]) == 320
        and model["model"]["conditioning"]["pre_encoded_keys"]
        == ["source_foa_latent"]
        # The complete-new-plan caption/structured branches may be unknowned
        # independently for CFG, but the aligned clean reference must never be
        # removed by generic model-level conditioning dropout.
        and float(training.get("cfg_dropout_prob", -1.0)) == 0.0
        and training.get("sceneplan_cfg_dropout")
        == {
            "mode": "independent",
            "caption_unknown_prob": 0.15,
            "structured_unknown_prob": 0.15,
        }
        and len(prompt_configs) == 1
        and prompt_configs[0].get("type") == "qwen_text"
        and prompt_configs[0].get("config", {}).get("enable_grad") is False
    ):
        raise RuntimeError("Editing DiT is not the 384-channel latest route")
    qwen_runtime = verify_frozen_qwen_runtime(
        prompt_configs[0]["config"]["model_path"]
    )
    scheduler = model["training"]["optimizer_configs"]["diffusion"]["scheduler"]
    if scheduler != {
        "type": "CosineAnnealingLR",
        "config": {"T_max": 30_000, "eta_min": 0.000002},
    }:
        raise RuntimeError(
            "Editing DiT scheduler retained incompatible inherited keys: "
            f"{scheduler}"
        )
    utilization_audit = _utilization_audit_report(
        args.utilization_audit, model_config=model_path
    )

    expected = {"train": 1_000_000, "validation": 20_000, "test": 5_000}
    reports = {
        split: _index_report(
            root / "training_index" / f"{split}.sqlite",
            split=split,
            expected_rows=rows,
        )
        for split, rows in expected.items()
    }
    current_shared_contract_sha256 = sha256_file(
        CURRENT_SHARED_CONTRACT.resolve(strict=True)
    )
    if any(
        report["provenance"]["shared_contract"]["index_build_sha256"]
        != INDEX_BUILD_SHARED_CONTRACT_SHA256
        or report["provenance"]["shared_contract"]["current_sha256"]
        != current_shared_contract_sha256
        for report in reports.values()
    ):
        raise RuntimeError("Editing splits disagree on shared-contract provenance")
    split_disjointness = _split_disjointness_audit(reports)
    train_plan_token_audit = _plan_token_audit_report(
        (
            args.train_plan_token_audit
            if args.train_plan_token_audit is not None
            else root / "audits/editing_train_plan_token_audit.json"
        ),
        train_report=reports["train"],
    )
    output_root = root / "contracts/full_training"
    config_paths = {}
    for split, report in reports.items():
        config = _dataset_config(report, split=split)
        validate_training_configs(model, config)
        path = output_root / f"{split}_dataset.json"
        _atomic_json(path, config)
        config_paths[split] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
        }
    audit = {
        "schema": "sceneplan_transfusion_editing_full_training_preflight",
        "schema_version": 2,
        "status": "PASS",
        "latest_route": {
            "editing_ar_inputs": ["source_foa_latent", "raw_edit_request"],
            "editing_ar_target": "complete_new_sceneplan",
            "old_sceneplan_input": False,
            "editing_dit_frame_channels": 384,
            "editing_dit_clean_source_always_present": True,
            "editing_ar_required_max_plan_tokens": 1024,
            "editing_evaluation_max_plan_tokens": 512,
            "max_plan_tokens_observed": int(
                train_plan_token_audit["token_length"]["max"]
            ),
        },
        "editing_dit_utilization_defaults": {
            "short_batch_size_per_gpu": DEFAULT_SHORT_BATCH_SIZE_PER_GPU,
            "long_batch_size_per_gpu": DEFAULT_LONG_BATCH_SIZE_PER_GPU,
            "num_workers_per_rank": DEFAULT_NUM_WORKERS_PER_RANK,
            "world_size": 5,
        },
        "editing_dit_utilization_benchmark": utilization_audit,
        "canonical_p10_checkpoint": str(checkpoint),
        "canonical_p10_checkpoint_sha256": CANONICAL_P10_SHA256,
        "model_config": str(model_path),
        "model_config_sha256": sha256_file(model_path),
        "shared_contract": {
            "path": str(CURRENT_SHARED_CONTRACT.resolve(strict=True)),
            "sha256": current_shared_contract_sha256,
            "index_build_sha256": INDEX_BUILD_SHARED_CONTRACT_SHA256,
            "index_build_role": "historical_pair_construction_provenance",
            "current_role": "active_execution_contract",
        },
        "frozen_qwen_runtime": qwen_runtime,
        "indices": reports,
        "split_disjointness": split_disjointness,
        "train_plan_token_audit": train_plan_token_audit,
        "dataset_configs": config_paths,
    }
    _atomic_json(output_root / "PREFLIGHT.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
