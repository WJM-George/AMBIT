#!/usr/bin/env python3
"""Freeze the fully audited revision-5 dataset at P9 and close P10/P11."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
import sys

for value in (SCRIPT_DIR, REPO_ROOT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from sceneplan_v2_common import DATASET_ROOT, atomic_write_json  # noqa: E402


SNAPSHOT_FILES = (
    "docs/sceneplan_v2/model_sceneplan_v1.schema.json",
    "docs/sceneplan_v2/sceneplan_dataset_contract_v5.json",
    "docs/sceneplan_v2/sceneplan_conditioning_amendment_v2_512.json",
    "docs/sceneplan_v2/speech_speaker_description_amendment_v1.json",
    "stable_audio_tools/configs/dataset_configs/construct_dataset/sceneplan_renderer_v2_1p124m.json",
    "stable_audio_tools/configs/dataset_configs/sceneplan_v2_train.json",
    "stable_audio_tools/configs/dataset_configs/sceneplan_v2_validation.json",
    "stable_audio_tools/configs/dataset_configs/sceneplan_v2_test.json",
    "stable_audio_tools/configs/model_configs/txt2audio/t2a/dit/qwen35_0p8b_300m_model_sceneplan_v1.json",
    "stable_audio_tools/configuration.py",
    "stable_audio_tools/data/dataset.py",
    "stable_audio_tools/data/model_sceneplan.py",
    "stable_audio_tools/data/sceneplan_v2.py",
    "stable_audio_tools/data/sceneplan_v2_dataset.py",
    "stable_audio_tools/models/conditioners.py",
    "stable_audio_tools/models/diffusion.py",
    "stable_audio_tools/models/dit.py",
    "stable_audio_tools/models/sceneplan_conditioning.py",
    "train.py",
    "scripts/t2a/data/sceneplan_v2_common.py",
    "scripts/t2a/data/build_sceneplan_manifests_v2.py",
    "scripts/t2a/data/build_model_sceneplan_manifests_v1.py",
    "scripts/t2a/data/migrate_model_sceneplan_speaker_descriptions_v1.py",
    "scripts/t2a/data/audit_model_sceneplan_manifests_v1.py",
    "scripts/t2a/data/audit_model_sceneplan_speaker_delta_v1.py",
    "dataset/captioning/sceneplan_speaker_v1/README.md",
    "dataset/captioning/sceneplan_speaker_v1/speaker_profile_prompt_v1.txt",
    "dataset/captioning/sceneplan_speaker_v1/build_speaker_profile_inputs.py",
    "dataset/captioning/sceneplan_speaker_v1/finalize_speaker_registry.py",
    "scripts/t2a/data/rebind_model_sceneplan_materialized_metadata_v1.py",
    "scripts/t2a/data/materialize_sceneplan_v2_shard.py",
    "scripts/t2a/data/sceneplan_v2_renderer.py",
    "scripts/t2a/data/materialize_model_sceneplan_v1_shard.py",
    "scripts/t2a/data/materialize_model_sceneplan_v1_worker.py",
    "scripts/t2a/data/run_model_sceneplan_v1_p8.py",
    "scripts/t2a/data/audit_model_sceneplan_materialized_v1.py",
    "scripts/t2a/data/audit_model_sceneplan_materialized_delta_v1.py",
    "scripts/t2a/data/build_model_sceneplan_training_index_v1.py",
    "scripts/t2a/data/verify_model_sceneplan_v1_loader.py",
    "scripts/t2a/data/freeze_model_sceneplan_v1_p9.py",
    "scripts/t2a/data/run_model_sceneplan_v1_p9.py",
    "scripts/t2a/data/wait_model_sceneplan_v1_p8_then_p9.py",
)


INVENTORY_SCHEMA = pa.schema(
    [
        ("path", pa.string()),
        ("role", pa.string()),
        ("num_bytes", pa.int64()),
        ("sha256", pa.string()),
        ("checksum_verification", pa.string()),
    ]
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_report(path: Path, schema: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != schema or value.get("ok") is not True:
        raise RuntimeError(f"required P9 gate is not all-pass: {path}")
    return value


def atomic_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    pq.write_table(
        pa.Table.from_pylist(rows, schema=INVENTORY_SCHEMA),
        temporary,
        compression="zstd",
    )
    if pq.read_metadata(temporary).num_rows != len(rows):
        raise RuntimeError("frozen artifact inventory row count changed after reopen")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument(
        "--checkpoint-root", type=Path, default=Path("/mnt/sdb/ckpts/sceneplan_dit_v2")
    )
    args = parser.parse_args()
    root = args.dataset_root.expanduser().resolve(strict=True)
    checkpoint_root = args.checkpoint_root.expanduser().resolve(strict=False)
    try:
        root.relative_to("/mnt/sdb")
        checkpoint_root.relative_to("/mnt/sdb")
    except ValueError as error:
        raise ValueError("P9 dataset/checkpoint roots must be on SDB") from error
    marker = root / "FROZEN_P9.json"
    if marker.is_file():
        print(marker.read_text(encoding="utf-8"), end="")
        return 0

    p75 = require_report(
        root / "sceneplans_model_v1/audit.json",
        "stable_audio_tools.model_sceneplan_manifest_audit",
    )
    p9 = require_report(
        root / "qc/p9_model_sceneplan_materialized_audit.json",
        "stable_audio_tools.model_sceneplan_materialized_dataset_audit",
    )
    loader = require_report(
        root / "qc/p9_model_sceneplan_loader_smoke.json",
        "stable_audio_tools.model_sceneplan_v1_loader_smoke",
    )
    p4_speech = require_report(
        root / "pilots/tts_2k/qc/asr_distil_large_v3_calibrated/speech_qc_summary.json",
        "stable_audio_tools.tts_v2_pilot_speech_qc_summary",
    )
    speech_ledger_qc = require_report(
        root / "split_ledgers/speech_v2/strong_qc_summary.json",
        "stable_audio_tools.sceneplan_speech_split_ledger_strong_qc",
    )
    p6_planned = require_report(
        root / "pilots/joint_4k/qc/planned_manifest_audit.json",
        "stable_audio_tools.sceneplan_manifest_audit",
    )
    p6_materialized = require_report(
        root / "pilots/joint_4k/qc/materialized_audit.json",
        "stable_audio_tools.sceneplan_materialized_dataset_audit",
    )
    p6_loader = require_report(
        root / "pilots/joint_4k/qc/loader_conditioner_smoke.json",
        "stable_audio_tools.sceneplan_v2_p6_loader_conditioner_smoke",
    )
    p8_path = root / "materialized/p8_orchestrator_summary.json"
    p8 = json.loads(p8_path.read_text(encoding="utf-8"))
    training_summary_path = root / "training_index/summary.json"
    training = json.loads(training_summary_path.read_text(encoding="utf-8"))
    if not (
        int(p75.get("dataset_contract_revision", -1)) == 5
        and int(p75.get("rows", -1)) == 1_124_000
        and int(p9.get("dataset_contract_revision", -1)) == 5
        and int(p9.get("rows", -1)) == 1_124_000
        and int(training.get("dataset_contract_revision", -1)) == 5
        and int(training.get("rows", -1)) == 1_124_000
    ):
        raise RuntimeError("revision-5 P7.5/P9/training-index totals are incomplete")
    if not (
        int(p75.get("speech_speaker_registry_rows", -1)) == 512_000
        and int(p75.get("speech_speaker_registry_rows_referenced", -1)) == 512_000
        and int(p75.get("unique_speaker_descriptions", -1)) > 1_000
        and int(p75.get("unique_speaker_profile_keys", -1)) >= 2_452
        and int(p75.get("constant_generic_speaker_description_rows", -1)) == 0
    ):
        raise RuntimeError("speech speaker-description registry gate is incomplete")
    if not (
        p8.get("schema") == "stable_audio_tools.model_sceneplan_p8_orchestrator_summary"
        and p8.get("status") == "complete"
        and int(p8.get("rows", -1)) == 1_124_000
        and p8.get("p10_training_started") is False
        and p8.get("p11_training_started") is False
    ):
        raise RuntimeError("formal P8 summary is incomplete")
    if int(p4_speech["rows"]) != 2_000 or p4_speech.get("status_counts") != {"pass": 2_000}:
        raise RuntimeError("P4 speech/FOA ASR gate is not 2000/2000 pass")
    if int(speech_ledger_qc["formal_rows"]) != 512_000:
        raise RuntimeError("strong-QC formal speech ledger is not 512,000 rows")
    if int(p6_planned["rows"]) != 4_000 or int(p6_materialized["rows"]) != 4_000:
        raise RuntimeError("P6 joint pilot is not exactly 4,000 planned/materialized rows")
    if (
        int(p6_loader.get("dataset_rows", -1)) != 4_000
        or p6_loader.get("p10_training_started") is not False
    ):
        raise RuntimeError("P6 loader/conditioner gate is incomplete")
    if loader.get("fusion_output_shape") != [2, 256, 432] or int(
        loader.get("structured_feature_dim", -1)
    ) != 9:
        raise RuntimeError("revision-5 real loader/conditioner smoke is incomplete")
    if list((root / "materialized/quarantine").glob("*/*.json")):
        raise RuntimeError("P8 quarantine is not empty")
    train_render_root = root / "materialized/renders/train"
    if train_render_root.exists() and any(path.is_file() for path in train_render_root.rglob("*")):
        raise RuntimeError("transient train FOA/render files remain at P9")
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    if list(checkpoint_root.rglob("*.ckpt")) or list(checkpoint_root.rglob("*.safetensors")):
        raise RuntimeError("P10 checkpoint root is not empty; training may have started")

    contract_root = root / "contracts"
    snapshot_root = contract_root / "p9_revision5_code_snapshot"
    for relative_text in SNAPSHOT_FILES:
        source = (REPO_ROOT / relative_text).resolve(strict=True)
        destination = snapshot_root / relative_text
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    qwen_root = Path("/mnt/sdc/ckpts/pretrained/Qwen/Qwen3.5-0.8B").resolve(strict=True)
    vae_checkpoint = Path(
        "/mnt/sdc/ckpts/compareVAE_ckpt/unwrapped_wdmix_1350000.ckpt"
    ).resolve(strict=True)
    dependencies = []
    for path in [*sorted(qwen_root.rglob("*")), vae_checkpoint]:
        if path.is_file():
            dependencies.append(
                {
                    "path": str(path),
                    "role": (
                        "frozen_qwen35_0p8b_dependency"
                        if path.is_relative_to(qwen_root)
                        else "frozen_foa_vae_dependency"
                    ),
                    "num_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    external_path = contract_root / "p9_revision5_external_dependency_manifest.json"
    atomic_write_json(
        external_path,
        {
            "schema": "stable_audio_tools.sceneplan_v2_external_dependency_manifest",
            "schema_version": 2,
            "dataset_contract_revision": 5,
            "dependencies": dependencies,
        },
    )
    approval_gate = {
        "schema": "stable_audio_tools.p10_user_approval_gate",
        "schema_version": 2,
        "dataset_contract_revision": 5,
        "state": "closed_waiting_for_user_acceptance",
        "p10_training_started": False,
        "p11_training_started": False,
        "initialization": "from_scratch",
        "legacy_300k_checkpoint": "reference_only_never_load",
        "model_config": str(
            REPO_ROOT
            / "stable_audio_tools/configs/model_configs/txt2audio/t2a/dit/"
            "qwen35_0p8b_300m_model_sceneplan_v1.json"
        ),
        "train_dataset_config": str(
            REPO_ROOT / "stable_audio_tools/configs/dataset_configs/sceneplan_v2_train.json"
        ),
        "validation_dataset_config": str(
            REPO_ROOT / "stable_audio_tools/configs/dataset_configs/sceneplan_v2_validation.json"
        ),
        "checkpoint_root": str(checkpoint_root),
    }
    atomic_write_json(contract_root / "P10_REQUIRES_USER_APPROVAL.json", approval_gate)

    inventory: dict[str, dict[str, Any]] = {}

    def add(
        path: Path,
        role: str,
        expected_sha: str | None = None,
        verified: str = "freeze_rehash",
    ) -> None:
        resolved = path.expanduser().resolve(strict=True)
        try:
            resolved.relative_to("/mnt/sdb")
        except ValueError as error:
            raise ValueError(f"frozen artifact is not on SDB: {resolved}") from error
        digest = expected_sha or sha256_file(resolved)
        if len(digest) != 64:
            raise RuntimeError(f"invalid artifact checksum: {resolved}")
        row = {
            "path": str(resolved),
            "role": role,
            "num_bytes": resolved.stat().st_size,
            "sha256": digest,
            "checksum_verification": verified,
        }
        previous = inventory.get(str(resolved))
        if previous is not None and previous["sha256"] != digest:
            raise RuntimeError(f"conflicting frozen checksums: {resolved}")
        inventory[str(resolved)] = row

    frozen_trees = (
        (root / "sceneplans_model_v1", "model_sceneplan_three_view"),
        (root / "sceneplans", "historical_revision4_sceneplan"),
        (root / "source_annotations", "source_description_annotation"),
        (root / "source_catalog", "source_catalog"),
        (root / "models", "frozen_quality_control_model"),
        (root / "split_ledgers", "split_ledger"),
        (root / "training_index", "dit_training_index"),
        (root / "qc", "quality_report"),
        (root / "audit", "audit_provenance"),
    )
    mutable_orchestrator_paths = {
        (root / "audit/p9_revision5_orchestrator.log").resolve(strict=False),
        (root / "audit/p9_revision5_orchestrator_state.json").resolve(strict=False),
        (root / "audit/p8_p9_revision5_completion_report.json").resolve(strict=False),
    }
    superseded_archive_root = (
        root / "audit/superseded_speaker_constant_20260819_0924"
    ).resolve(strict=False)
    for tree, role in frozen_trees:
        for path in sorted(tree.rglob("*")):
            if (
                path.is_file()
                and not path.resolve().is_relative_to(superseded_archive_root)
                and path.resolve() not in mutable_orchestrator_paths
            ):
                add(path, role)
    for path in (
        root / "pilots/tts_2k/render_summary.json",
        root / "pilots/tts_2k/qc/asr_distil_large_v3_calibrated/speech_qc_summary.json",
        root / "pilots/tts_2k/qc/cutoff_calibration_distil_large_v3/summary.json",
        root / "pilots/joint_4k/qc/planned_manifest_audit.json",
        root / "pilots/joint_4k/qc/materialized_audit.json",
        root / "pilots/joint_4k/qc/loader_conditioner_smoke.json",
        root / "pilots/joint_4k/training_index/summary.json",
        root / "pilots/joint_4k/materialized/p6_orchestrator_summary.json",
        p8_path,
    ):
        add(path, "stage_gate_quality_report")
    generated_freeze_outputs = {
        (contract_root / "p9_revision5_frozen_artifact_manifest.parquet").resolve(
            strict=False
        ),
        (contract_root / "freeze_manifest.json").resolve(strict=False),
    }
    for path in sorted(contract_root.rglob("*")):
        if path.is_file() and path.resolve() not in generated_freeze_outputs:
            add(path, "contract_or_code_snapshot")

    manifests = sorted((root / "materialized/manifests").glob("*/materialized-*.parquet"))
    if len(manifests) != 1_099:
        raise RuntimeError("P9 freeze expected exactly 1,099 materialized manifests")
    for manifest in manifests:
        add(manifest, "materialized_manifest")
        rows = pq.read_table(
            manifest,
            columns=["split", "latent_ref", "latent_shard_sha256", "render_result_json"],
        ).to_pylist()
        latent_refs = {str(row["latent_ref"]).split("#", 1)[0] for row in rows}
        latent_hashes = {str(row["latent_shard_sha256"]) for row in rows}
        if len(latent_refs) != 1 or len(latent_hashes) != 1:
            raise RuntimeError(f"materialized manifest has inconsistent latent lineage: {manifest}")
        add(
            Path(latent_refs.pop()),
            "variable_latent_shard",
            latent_hashes.pop(),
            "p9_exhaustive_checksum_verified",
        )
        for row in rows:
            if row["split"] == "train":
                continue
            result = json.loads(row["render_result_json"])
            result_path = Path(result["foa_path"]).parent / "render_result.json"
            add(result_path, "retained_eval_render_result")
            add(
                Path(result["foa_path"]),
                "retained_eval_foa",
                str(result["foa_sha256"]),
                "p9_exhaustive_checksum_audio_qc_verified",
            )
            for reference in result["stem_refs"]:
                add(
                    Path(reference["path"]),
                    "retained_eval_source_stem",
                    str(reference["sha256"]),
                    "p9_exhaustive_checksum_audio_qc_verified",
                )
    for path in sorted((root / "materialized/work_done").rglob("*.json")):
        add(path, "materialization_done_marker")
    for path in sorted((root / "materialized/workers").glob("*.json")):
        add(path, "materialization_worker_report")
    for path in sorted((root / "materialized/logs").glob("*.log")):
        add(path, "materialization_worker_log")
    add(root / "materialized/p8_orchestrator_state.json", "materialization_state")
    add(root / "materialized/p8_orchestrator_summary.json", "materialization_summary")

    rows = sorted(inventory.values(), key=lambda row: row["path"])
    inventory_path = contract_root / "p9_revision5_frozen_artifact_manifest.parquet"
    atomic_parquet(inventory_path, rows)
    freeze_manifest = {
        "schema": "stable_audio_tools.sceneplan_v2_p9_freeze_manifest",
        "schema_version": 3,
        "dataset_id": "sceneplan_renderer_v2_1p124m",
        "dataset_contract_revision": 5,
        "historical_base_contract_revision": 4,
        "state": "P9_complete_frozen_waiting_for_user_acceptance",
        "p10_training_started": False,
        "p11_training_started": False,
        "rows": 1_124_000,
        "splits": {"train": 1_100_000, "validation": 20_000, "test": 4_000},
        "speech_rows": {"train": 500_000, "validation": 10_000, "test": 2_000},
        "no_speech_rows": {"train": 600_000, "validation": 10_000, "test": 2_000},
        "max_model_samples": 442_368,
        "max_latent_frames": 432,
        "structured_feature_dim": 9,
        "renderer_caption_max_tokens": 512,
        "speech_speaker_descriptions_registry_driven": True,
        "speech_speaker_registry_rows": 512_000,
        "unique_speaker_descriptions": int(p75["unique_speaker_descriptions"]),
        "unique_speaker_profile_keys": int(p75["unique_speaker_profile_keys"]),
        "random_crop": False,
        "artifact_manifest": str(inventory_path),
        "artifact_manifest_sha256": sha256_file(inventory_path),
        "external_dependency_manifest": str(external_path),
        "external_dependency_manifest_sha256": sha256_file(external_path),
        "artifact_count": len(rows),
        "artifact_bytes": sum(int(row["num_bytes"]) for row in rows),
        "p75_report": str(root / "sceneplans_model_v1/audit.json"),
        "p75_report_sha256": sha256_file(root / "sceneplans_model_v1/audit.json"),
        "p8_summary": str(p8_path),
        "p8_summary_sha256": sha256_file(p8_path),
        "p9_report": str(root / "qc/p9_model_sceneplan_materialized_audit.json"),
        "p9_report_sha256": sha256_file(root / "qc/p9_model_sceneplan_materialized_audit.json"),
        "loader_smoke_report": str(root / "qc/p9_model_sceneplan_loader_smoke.json"),
        "loader_smoke_report_sha256": sha256_file(root / "qc/p9_model_sceneplan_loader_smoke.json"),
        "training_index_summary": str(training_summary_path),
        "training_index_summary_sha256": sha256_file(training_summary_path),
        "p10_approval_gate": str(contract_root / "P10_REQUIRES_USER_APPROVAL.json"),
        "model_initialization": "from_scratch",
        "legacy_300k_checkpoint_usage": "reference_only_never_warm_start",
    }
    freeze_path = contract_root / "freeze_manifest.json"
    atomic_write_json(freeze_path, freeze_manifest)
    marker_value = {
        "schema": "stable_audio_tools.sceneplan_v2_p9_marker",
        "schema_version": 3,
        "dataset_contract_revision": 5,
        "state": freeze_manifest["state"],
        "freeze_manifest": str(freeze_path),
        "freeze_manifest_sha256": sha256_file(freeze_path),
        "p10_training_started": False,
        "p11_training_started": False,
    }
    atomic_write_json(marker, marker_value)
    print(json.dumps({**freeze_manifest, "marker": str(marker)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
