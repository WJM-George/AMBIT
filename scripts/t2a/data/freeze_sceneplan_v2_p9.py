#!/usr/bin/env python3
"""Freeze the fully audited ScenePlan-v2 dataset at P9 and close the P10 gate."""

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
    "docs/sceneplan_v2/README.md",
    "docs/sceneplan_v2/sceneplan_renderer_sample_v2.schema.json",
    "docs/sceneplan_v2/speech_dry_asset_v2.schema.json",
    "docs/sceneplan_v2/sceneplan_compiler_output_v2.schema.json",
    "stable_audio_tools/configs/dataset_configs/construct_dataset/sceneplan_renderer_v2_1p124m.json",
    "stable_audio_tools/configs/dataset_configs/sceneplan_v2_train.json",
    "stable_audio_tools/configs/dataset_configs/sceneplan_v2_validation.json",
    "stable_audio_tools/configs/dataset_configs/sceneplan_v2_test.json",
    "stable_audio_tools/configs/model_configs/txt2audio/t2a/dit/qwen35_0p8b_300m_sceneplan_v2.json",
    "stable_audio_tools/configs/model_configs/txt2audio/t2a/dit/qwen35_0p8b_300m.json",
    "stable_audio_tools/configuration.py",
    "stable_audio_tools/data/dataset.py",
    "stable_audio_tools/data/sceneplan_v2.py",
    "stable_audio_tools/data/sceneplan_v2_dataset.py",
    "stable_audio_tools/models/conditioners.py",
    "stable_audio_tools/models/diffusion.py",
    "stable_audio_tools/models/dit.py",
    "stable_audio_tools/models/sceneplan_conditioning.py",
    "train.py",
    "scripts/t2a/data/sceneplan_v2_common.py",
    "scripts/t2a/data/build_sceneplan_speech_catalog_v2.py",
    "scripts/t2a/data/select_sceneplan_speech_splits_v2.py",
    "scripts/t2a/data/audit_sceneplan_speech_catalog_v2.py",
    "scripts/t2a/data/finalize_sceneplan_speech_ledger_v2.py",
    "scripts/t2a/data/build_sceneplan_nonspeech_catalog_v2.py",
    "scripts/t2a/data/audit_sceneplan_nonspeech_signal_v2.py",
    "scripts/t2a/data/render_tts_v2_pilot.py",
    "scripts/t2a/data/audit_tts_v2_pilot.py",
    "scripts/t2a/data/calibrate_tts_v2_cutoff_gate.py",
    "scripts/t2a/data/replace_tts_v2_pilot_quarantine.py",
    "scripts/t2a/data/build_sceneplan_manifests_v2.py",
    "scripts/t2a/data/materialize_sceneplan_v2_shard.py",
    "scripts/t2a/data/materialize_sceneplan_v2_worker.py",
    "scripts/t2a/data/run_sceneplan_v2_p6.py",
    "scripts/t2a/data/run_sceneplan_v2_p8.py",
    "scripts/t2a/data/audit_sceneplan_manifests_v2.py",
    "scripts/t2a/data/audit_sceneplan_materialized_v2.py",
    "scripts/t2a/data/build_sceneplan_training_index_v2.py",
    "scripts/t2a/data/sceneplan_v2_renderer.py",
    "scripts/t2a/data/validate_sceneplan_renderer_v2_contract.py",
    "scripts/t2a/data/verify_sceneplan_renderer_v2_acoustics.py",
    "scripts/t2a/data/verify_sceneplan_v2_conditioning.py",
    "scripts/t2a/data/verify_sceneplan_v2_p6_loader.py",
    "scripts/t2a/data/freeze_sceneplan_v2_p9.py",
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
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_ok(path: Path, expected_schema: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != expected_schema or value.get("ok") is not True:
        raise RuntimeError(f"required P9 gate report is not all-pass: {path}")
    return value


def atomic_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    pq.write_table(pa.Table.from_pylist(rows, schema=INVENTORY_SCHEMA), temporary, compression="zstd")
    if pq.read_metadata(temporary).num_rows != len(rows):
        raise RuntimeError("frozen artifact inventory row count changed after reopen")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--checkpoint-root", type=Path, default=Path("/mnt/sdb/ckpts/sceneplan_dit_v2"))
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
    p7 = require_ok(
        root / "qc/p7_sceneplan_manifest_audit.json",
        "stable_audio_tools.sceneplan_manifest_audit",
    )
    p9 = require_ok(
        root / "qc/p9_materialized_dataset_audit.json",
        "stable_audio_tools.sceneplan_materialized_dataset_audit",
    )
    p4_speech = require_ok(
        root
        / "pilots/tts_2k/qc/asr_distil_large_v3_calibrated/speech_qc_summary.json",
        "stable_audio_tools.tts_v2_pilot_speech_qc_summary",
    )
    speech_ledger_qc = require_ok(
        root / "split_ledgers/speech_v2/strong_qc_summary.json",
        "stable_audio_tools.sceneplan_speech_split_ledger_strong_qc",
    )
    p6_planned = require_ok(
        root / "pilots/joint_4k/qc/planned_manifest_audit.json",
        "stable_audio_tools.sceneplan_manifest_audit",
    )
    p6_materialized = require_ok(
        root / "pilots/joint_4k/qc/materialized_audit.json",
        "stable_audio_tools.sceneplan_materialized_dataset_audit",
    )
    p6_loader = require_ok(
        root / "pilots/joint_4k/qc/loader_conditioner_smoke.json",
        "stable_audio_tools.sceneplan_v2_p6_loader_conditioner_smoke",
    )
    p6_orchestrator = json.loads(
        (root / "pilots/joint_4k/materialized/p6_orchestrator_summary.json").read_text(
            encoding="utf-8"
        )
    )
    p8_summary_path = root / "materialized/p8_orchestrator_summary.json"
    p8 = json.loads(p8_summary_path.read_text(encoding="utf-8"))
    if int(p7["rows"]) != 1_124_000 or int(p9["rows"]) != 1_124_000:
        raise RuntimeError("P7/P9 report row totals are not 1,124,000")
    if int(p4_speech["rows"]) != 2_000 or p4_speech.get("status_counts") != {"pass": 2_000}:
        raise RuntimeError("P4 complete-speech/FOA strong-ASR gate is not 2000/2000 pass")
    if int(speech_ledger_qc["formal_rows"]) != 512_000:
        raise RuntimeError("final strong-QC speech ledger is not exactly 512,000 rows")
    if int(p6_planned["rows"]) != 4_000 or int(p6_materialized["rows"]) != 4_000:
        raise RuntimeError("P6 planned/materialized joint pilot is not exactly 4,000 rows")
    if (
        int(p6_loader.get("dataset_rows", -1)) != 4_000
        or p6_loader.get("collated_latent_shape") != [2, 64, 432]
        or p6_loader.get("fusion_output_shape") != [2, 256, 432]
        or p6_loader.get("p10_training_started") is not False
    ):
        raise RuntimeError("P6 real loader/conditioner smoke gate is incomplete")
    if (
        p6_orchestrator.get("status") != "complete"
        or int(p6_orchestrator.get("rows", -1)) != 4_000
        or p6_orchestrator.get("p10_training_started") is not False
    ):
        raise RuntimeError("P6 materialization orchestrator gate is not complete")
    if (
        p8.get("status") != "complete"
        or int(p8.get("rows", -1)) != 1_124_000
        or p8.get("p10_training_started") is not False
    ):
        raise RuntimeError("P8 orchestrator is not complete or the P10 stop gate drifted")
    training_summary_path = root / "training_index/summary.json"
    training_summary = json.loads(training_summary_path.read_text(encoding="utf-8"))
    if int(training_summary.get("rows", -1)) != 1_124_000:
        raise RuntimeError("frozen training-index summary row total mismatch")
    if list((root / "materialized/quarantine").glob("*/*.json")):
        raise RuntimeError("materialization quarantine is not empty at P9")
    train_render_root = root / "materialized/renders/train"
    if train_render_root.exists() and any(path.is_file() for path in train_render_root.rglob("*")):
        raise RuntimeError("transient train FOA/render files remain at P9")
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    if list(checkpoint_root.rglob("*.ckpt")) or list(checkpoint_root.rglob("*.safetensors")):
        raise RuntimeError("P10 checkpoint root is not empty; training may already have started")

    contract_root = root / "contracts"
    snapshot_root = contract_root / "p9_code_snapshot"
    for relative_text in SNAPSHOT_FILES:
        source = (REPO_ROOT / relative_text).resolve(strict=True)
        destination = snapshot_root / relative_text
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    qwen_root = Path("/mnt/sdc/ckpts/pretrained/Qwen/Qwen3.5-0.8B").resolve(strict=True)
    vae_checkpoint = Path(
        "/mnt/sdc/ckpts/compareVAE_ckpt/unwrapped_wdmix_1350000.ckpt"
    ).resolve(strict=True)
    external_dependency_rows = []
    for path in [*sorted(qwen_root.rglob("*")), vae_checkpoint]:
        if not path.is_file():
            continue
        external_dependency_rows.append(
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
    external_dependency_path = contract_root / "external_dependency_manifest.json"
    atomic_write_json(
        external_dependency_path,
        {
            "schema": "stable_audio_tools.sceneplan_v2_external_dependency_manifest",
            "schema_version": 1,
            "dependencies": external_dependency_rows,
        },
    )
    approval_gate = {
        "schema": "stable_audio_tools.p10_user_approval_gate",
        "schema_version": 1,
        "state": "closed_waiting_for_user_acceptance",
        "p10_training_started": False,
        "initialization": "from_scratch",
        "legacy_300k_checkpoint": "reference_only_never_load",
        "model_config": str(
            REPO_ROOT
            / "stable_audio_tools/configs/model_configs/txt2audio/t2a/dit/"
            "qwen35_0p8b_300m_sceneplan_v2.json"
        ),
        "train_dataset_config": str(
            REPO_ROOT / "stable_audio_tools/configs/dataset_configs/sceneplan_v2_train.json"
        ),
        "validation_dataset_config": str(
            REPO_ROOT
            / "stable_audio_tools/configs/dataset_configs/sceneplan_v2_validation.json"
        ),
        "checkpoint_root": str(checkpoint_root),
    }
    atomic_write_json(contract_root / "P10_REQUIRES_USER_APPROVAL.json", approval_gate)

    inventory: dict[str, dict[str, Any]] = {}

    def add(path: Path, role: str, expected_sha: str | None = None, verified: str = "freeze_rehash") -> None:
        resolved = path.expanduser().resolve(strict=True)
        try:
            resolved.relative_to("/mnt/sdb")
        except ValueError as error:
            raise ValueError(f"frozen artifact is not on SDB: {resolved}") from error
        digest = expected_sha or sha256_file(resolved)
        if len(digest) != 64:
            raise RuntimeError(f"invalid artifact checksum: {resolved}")
        key = str(resolved)
        row = {
            "path": key,
            "role": role,
            "num_bytes": resolved.stat().st_size,
            "sha256": digest,
            "checksum_verification": verified,
        }
        previous = inventory.get(key)
        if previous is not None and previous["sha256"] != digest:
            raise RuntimeError(f"conflicting frozen checksums: {resolved}")
        inventory[key] = row

    for path in sorted((root / "sceneplans").rglob("*")):
        if path.is_file():
            add(path, "sceneplan_manifest")
    for path in sorted((root / "source_catalog").rglob("*")):
        if path.is_file():
            add(path, "source_catalog")
    for path in sorted((root / "models").rglob("*")):
        if path.is_file():
            add(path, "frozen_quality_control_model")
    for path in sorted((root / "split_ledgers").rglob("*")):
        if path.is_file():
            add(path, "split_ledger")
    for path in sorted((root / "training_index").glob("*")):
        if path.is_file():
            add(path, "training_index")
    for path in sorted((root / "qc").rglob("*")):
        if path.is_file():
            add(path, "quality_report")
    for path in sorted((root / "audit").rglob("*")):
        if path.is_file():
            add(path, "audit_provenance")
    for path in (
        root / "pilots/tts_2k/render_summary.json",
        root
        / "pilots/tts_2k/qc/asr_distil_large_v3_calibrated/speech_qc_summary.json",
        root
        / "pilots/tts_2k/qc/cutoff_calibration_distil_large_v3/summary.json",
        root / "pilots/joint_4k/qc/planned_manifest_audit.json",
        root / "pilots/joint_4k/qc/materialized_audit.json",
        root / "pilots/joint_4k/qc/loader_conditioner_smoke.json",
        root / "pilots/joint_4k/training_index/summary.json",
        root / "pilots/joint_4k/materialized/p6_orchestrator_summary.json",
        p8_summary_path,
    ):
        add(path, "stage_gate_quality_report")
    for path in sorted(contract_root.rglob("*")):
        if path.is_file():
            add(path, "contract_or_code_snapshot")

    manifests = sorted(
        (root / "materialized/manifests").glob("*/materialized-*.parquet")
    )
    for manifest in manifests:
        add(manifest, "materialized_manifest")
        rows = pq.read_table(
            manifest,
            columns=[
                "split",
                "latent_ref",
                "latent_shard_sha256",
                "render_result_json",
            ],
        ).to_pylist()
        for row in rows:
            latent_path = Path(str(row["latent_ref"]).split("#", 1)[0])
            add(
                latent_path,
                "variable_latent_shard",
                str(row["latent_shard_sha256"]),
                "p9_exhaustive_checksum_verified",
            )
            if row["split"] == "train":
                continue
            result = json.loads(row["render_result_json"])
            add(
                Path(result["foa_path"]).parent / "render_result.json",
                "retained_eval_render_result",
            )
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

    rows = sorted(inventory.values(), key=lambda row: row["path"])
    inventory_path = contract_root / "frozen_artifact_manifest.parquet"
    atomic_parquet(inventory_path, rows)
    inventory_sha = sha256_file(inventory_path)
    freeze_manifest = {
        "schema": "stable_audio_tools.sceneplan_v2_p9_freeze_manifest",
        "schema_version": 2,
        "dataset_id": "sceneplan_renderer_v2_1p124m",
        "contract_revision": 4,
        "state": "P9_complete_frozen_waiting_for_user_acceptance",
        "p10_training_started": False,
        "rows": 1_124_000,
        "splits": {"train": 1_100_000, "validation": 20_000, "test": 4_000},
        "speech_rows": {"train": 500_000, "validation": 10_000, "test": 2_000},
        "no_speech_rows": {"train": 600_000, "validation": 10_000, "test": 2_000},
        "max_model_samples": 442_368,
        "max_latent_frames": 432,
        "random_crop": False,
        "artifact_manifest": str(inventory_path),
        "artifact_manifest_sha256": inventory_sha,
        "external_dependency_manifest": str(external_dependency_path),
        "external_dependency_manifest_sha256": sha256_file(external_dependency_path),
        "artifact_count": len(rows),
        "artifact_bytes": sum(int(row["num_bytes"]) for row in rows),
        "p4_complete_speech_foa_asr_report": str(
            root
            / "pilots/tts_2k/qc/asr_distil_large_v3_calibrated/speech_qc_summary.json"
        ),
        "p6_planned_report": str(
            root / "pilots/joint_4k/qc/planned_manifest_audit.json"
        ),
        "p6_materialized_report": str(
            root / "pilots/joint_4k/qc/materialized_audit.json"
        ),
        "p6_loader_conditioner_report": str(
            root / "pilots/joint_4k/qc/loader_conditioner_smoke.json"
        ),
        "p8_summary": str(p8_summary_path),
        "p7_report": str(root / "qc/p7_sceneplan_manifest_audit.json"),
        "p7_report_sha256": sha256_file(root / "qc/p7_sceneplan_manifest_audit.json"),
        "p9_report": str(root / "qc/p9_materialized_dataset_audit.json"),
        "p9_report_sha256": sha256_file(root / "qc/p9_materialized_dataset_audit.json"),
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
        "schema_version": 2,
        "state": freeze_manifest["state"],
        "freeze_manifest": str(freeze_path),
        "freeze_manifest_sha256": sha256_file(freeze_path),
        "p10_training_started": False,
    }
    atomic_write_json(marker, marker_value)
    print(json.dumps({**freeze_manifest, "marker": str(marker)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
