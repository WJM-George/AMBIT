#!/usr/bin/env python3
"""Run revision-5 P9 audit, index, loader smoke, and freeze in order."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from sceneplan_v2_common import DATASET_ROOT, atomic_write_json, require_dataset_not_frozen  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    require_dataset_not_frozen()
    root = DATASET_ROOT.resolve(strict=True)
    p8_path = root / "materialized/p8_orchestrator_summary.json"
    p8 = json.loads(p8_path.read_text(encoding="utf-8"))
    if not (
        p8.get("schema") == "stable_audio_tools.model_sceneplan_p8_orchestrator_summary"
        and p8.get("status") == "complete"
        and int(p8.get("rows", -1)) == 1_124_000
        and int(p8.get("dataset_contract_revision", -1)) == 5
    ):
        raise RuntimeError("P9 cannot start before formal revision-5 P8 completes")
    state_path = root / "audit/p9_revision5_orchestrator_state.json"
    log_path = root / "audit/p9_revision5_orchestrator.log"
    stages = [
        (
            "materialized_speaker_delta_audit",
            [sys.executable, str(SCRIPT_DIR / "audit_model_sceneplan_materialized_delta_v1.py")],
        ),
        (
            "training_index",
            [sys.executable, str(SCRIPT_DIR / "build_model_sceneplan_training_index_v1.py")],
        ),
        (
            "loader_smoke",
            [sys.executable, str(SCRIPT_DIR / "verify_model_sceneplan_v1_loader.py")],
        ),
        (
            "freeze",
            [sys.executable, str(SCRIPT_DIR / "freeze_model_sceneplan_v1_p9.py")],
        ),
    ]
    started = time.time()
    completed: list[str] = []
    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        for name, command in stages:
            atomic_write_json(
                state_path,
                {
                    "schema": "stable_audio_tools.model_sceneplan_p9_orchestrator_state",
                    "schema_version": 1,
                    "dataset_contract_revision": 5,
                    "status": "running",
                    "stage": name,
                    "completed_stages": completed,
                    "elapsed_sec": round(time.time() - started, 3),
                    "p10_training_started": False,
                    "p11_training_started": False,
                },
            )
            print(json.dumps({"p9_stage": name, "state": "starting"}), flush=True)
            result = subprocess.run(
                command,
                cwd=REPO_ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                atomic_write_json(
                    state_path,
                    {
                        "schema": "stable_audio_tools.model_sceneplan_p9_orchestrator_state",
                        "schema_version": 1,
                        "dataset_contract_revision": 5,
                        "status": "failed",
                        "stage": name,
                        "returncode": result.returncode,
                        "completed_stages": completed,
                        "elapsed_sec": round(time.time() - started, 3),
                        "p10_training_started": False,
                        "p11_training_started": False,
                    },
                )
                raise RuntimeError(f"P9 stage failed: {name}; see {log_path}")
            completed.append(name)
            print(json.dumps({"p9_stage": name, "state": "complete"}), flush=True)
    marker = root / "FROZEN_P9.json"
    marker_value = json.loads(marker.read_text(encoding="utf-8"))
    if not (
        int(marker_value.get("dataset_contract_revision", -1)) == 5
        and marker_value.get("state") == "P9_complete_frozen_waiting_for_user_acceptance"
        and marker_value.get("p10_training_started") is False
        and marker_value.get("p11_training_started") is False
    ):
        raise RuntimeError("P9 freeze marker is incomplete")
    report = {
        "schema": "stable_audio_tools.p8_p9_revision5_completion_report",
        "schema_version": 1,
        "dataset_contract_revision": 5,
        "state": "P8_and_P9_complete_frozen_waiting_for_user_acceptance_before_P10",
        "rows": 1_124_000,
        "p8_summary": str(p8_path),
        "p8_summary_sha256": sha256_file(p8_path),
        "p9_materialized_audit": str(
            root / "qc/p9_model_sceneplan_materialized_audit.json"
        ),
        "p9_materialized_audit_sha256": sha256_file(
            root / "qc/p9_model_sceneplan_materialized_audit.json"
        ),
        "training_index_summary": str(root / "training_index/summary.json"),
        "training_index_summary_sha256": sha256_file(
            root / "training_index/summary.json"
        ),
        "loader_smoke": str(root / "qc/p9_model_sceneplan_loader_smoke.json"),
        "loader_smoke_sha256": sha256_file(
            root / "qc/p9_model_sceneplan_loader_smoke.json"
        ),
        "freeze_marker": str(marker),
        "freeze_marker_sha256": sha256_file(marker),
        "elapsed_sec": round(time.time() - started, 3),
        "p10_training_started": False,
        "p11_training_started": False,
    }
    report_path = root / "audit/p8_p9_revision5_completion_report.json"
    atomic_write_json(report_path, report)
    atomic_write_json(
        state_path,
        {
            "schema": "stable_audio_tools.model_sceneplan_p9_orchestrator_state",
            "schema_version": 1,
            "dataset_contract_revision": 5,
            "status": "complete",
            "completed_stages": completed,
            "completion_report": str(report_path),
            "elapsed_sec": report["elapsed_sec"],
            "p10_training_started": False,
            "p11_training_started": False,
        },
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
