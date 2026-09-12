#!/usr/bin/env python3
"""Wait for the approved P7 run, then finalize and audit P7.5.

This orchestrator deliberately stops after the compact 1.124M ScenePlan audit.
It never invokes FOA rendering, VAE encoding, P8, P9, or model training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
DATASET_ROOT = Path("/mnt/sdb/audio_dataset/sceneplan_v2_1p124m")
SOURCE_ROOT = DATASET_ROOT / "source_annotations/nonspeech_instruct_v2"
ANNOTATION_ROOT = SOURCE_ROOT / "annotations"
REGISTRY_ROOT = SOURCE_ROOT / "registry"
SCENEPLAN_ROOT = DATASET_ROOT / "sceneplans_model_v1"
EXPECTED_SHARD_ROWS = (218_376, 218_376, 218_375, 218_375)
ANNOTATION_PATHS = tuple(
    ANNOTATION_ROOT
    / f"source_descriptions_instruct.shard{shard:03d}-of-004.jsonl"
    for shard in range(4)
)
SUMMARY_PATHS = tuple(path.with_suffix(".summary.json") for path in ANNOTATION_PATHS)
SPOKEN_LABELS = ANNOTATION_ROOT / "spoken_language_background_v1.jsonl"
STATE_PATH = SOURCE_ROOT / "p7_p75_orchestrator_state.json"
FINAL_REPORT = DATASET_ROOT / "audit/p7_p75_completion_report.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as sink:
        json.dump(value, sink, ensure_ascii=False, indent=2)
        sink.write("\n")
        sink.flush()
        os.fsync(sink.fileno())
    os.replace(temporary, path)


def line_count(path: Path) -> tuple[int, bool]:
    if not path.is_file():
        return 0, False
    count = 0
    incomplete_tail = False
    with path.open("rb") as source:
        for raw in source:
            incomplete_tail = not raw.endswith(b"\n")
            if raw.strip() and not incomplete_tail:
                count += 1
    return count, incomplete_tail


def annotation_state() -> dict[str, Any]:
    counted = [line_count(path) for path in ANNOTATION_PATHS]
    counts = [value[0] for value in counted]
    incomplete_tails = [value[1] for value in counted]
    for shard, (actual, expected) in enumerate(zip(counts, EXPECTED_SHARD_ROWS)):
        if actual > expected:
            raise RuntimeError(
                f"P7 shard {shard} contains {actual} rows, expected at most {expected}"
            )
    summaries_present = [path.is_file() for path in SUMMARY_PATHS]
    return {
        "stage": "waiting_for_p7_annotations",
        "annotation_rows": sum(counts),
        "expected_annotation_rows": sum(EXPECTED_SHARD_ROWS),
        "rows_by_shard": counts,
        "expected_rows_by_shard": list(EXPECTED_SHARD_ROWS),
        "incomplete_tails_while_writing": incomplete_tails,
        "worker_summaries_present": summaries_present,
        "p8_started": False,
        "p9_started": False,
    }


def validate_worker_summaries() -> None:
    for shard, (path, expected_rows) in enumerate(
        zip(SUMMARY_PATHS, EXPECTED_SHARD_ROWS)
    ):
        value = json.loads(path.read_text(encoding="utf-8"))
        if (
            value.get("schema")
            != "stable_audio_tools.sceneplan_a2t_transformers_run"
            or int(value.get("shard", -1)) != shard
            or int(value.get("num_shards", -1)) != 4
            or int(value.get("assigned_shard_rows_scanned", -1)) != expected_rows
            or value.get("input_scan_complete") is not True
            or int(value.get("generation_capped", -1)) != 0
        ):
            raise RuntimeError(f"invalid completed P7 worker summary: {path}")


def annotation_workers_running() -> bool:
    for command_path in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            command = command_path.read_bytes().replace(b"\0", b" ")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if b"caption_transformers.py" in command:
            return True
    return False


def wait_for_annotations(poll_seconds: int) -> None:
    last_report = 0.0
    missing_worker_polls = 0
    while True:
        state = annotation_state()
        complete_rows = state["rows_by_shard"] == list(EXPECTED_SHARD_ROWS)
        complete_summaries = all(state["worker_summaries_present"])
        clean_tails = not any(state["incomplete_tails_while_writing"])
        atomic_json(STATE_PATH, state)
        now = time.monotonic()
        if now - last_report >= 300 or (complete_rows and complete_summaries):
            print(json.dumps(state, ensure_ascii=False), flush=True)
            last_report = now
        if complete_rows and complete_summaries and clean_tails:
            validate_worker_summaries()
            return
        missing_worker_polls = (
            0 if annotation_workers_running() else missing_worker_polls + 1
        )
        if missing_worker_polls >= 2:
            raise RuntimeError(
                "P7 annotation workers exited before all durable rows/summaries arrived"
            )
        time.sleep(poll_seconds)


def run_stage(name: str, command: list[str]) -> None:
    state = {
        "stage": name,
        "command": command,
        "p8_started": False,
        "p9_started": False,
    }
    atomic_json(STATE_PATH, state)
    print(json.dumps(state, ensure_ascii=False), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument(
        "--resume-from-finalized-registry",
        action="store_true",
        help="Skip completed P7 classification/finalization and restart P7.5.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 10 <= args.poll_seconds <= 60:
        raise ValueError("--poll-seconds must be between 10 and 60")
    # Keep the virtual-environment launcher path intact.  Resolving it can
    # follow the venv's Python symlink back to the base Conda interpreter and
    # silently drop packages installed only in the venv (for example pyarrow).
    python = sys.executable
    classifier = REPO_ROOT / "dataset/captioning/sceneplan_a2t_v2/classify_spoken_language.py"
    finalizer = REPO_ROOT / "dataset/captioning/sceneplan_a2t_v2/finalize_source_registry.py"
    builder = REPO_ROOT / "scripts/t2a/data/build_model_sceneplan_manifests_v1.py"
    auditor = REPO_ROOT / "scripts/t2a/data/audit_model_sceneplan_manifests_v1.py"

    if args.resume_from_finalized_registry:
        registry_audit = REGISTRY_ROOT / "finalizer_audit.json"
        value = json.loads(registry_audit.read_text(encoding="utf-8"))
        if (
            value.get("registry_finalized") is not True
            or value.get("complete") is not True
            or int(value.get("expected_rows", -1)) != 873_502
            or int(value.get("annotation_rows", -1)) != 873_502
            or int(value.get("missing_annotation_rows", -1)) != 0
            or int(value.get("missing_spoken_label_rows", -1)) != 0
        ):
            raise RuntimeError("cannot resume P7.5 from an invalid P7 registry")
        atomic_json(
            STATE_PATH,
            {
                "stage": "resume_p75_from_finalized_p7_registry",
                "registry_audit": str(registry_audit),
                "p8_started": False,
                "p9_started": False,
            },
        )
    else:
        wait_for_annotations(args.poll_seconds)
        classify_command = [python, str(classifier)]
        for path in ANNOTATION_PATHS:
            classify_command.extend(["--input-jsonl", str(path)])
        classify_command.extend(["--out", str(SPOKEN_LABELS)])
        run_stage("classify_spoken_language_background", classify_command)
        run_stage(
            "finalize_p7_registry",
            [
                python,
                str(finalizer),
                "--spoken-labels-jsonl",
                str(SPOKEN_LABELS),
                "--finalize",
            ],
        )
    run_stage(
        "build_p75_model_sceneplans",
        [python, str(builder), "--mode", "full"],
    )
    run_stage(
        "audit_p75_model_sceneplans",
        [python, str(auditor), "--mode", "full"],
    )

    registry_audit = REGISTRY_ROOT / "finalizer_audit.json"
    sceneplan_summary = SCENEPLAN_ROOT / "summary.json"
    sceneplan_audit = SCENEPLAN_ROOT / "audit.json"
    registry_value = json.loads(registry_audit.read_text(encoding="utf-8"))
    build_value = json.loads(sceneplan_summary.read_text(encoding="utf-8"))
    audit_value = json.loads(sceneplan_audit.read_text(encoding="utf-8"))
    if (
        registry_value.get("registry_finalized") is not True
        or int(registry_value.get("expected_rows", -1)) != 873_502
        or int(build_value.get("rows", -1)) != 1_124_000
        or audit_value.get("ok") is not True
        or int(audit_value.get("rows", -1)) != 1_124_000
    ):
        raise RuntimeError("P7/P7.5 completion artifacts do not satisfy frozen counts")
    report = {
        "schema": "stable_audio_tools.p7_p75_completion_report",
        "schema_version": 1,
        "state": "P7_and_P7.5_complete_waiting_before_P8",
        "p7_registry_rows": 873_502,
        "p75_sceneplan_rows": 1_124_000,
        "registry_audit": str(registry_audit),
        "registry_audit_sha256": sha256_file(registry_audit),
        "sceneplan_summary": str(sceneplan_summary),
        "sceneplan_summary_sha256": sha256_file(sceneplan_summary),
        "sceneplan_audit": str(sceneplan_audit),
        "sceneplan_audit_sha256": sha256_file(sceneplan_audit),
        "foa_materialization_started": False,
        "latent_materialization_started": False,
        "p8_started": False,
        "p9_started": False,
        "p10_training_started": False,
        "p11_training_started": False,
    }
    atomic_json(FINAL_REPORT, report)
    atomic_json(STATE_PATH, report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
