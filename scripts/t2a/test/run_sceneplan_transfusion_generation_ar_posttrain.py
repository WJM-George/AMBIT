#!/usr/bin/env python3
"""Wait for Generation AR training, select on validation, then run smoke and 8K."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stable_audio_tools.data.sceneplan_transfusion_generation_ar_contract import (  # noqa: E402
    canonical_sha256 as _canonical_sha256,
    expected_checkpoint_steps as _expected_checkpoint_steps,
    expected_selection_candidate_steps as _expected_selection_steps,
    sha256_file as _sha256_file,
    validate_parent_lineage_artifacts as _validated_parent_checkpoint,
)

DEFAULT_EVALUATION_ROOT = Path(
    os.environ.get("AMBIT_DATA_ROOT", "data") + "/sceneplan_v2_1p124m/transfusion_shared_v1/"
    "generation_ar/evaluation"
)
DEFAULT_VALIDATION_MANIFEST = Path(
    os.environ.get("AMBIT_DATA_ROOT", "data") + "/sceneplan_v2_1p124m/transfusion_shared_v1/"
    "generation_ar/validation.sqlite"
)
DEFAULT_TEST_MANIFEST = Path(
    os.environ.get("AMBIT_DATA_ROOT", "data") + "/sceneplan_v2_1p124m/transfusion_shared_v1/"
    "generation_ar/test.sqlite"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, default=DEFAULT_EVALUATION_ROOT)
    parser.add_argument(
        "--execution-root",
        type=Path,
        default=REPO_ROOT,
        help="Immutable source tree used by selector and evaluator.",
    )
    parser.add_argument(
        "--audio-execution-root",
        type=Path,
        help=(
            "Optional immutable source tree for the P10 audio evaluator. "
            "Defaults to --execution-root."
        ),
    )
    parser.add_argument(
        "--venv-root",
        type=Path,
        default=REPO_ROOT / ".venv",
        help="Environment containing torchrun (kept outside the source snapshot).",
    )
    parser.add_argument(
        "--validation-manifest", type=Path, default=DEFAULT_VALIDATION_MANIFEST
    )
    parser.add_argument("--test-manifest", type=Path, default=DEFAULT_TEST_MANIFEST)
    parser.add_argument("--poll-seconds", type=int, default=30)
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


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _has_pass_summary(path: Path) -> bool:
    try:
        return _read_json(path).get("status") == "PASS"
    except (OSError, ValueError, TypeError, RuntimeError):
        return False


def _validated_selection(
    *, selection_path: Path, run_dir: Path, execution_root: Path
) -> dict[str, Any]:
    selection = _read_json(selection_path.resolve(strict=True))
    contract_path = (run_dir / "RUN_CONTRACT.json").resolve(strict=True)
    final_path = (run_dir / "FINAL.json").resolve(strict=True)
    contract = _read_json(contract_path)
    requested_steps = int(contract.get("requested_steps", -1))
    expected_selection_steps = _expected_selection_steps(contract)
    parent_checkpoint = _validated_parent_checkpoint(contract)
    expected_candidate_policy = (
        "selected_parent_plus_every_stage_half_epoch_full_32k_validation"
        if parent_checkpoint is not None
        else (
            "every_half_epoch_full_32k_validation"
            if contract.get("checkpoint_policy") == "interval"
            else "half_epoch_and_epoch_end_only"
        )
    )
    candidates = list(selection.get("candidates") or ())
    if (
        selection.get("schema")
        != "stable_audio_tools.sceneplan_transfusion_generation_ar_checkpoint_selection"
        or selection.get("status") != "COMPLETE"
        or selection.get("training_run_dir") != str(run_dir)
        or selection.get("training_run_contract_sha256") != _sha256_file(contract_path)
        or selection.get("training_final_sha256") != _sha256_file(final_path)
        or int(selection.get("training_final_step", -1)) != requested_steps
        or selection.get("candidate_policy") != expected_candidate_policy
        or sorted(int(row.get("step", -1)) for row in candidates)
        != expected_selection_steps
        or int(selection.get("validation_rows", -1)) != 32_000
    ):
        raise RuntimeError("existing checkpoint selection identity mismatch")
    expected_validation = min(
        candidates,
        key=lambda row: (
            float(row["validation"]["loss"]),
            -float(row["validation"]["token_accuracy"]),
            -int(row["step"]),
        ),
    )
    selected_checkpoint = Path(str(selection.get("selected_checkpoint", ""))).resolve(
        strict=True
    )
    if (
        Path(str(expected_validation["checkpoint"])).resolve() != selected_checkpoint
        or selection.get("selected_checkpoint_sha256")
        != _sha256_file(selected_checkpoint)
        or int(selection.get("selected_checkpoint_step", -1))
        != int(expected_validation["step"])
        or selection.get("selected_validation") != expected_validation["validation"]
        or selection.get("selected_lineage_role")
        != expected_validation.get("lineage_role")
    ):
        raise RuntimeError("existing selection is not the validated optimum")
    for candidate in candidates:
        candidate_path = Path(str(candidate["checkpoint"])).resolve(strict=True)
        if (
            (
                candidate_path.parent != (run_dir / "checkpoints").resolve(strict=True)
                and (parent_checkpoint is None or candidate_path != parent_checkpoint)
            )
            or candidate.get("checkpoint_sha256") != _sha256_file(candidate_path)
            or int(candidate.get("validation", {}).get("sequences", -1)) != 32_000
        ):
            raise RuntimeError("existing selection candidate identity mismatch")
    for relative, expected in dict(selection.get("source_sha256") or {}).items():
        source = (execution_root / str(relative)).resolve(strict=True)
        if _sha256_file(source) != str(expected):
            raise RuntimeError(f"checkpoint selector source changed: {relative}")
    return selection


def _validated_plan_evaluation(
    *,
    output_dir: Path,
    split: str,
    rows: int,
    manifest: Path,
    checkpoint_sha256: str,
    selection_path: Path,
    execution_root: Path,
) -> dict[str, Any]:
    contract_path = (output_dir / "RUN_CONTRACT.json").resolve(strict=True)
    summary_path = (output_dir / "SUMMARY.json").resolve(strict=True)
    teacher_path = (output_dir / "TEACHER_FORCED.json").resolve(strict=True)
    contract = _read_json(contract_path)
    summary = _read_json(summary_path)
    teacher = _read_json(teacher_path)
    if (
        int(contract.get("schema_version", -1)) != 2
        or contract.get("checkpoint_sha256") != checkpoint_sha256
        or contract.get("evaluation_split") != split
        or int(contract.get("row_limit", -1)) != int(rows)
        or Path(str(contract.get("evaluation_manifest", {}).get("path", ""))).resolve()
        != manifest
        or contract.get("evaluation_manifest_sha256") != _sha256_file(manifest)
        or contract.get("training_completion_proof", {}).get(
            "selection_manifest_sha256"
        )
        != _sha256_file(selection_path)
        or summary.get("status") != "PASS"
        or summary.get("checkpoint_sha256") != checkpoint_sha256
        or summary.get("evaluation_split") != split
        or int(summary.get("rows", -1)) != int(rows)
        or summary.get("teacher_forced") != teacher
    ):
        raise RuntimeError(f"existing {split} evaluation identity mismatch")
    for relative, expected in dict(contract.get("source_sha256") or {}).items():
        source = (execution_root / str(relative)).resolve(strict=True)
        if _sha256_file(source) != str(expected):
            raise RuntimeError(f"plan evaluator source changed: {relative}")
    contract_sha = _canonical_sha256(contract)
    seen: set[int] = set()
    prediction_dir = output_dir / "predictions"
    for rank in range(3):
        shard_path = (prediction_dir / f"rank_{rank:03d}.sqlite").resolve(strict=True)
        connection = sqlite3.connect(f"file:{shard_path}?mode=ro&immutable=1", uri=True)
        try:
            metadata = dict(connection.execute("SELECT key,value FROM metadata"))
            if (
                metadata.get("status") != "COMPLETE"
                or metadata.get("run_contract_sha256") != contract_sha
            ):
                raise RuntimeError(f"incomplete prediction shard: {shard_path}")
            for ordinal, status in connection.execute(
                "SELECT ordinal,status FROM predictions ORDER BY ordinal"
            ):
                ordinal = int(ordinal)
                if status != "ok" or ordinal % 3 != rank or ordinal in seen:
                    raise RuntimeError(f"invalid prediction row {ordinal}")
                seen.add(ordinal)
        finally:
            connection.close()
    if seen != set(range(int(rows))):
        raise RuntimeError(f"existing {split} evaluation coverage mismatch")
    return summary


def _validated_audio_evaluation(
    *, output_dir: Path, checkpoint_sha256: str
) -> dict[str, Any]:
    summary = _read_json((output_dir / "SUMMARY.json").resolve(strict=True))
    if (
        summary.get("status") != "PASS"
        or summary.get("checkpoint_sha256") != checkpoint_sha256
        or int(summary.get("coverage", {}).get("rows", -1)) != 8_000
    ):
        raise RuntimeError("existing P10 audio evaluation identity mismatch")
    return summary


def _training_is_running(run_dir: Path) -> bool:
    marker = str(run_dir).encode("utf-8")
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (entry / "cmdline").read_bytes()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if (
            b"train_sceneplan_transfusion_generation_ar.py" in command
            and marker in command
        ):
            return True
    return False


def _run_stage(
    *,
    name: str,
    command: list[str],
    log_dir: Path,
    environment: dict[str, str],
    execution_root: Path,
) -> None:
    log_path = log_dir / f"{name}.log"
    started = time.time()
    with log_path.open("a", encoding="utf-8") as log:
        log.write(
            json.dumps(
                {
                    "event": "stage_start",
                    "stage": name,
                    "time_unix": started,
                    "command": command,
                },
                sort_keys=True,
            )
            + "\n"
        )
        log.flush()
        completed = subprocess.run(
            command,
            cwd=execution_root,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
        log.write(
            json.dumps(
                {
                    "event": "stage_end",
                    "stage": name,
                    "time_unix": time.time(),
                    "exit_code": completed.returncode,
                },
                sort_keys=True,
            )
            + "\n"
        )
        log.flush()
        os.fsync(log.fileno())
    if completed.returncode != 0:
        raise RuntimeError(
            f"post-training stage {name} failed with exit code "
            f"{completed.returncode}; see {log_path}"
        )


def main() -> int:
    args = _parse_args()
    if not 1 <= int(args.poll_seconds) <= 60:
        raise ValueError("poll interval must be between 1 and 60 seconds")
    run_dir = args.run_dir.expanduser().resolve(strict=True)
    execution_root = args.execution_root.expanduser().resolve(strict=True)
    audio_execution_root = (
        args.audio_execution_root.expanduser().resolve(strict=True)
        if args.audio_execution_root is not None
        else execution_root
    )
    venv_root = args.venv_root.expanduser().resolve(strict=True)
    validation_manifest = args.validation_manifest.expanduser().resolve(strict=True)
    test_manifest = args.test_manifest.expanduser().resolve(strict=True)
    evaluation_root = args.evaluation_root.expanduser().resolve(strict=False)
    evaluation_root.mkdir(parents=True, exist_ok=True)
    post_dir = run_dir / "posttrain"
    post_dir.mkdir(exist_ok=True)
    status_path = post_dir / "POSTTRAIN_STATUS.json"
    final_path = run_dir / "FINAL.json"
    _atomic_json(
        status_path,
        {
            "status": "WAITING_FOR_TRAINING",
            "run_dir": str(run_dir),
            "execution_root": str(execution_root),
            "audio_execution_root": str(audio_execution_root),
            "started_unix": time.time(),
        },
    )
    while True:
        if final_path.is_file():
            if _training_is_running(run_dir):
                time.sleep(int(args.poll_seconds))
                continue
            break
        if not _training_is_running(run_dir):
            _atomic_json(
                status_path,
                {
                    "status": "FAILED",
                    "reason": "training process exited without FINAL.json",
                    "run_dir": str(run_dir),
                    "failed_unix": time.time(),
                },
            )
            raise RuntimeError("Generation AR training exited without FINAL.json")
        time.sleep(int(args.poll_seconds))

    training_contract = json.loads(
        (run_dir / "RUN_CONTRACT.json").read_text(encoding="utf-8")
    )
    training_final = json.loads(final_path.read_text(encoding="utf-8"))
    if (
        training_final.get("event") != "complete"
        or training_final.get("mode") != "full"
        or int(training_final.get("step", -1))
        != int(training_contract.get("requested_steps", -2))
    ):
        raise RuntimeError("Generation AR FINAL.json is not a completed full run")

    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = "0,1,2"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    torchrun = str((venv_root / "bin/torchrun").resolve(strict=True))
    python = str((venv_root / "bin/python").resolve(strict=True))
    selector = str(
        (
            execution_root
            / "scripts/t2a/test/select_sceneplan_transfusion_generation_ar_checkpoint.py"
        ).resolve(strict=True)
    )
    evaluator = str(
        (
            execution_root
            / "scripts/t2a/test/evaluate_sceneplan_transfusion_generation_ar_8k.py"
        ).resolve(strict=True)
    )
    audio_evaluator = str(
        (
            audio_execution_root / "scripts/t2a/test/"
            "evaluate_sceneplan_transfusion_generation_ar_p10_audio_8k.py"
        ).resolve(strict=True)
    )
    selection_path = run_dir / "CHECKPOINT_SELECTION.json"
    _atomic_json(
        status_path,
        {
            "status": "SELECTING_CHECKPOINT",
            "run_dir": str(run_dir),
            "training_final": str(final_path),
            "training_final_sha256": _sha256_file(final_path),
        },
    )
    if not selection_path.is_file():
        _run_stage(
            name="select_checkpoint_full_validation",
            command=[
                torchrun,
                "--standalone",
                "--nproc_per_node=3",
                selector,
                "--run-dir",
                str(run_dir),
                "--batch-size",
                "64",
                "--num-workers",
                "4",
            ],
            log_dir=post_dir,
            environment=environment,
            execution_root=execution_root,
        )
    selection = _validated_selection(
        selection_path=selection_path,
        run_dir=run_dir,
        execution_root=execution_root,
    )
    selected_checkpoint = Path(selection["selected_checkpoint"]).resolve(strict=True)
    selected_checkpoint_sha256 = _sha256_file(selected_checkpoint)
    if selection.get("selected_lineage_role") == "parent":
        parent = dict(training_contract.get("parent_lineage") or {})
        parent_run_dir = Path(str(parent.get("run_dir", ""))).resolve(strict=True)
        parent_status_path = (
            parent_run_dir / "posttrain/POSTTRAIN_STATUS.json"
        ).resolve(strict=True)
        parent_status = _read_json(parent_status_path)
        required_parent_artifacts = [
            "selection_manifest",
            "smoke_summary",
            "test8k_summary",
            "p10_audio8k_summary",
        ]
        if (
            parent_status.get("status") != "GENERATION_EVALUATION_COMPLETE"
            or parent_status.get("selected_checkpoint_sha256")
            != selected_checkpoint_sha256
        ):
            raise RuntimeError(
                "validation retained the parent, but its Generation evaluation "
                "is not complete"
            )
        for key in required_parent_artifacts:
            artifact = Path(str(parent_status.get(key, ""))).resolve(strict=True)
            expected_sha = parent_status.get(f"{key}_sha256")
            if not isinstance(expected_sha, str) or expected_sha != _sha256_file(
                artifact
            ):
                raise RuntimeError(f"retained parent artifact changed: {key}")
        parent_execution_root = Path(
            str(parent_status.get("execution_root", ""))
        ).resolve(strict=True)
        parent_audio_execution_root = Path(
            str(parent_status.get("audio_execution_root", ""))
        ).resolve(strict=True)
        parent_selection_path = Path(
            str(parent_status["selection_manifest"])
        ).resolve(strict=True)
        parent_test_dir = Path(str(parent_status["test8k_summary"])).resolve(
            strict=True
        ).parent
        parent_audio_dir = Path(
            str(parent_status["p10_audio8k_summary"])
        ).resolve(strict=True).parent
        parent_selection = _validated_selection(
            selection_path=parent_selection_path,
            run_dir=parent_run_dir,
            execution_root=parent_execution_root,
        )
        if (
            parent_selection.get("selected_checkpoint_sha256")
            != selected_checkpoint_sha256
        ):
            raise RuntimeError("retained parent checkpoint selection changed")
        _validated_plan_evaluation(
            output_dir=parent_test_dir,
            split="test",
            rows=8_000,
            manifest=test_manifest,
            checkpoint_sha256=selected_checkpoint_sha256,
            selection_path=parent_selection_path,
            execution_root=parent_execution_root,
        )
        _atomic_json(
            status_path,
            {
                "status": "VERIFYING_RETAINED_PARENT_EVALUATION",
                "selected_checkpoint": str(selected_checkpoint),
                "parent_posttrain_status": str(parent_status_path),
            },
        )
        _run_stage(
            name="verify_retained_parent_p10_audio_8k",
            command=[
                python,
                audio_evaluator,
                "--plan-evaluation-dir",
                str(parent_test_dir),
                "--output-dir",
                str(parent_audio_dir),
                "--test-manifest",
                str(test_manifest),
                "--source-root",
                str(parent_audio_execution_root),
                "--verify-only",
            ],
            log_dir=post_dir,
            environment=environment,
            execution_root=audio_execution_root,
        )
        _validated_audio_evaluation(
            output_dir=parent_audio_dir,
            checkpoint_sha256=selected_checkpoint_sha256,
        )
        retained = {
            "status": "GENERATION_EVALUATION_COMPLETE",
            "execution_root": str(execution_root),
            "audio_execution_root": str(audio_execution_root),
            "selected_checkpoint": str(selected_checkpoint),
            "selected_checkpoint_sha256": selected_checkpoint_sha256,
            "selected_lineage_role": "parent",
            "evaluation_reused_from_parent": True,
            "parent_posttrain_status": str(parent_status_path),
            "parent_posttrain_status_sha256": _sha256_file(parent_status_path),
            "selection_manifest": str(selection_path),
            "selection_manifest_sha256": _sha256_file(selection_path),
            "smoke_summary": parent_status["smoke_summary"],
            "smoke_summary_sha256": parent_status["smoke_summary_sha256"],
            "test8k_summary": parent_status["test8k_summary"],
            "test8k_summary_sha256": parent_status["test8k_summary_sha256"],
            "p10_audio8k_summary": parent_status["p10_audio8k_summary"],
            "p10_audio8k_summary_sha256": parent_status["p10_audio8k_summary_sha256"],
            "completed_unix": time.time(),
        }
        _atomic_json(status_path, retained)
        print(json.dumps(retained, sort_keys=True))
        return 0
    run_contract_sha256 = _sha256_file(run_dir / "RUN_CONTRACT.json")
    evaluation_identity = (
        f"{run_dir.name}_{run_contract_sha256[:12]}_{selected_checkpoint_sha256[:12]}"
    )
    evaluation_dir = evaluation_root / evaluation_identity
    smoke_dir = evaluation_dir / "validation_smoke12"
    full_dir = evaluation_dir / "test8k"
    audio_dir = evaluation_dir / "p10_audio_8k"
    common = [
        torchrun,
        "--standalone",
        "--nproc_per_node=3",
        evaluator,
        "--checkpoint",
        str(selected_checkpoint),
        "--selection-manifest",
        str(selection_path),
        "--batch-size",
        "32",
        "--teacher-batch-size",
        "64",
        "--num-workers",
        "4",
        "--max-plan-tokens",
        "512",
    ]
    _atomic_json(
        status_path,
        {
            "status": "RUNNING_SMOKE12",
            "selected_checkpoint": str(selected_checkpoint),
            "selection_manifest": str(selection_path),
        },
    )
    if not _has_pass_summary(smoke_dir / "SUMMARY.json"):
        _run_stage(
            name="evaluate_validation_smoke12",
            command=common
            + [
                "--output-dir",
                str(smoke_dir),
                "--evaluation-manifest",
                str(validation_manifest),
                "--split",
                "validation",
                "--max-rows",
                "12",
            ],
            log_dir=post_dir,
            environment=environment,
            execution_root=execution_root,
        )
    _validated_plan_evaluation(
        output_dir=smoke_dir,
        split="validation",
        rows=12,
        manifest=validation_manifest,
        checkpoint_sha256=selected_checkpoint_sha256,
        selection_path=selection_path,
        execution_root=execution_root,
    )

    _atomic_json(
        status_path,
        {
            "status": "RUNNING_TEST8K",
            "selected_checkpoint": str(selected_checkpoint),
            "selection_manifest": str(selection_path),
            "smoke_summary": str(smoke_dir / "SUMMARY.json"),
        },
    )
    if not _has_pass_summary(full_dir / "SUMMARY.json"):
        _run_stage(
            name="evaluate_test8k",
            command=common
            + [
                "--output-dir",
                str(full_dir),
                "--evaluation-manifest",
                str(test_manifest),
                "--split",
                "test",
            ],
            log_dir=post_dir,
            environment=environment,
            execution_root=execution_root,
        )
    _validated_plan_evaluation(
        output_dir=full_dir,
        split="test",
        rows=8_000,
        manifest=test_manifest,
        checkpoint_sha256=selected_checkpoint_sha256,
        selection_path=selection_path,
        execution_root=execution_root,
    )

    _atomic_json(
        status_path,
        {
            "status": "RUNNING_P10_AUDIO_8K",
            "selected_checkpoint": str(selected_checkpoint),
            "selected_checkpoint_sha256": selected_checkpoint_sha256,
            "plan_evaluation": str(full_dir),
            "audio_evaluation": str(audio_dir),
        },
    )
    if not _has_pass_summary(audio_dir / "SUMMARY.json"):
        _run_stage(
            name="evaluate_p10_audio_8k",
            command=[
                torchrun,
                "--standalone",
                "--nproc_per_node=3",
                audio_evaluator,
                "--plan-evaluation-dir",
                str(full_dir),
                "--output-dir",
                str(audio_dir),
                "--test-manifest",
                str(test_manifest),
            ],
            log_dir=post_dir,
            environment=environment,
            execution_root=audio_execution_root,
        )
    _run_stage(
        name="verify_p10_audio_8k",
        command=[
            python,
            audio_evaluator,
            "--plan-evaluation-dir",
            str(full_dir),
            "--output-dir",
            str(audio_dir),
            "--test-manifest",
            str(test_manifest),
            "--verify-only",
        ],
        log_dir=post_dir,
        environment=environment,
        execution_root=audio_execution_root,
    )
    _validated_audio_evaluation(
        output_dir=audio_dir,
        checkpoint_sha256=selected_checkpoint_sha256,
    )

    _atomic_json(
        status_path,
        {
            "status": "GENERATION_EVALUATION_COMPLETE",
            "execution_root": str(execution_root),
            "audio_execution_root": str(audio_execution_root),
            "source_snapshot_manifest": (
                str(execution_root / "SOURCE_SNAPSHOT_MANIFEST.json")
                if (execution_root / "SOURCE_SNAPSHOT_MANIFEST.json").is_file()
                else None
            ),
            "selected_checkpoint": str(selected_checkpoint),
            "selected_checkpoint_sha256": selected_checkpoint_sha256,
            "selected_lineage_role": selection.get("selected_lineage_role"),
            "selection_manifest": str(selection_path),
            "selection_manifest_sha256": _sha256_file(selection_path),
            "smoke_summary": str(smoke_dir / "SUMMARY.json"),
            "smoke_summary_sha256": _sha256_file(smoke_dir / "SUMMARY.json"),
            "test8k_summary": str(full_dir / "SUMMARY.json"),
            "test8k_summary_sha256": _sha256_file(full_dir / "SUMMARY.json"),
            "p10_audio8k_summary": str(audio_dir / "SUMMARY.json"),
            "p10_audio8k_summary_sha256": _sha256_file(audio_dir / "SUMMARY.json"),
            "completed_unix": time.time(),
        },
    )
    print(
        json.dumps(json.loads(status_path.read_text(encoding="utf-8")), sort_keys=True)
    )
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
    except BaseException as exc:
        # Keep the durable status truthful even when a subprocess or identity
        # gate fails.  KeyboardInterrupt/SystemExit are recorded as well.
        try:
            run_index = sys.argv.index("--run-dir") + 1
            failed_run_dir = Path(sys.argv[run_index]).expanduser().resolve()
            failed_status = failed_run_dir / "posttrain/POSTTRAIN_STATUS.json"
            _atomic_json(
                failed_status,
                {
                    "status": "FAILED",
                    "reason": f"{type(exc).__name__}: {exc}",
                    "run_dir": str(failed_run_dir),
                    "failed_unix": time.time(),
                },
            )
        except Exception:
            pass
        raise
    raise SystemExit(exit_code)
