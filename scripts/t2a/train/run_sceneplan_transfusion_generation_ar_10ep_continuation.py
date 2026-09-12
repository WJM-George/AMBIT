#!/usr/bin/env python3
"""Queue the immutable Generation AR epoch-6--10 continuation after P10 baseline."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any


SNAPSHOT_ROOT = Path(__file__).resolve().parents[3]
PARENT_RUN = Path(
    "/mnt/sdc/ckpts/transfusion_sceneplan/generation_ar/"
    "p10v11_shared_gen_ar_full_1p6m_5ep_s42_20260904_v1"
)
PARENT_CHECKPOINT = PARENT_RUN / "checkpoints/step_00041670.pt"
PARENT_SELECTION = PARENT_RUN / "CHECKPOINT_SELECTION.json"
PARENT_POSTTRAIN_STATUS = PARENT_RUN / "posttrain/POSTTRAIN_STATUS.json"
RUN_DIR = Path(
    "/mnt/sdc/ckpts/transfusion_sceneplan/generation_ar/"
    "p10v11_shared_gen_ar_full_1p6m_10ep_s42_20260904_stage2_v1"
)
TRAIN_MANIFEST = Path(
    "/mnt/sdb/audio_dataset/sceneplan_v2_1p124m/transfusion_shared_v1/"
    "generation_ar/train.sqlite"
)
VALIDATION_MANIFEST = TRAIN_MANIFEST.with_name("validation.sqlite")
TEST_MANIFEST = TRAIN_MANIFEST.with_name("test.sqlite")
VENV_ROOT = Path("/mnt/sdc/stable-audio-tools-venv")
QWEN_ROOT = Path("/mnt/sdc/ckpts/pretrained/Qwen/Qwen3.5-0.8B")
P10_CHECKPOINT = Path(
    "/mnt/sdc/ckpts/dit/sceneplan_dit_v11_semantic_v2_protected_resume_150k/"
    "checkpoints/epoch=48-step=150000.ckpt"
)
P10_MODEL_CONFIG = Path(
    "/mnt/sdc/stable-audio-tools-workspace/stable_audio_tools/configs/"
    "model_configs/txt2audio/t2a/dit/"
    "qwen35_0p8b_300m_model_sceneplan_44_soundexp_noalign_15s_"
    "resume_cosine_40k.json"
)
EXPECTED_PARENT = {
    "checkpoint_sha256": "037efa7999d81d28a8d75e4ad80c40e6160169b5f9cf3d748de7fe0003336e84",
    "selection_sha256": "4045e96fd16727dde92866ad819e5f70c38a7abc40fc9fd1adb6b722dd582461",
    "run_contract_sha256": "7b356444c9a629b51b6862182c412099cfb9c0521c5d4c8edb09fca188600dfc",
    "final_sha256": "5c2eaca20a800892217e88977f0112303ef2bb612905bd5d50bf367b0d5f9b13",
}
TRAIN_BATCH_SIZE_PER_RANK = 64
TRAIN_NUM_WORKERS_PER_RANK = 8
ONLINE_VALIDATION_EVERY_STEPS = 4167


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _snapshot_tree_sha256(root: Path) -> str:
    """Match `find -print0 | sort -z | xargs -0 sha256sum | sha256sum`."""

    digest = hashlib.sha256()
    files = sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file()
            and not path.is_symlink()
            and path.name != "SOURCE_SNAPSHOT_MANIFEST.json"
        ),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    for path in files:
        relative = path.relative_to(root).as_posix()
        digest.update(f"{_sha256_file(path)}  ./{relative}\n".encode("utf-8"))
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
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


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def _validate_snapshot() -> dict[str, Any]:
    manifest_path = (SNAPSHOT_ROOT / "SOURCE_SNAPSHOT_MANIFEST.json").resolve(
        strict=True
    )
    manifest = _read_json(manifest_path)
    if (
        manifest.get("status") != "IMMUTABLE_FOR_GENERATION_AR_RUN"
        or Path(str(manifest.get("snapshot_root", ""))).resolve() != SNAPSHOT_ROOT
        or _snapshot_tree_sha256(SNAPSHOT_ROOT)
        != manifest.get("tree_sha256_excluding_this_manifest")
    ):
        raise RuntimeError("Generation AR continuation source snapshot changed")
    for relative, expected in dict(manifest.get("key_source_sha256") or {}).items():
        if _sha256_file(SNAPSHOT_ROOT / relative) != expected:
            raise RuntimeError(f"Generation AR snapshot key source changed: {relative}")
    return manifest


def _validate_external_artifacts(manifest: dict[str, Any]) -> None:
    data_paths = {
        "train.sqlite": TRAIN_MANIFEST,
        "validation.sqlite": VALIDATION_MANIFEST,
        "test.sqlite": TEST_MANIFEST,
    }
    for name, expected in dict(manifest["generation_data_sha256"]).items():
        if _sha256_file(data_paths[name].resolve(strict=True)) != expected:
            raise RuntimeError(f"Generation AR immutable data changed: {name}")
    for name, expected in dict(manifest["qwen3_5_0p8b_sha256"]).items():
        if _sha256_file((QWEN_ROOT / name).resolve(strict=True)) != expected:
            raise RuntimeError(f"Generation AR frozen Qwen artifact changed: {name}")
    if (
        _sha256_file(P10_CHECKPOINT.resolve(strict=True))
        != manifest["p10_v11_checkpoint_sha256"]
        or _sha256_file(P10_MODEL_CONFIG.resolve(strict=True))
        != manifest["p10_v11_model_config_sha256"]
    ):
        raise RuntimeError("Generation AR frozen P10-v11 artifact changed")


def _validate_parent() -> dict[str, Any]:
    artifacts = {
        "checkpoint_sha256": PARENT_CHECKPOINT,
        "selection_sha256": PARENT_SELECTION,
        "run_contract_sha256": PARENT_RUN / "RUN_CONTRACT.json",
        "final_sha256": PARENT_RUN / "FINAL.json",
    }
    for key, path in artifacts.items():
        if _sha256_file(path.resolve(strict=True)) != EXPECTED_PARENT[key]:
            raise RuntimeError(f"Generation AR continuation parent changed: {key}")
    selection = _read_json(PARENT_SELECTION)
    if (
        selection.get("status") != "COMPLETE"
        or Path(str(selection.get("selected_checkpoint", ""))).resolve()
        != PARENT_CHECKPOINT
        or int(selection.get("selected_checkpoint_step", -1)) != 41670
    ):
        raise RuntimeError("Generation AR continuation parent selection is invalid")
    return selection


def _run(command: list[str], *, log_path: Path, environment: dict[str, str]) -> None:
    with log_path.open("a", encoding="utf-8") as log:
        log.write(
            json.dumps(
                {"event": "stage_start", "time_unix": time.time(), "command": command},
                sort_keys=True,
            )
            + "\n"
        )
        log.flush()
        completed = subprocess.run(
            command,
            cwd=SNAPSHOT_ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
        log.write(
            json.dumps(
                {
                    "event": "stage_end",
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
            f"Generation AR queued stage failed ({completed.returncode}): {log_path}"
        )


def _rewind_metrics_to_checkpoint(global_step: int) -> None:
    metrics_path = RUN_DIR / "metrics.jsonl"
    if not metrics_path.is_file():
        return
    lines = metrics_path.read_text(encoding="utf-8").splitlines()
    retained: list[str] = []
    for line in lines:
        try:
            event = json.loads(line)
        except (TypeError, ValueError):
            continue
        if isinstance(event, dict) and int(event.get("step", -1)) <= int(global_step):
            retained.append(json.dumps(event, ensure_ascii=False, sort_keys=True))
    history = RUN_DIR / "queue/attempt_history"
    history.mkdir(parents=True, exist_ok=True)
    archived = history / f"metrics_before_resume_{time.time_ns()}.jsonl"
    shutil.copy2(metrics_path, archived)
    _atomic_text(metrics_path, "".join(f"{line}\n" for line in retained))


def _latest_resume_checkpoint() -> Path | None:
    contract_path = RUN_DIR / "RUN_CONTRACT.json"
    final_path = RUN_DIR / "FINAL.json"
    if final_path.is_file() or not contract_path.is_file():
        return None
    latest_path = RUN_DIR / "checkpoints/LATEST.json"
    if not latest_path.is_file():
        _rewind_metrics_to_checkpoint(41670)
        return None
    latest = _read_json(latest_path.resolve(strict=True))
    checkpoint = Path(str(latest.get("checkpoint", ""))).resolve(strict=True)
    if checkpoint.parent != (RUN_DIR / "checkpoints").resolve(strict=True):
        raise RuntimeError("Generation AR continuation LATEST escaped its run")
    if latest.get("checkpoint_sha256") != _sha256_file(checkpoint):
        raise RuntimeError("Generation AR continuation LATEST checksum changed")
    _rewind_metrics_to_checkpoint(int(latest.get("global_step", -1)))
    return checkpoint


def main() -> int:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    queue_dir = RUN_DIR / "queue"
    queue_dir.mkdir(exist_ok=True)
    lock_handle = (queue_dir / "LOCK").open("w", encoding="utf-8")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError(
            "Generation AR continuation queue is already active"
        ) from error
    status_path = queue_dir / "STATUS.json"
    if status_path.is_file():
        attempt_history = queue_dir / "attempt_history"
        attempt_history.mkdir(exist_ok=True)
        shutil.copy2(
            status_path,
            attempt_history / f"status_before_restart_{time.time_ns()}.json",
        )
    manifest = _validate_snapshot()
    _validate_external_artifacts(manifest)
    _validate_parent()
    _atomic_json(
        status_path,
        {
            "status": "WAITING_FOR_PARENT_P10_BASELINE",
            "parent_run": str(PARENT_RUN),
            "run_dir": str(RUN_DIR),
            "snapshot_root": str(SNAPSHOT_ROOT),
            "snapshot_manifest_sha256": _sha256_file(
                SNAPSHOT_ROOT / "SOURCE_SNAPSHOT_MANIFEST.json"
            ),
            "snapshot_tree_sha256": manifest["tree_sha256_excluding_this_manifest"],
            "started_unix": time.time(),
        },
    )
    while True:
        try:
            parent_status = _read_json(PARENT_POSTTRAIN_STATUS)
        except (FileNotFoundError, ValueError, TypeError, RuntimeError):
            parent_status = {}
        if parent_status.get("status") == "GENERATION_EVALUATION_COMPLETE":
            if (
                parent_status.get("selected_checkpoint_sha256")
                != EXPECTED_PARENT["checkpoint_sha256"]
                or int(
                    _read_json(Path(parent_status["p10_audio8k_summary"]))
                    .get("coverage", {})
                    .get("rows", -1)
                )
                != 8000
            ):
                raise RuntimeError(
                    "completed parent P10 status is not the frozen baseline"
                )
            break
        if parent_status.get("status") == "FAILED":
            raise RuntimeError("parent five-epoch P10 baseline failed")
        time.sleep(60)

    manifest = _validate_snapshot()
    _validate_external_artifacts(manifest)
    _validate_parent()
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = "0,1,2"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    torchrun = str((VENV_ROOT / "bin/torchrun").resolve(strict=True))
    trainer = str(
        (
            SNAPSHOT_ROOT
            / "scripts/t2a/train/train_sceneplan_transfusion_generation_ar.py"
        ).resolve(strict=True)
    )
    training_command = [
        torchrun,
        "--standalone",
        "--nproc_per_node=3",
        trainer,
        "--mode",
        "full",
        "--run-dir",
        str(RUN_DIR),
        "--train-manifest",
        str(TRAIN_MANIFEST),
        "--validation-manifest",
        str(VALIDATION_MANIFEST),
        "--batch-size",
        str(TRAIN_BATCH_SIZE_PER_RANK),
        "--grad-accum",
        "1",
        "--epochs",
        "10",
        "--learning-rate",
        "0.00003",
        "--weight-decay",
        "0.01",
        "--warmup-steps",
        "0",
        "--num-workers",
        str(TRAIN_NUM_WORKERS_PER_RANK),
        "--log-every",
        "10",
        "--validate-every",
        str(ONLINE_VALIDATION_EVERY_STEPS),
        "--validation-batches",
        "64",
        "--length-bucket-batches",
        "64",
        "--save-every",
        "4167",
        "--checkpoint-policy",
        "interval",
        "--seed",
        "42",
        "--no-activation-checkpointing",
    ]
    resume = _latest_resume_checkpoint()
    if resume is None and not (RUN_DIR / "FINAL.json").is_file():
        training_command.extend(
            [
                "--extend-from",
                str(PARENT_CHECKPOINT),
                "--parent-selection-manifest",
                str(PARENT_SELECTION),
            ]
        )
    elif resume is not None:
        training_command.extend(["--resume", str(resume)])
    if not (RUN_DIR / "FINAL.json").is_file():
        _atomic_json(
            status_path,
            {
                "status": "RUNNING_EPOCHS_6_TO_10",
                "command": training_command,
                "run_dir": str(RUN_DIR),
                "started_unix": time.time(),
            },
        )
        _run(
            training_command,
            log_path=queue_dir / "train_epochs_6_to_10.log",
            environment=environment,
        )

    postrunner = str(
        (
            SNAPSHOT_ROOT
            / "scripts/t2a/test/run_sceneplan_transfusion_generation_ar_posttrain.py"
        ).resolve(strict=True)
    )
    posttrain_command = [
        str((VENV_ROOT / "bin/python").resolve(strict=True)),
        postrunner,
        "--run-dir",
        str(RUN_DIR),
        "--execution-root",
        str(SNAPSHOT_ROOT),
        "--audio-execution-root",
        str(SNAPSHOT_ROOT),
        "--venv-root",
        str(VENV_ROOT),
        "--validation-manifest",
        str(VALIDATION_MANIFEST),
        "--test-manifest",
        str(TEST_MANIFEST),
        "--poll-seconds",
        "30",
    ]
    _atomic_json(
        status_path,
        {
            "status": "RUNNING_10_EPOCH_POSTTRAIN",
            "command": posttrain_command,
            "run_dir": str(RUN_DIR),
            "started_unix": time.time(),
        },
    )
    _run(
        posttrain_command,
        log_path=queue_dir / "posttrain_10epoch.log",
        environment=environment,
    )
    post_status = _read_json(RUN_DIR / "posttrain/POSTTRAIN_STATUS.json")
    if post_status.get("status") != "GENERATION_EVALUATION_COMPLETE":
        raise RuntimeError("10-epoch Generation AR post-training closure is incomplete")
    _atomic_json(
        status_path,
        {
            "status": "COMPLETE",
            "run_dir": str(RUN_DIR),
            "posttrain_status": str(RUN_DIR / "posttrain/POSTTRAIN_STATUS.json"),
            "posttrain_status_sha256": _sha256_file(
                RUN_DIR / "posttrain/POSTTRAIN_STATUS.json"
            ),
            "completed_unix": time.time(),
        },
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BaseException as error:
        try:
            _atomic_json(
                RUN_DIR / "queue/STATUS.json",
                {
                    "status": "FAILED",
                    "reason": f"{type(error).__name__}: {error}",
                    "failed_unix": time.time(),
                },
            )
        except Exception:
            pass
        raise
