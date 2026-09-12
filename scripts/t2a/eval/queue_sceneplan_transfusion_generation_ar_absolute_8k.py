#!/usr/bin/env python3
"""Queue the absolute-GT Generation-AR audio benchmark after post-training.

The queue is deliberately passive while training and its existing post-train
closure own GPUs 0--2.  Once that run is fully complete, it resolves the
selected checkpoint's 8K ScenePlan evaluation and launches the AR-only
persisted-FOA renderer plus the frozen cross-system metric suite.  The already
complete high-precision GT-ScenePlan P10 oracle is reused from frozen metrics.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback
from typing import Any


CONTRACT = "p10v11_generation_ar_absolute_gt_8k_queue_ar_only_v1"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpu-list", default="0,1,2")
    parser.add_argument("--poll-seconds", type=int, default=60)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON artifact is not an object: {path}")
    return value


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _source_paths(repo_root: Path) -> dict[str, Path]:
    relative = {
        "queue": "scripts/t2a/eval/queue_sceneplan_transfusion_generation_ar_absolute_8k.py",
        "runner": "scripts/t2a/eval/run_sceneplan_transfusion_generation_ar_absolute_8k_3gpu.sh",
        "renderer": "scripts/t2a/eval/render_sceneplan_transfusion_generation_ar_absolute_8k.py",
        "scorer": "scripts/t2a/eval/score_sceneplan_transfusion_generation_ar_absolute_8k.py",
        "p10_audio_evaluator": "scripts/t2a/test/evaluate_sceneplan_transfusion_generation_ar_p10_audio_8k.py",
        "content_scorer": "scripts/t2a/eval/baselines/score_p10_final_content_benchmark.py",
        "content_helpers": "scripts/t2a/eval/baselines/score_p10_60k_cross_system.py",
        "frechet_helper": "scripts/t2a/eval/baselines/score_p10_v11_stratified_3000_public_baselines.py",
        "panel_common": "scripts/t2a/eval/sceneplan_dit_p10_panel_common.py",
        "audio_io": "scripts/t2a/eval/sceneplan_44_eval_common.py",
        "spatial_core": "scripts/t2a/eval/score_sceneplan_dit_p10_core.py",
        "p10_executor": "stable_audio_tools/inference/sceneplan_cot.py",
        "p10_finalize": "stable_audio_tools/data/sceneplan_p11_single_turn.py",
        "fad_metrics": "stable_audio_tools/training/metrics/fad_metrics.py",
    }
    return {
        key: (repo_root / path).resolve(strict=True)
        for key, path in sorted(relative.items())
    }


def _snapshot_sources(
    paths: dict[str, Path], snapshot_root: Path
) -> dict[str, dict[str, Any]]:
    snapshot_root.mkdir(parents=True, exist_ok=True)
    identity: dict[str, dict[str, Any]] = {}
    for key, path in paths.items():
        digest = _sha256(path)
        archived = snapshot_root / f"{key}__{digest[:16]}{path.suffix}"
        if not archived.exists():
            shutil.copy2(path, archived)
            archived.chmod(0o444)
        if _sha256(archived) != digest:
            raise RuntimeError(f"source snapshot hash mismatch: {key}")
        identity[key] = {
            "path": str(path),
            "sha256": digest,
            "bytes": int(path.stat().st_size),
            "snapshot": str(archived),
        }
    return identity


def _verify_sources(identity: dict[str, dict[str, Any]]) -> None:
    for key, value in identity.items():
        path = Path(str(value["path"])).resolve(strict=True)
        snapshot = Path(str(value["snapshot"])).resolve(strict=True)
        if _sha256(path) != value["sha256"] or _sha256(snapshot) != value["sha256"]:
            raise RuntimeError(f"queued evaluation source changed: {key}")


def _gpu_processes(gpu_list: str) -> list[str]:
    command = [
        "nvidia-smi",
        "-i",
        gpu_list,
        "--query-compute-apps=pid",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _completed_comparison(
    output_dir: Path,
    *,
    checkpoint_sha256: str,
    test_summary: Path,
) -> Path | None:
    marker = output_dir / "ABSOLUTE_GT_EVALUATION_COMPLETE"
    comparison = output_dir / "metrics/absolute_gt/COMPARISON.json"
    if not marker.is_file() or not comparison.is_file():
        return None
    try:
        if marker.read_text(encoding="utf-8") != "PASS\n":
            return None
        report = _read_json(comparison)
        identity = dict(report["absolute_gt_identity"])
        render_contract = _read_json(
            Path(str(identity["render_run_contract"])).resolve(strict=True)
        )
        if not (
            report.get("status") == "PASS"
            and identity.get("generation_ar_checkpoint_sha256")
            == checkpoint_sha256
            and render_contract["plan_evaluation"].get("summary")
            == str(test_summary)
            and render_contract["plan_evaluation"].get("summary_sha256")
            == _sha256(test_summary)
        ):
            return None
        return comparison.resolve(strict=True)
    except (KeyError, OSError, TypeError, ValueError):
        return None


def _wait_for_posttrain(
    run_dir: Path, status_path: Path, poll_seconds: int
) -> dict[str, Any]:
    queue_status_path = run_dir / "queue/STATUS.json"
    posttrain_status_path = run_dir / "posttrain/POSTTRAIN_STATUS.json"
    while True:
        queue_status = (
            _read_json(queue_status_path) if queue_status_path.is_file() else {}
        )
        if queue_status.get("status") == "FAILED":
            raise RuntimeError("Generation-AR training/posttrain queue failed")
        posttrain = (
            _read_json(posttrain_status_path)
            if posttrain_status_path.is_file()
            else {}
        )
        if (
            queue_status.get("status") == "COMPLETE"
            and posttrain.get("status") == "GENERATION_EVALUATION_COMPLETE"
        ):
            return posttrain
        _atomic_json(
            status_path,
            {
                "contract": CONTRACT,
                "status": "WAITING_FOR_GENERATION_POSTTRAIN",
                "run_dir": str(run_dir),
                "observed_generation_status": queue_status.get("status"),
                "observed_posttrain_status": posttrain.get("status"),
                "updated_unix": time.time(),
            },
        )
        time.sleep(poll_seconds)


def main() -> int:
    args = _parse_args()
    if args.poll_seconds < 10:
        raise ValueError("poll-seconds must be at least 10")
    if len([value for value in args.gpu_list.split(",") if value]) != 3:
        raise ValueError("gpu-list must contain exactly three GPU indices")
    run_dir = args.run_dir.expanduser().resolve(strict=True)
    repo_root = args.repo_root.expanduser().resolve(strict=True)
    output_root = args.output_root.expanduser().resolve()
    queue_root = output_root / "queue" / run_dir.name
    queue_root.mkdir(parents=True, exist_ok=True)
    lock_handle = (queue_root / "LOCK").open("w", encoding="utf-8")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("absolute-GT queue is already active") from error
    status_path = queue_root / "STATUS.json"
    source_identity = _snapshot_sources(
        _source_paths(repo_root), queue_root / "SOURCE_SNAPSHOT"
    )
    _atomic_json(
        queue_root / "QUEUE_CONTRACT.json",
        {
            "contract": CONTRACT,
            "run_dir": str(run_dir),
            "repo_root": str(repo_root),
            "output_root": str(output_root),
            "gpu_list": args.gpu_list,
            "source": source_identity,
            "created_unix": time.time(),
        },
    )

    posttrain = _wait_for_posttrain(run_dir, status_path, args.poll_seconds)
    _verify_sources(source_identity)
    selected_checkpoint = Path(str(posttrain["selected_checkpoint"])).resolve(
        strict=True
    )
    selected_sha = _sha256(selected_checkpoint)
    if selected_sha != posttrain.get("selected_checkpoint_sha256"):
        raise RuntimeError("selected Generation-AR checkpoint hash changed")
    test_summary = Path(str(posttrain["test8k_summary"])).resolve(strict=True)
    if _sha256(test_summary) != posttrain.get("test8k_summary_sha256"):
        raise RuntimeError("selected Generation-AR 8K summary hash changed")
    test_value = _read_json(test_summary)
    if test_value.get("status") != "PASS" or int(test_value.get("rows", -1)) != 8_000:
        raise RuntimeError("selected Generation-AR 8K evaluation is incomplete")

    output_dir = output_root / "by_checkpoint" / selected_sha
    existing = _completed_comparison(
        output_dir,
        checkpoint_sha256=selected_sha,
        test_summary=test_summary,
    )
    if existing is not None:
        _atomic_json(
            status_path,
            {
                "contract": CONTRACT,
                "status": "COMPLETE",
                "selected_checkpoint": str(selected_checkpoint),
                "selected_checkpoint_sha256": selected_sha,
                "output_dir": str(output_dir),
                "comparison": str(existing),
                "comparison_sha256": _sha256(existing),
                "reused_completed_checkpoint_evaluation": True,
                "completed_unix": time.time(),
            },
        )
        return 0

    while True:
        processes = _gpu_processes(args.gpu_list)
        if not processes:
            break
        _atomic_json(
            status_path,
            {
                "contract": CONTRACT,
                "status": "WAITING_FOR_GPUS",
                "gpu_list": args.gpu_list,
                "active_compute_pids": processes,
                "selected_checkpoint": str(selected_checkpoint),
                "updated_unix": time.time(),
            },
        )
        time.sleep(args.poll_seconds)

    _verify_sources(source_identity)
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        "bash",
        str(source_identity["runner"]["path"]),
    ]
    environment = dict(os.environ)
    environment.update(
        {
            "REPO_ROOT": str(repo_root),
            "PLAN_EVALUATION_DIR": str(test_summary.parent),
            "OUTPUT_DIR": str(output_dir),
            "GPU_LIST": args.gpu_list,
        }
    )
    _atomic_json(
        status_path,
        {
            "contract": CONTRACT,
            "status": "RUNNING_ABSOLUTE_GT_EVALUATION",
            "command": command,
            "plan_evaluation_dir": str(test_summary.parent),
            "selected_checkpoint": str(selected_checkpoint),
            "selected_checkpoint_sha256": selected_sha,
            "output_dir": str(output_dir),
            "started_unix": time.time(),
        },
    )
    log_path = queue_root / "absolute_gt_pipeline.log"
    with log_path.open("a", encoding="utf-8") as log:
        subprocess.run(
            command,
            cwd=str(repo_root),
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    comparison = (output_dir / "metrics/absolute_gt/COMPARISON.json").resolve(
        strict=True
    )
    report = _read_json(comparison)
    if report.get("status") != "PASS":
        raise RuntimeError("absolute-GT comparison did not pass")
    _atomic_json(
        status_path,
        {
            "contract": CONTRACT,
            "status": "COMPLETE",
            "selected_checkpoint": str(selected_checkpoint),
            "selected_checkpoint_sha256": selected_sha,
            "output_dir": str(output_dir),
            "comparison": str(comparison),
            "comparison_sha256": _sha256(comparison),
            "completed_unix": time.time(),
        },
    )
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
    except Exception as error:
        try:
            parsed = _parse_args()
            failure_root = (
                parsed.output_root.expanduser().resolve()
                / "queue"
                / parsed.run_dir.expanduser().resolve(strict=False).name
            )
            _atomic_json(
                failure_root / "STATUS.json",
                {
                    "contract": CONTRACT,
                    "status": "FAILED",
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(),
                    "failed_unix": time.time(),
                },
            )
        except Exception:
            pass
        raise
    raise SystemExit(exit_code)
