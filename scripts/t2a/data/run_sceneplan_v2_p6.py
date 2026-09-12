#!/usr/bin/env python3
"""Materialize the audited 4k joint P6 pilot on four GPUs, retaining FOA/stems."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from sceneplan_v2_common import (  # noqa: E402
    DATASET_ROOT,
    atomic_write_json,
    require_dataset_not_frozen,
)


def pid_alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except OSError:
        return False


def terminate_groups(processes: list[subprocess.Popen]) -> None:
    for process in processes:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.time() + 30
    for process in processes:
        remaining = max(0.0, deadline - time.time())
        try:
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def main() -> int:
    require_dataset_not_frozen()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sceneplan-root", type=Path, default=DATASET_ROOT / "pilots/joint_4k/sceneplans"
    )
    parser.add_argument(
        "--output-root", type=Path, default=DATASET_ROOT / "pilots/joint_4k/materialized"
    )
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--jobs-per-gpu", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    sceneplan_root = args.sceneplan_root.expanduser().resolve(strict=True)
    output_root = args.output_root.expanduser().resolve(strict=False)
    try:
        sceneplan_root.relative_to(os.environ.get("AMBIT_DATA_ROOT", "data"))
        output_root.relative_to(os.environ.get("AMBIT_DATA_ROOT", "data"))
    except ValueError as error:
        raise ValueError("P6 ScenePlans/materialization must remain on SDB") from error
    ready = json.loads((sceneplan_root / "READY").read_text(encoding="utf-8"))
    if ready.get("mode") != "pilot" or int(ready.get("rows", -1)) != 4_000:
        raise RuntimeError("P6 ScenePlan READY marker is not the frozen 4k pilot")
    planned_audit = json.loads(
        (DATASET_ROOT / "pilots/joint_4k/qc/planned_manifest_audit.json").read_text(
            encoding="utf-8"
        )
    )
    if planned_audit.get("ok") is not True or int(planned_audit.get("rows", -1)) != 4_000:
        raise RuntimeError("P6 planned ScenePlan audit is not all-pass")
    shards = sorted((sceneplan_root / "train").glob("sceneplans-train-*.parquet"))
    gpu_indices = [int(value) for value in args.gpus.split(",") if value.strip()]
    if len(shards) != 4 or len(gpu_indices) != 4 or len(set(gpu_indices)) != 4:
        raise RuntimeError("formal P6 requires exactly four pilot shards and four distinct GPUs")
    output_root.mkdir(parents=True, exist_ok=True)
    state_path = output_root / "p6_orchestrator_state.json"
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        live = [int(pid) for pid in state.get("worker_pids", []) if pid_alive(int(pid))]
        if live:
            raise RuntimeError(f"P6 worker processes are already live: {live}")
    log_root = output_root / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    script = SCRIPT_DIR / "materialize_sceneplan_v2_shard.py"
    environment = dict(os.environ)
    environment.update(
        {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    processes: list[subprocess.Popen] = []
    handles = []
    started = time.time()
    try:
        for worker, (gpu, shard) in enumerate(zip(gpu_indices, shards)):
            log_path = log_root / f"worker-{worker:02d}.log"
            handle = log_path.open("a", encoding="utf-8", buffering=1)
            handles.append(handle)
            command = [
                sys.executable,
                str(script),
                "--sceneplan-shard",
                str(shard),
                "--output-root",
                str(output_root),
                "--gpu",
                str(gpu),
                "--jobs",
                str(args.jobs_per_gpu),
                "--batch-size",
                str(args.batch_size),
                "--retain-stems",
            ]
            processes.append(
                subprocess.Popen(
                    command,
                    cwd=REPO_ROOT,
                    env=environment,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                    start_new_session=True,
                )
            )
        atomic_write_json(
            state_path,
            {
                "schema": "stable_audio_tools.sceneplan_p6_orchestrator_state",
                "schema_version": 2,
                "status": "running",
                "worker_pids": [process.pid for process in processes],
                "gpus": gpu_indices,
                "shards": [str(path) for path in shards],
                "started_unix": started,
            },
        )
        while True:
            codes = [process.poll() for process in processes]
            if any(code not in (None, 0) for code in codes):
                raise RuntimeError(f"P6 worker failure: {codes}")
            done = sorted((output_root / "work_done/train").glob("work-*.json"))
            print(
                json.dumps(
                    {
                        "p6_done_shards": len(done),
                        "p6_total_shards": 4,
                        "elapsed_sec": round(time.time() - started, 1),
                    }
                ),
                flush=True,
            )
            if all(code == 0 for code in codes):
                break
            time.sleep(15)
    except Exception:
        terminate_groups(processes)
        atomic_write_json(
            state_path,
            {
                "schema": "stable_audio_tools.sceneplan_p6_orchestrator_state",
                "schema_version": 2,
                "status": "failed",
                "exit_codes": [process.poll() for process in processes],
                "elapsed_sec": round(time.time() - started, 3),
            },
        )
        raise
    finally:
        for handle in handles:
            handle.close()
    done = sorted((output_root / "work_done/train").glob("work-*.json"))
    rows = sum(int(json.loads(path.read_text(encoding="utf-8"))["rows"]) for path in done)
    quarantines = list((output_root / "quarantine").glob("*/*.json"))
    if len(done) != 4 or rows != 4_000 or quarantines:
        raise RuntimeError(
            f"P6 completion mismatch: shards={len(done)}, rows={rows}, quarantine={quarantines}"
        )
    summary = {
        "schema": "stable_audio_tools.sceneplan_p6_orchestrator_summary",
        "schema_version": 2,
        "status": "complete",
        "rows": rows,
        "shards": len(done),
        "gpus": gpu_indices,
        "retain_foa": True,
        "retain_source_stems": True,
        "p10_training_started": False,
        "elapsed_sec": round(time.time() - started, 3),
    }
    atomic_write_json(output_root / "p6_orchestrator_summary.json", summary)
    atomic_write_json(state_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
