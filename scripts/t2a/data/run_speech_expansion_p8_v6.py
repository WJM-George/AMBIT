#!/usr/bin/env python3
"""Launch and monitor eight-GPU P8 for the revision-6 500k delta."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.t2a.data.sceneplan_v2_common import atomic_write_json  # noqa: E402


DATASET_ROOT = Path("/mnt/sdb/audio_dataset/sceneplan_v2_1p124m")
REVISION_ROOT = DATASET_ROOT / "revisions/speech_expansion_noalign_15s_v1"
DEFAULT_SCENEPLANS = REVISION_ROOT / "sceneplans_model_v2_delta"
DEFAULT_OUTPUT = REVISION_ROOT / "materialized_delta"
DEFAULT_TMPFS = Path("/dev/shm/sceneplan_speech_expansion_p8")


def pid_alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except OSError:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sceneplan-root", type=Path, default=DEFAULT_SCENEPLANS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--jobs-per-worker", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--train-render-root", type=Path, default=DEFAULT_TMPFS)
    args = parser.parse_args()
    sceneplans = args.sceneplan_root.expanduser().resolve(strict=True)
    output = args.output_root.expanduser().resolve(strict=False)
    transient = args.train_render_root.expanduser().resolve(strict=False)
    revisions = (DATASET_ROOT / "revisions").resolve(strict=True)
    for path in (sceneplans, output):
        try:
            path.relative_to(revisions)
        except ValueError as error:
            raise ValueError(f"revision-6 P8 path is outside revisions: {path}") from error
    try:
        transient.relative_to("/dev/shm")
    except ValueError as error:
        raise ValueError("transient train FOA must remain under /dev/shm") from error
    ready = json.loads((sceneplans / "READY").read_text(encoding="utf-8"))
    audit = json.loads((sceneplans / "audit.json").read_text(encoding="utf-8"))
    if (
        int(ready.get("rows", -1)) != 500_000
        or audit.get("ok") is not True
        or int(audit.get("dataset_contract_revision", -1)) != 6
        or int(audit.get("rows", -1)) != 500_000
        or audit.get("p8_started") is not False
    ):
        raise RuntimeError("revision-6 P7.5 audit is not an all-pass P8 gate")
    gpus = [int(value) for value in args.gpus.split(",") if value.strip()]
    if len(gpus) != 8 or len(set(gpus)) != 8:
        raise ValueError("revision-6 P8 requires eight distinct GPUs")
    shards = sorted((sceneplans / "train").glob("model-sceneplans-train-*.jsonl"))
    if len(shards) != int(audit["shards"]):
        raise RuntimeError("revision-6 P8 shard coverage changed")
    transient.mkdir(parents=True, exist_ok=True)
    stat = os.statvfs(transient)
    free = stat.f_bavail * stat.f_frsize
    if free < 96 * 1024**3:
        raise RuntimeError(f"tmpfs has less than 96 GiB free: {free}")
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / "p8_orchestrator_state.json"
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        live = [pid for pid in state.get("worker_pids", []) if pid_alive(int(pid))]
        if live:
            raise RuntimeError(f"revision-6 P8 workers are already live: {live}")
    worker = SCRIPT_DIR / "materialize_model_sceneplan_v1_worker.py"
    log_root = output / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment.update(
        {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    processes: list[subprocess.Popen[str]] = []
    logs = []
    started = time.monotonic()
    try:
        for worker_index, gpu in enumerate(gpus):
            log_path = log_root / f"worker-{worker_index:02d}.log"
            handle = log_path.open("a", encoding="utf-8", buffering=1)
            logs.append(handle)
            command = [
                sys.executable,
                str(worker),
                "--sceneplan-root",
                str(sceneplans),
                "--output-root",
                str(output),
                "--splits",
                "train",
                "--gpu",
                str(gpu),
                "--worker-index",
                str(worker_index),
                "--worker-count",
                str(len(gpus)),
                "--jobs",
                str(args.jobs_per_worker),
                "--batch-size",
                str(args.batch_size),
                "--train-render-root",
                str(transient),
                "--supplement-mode",
                "--contract-revision",
                "6",
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
                "schema": "stable_audio_tools.speech_expansion_p8_state",
                "schema_version": 1,
                "dataset_contract_revision": 6,
                "status": "running",
                "worker_pids": [process.pid for process in processes],
                "gpus": gpus,
                "planned_shards": len(shards),
                "jobs_per_worker": args.jobs_per_worker,
                "batch_size": args.batch_size,
                "started_unix": time.time(),
            },
        )
        last_done = -1
        while True:
            codes = [process.poll() for process in processes]
            if any(code not in (None, 0) for code in codes):
                raise RuntimeError(f"revision-6 P8 worker failure: {codes}")
            done = sorted((output / "work_done/train").glob("work-*.json"))
            if len(done) != last_done:
                rows = sum(
                    int(json.loads(path.read_text(encoding="utf-8"))["rows"])
                    for path in done
                )
                storage = os.statvfs("/mnt/sdb")
                free_fraction = storage.f_bavail / storage.f_blocks
                if free_fraction < 0.20:
                    raise RuntimeError(
                        f"SDB free fraction fell below 20%: {free_fraction}"
                    )
                print(
                    json.dumps(
                        {
                            "done_shards": len(done),
                            "total_shards": len(shards),
                            "rows": rows,
                            "sdb_free_fraction": free_fraction,
                            "elapsed_sec": round(time.monotonic() - started, 1),
                        }
                    ),
                    flush=True,
                )
                last_done = len(done)
            if all(code == 0 for code in codes):
                break
            time.sleep(15)
    except Exception:
        for process in processes:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        for process in processes:
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        atomic_write_json(
            state_path,
            {
                "schema": "stable_audio_tools.speech_expansion_p8_state",
                "schema_version": 1,
                "dataset_contract_revision": 6,
                "status": "failed",
                "worker_pids": [process.pid for process in processes],
                "exit_codes": [process.poll() for process in processes],
                "elapsed_sec": round(time.monotonic() - started, 3),
            },
        )
        raise
    finally:
        for handle in logs:
            handle.close()
    done = sorted((output / "work_done/train").glob("work-*.json"))
    rows = sum(int(json.loads(path.read_text(encoding="utf-8"))["rows"]) for path in done)
    if len(done) != len(shards) or rows != 500_000:
        raise RuntimeError("revision-6 P8 workers exited with incomplete coverage")
    result = {
        "schema": "stable_audio_tools.speech_expansion_p8_summary",
        "schema_version": 1,
        "dataset_contract_revision": 6,
        "status": "complete",
        "rows": rows,
        "shards": len(done),
        "workers": len(processes),
        "gpus": gpus,
        "jobs_per_worker": args.jobs_per_worker,
        "batch_size": args.batch_size,
        "train_foa_retained": False,
        "elapsed_sec": round(time.monotonic() - started, 3),
        "p10_training_started": False,
    }
    atomic_write_json(output / "P8_SUMMARY.json", result)
    atomic_write_json(state_path, result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
