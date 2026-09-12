#!/usr/bin/env python3
"""Render and frozen-VAE encode the revision-6 validation/test delta."""

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
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.t2a.data.sceneplan_v2_common import atomic_write_json  # noqa: E402


DATASET_ROOT = Path("/mnt/sdb/audio_dataset/sceneplan_v2_1p124m")
REVISION_ROOT = DATASET_ROOT / "revisions/speech_expansion_noalign_15s_v1"
DEFAULT_SCENEPLANS = REVISION_ROOT / "sceneplans_model_v2_eval_delta"
DEFAULT_OUTPUT = REVISION_ROOT / "materialized_eval_delta"
EXPECTED = {"validation": 12_000, "test": 4_000}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sceneplan-root", type=Path, default=DEFAULT_SCENEPLANS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--jobs-per-worker", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    sceneplans = args.sceneplan_root.expanduser().resolve(strict=True)
    output = args.output_root.expanduser().resolve(strict=False)
    for path in (sceneplans, output):
        try:
            path.relative_to((DATASET_ROOT / "revisions").resolve(strict=True))
        except ValueError as error:
            raise ValueError(f"eval P8 path is outside SDB revision root: {path}") from error
    audit = json.loads((sceneplans / "audit.json").read_text(encoding="utf-8"))
    if audit.get("ok") is not True or audit.get("split_counts") != EXPECTED:
        raise RuntimeError("eval ScenePlan audit is not an all-pass P8 gate")
    gpus = [int(value) for value in args.gpus.split(",") if value.strip()]
    if len(gpus) != 8 or len(set(gpus)) != 8:
        raise ValueError("eval P8 requires eight distinct GPUs")
    shards = {
        split: sorted((sceneplans / split).glob(f"model-sceneplans-{split}-*.jsonl"))
        for split in EXPECTED
    }
    if any(not values for values in shards.values()):
        raise RuntimeError("eval P8 has an empty split shard set")
    output.mkdir(parents=True, exist_ok=True)
    log_root = output / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    worker = Path(__file__).resolve().parent / "materialize_model_sceneplan_v1_worker.py"
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
    state_path = output / "p8_orchestrator_state.json"
    try:
        for worker_index, gpu in enumerate(gpus):
            handle = (log_root / f"worker-{worker_index:02d}.log").open(
                "a", encoding="utf-8", buffering=1
            )
            logs.append(handle)
            command = [
                sys.executable,
                str(worker),
                "--sceneplan-root", str(sceneplans),
                "--output-root", str(output),
                "--splits", "validation", "test",
                "--gpu", str(gpu),
                "--worker-index", str(worker_index),
                "--worker-count", str(len(gpus)),
                "--jobs", str(args.jobs_per_worker),
                "--batch-size", str(args.batch_size),
                "--supplement-mode",
                "--contract-revision", "6",
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
                "schema": "stable_audio_tools.sceneplan_eval_p8_state",
                "schema_version": 1,
                "status": "running",
                "worker_pids": [process.pid for process in processes],
                "gpus": gpus,
                "planned_shards": {key: len(value) for key, value in shards.items()},
                "started_unix": time.time(),
            },
        )
        last = None
        while True:
            codes = [process.poll() for process in processes]
            if any(code not in (None, 0) for code in codes):
                raise RuntimeError(f"eval P8 worker failure: {codes}")
            state = {
                split: len(list((output / "work_done" / split).glob("work-*.json")))
                for split in EXPECTED
            }
            if state != last:
                print(
                    json.dumps(
                        {
                            "done_shards": state,
                            "total_shards": {key: len(value) for key, value in shards.items()},
                            "elapsed_sec": round(time.monotonic() - started, 1),
                        }
                    ),
                    flush=True,
                )
                last = state
            if all(code == 0 for code in codes):
                break
            time.sleep(10)
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
                "schema": "stable_audio_tools.sceneplan_eval_p8_state",
                "schema_version": 1,
                "status": "failed",
                "exit_codes": [process.poll() for process in processes],
            },
        )
        raise
    finally:
        for handle in logs:
            handle.close()
    split_rows = {}
    for split, expected in EXPECTED.items():
        paths = sorted((output / "work_done" / split).glob("work-*.json"))
        rows = sum(int(json.loads(path.read_text(encoding="utf-8"))["rows"]) for path in paths)
        if len(paths) != len(shards[split]) or rows != expected:
            raise RuntimeError(f"{split}: eval P8 coverage changed")
        split_rows[split] = rows
    result = {
        "schema": "stable_audio_tools.sceneplan_eval_p8_summary",
        "schema_version": 1,
        "dataset_contract_revision": 6,
        "status": "complete",
        "rows": sum(split_rows.values()),
        "split_counts": split_rows,
        "workers": len(processes),
        "gpus": gpus,
        "jobs_per_worker": args.jobs_per_worker,
        "batch_size": args.batch_size,
        "eval_foa_retained": True,
        "elapsed_sec": round(time.monotonic() - started, 3),
        "p10_training_started": False,
    }
    atomic_write_json(output / "P8_SUMMARY.json", result)
    atomic_write_json(state_path, result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
