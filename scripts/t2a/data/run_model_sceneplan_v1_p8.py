#!/usr/bin/env python3
"""Launch and monitor the formal eight-GPU revision-5 P8 materialization."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from sceneplan_v2_common import DATASET_ROOT, atomic_write_json, require_dataset_not_frozen  # noqa: E402
from materialize_model_sceneplan_v1_shard import load_shard_rows  # noqa: E402


def pid_alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except OSError:
        return False


def main() -> int:
    require_dataset_not_frozen()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sceneplan-root", type=Path, default=DATASET_ROOT / "sceneplans_model_v1"
    )
    parser.add_argument("--output-root", type=Path, default=DATASET_ROOT / "materialized")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--jobs-per-worker", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--train-render-root",
        type=Path,
        default=Path("/dev/shm/sceneplan_v2_p8_train_renders"),
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "validation", "test"),
        default=["validation", "test", "train"],
    )
    args = parser.parse_args()
    sceneplan_root = args.sceneplan_root.expanduser().resolve(strict=True)
    output_root = args.output_root.expanduser().resolve(strict=False)
    train_render_root = args.train_render_root.expanduser().resolve(strict=False)
    try:
        sceneplan_root.relative_to("/mnt/sdb")
        output_root.relative_to("/mnt/sdb")
    except ValueError as error:
        raise ValueError("formal P8 inputs and outputs must be on SDB") from error
    try:
        train_render_root.relative_to("/dev/shm")
    except ValueError as error:
        raise ValueError("formal transient train renders must remain under /dev/shm") from error
    train_render_root.mkdir(parents=True, exist_ok=True)
    transient_stat = os.statvfs(train_render_root)
    transient_free = transient_stat.f_bavail * transient_stat.f_frsize
    if transient_free < 64 * 1024**3:
        raise RuntimeError(
            f"transient train render root has less than 64 GiB free: {transient_free}"
        )
    if not (sceneplan_root / "READY").is_file():
        raise RuntimeError("revised P7.5 model ScenePlans are not READY")
    audit_path = sceneplan_root / "audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if (
        audit.get("schema") != "stable_audio_tools.model_sceneplan_manifest_audit"
        or audit.get("ok") is not True
        or int(audit.get("dataset_contract_revision", -1)) != 5
        or int(audit.get("rows", -1)) != 1_124_000
        or audit.get("p8_started") is not False
    ):
        raise RuntimeError("revised P7.5 independent audit is not an all-pass P8 gate")
    gpu_indices = [int(value) for value in args.gpus.split(",") if value.strip()]
    if len(gpu_indices) != 8 or len(set(gpu_indices)) != 8:
        raise ValueError("formal P8 requires exactly eight distinct GPU indices")
    planned_shards = [
        path
        for split in args.splits
        for path in sorted(
            (sceneplan_root / split).glob(f"model-sceneplans-{split}-*.jsonl")
        )
    ]
    if len(planned_shards) != 1_099:
        raise RuntimeError(f"formal P8 expected 1,099 shards, found {len(planned_shards)}")
    # Bind the adapter's exact hash convention to the frozen P7.5 index before
    # any expensive rendering starts.  This catches both text-only caption
    # hashes and any future three-view canonicalization drift.
    probe_paths = [
        planned_shards[0],
        next(path for path in planned_shards if path.parent.name == "test"),
        next(path for path in planned_shards if path.parent.name == "train"),
        planned_shards[-1],
    ]
    probes = [load_shard_rows(path, max_rows=1)[0] for path in probe_paths]
    probe_ids = [str(row["sample_id"]) for row in probes]
    index_table = pq.read_table(
        sceneplan_root / "index.parquet",
        columns=[
            "sample_id",
            "model_sceneplan_sha256",
            "render_recipe_sha256",
            "renderer_caption_sha256",
        ],
    )
    index_table = index_table.filter(
        pc.is_in(index_table["sample_id"], value_set=pa.array(probe_ids))
    )
    indexed = {str(row["sample_id"]): row for row in index_table.to_pylist()}
    for probe in probes:
        frozen = indexed.get(str(probe["sample_id"]))
        if frozen is None or any(
            probe[key] != frozen[key]
            for key in (
                "model_sceneplan_sha256",
                "render_recipe_sha256",
                "renderer_caption_sha256",
            )
        ):
            raise RuntimeError(
                f"P8 adapter/P7.5 three-view hash mismatch: {probe['sample_id']}"
            )
    output_root.mkdir(parents=True, exist_ok=True)
    state_path = output_root / "p8_orchestrator_state.json"
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        live = [int(pid) for pid in state.get("worker_pids", []) if pid_alive(int(pid))]
        if live:
            raise RuntimeError(f"P8 worker processes are already live: {live}")
    worker_script = SCRIPT_DIR / "materialize_model_sceneplan_v1_worker.py"
    log_root = output_root / "logs"
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
    processes: list[subprocess.Popen] = []
    logs = []
    started = time.time()
    try:
        for worker_index, gpu in enumerate(gpu_indices):
            log_path = log_root / f"worker-{worker_index:02d}.log"
            handle = log_path.open("a", encoding="utf-8", buffering=1)
            logs.append(handle)
            command = [
                sys.executable,
                str(worker_script),
                "--sceneplan-root",
                str(sceneplan_root),
                "--output-root",
                str(output_root),
                "--gpu",
                str(gpu),
                "--worker-index",
                str(worker_index),
                "--worker-count",
                str(len(gpu_indices)),
                "--jobs",
                str(args.jobs_per_worker),
                "--batch-size",
                str(args.batch_size),
                "--train-render-root",
                str(train_render_root),
                "--splits",
                *args.splits,
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
                "schema": "stable_audio_tools.model_sceneplan_p8_orchestrator_state",
                "schema_version": 1,
                "dataset_contract_revision": 5,
                "status": "running",
                "worker_pids": [process.pid for process in processes],
                "gpus": gpu_indices,
                "splits": args.splits,
                "planned_shards": len(planned_shards),
                "train_transient_render_root": str(train_render_root),
                "started_unix": started,
            },
        )
        last_done = -1
        while True:
            exit_codes = [process.poll() for process in processes]
            failures = [code for code in exit_codes if code not in (None, 0)]
            if failures:
                raise RuntimeError(f"P8 worker failure exit codes: {exit_codes}")
            done_paths = [
                path
                for split in args.splits
                for path in (output_root / "work_done" / split).glob("work-*.json")
            ]
            done_count = len(done_paths)
            if done_count != last_done:
                rows = 0
                by_split: dict[str, int] = {}
                for path in done_paths:
                    value = json.loads(path.read_text(encoding="utf-8"))
                    rows += int(value["rows"])
                    by_split[value["split"]] = by_split.get(value["split"], 0) + int(
                        value["rows"]
                    )
                stat = os.statvfs("/mnt/sdb")
                free_fraction = stat.f_bavail / stat.f_blocks
                if free_fraction < 0.20:
                    raise RuntimeError(
                        f"SDB free fraction fell below 20%: {free_fraction}"
                    )
                print(
                    json.dumps(
                        {
                            "p8_done_shards": done_count,
                            "p8_total_shards": len(planned_shards),
                            "rows": rows,
                            "rows_by_split": by_split,
                            "sdb_free_fraction": free_fraction,
                            "elapsed_sec": round(time.time() - started, 1),
                        }
                    ),
                    flush=True,
                )
                last_done = done_count
            if all(code == 0 for code in exit_codes):
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
                "schema": "stable_audio_tools.model_sceneplan_p8_orchestrator_state",
                "schema_version": 1,
                "dataset_contract_revision": 5,
                "status": "failed",
                "worker_pids": [process.pid for process in processes],
                "exit_codes": [process.poll() for process in processes],
                "elapsed_sec": round(time.time() - started, 3),
            },
        )
        raise
    finally:
        for handle in logs:
            handle.close()
    done_paths = [
        path
        for split in args.splits
        for path in (output_root / "work_done" / split).glob("work-*.json")
    ]
    if len(done_paths) != len(planned_shards):
        raise RuntimeError("P8 workers exited cleanly but done-shard count is incomplete")
    rows = sum(
        int(json.loads(path.read_text(encoding="utf-8"))["rows"])
        for path in done_paths
    )
    if rows != 1_124_000:
        raise RuntimeError(f"P8 materialized rows {rows} != 1,124,000")
    summary = {
        "schema": "stable_audio_tools.model_sceneplan_p8_orchestrator_summary",
        "schema_version": 1,
        "dataset_contract_revision": 5,
        "status": "complete",
        "rows": rows,
        "shards": len(done_paths),
        "workers": len(processes),
        "gpus": gpu_indices,
        "splits": args.splits,
        "train_transient_render_root": str(train_render_root),
        "elapsed_sec": round(time.time() - started, 3),
        "p10_training_started": False,
        "p11_training_started": False,
    }
    atomic_write_json(output_root / "p8_orchestrator_summary.json", summary)
    atomic_write_json(state_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
