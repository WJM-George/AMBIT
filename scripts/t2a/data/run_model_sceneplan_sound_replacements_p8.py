#!/usr/bin/env python3
"""Materialize all split-specific Sound replacement rows on eight GPUs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pyarrow.parquet as pq

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from sceneplan_v2_common import DATASET_ROOT, atomic_write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sceneplan-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--jobs-per-worker", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--train-render-root", type=Path,
        default=Path("/dev/shm/sceneplan_sound_expansion_v1"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sceneplan_root = args.sceneplan_root.expanduser().resolve(strict=True)
    output_root = args.output_root.expanduser().resolve()
    revision_root = (DATASET_ROOT / "revisions").resolve(strict=True)
    for name, path in (("sceneplan-root", sceneplan_root), ("output-root", output_root)):
        try:
            path.relative_to(revision_root)
        except ValueError as error:
            raise ValueError(f"{name} must stay below {revision_root}") from error
    if not (DATASET_ROOT / "FROZEN_P9.json").is_file():
        raise RuntimeError("Sound replacement P8 requires the frozen base P9")
    ready = json.loads((sceneplan_root / "READY").read_text(encoding="utf-8"))
    summary = json.loads((sceneplan_root / "summary.json").read_text(encoding="utf-8"))
    expected_rows = int(summary.get("rows", -1))
    if (
        ready.get("schema") != "stable_audio_tools.sound_replacement_sceneplans_ready"
        or ready.get("status") != "PASS"
        or summary.get("status") != "PASS"
        or expected_rows <= 0
        or summary.get("base_p9_mutated") is not False
    ):
        raise RuntimeError("Sound replacement P7.5 gate is not PASS")
    split_expected = {str(k): int(v) for k, v in summary["split_counts"].items()}
    shards_by_split = {
        split: sorted((sceneplan_root / split).glob(f"model-sceneplans-{split}-*.jsonl"))
        for split in ("train", "validation", "test")
    }
    if any(
        len(shards_by_split[split]) != (split_expected.get(split, 0) + 1023) // 1024
        for split in shards_by_split
    ):
        raise RuntimeError("Sound replacement ScenePlan shard count changed")
    all_shards = [path for values in shards_by_split.values() for path in values]
    gpus = [int(value) for value in args.gpus.split(",") if value.strip()]
    if len(gpus) != 8 or len(set(gpus)) != 8:
        raise ValueError("Sound replacement P8 requires eight distinct GPUs")
    if len(all_shards) < len(gpus):
        raise RuntimeError("not enough replacement shards for eight workers")
    render_root = args.train_render_root.expanduser().resolve()
    try:
        render_root.relative_to(Path("/dev/shm"))
    except ValueError as error:
        raise ValueError("transient train renders must stay under /dev/shm") from error
    render_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    logs_root = output_root / "logs"
    logs_root.mkdir(parents=True, exist_ok=True)
    worker_script = SCRIPT_DIR / "materialize_model_sceneplan_v1_worker.py"
    environment = dict(os.environ)
    environment.update(
        {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
    processes: list[subprocess.Popen[str]] = []
    handles = []
    started = time.time()
    try:
        for worker_index, gpu in enumerate(gpus):
            log_path = logs_root / f"worker-{worker_index:02d}.log"
            handle = log_path.open("a", encoding="utf-8", buffering=1)
            handles.append(handle)
            command = [
                sys.executable, str(worker_script),
                "--sceneplan-root", str(sceneplan_root),
                "--output-root", str(output_root),
                "--splits", "validation", "train",
                "--gpu", str(gpu),
                "--worker-index", str(worker_index),
                "--worker-count", str(len(gpus)),
                "--jobs", str(args.jobs_per_worker),
                "--batch-size", str(args.batch_size),
                "--train-render-root", str(render_root),
                "--supplement-mode",
            ]
            processes.append(
                subprocess.Popen(
                    command, cwd=REPO_ROOT, env=environment,
                    stdout=handle, stderr=subprocess.STDOUT, text=True,
                    start_new_session=True,
                )
            )
        atomic_write_json(
            output_root / "p8_orchestrator_state.json",
            {
                "schema": "stable_audio_tools.sound_replacement_p8_state",
                "schema_version": 1, "status": "running",
                "expected_rows": expected_rows,
                "split_expected": split_expected,
                "worker_pids": [process.pid for process in processes],
                "gpus": gpus, "base_p9_mutated": False,
                "started_unix": started,
            },
        )
        while True:
            codes = [process.poll() for process in processes]
            failure = next((code for code in codes if code not in (None, 0)), None)
            if failure is not None:
                for process in processes:
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGTERM)
                for process in processes:
                    if process.poll() is None:
                        process.wait(timeout=60)
                raise RuntimeError(f"Sound replacement P8 worker failed with {failure}")
            if all(code == 0 for code in codes):
                break
            time.sleep(2.0)
    finally:
        for handle in handles:
            handle.close()
    worker_summaries = [
        json.loads((output_root / "workers" / f"worker-{index:02d}.json").read_text(encoding="utf-8"))
        for index in range(8)
    ]
    actual: dict[str, int] = {}
    manifest_count = 0
    for split in ("train", "validation", "test"):
        manifests = sorted(
            (output_root / "manifests" / split).glob(f"materialized-{split}-*.parquet")
        )
        manifest_count += len(manifests)
        actual[split] = sum(pq.read_metadata(path).num_rows for path in manifests)
        if len(manifests) != len(shards_by_split[split]):
            raise RuntimeError(f"{split}: materialized shard coverage changed")
    quarantines = list((output_root / "quarantine").glob("*/*.json"))
    if (
        actual != split_expected
        or sum(actual.values()) != expected_rows
        or quarantines
        or sum(int(row["rows"]) for row in worker_summaries) != expected_rows
    ):
        raise RuntimeError("Sound replacement P8 coverage/quarantine gate failed")
    final = {
        "schema": "stable_audio_tools.sound_replacement_p8_summary",
        "schema_version": 1, "status": "PASS",
        "dataset_contract_revision": 5,
        "rows": expected_rows, "split_counts": actual,
        "sceneplan_shards": len(all_shards),
        "materialized_shards": manifest_count,
        "quarantine_rows": 0, "workers": worker_summaries,
        "gpus": gpus, "jobs_per_worker": args.jobs_per_worker,
        "vae_batch_size": args.batch_size,
        "base_p9_mutated": False,
        "elapsed_sec": round(time.time() - started, 3),
    }
    atomic_write_json(output_root / "P8_SUMMARY.json", final)
    atomic_write_json(
        output_root / "p8_orchestrator_state.json",
        {
            "schema": "stable_audio_tools.sound_replacement_p8_state",
            "schema_version": 1, "status": "complete",
            "rows": expected_rows,
            "summary": str(output_root / "P8_SUMMARY.json"),
            "base_p9_mutated": False,
        },
    )
    print(json.dumps(final, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
