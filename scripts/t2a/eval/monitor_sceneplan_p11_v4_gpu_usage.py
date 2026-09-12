#!/usr/bin/env python3
"""Record auditable per-GPU usage during the P11-v4 parallel eval phase."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


SCHEMA = "stable_audio_tools.p11_v4_eval_gpu_telemetry"
SCHEMA_VERSION = 1
GPU_QUERY = (
    "index,uuid,name,utilization.gpu,memory.used,memory.total,"
    "power.draw,temperature.gpu"
)
PROCESS_QUERY = "gpu_uuid,pid,used_gpu_memory"


def _json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rows(command: Sequence[str]) -> list[list[str]]:
    completed = subprocess.run(
        command,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return [
        [field.strip() for field in row]
        for row in csv.reader(completed.stdout.splitlines())
        if row and any(field.strip() for field in row)
    ]


def _float(value: str) -> float | None:
    if value.lower() in {"n/a", "[not supported]", "not supported"}:
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _sample(started: float) -> dict[str, Any]:
    gpu_rows = _rows(
        [
            "nvidia-smi",
            f"--query-gpu={GPU_QUERY}",
            "--format=csv,noheader,nounits",
        ]
    )
    process_rows = _rows(
        [
            "nvidia-smi",
            f"--query-compute-apps={PROCESS_QUERY}",
            "--format=csv,noheader,nounits",
        ]
    )
    gpus = []
    for row in gpu_rows:
        if len(row) != 8:
            raise RuntimeError(f"unexpected GPU telemetry row: {row!r}")
        gpus.append(
            {
                "index": int(row[0]),
                "uuid": row[1],
                "name": row[2],
                "utilization_percent": _float(row[3]),
                "memory_used_mib": _float(row[4]),
                "memory_total_mib": _float(row[5]),
                "power_draw_w": _float(row[6]),
                "temperature_c": _float(row[7]),
            }
        )
    processes = []
    for row in process_rows:
        if len(row) != 3:
            raise RuntimeError(f"unexpected compute-process row: {row!r}")
        processes.append(
            {
                "gpu_uuid": row[0],
                "pid": int(row[1]),
                "used_gpu_memory_mib": _float(row[2]),
            }
        )
    return {
        "wall_time_ns": time.time_ns(),
        "elapsed_seconds": time.monotonic() - started,
        "gpus": sorted(gpus, key=lambda value: value["index"]),
        "compute_processes": processes,
    }


def _mean(values: Sequence[float]) -> float | None:
    return float(statistics.fmean(values)) if values else None


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * quantile))
    return float(ordered[index])


def _values(rows: Sequence[dict[str, Any]], key: str) -> list[float]:
    return [float(row[key]) for row in rows if row.get(key) is not None]


def _parse_assignment(value: str) -> tuple[int, str]:
    index, separator, task = value.partition("=")
    if not separator or not index.isdigit() or not task.strip():
        raise argparse.ArgumentTypeError("--assignment must be INDEX=TASK")
    return int(index), task.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval-sec", type=float, default=5.0)
    parser.add_argument("--expected-gpus", type=int, default=8)
    parser.add_argument("--phase", default="initial_parallel_diagnostics")
    parser.add_argument(
        "--assignment", action="append", type=_parse_assignment, required=True
    )
    args = parser.parse_args()
    if args.interval_sec < 0.25:
        raise ValueError("--interval-sec must be at least 0.25 seconds")
    if args.expected_gpus <= 0:
        raise ValueError("--expected-gpus must be positive")
    assignments = dict(args.assignment)
    if len(assignments) != len(args.assignment):
        raise ValueError("GPU assignment indices must be unique")
    expected_indices = set(range(args.expected_gpus))
    if set(assignments) != expected_indices:
        raise ValueError(
            f"assignments must cover exactly {sorted(expected_indices)}"
        )

    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing telemetry report: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    stop = threading.Event()

    def request_stop(_signum: int, _frame: Any) -> None:
        stop.set()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    started_at = datetime.now(timezone.utc).isoformat()
    started = time.monotonic()
    samples: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    while not stop.is_set():
        try:
            samples.append(_sample(started))
        except Exception as error:  # noqa: BLE001 - retain telemetry failures.
            errors.append(
                {
                    "wall_time_ns": time.time_ns(),
                    "elapsed_seconds": time.monotonic() - started,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
        stop.wait(args.interval_sec)

    by_gpu: dict[int, list[dict[str, Any]]] = {
        index: [] for index in sorted(expected_indices)
    }
    process_samples = {index: 0 for index in sorted(expected_indices)}
    process_ids = {index: set() for index in sorted(expected_indices)}
    uuid_to_index: dict[str, int] = {}
    sample_index_sets = []
    for sample in samples:
        indices = {int(gpu["index"]) for gpu in sample["gpus"]}
        sample_index_sets.append(indices)
        for gpu in sample["gpus"]:
            index = int(gpu["index"])
            if index in by_gpu:
                by_gpu[index].append(gpu)
                uuid_to_index[str(gpu["uuid"])] = index
        active_indices = set()
        for process in sample["compute_processes"]:
            index = uuid_to_index.get(str(process["gpu_uuid"]))
            if index in by_gpu:
                active_indices.add(index)
                process_ids[index].add(int(process["pid"]))
        for index in active_indices:
            process_samples[index] += 1

    per_gpu = {}
    for index, rows in by_gpu.items():
        utilization = _values(rows, "utilization_percent")
        memory = _values(rows, "memory_used_mib")
        power = _values(rows, "power_draw_w")
        per_gpu[str(index)] = {
            "assignment": assignments[index],
            "samples": len(rows),
            "gpu": (
                {
                    "uuid": rows[0]["uuid"],
                    "name": rows[0]["name"],
                    "memory_total_mib": rows[0]["memory_total_mib"],
                }
                if rows
                else None
            ),
            "utilization_percent": {
                "mean": _mean(utilization),
                "p50": _percentile(utilization, 0.50),
                "p90": _percentile(utilization, 0.90),
                "max": max(utilization) if utilization else None,
            },
            "memory_used_mib": {
                "mean": _mean(memory),
                "max": max(memory) if memory else None,
            },
            "power_draw_w": {
                "mean": _mean(power),
                "max": max(power) if power else None,
            },
            "compute_process_samples": process_samples[index],
            "compute_process_sample_rate": (
                process_samples[index] / len(rows) if rows else 0.0
            ),
            "observed_compute_pids": sorted(process_ids[index]),
        }

    gates = {
        "samples_recorded": len(samples) >= 2,
        "no_sampling_errors": not errors,
        "every_sample_has_exact_gpu_inventory": bool(sample_index_sets)
        and all(indices == expected_indices for indices in sample_index_sets),
        "every_gpu_observed_compute_process": all(
            process_samples[index] > 0 for index in expected_indices
        ),
        "every_gpu_observed_nonzero_utilization": all(
            bool(_values(by_gpu[index], "utilization_percent"))
            and max(_values(by_gpu[index], "utilization_percent")) > 0.0
            for index in expected_indices
        ),
    }
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": "PASS" if all(gates.values()) else "FAIL",
        "scope": "P11-v4 initial eight-way post-training diagnostic phase",
        "phase": args.phase,
        "started_at_utc": started_at,
        "ended_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.monotonic() - started,
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "interval_seconds": args.interval_sec,
        "expected_gpu_count": args.expected_gpus,
        "assignments": {str(key): value for key, value in sorted(assignments.items())},
        "gates": gates,
        "sample_count": len(samples),
        "sampling_errors": errors,
        "per_gpu": per_gpu,
        "source": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256_file(Path(__file__).resolve()),
        },
        "invocation": " ".join(sys.argv),
        "samples": samples,
    }
    report["report_sha256_without_self"] = _json_sha256(report)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    print(
        json.dumps(
            {
                "status": report["status"],
                "output": str(output),
                "elapsed_seconds": report["elapsed_seconds"],
                "sample_count": report["sample_count"],
                "gates": gates,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
