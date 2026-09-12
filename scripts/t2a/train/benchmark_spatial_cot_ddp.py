#!/usr/bin/env python3
"""Benchmark the real Spatial-CoT family training path on eight GPUs."""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
# Keep the historical Spatial-CoT default while allowing another fully gated
# eight-GPU launcher to reuse the exact same measurement harness.
LAUNCHER = Path(
    os.environ.get(
        "SAT_BENCHMARK_LAUNCHER",
        REPO_ROOT / "scripts/t2a/train/run_t2a_spatial_chat_500m_8gpu.sh",
    )
).expanduser().resolve()
RESULT_PREFIX = "SAT_BENCHMARK_RESULT="
BENCHMARK_START = "SAT_BENCHMARK_START=1"
RESOURCE_FAILURE_PATTERN = re.compile(
    r"out of memory|CUDA error|CUBLAS_STATUS_ALLOC_FAILED|NCCL.*error",
    flags=re.IGNORECASE,
)


def _sample_gpus(
    *,
    stop: threading.Event,
    samples: list[dict[str, float]],
    visible_devices: str,
    num_gpus: int,
) -> None:
    requested = [item.strip() for item in visible_devices.split(",") if item.strip()]
    numeric_devices = (
        {int(item) for item in requested[:num_gpus]}
        if requested and all(item.isdigit() for item in requested[:num_gpus])
        else None
    )
    while not stop.is_set():
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.used,power.draw",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode == 0:
            rows = []
            for line in completed.stdout.splitlines():
                fields = [field.strip() for field in line.split(",")]
                if len(fields) != 4:
                    continue
                try:
                    index = int(fields[0])
                    if numeric_devices is not None and index not in numeric_devices:
                        continue
                    rows.append(
                        {
                            "utilization": float(fields[1]),
                            "memory_mib": float(fields[2]),
                            "power_w": float(fields[3]),
                        }
                    )
                except ValueError:
                    continue
            if numeric_devices is None:
                rows = rows[:num_gpus]
            samples.extend(rows)
        stop.wait(0.5)


def _summarize_gpu_samples(samples: list[dict[str, float]]) -> dict[str, float | int]:
    if not samples:
        return {"gpu_sample_count": 0}
    utilizations = sorted(sample["utilization"] for sample in samples)
    p10_index = max(0, int(0.10 * (len(utilizations) - 1)))
    p50_index = max(0, int(0.50 * (len(utilizations) - 1)))
    return {
        "gpu_sample_count": len(samples),
        "gpu_utilization_mean_percent": sum(utilizations) / len(utilizations),
        "gpu_utilization_p10_percent": utilizations[p10_index],
        "gpu_utilization_p50_percent": utilizations[p50_index],
        "gpu_utilization_peak_percent": max(utilizations),
        "gpu_power_mean_w": sum(sample["power_w"] for sample in samples)
        / len(samples),
        "gpu_memory_used_peak_mib_sampled": max(
            sample["memory_mib"] for sample in samples
        ),
    }


def _run_variant(
    *,
    batch: int,
    bucket: int,
    hook: str,
    args: argparse.Namespace,
    tag: str,
) -> dict:
    variant = f"batch{batch}_bucket{bucket}_{hook}"
    run_name = f"spcot_ddp_benchmark_{tag}_{variant}"
    log_path = args.output_dir / f"{variant}.log"
    environment = os.environ.copy()
    environment.update(
        {
            "RUN_NAME": run_name,
            "RUN_LABEL": f"spcot-bench-{variant}",
            "CUDA_VISIBLE_DEVICES": args.cuda_visible_devices,
            "NUM_GPUS": str(args.num_gpus),
            "BATCH_SIZE": str(batch),
            "NUM_WORKERS": str(args.num_workers),
            "MAX_STEPS": str(args.steps),
            "CHECKPOINT_EVERY": str(max(100_000, args.steps + 1)),
            "SAVE_TOP_K": "0",
            "LOGGER": "none",
            "BENCHMARK": "1",
            "BENCHMARK_WARMUP_BATCHES": str(args.warmup_steps),
            "TRAINING_GATE": "0",
            "GRADIENT_CLIP_VAL": "1.0",
            "VAL_EVERY": "-1",
            "LIMIT_VAL_BATCHES": "0",
            "OVERFIT_BATCHES": "0",
            "TRAINING_STRATEGY": "ddp_static",
            "DDP_BUCKET_CAP_MB": str(bucket),
            "DDP_COMM_HOOK": hook,
            "BIND_TO_GPU_NUMA": "1",
            "MIN_FREE_DISK_GIB": str(args.min_free_disk_gib),
        }
    )
    if args.dry_run:
        return {
            "status": "DRY_RUN",
            "variant": variant,
            "run_name": run_name,
            "environment": {
                key: environment[key]
                for key in (
                    "NUM_GPUS",
                    "BATCH_SIZE",
                    "MAX_STEPS",
                    "DDP_BUCKET_CAP_MB",
                    "DDP_COMM_HOOK",
                )
            },
        }

    print(f"[spcot-ddp-benchmark] starting {variant}", flush=True)
    process = subprocess.Popen(
        [str(LAUNCHER)],
        cwd=REPO_ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    output_lines = []
    gpu_samples: list[dict[str, float]] = []
    sampler_stop = threading.Event()
    sampler_thread = None
    resource_failure_seen = False
    force_kill_timer = None

    def signal_process_group(sig: int) -> None:
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            pass

    def force_kill() -> None:
        if process.poll() is None:
            signal_process_group(signal.SIGKILL)

    assert process.stdout is not None
    with log_path.open("w", encoding="utf-8") as log:
        for line in process.stdout:
            output_lines.append(line)
            log.write(line)
            log.flush()
            sys.stdout.write(line)
            sys.stdout.flush()
            if BENCHMARK_START in line and sampler_thread is None:
                sampler_thread = threading.Thread(
                    target=_sample_gpus,
                    kwargs={
                        "stop": sampler_stop,
                        "samples": gpu_samples,
                        "visible_devices": args.cuda_visible_devices,
                        "num_gpus": args.num_gpus,
                    },
                    daemon=True,
                )
                sampler_thread.start()
            if RESULT_PREFIX in line:
                sampler_stop.set()
            if RESOURCE_FAILURE_PATTERN.search(line) and not resource_failure_seen:
                resource_failure_seen = True
                sampler_stop.set()
                if process.poll() is None:
                    signal_process_group(signal.SIGINT)
                    force_kill_timer = threading.Timer(20.0, force_kill)
                    force_kill_timer.daemon = True
                    force_kill_timer.start()
    return_code = process.wait()
    sampler_stop.set()
    if sampler_thread is not None:
        sampler_thread.join(timeout=3.0)
    if force_kill_timer is not None:
        force_kill_timer.cancel()
    gpu_summary = _summarize_gpu_samples(gpu_samples)
    results = [
        json.loads(line.split(RESULT_PREFIX, 1)[1])
        for line in output_lines
        if RESULT_PREFIX in line
    ]
    if return_code == 0 and len(results) == 1:
        result = dict(results[0])
        result.update(
            {
                "status": "PASS",
                "variant": variant,
                "bucket_cap_mb": bucket,
                "comm_hook": hook,
                "run_name": run_name,
                "log_path": str(log_path),
                **gpu_summary,
            }
        )
        return result

    joined_tail = "".join(output_lines[-100:])
    resource_failure = resource_failure_seen or bool(
        RESOURCE_FAILURE_PATTERN.search(joined_tail)
    )
    return {
        "status": "RESOURCE_FAIL" if resource_failure else "FAIL",
        "variant": variant,
        "bucket_cap_mb": bucket,
        "comm_hook": hook,
        "run_name": run_name,
        "return_code": return_code,
        "log_path": str(log_path),
        "tail": joined_tail,
        **gpu_summary,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--steps", type=int, default=7)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--buckets", type=int, nargs="+", default=[50, 100])
    parser.add_argument(
        "--hooks",
        nargs="+",
        choices=("none", "bf16"),
        default=["none", "bf16"],
    )
    parser.add_argument(
        "--cuda-visible-devices", default="0,1,2,3,4,5,6,7"
    )
    parser.add_argument("--min-free-disk-gib", type=float, default=25.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/mnt/sdc/sat_gates/spatial_cot/ddp_benchmark"),
    )
    parser.add_argument("--tag")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="probe batch viability with the conservative 50MB/no-hook variant only",
    )
    parser.add_argument(
        "--stop-after-resource-fail",
        action="store_true",
        help="stop probing larger (ascending) batches after the first resource failure",
    )
    args = parser.parse_args()
    if any(batch <= 0 for batch in args.batches):
        raise ValueError("all batches must be positive")
    if any(bucket <= 0 for bucket in args.buckets):
        raise ValueError("all bucket sizes must be positive")
    if args.steps <= args.warmup_steps or args.warmup_steps < 1:
        raise ValueError("steps must exceed a positive warmup-steps")
    if args.num_gpus != 8:
        raise ValueError("this gate is intentionally the eight-GPU benchmark")

    tag = args.tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output_dir = args.output_dir / tag
    args.output_dir.mkdir(parents=True, exist_ok=False)

    combinations = (
        [(50, "none")]
        if args.baseline_only
        else [(bucket, hook) for hook in args.hooks for bucket in args.buckets]
    )
    results = []
    for batch in args.batches:
        baseline_bucket, baseline_hook = combinations[0]
        baseline = _run_variant(
            batch=batch,
            bucket=baseline_bucket,
            hook=baseline_hook,
            args=args,
            tag=tag,
        )
        results.append(baseline)
        if baseline["status"] not in {"PASS", "DRY_RUN"}:
            # Batch viability is decided with the conservative baseline. Avoid
            # repeating an OOM four times; smaller viable batches still run all
            # communication variants.
            if args.stop_after_resource_fail and baseline["status"] == "RESOURCE_FAIL":
                break
            continue
        for bucket, hook in combinations[1:]:
            results.append(
                _run_variant(
                    batch=batch,
                    bucket=bucket,
                    hook=hook,
                    args=args,
                    tag=tag,
                )
            )

    passed = [item for item in results if item["status"] == "PASS"]
    viable_batches = sorted({int(item["batch_size_per_gpu"]) for item in passed})
    complete_batches = [
        batch
        for batch in viable_batches
        if sum(
            item["status"] == "PASS" and item["batch_size_per_gpu"] == batch
            for item in results
        )
        == len(combinations)
    ]
    best = (
        max(passed, key=lambda item: item["global_training_examples_per_second"])
        if passed
        else None
    )
    status = "DRY_RUN" if args.dry_run else ("PASS" if complete_batches else "FAIL")
    report = {
        "status": status,
        "tag": tag,
        "baseline_only": args.baseline_only,
        "stop_after_resource_fail": args.stop_after_resource_fail,
        "complete_batches": complete_batches,
        "best_variant": best,
        "results": results,
    }
    result_path = args.output_dir / "RESULT.json"
    result_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"SAT_SPATIAL_COT_DDP_MATRIX={json.dumps(report, sort_keys=True)}")
    return 0 if status in {"PASS", "DRY_RUN"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
