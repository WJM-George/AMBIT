#!/usr/bin/env python3
"""Freeze the overfit, throughput, and resume gates before full P10."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
REVISION_ROOT = Path(
    "/mnt/sdb/audio_dataset/sceneplan_v2_1p124m/"
    "revisions/speech_expansion_noalign_15s_v1"
)
DEFAULT_OVERFIT = Path(
    "/mnt/sdb/model_archives/p10_pre_v11_20260831/pilots/"
    "sceneplan_dit_speechexp_noalign_15s_overfit10_20260828/OVERFIT_GATE.json"
)
DEFAULT_RESUME = Path(
    "/mnt/sdb/model_archives/p10_pre_v11_20260831/preflights/"
    "sceneplan_dit_speechexp_noalign_15s_resume_preflight_20260828/RESUME_GATE.json"
)
MODEL_CONFIG = REPO_ROOT / (
    "stable_audio_tools/configs/model_configs/txt2audio/t2a/dit/"
    "qwen35_0p8b_300m_model_sceneplan_44_soundexp_noalign_15s.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def frozen(path: Path) -> dict[str, str]:
    path = path.expanduser().resolve(strict=True)
    return {"path": str(path), "sha256": sha256(path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-result", type=Path, required=True)
    parser.add_argument("--overfit-gate", type=Path, default=DEFAULT_OVERFIT)
    parser.add_argument("--resume-gate", type=Path, default=DEFAULT_RESUME)
    parser.add_argument(
        "--output", type=Path, default=REVISION_ROOT / "P10_PREFLIGHT_GATE.json"
    )
    args = parser.parse_args()
    overfit_path = args.overfit_gate.expanduser().resolve(strict=True)
    resume_path = args.resume_gate.expanduser().resolve(strict=True)
    benchmark_path = args.benchmark_result.expanduser().resolve(strict=True)
    overfit = json.loads(overfit_path.read_text(encoding="utf-8"))
    resume = json.loads(resume_path.read_text(encoding="utf-8"))
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    if not (
        overfit.get("status") == "PASS"
        and overfit.get("dataset_contract_revision") == 6
        and overfit.get("rows") == 10
        and overfit.get("short_rows") == 5
        and overfit.get("long_rows") == 5
        and overfit.get("initialization") == "from_scratch"
        and overfit.get("forced_aligner") is False
        and resume.get("status") == "PASS"
        and resume.get("dataset_contract_revision") == 6
        and benchmark.get("status") == "PASS"
    ):
        raise RuntimeError("one or more revision-6 P10 preflight stages failed")
    best = benchmark.get("best_variant") or {}
    log_path = Path(best.get("log_path") or "").resolve(strict=True)
    raw = log_path.read_text(encoding="utf-8", errors="replace")
    fast = re.findall(r"SAT_QWEN35_FAST_PATH=(\{[^\r\n]+\})", raw)
    if len(fast) != 1 or json.loads(fast[0]) != {
        "causal_conv1d_version": "1.7.0",
        "fast_path_available": True,
        "flash_linear_attention_version": "0.5.2",
    }:
        raise RuntimeError("throughput gate did not use the frozen Qwen3.5 fast path")
    if not (
        best.get("status") == "PASS"
        and int(best.get("world_size", -1)) == 8
        and int(best.get("batch_size_per_gpu", -1)) == 72
        and int(best.get("observed_batch_size_per_gpu_min", -1)) == 48
        and int(best.get("observed_batch_size_per_gpu_max", -1)) == 72
        and int(best.get("observed_sequence_length_min", -1)) == 432
        and int(best.get("observed_sequence_length_max", -1)) == 648
        and int(best.get("measured_optimizer_steps", 0)) >= 20
        and float(best.get("global_samples_per_second", 0.0)) >= 180.0
        and float(best.get("global_sequence_positions_per_second", 0.0))
        >= 80_000.0
        and float(best.get("peak_reserved_gib", 100.0)) < 47.0
        and int(best.get("gpu_sample_count", 0)) >= 80
        and float(best.get("gpu_utilization_mean_percent", 0.0)) >= 85.0
        and float(best.get("gpu_utilization_p50_percent", 0.0)) >= 90.0
    ):
        raise RuntimeError(f"revision-6 throughput/utilization gate failed: {best}")
    health = [
        json.loads(value)
        for value in re.findall(
            r"SAT_TRAINING_GATE_RESULT=(\{[^\r\n]+\})", raw
        )
    ]
    if len(health) != 1 or health[0].get("status") != "PASS":
        raise RuntimeError("throughput run did not pass its numerical health gate")

    train_config = REVISION_ROOT / (
        "p10_configs/sceneplan_v2_speech_expansion_noalign_15s_v1_train.json"
    )
    validation_config = REVISION_ROOT / (
        "p10_configs/sceneplan_v2_speech_expansion_noalign_15s_v1_validation.json"
    )
    gate = {
        "schema": "stable_audio_tools.sceneplan_44_revision6_p10_preflight",
        "schema_version": 1,
        "status": "PASS",
        "dataset_contract_revision": 6,
        "overfit_10_sample": "PASS",
        "throughput_8gpu": "PASS",
        "checkpoint_resume": "PASS",
        "overfit_gate": frozen(overfit_path),
        "benchmark_gate": frozen(benchmark_path),
        "benchmark_log": frozen(log_path),
        "resume_gate": frozen(resume_path),
        "selected_runtime": {
            "world_size": 8,
            "short_batch_size_per_gpu": 72,
            "long_batch_size_per_gpu": 48,
            "short_bucket_frames": 432,
            "long_bucket_frames": 648,
            "num_workers_per_rank": 12,
            "precision": "bf16-mixed",
            "strategy": "ddp_static",
            "ddp_bucket_cap_mb": 50,
            "qwen35_fast_path": True,
        },
        "measured_runtime": best,
        "frozen_inputs": {
            "model_config": frozen(MODEL_CONFIG),
            "train_dataset_config": frozen(train_config),
            "validation_dataset_config": frozen(validation_config),
        },
        "authorization": "ready_to_start_full_P10_from_scratch",
    }
    output = args.output.expanduser().resolve(strict=False)
    if output != (REVISION_ROOT / "P10_PREFLIGHT_GATE.json").resolve(strict=False):
        raise ValueError("P10 preflight gate must live at the canonical revision path")
    atomic_json(output, gate)
    print(json.dumps(gate, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
