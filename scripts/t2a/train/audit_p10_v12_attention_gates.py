#!/usr/bin/env python3
"""Turn final P10-v12 soft-block diagnostics into an attention health gate.

This gate proves that all zero-initialized per-layer gates learned a bounded,
non-trivial signal without saturating.  It is a training-health prerequisite;
matched audio evaluation is still required to prove a quality improvement.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any


PREFIX = "SAT_TRAINING_GATE_RESULT="


def load_last_training_gate(path: Path) -> dict[str, Any]:
    payloads = []
    text = path.read_text(encoding="utf-8", errors="replace").replace("\r", "\n")
    for line in text.splitlines():
        marker = line.find(PREFIX)
        if marker >= 0:
            payloads.append(json.loads(line[marker + len(PREFIX) :]))
    if not payloads:
        raise RuntimeError(f"no {PREFIX.rstrip('=')} record in {path}")
    return payloads[-1]


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def make_audit(
    training_gate: dict[str, Any],
    *,
    minimum_observations: int,
) -> dict[str, Any]:
    objectives = training_gate.get("objectives") or {}
    metric_windows = training_gate.get("metric_windows") or {}
    required = [
        "train/softblock_event_bias_abs_mean",
        "train/softblock_speech_bias_abs_mean",
        "train/softblock_bias_abs_max",
        "train/softblock_saturation_fraction",
        "train/softblock_active_layers",
    ]
    missing = sorted(
        name
        for name in required
        if name not in objectives
        or name not in metric_windows
        or metric_windows[name].get("last_mean") is None
    )
    if missing:
        return {
            "schema": "stable_audio_tools.p10_v12_attention_gate",
            "schema_version": 1,
            "status": "FAIL",
            "reason": "missing soft-block diagnostics",
            "missing_metrics": missing,
            "matched_audio_evaluation_required": True,
        }

    # Use the final HealthGate window, not a potentially lucky last batch.
    values = {
        name: float(metric_windows[name]["last_mean"])
        for name in required
    }
    finite = all(math.isfinite(value) for value in values.values())
    checks = {
        "training_gate_pass": training_gate.get("status") == "PASS",
        "metric_observations": int(training_gate.get("metric_observations", 0))
        >= int(minimum_observations),
        "finite": finite,
        "all_fifteen_layers_active": values["train/softblock_active_layers"]
        == 15.0,
        "event_gate_is_nontrivial": values[
            "train/softblock_event_bias_abs_mean"
        ]
        >= 1.0e-4,
        "speech_gate_is_nontrivial": values[
            "train/softblock_speech_bias_abs_mean"
        ]
        >= 1.0e-4,
        "bias_is_bounded": values["train/softblock_bias_abs_max"] <= 2.0,
        "gates_are_not_saturated": values[
            "train/softblock_saturation_fraction"
        ]
        <= 0.25,
    }
    return {
        "schema": "stable_audio_tools.p10_v12_attention_gate",
        "schema_version": 1,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "aggregation": "final_training_gate_last_window_mean",
        "checks": checks,
        "thresholds": {
            "minimum_metric_observations": int(minimum_observations),
            "active_layers": 15,
            "event_bias_abs_mean_min": 1.0e-4,
            "speech_bias_abs_mean_min": 1.0e-4,
            "bias_abs_max": 2.0,
            "saturation_fraction_max": 0.25,
        },
        "metrics": values,
        "matched_audio_evaluation_required": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-observations", type=int, default=100)
    args = parser.parse_args()
    if args.minimum_observations <= 0:
        parser.error("--minimum-observations must be positive")
    log = args.train_log.expanduser().resolve(strict=True)
    audit = make_audit(
        load_last_training_gate(log),
        minimum_observations=args.minimum_observations,
    )
    audit["train_log"] = str(log)
    atomic_json(args.output.expanduser().resolve(), audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if audit["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
