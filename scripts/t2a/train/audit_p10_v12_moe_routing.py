#!/usr/bin/env python3
"""Turn final P10-v12 MoE training diagnostics into a routing health gate.

This gate rejects dead/dominant experts, maximum-entropy averaging, and a
collapsed prior/evidence blend.  It is a training-health prerequisite only;
multi-source evaluation is still required to prove useful specialization.
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
        "train/moe_router_entropy",
        "train/moe_router_max_probability",
        "train/moe_top1_weight_mean",
        "train/moe_conflict_gate_saturation_fraction",
        "train/moe_conflict_logit_l2",
        "train/moe_prior_evidence_top1_agreement",
        "train/moe_active_layers",
        *(f"train/moe_expert_{index}_dispatch" for index in range(4)),
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
            "schema": "stable_audio_tools.p10_v12_moe_routing_gate",
            "schema_version": 2,
            "status": "FAIL",
            "reason": "missing collapse diagnostics",
            "missing_metrics": missing,
            "balanced_1200_is_not_a_specialization_proof": True,
            "multisource_evaluation_required": True,
        }

    # Gate stable behavior across the final training window.  ``objectives``
    # is only the latest batch and is retained above as a completeness check.
    values = {
        name: float(metric_windows[name]["last_mean"])
        for name in required
    }
    finite = all(math.isfinite(value) for value in values.values())
    dispatch = [values[f"train/moe_expert_{index}_dispatch"] for index in range(4)]
    entropy = values["train/moe_router_entropy"]
    maximum_entropy = math.log(4.0)
    agreement = values["train/moe_prior_evidence_top1_agreement"]
    checks = {
        "training_gate_pass": training_gate.get("status") == "PASS",
        "metric_observations": int(training_gate.get("metric_observations", 0))
        >= int(minimum_observations),
        "finite": finite,
        "four_active_layers": values["train/moe_active_layers"] == 4.0,
        "no_dead_or_dominant_expert": min(dispatch) >= 0.05
        and max(dispatch) <= 0.60,
        "not_maximum_entropy_averaging": 0.35
        <= entropy
        <= maximum_entropy - 0.01,
        "routing_probability_is_selective": values[
            "train/moe_router_max_probability"
        ]
        >= 0.30,
        "top1_weight_is_selective": values["train/moe_top1_weight_mean"]
        >= 0.55,
        "top2_route_remains_effective": values["train/moe_top1_weight_mean"]
        <= 0.90,
        "conflict_gate_not_saturated": values[
            "train/moe_conflict_gate_saturation_fraction"
        ]
        <= 0.25,
        "conflict_logit_bounded": values["train/moe_conflict_logit_l2"] <= 9.0,
        "prior_and_evidence_both_nontrivial": 0.05 <= agreement <= 0.95,
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    return {
        "schema": "stable_audio_tools.p10_v12_moe_routing_gate",
        "schema_version": 2,
        "status": status,
        "aggregation": "final_training_gate_last_window_mean",
        "checks": checks,
        "thresholds": {
            "minimum_metric_observations": int(minimum_observations),
            "expert_dispatch_min": 0.05,
            "expert_dispatch_max": 0.60,
            "router_entropy_min": 0.35,
            "router_entropy_max": maximum_entropy - 0.01,
            "router_max_probability_min": 0.30,
            "top1_weight_mean_min": 0.55,
            "top1_weight_mean_max": 0.90,
            "conflict_gate_saturation_fraction_max": 0.25,
            "conflict_logit_l2_max": 9.0,
            "prior_evidence_top1_agreement_range": [0.05, 0.95],
        },
        "metrics": values,
        "expert_dispatch": dispatch,
        "maximum_entropy": maximum_entropy,
        "balanced_1200_is_not_a_specialization_proof": True,
        "multisource_evaluation_required": True,
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
