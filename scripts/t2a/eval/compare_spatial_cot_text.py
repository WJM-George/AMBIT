#!/usr/bin/env python3
"""Compare two Spatial-CoT text evaluations on the same immutable panel."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.t2a.eval.spatial_cot_text_gate import (
    atomic_json,
    compare_free_results,
    compare_teacher_results,
    load_text_panel,
    read_json,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("teacher_forced_counterfactual", "free_decode"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    panel, _ = load_text_panel(args.panel)
    baseline = read_json(args.baseline.expanduser().resolve())
    candidate = read_json(args.candidate.expanduser().resolve())
    if args.mode == "teacher_forced_counterfactual":
        report = compare_teacher_results(
            baseline,
            candidate,
            thresholds=panel["gate"]["teacher_forced"],
        )
    else:
        report = compare_free_results(
            baseline,
            candidate,
            thresholds=panel["gate"]["free_decode"],
        )
    atomic_json(args.output.expanduser().resolve(), report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
