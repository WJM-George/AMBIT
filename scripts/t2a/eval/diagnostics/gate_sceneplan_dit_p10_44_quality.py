#!/usr/bin/env python3
"""Fail closed when a ScenePlan 4+4 checkpoint is only numerically trainable.

The existing training and evaluation receipts use PASS to mean that execution
completed.  This gate instead asks whether the flow field uses both condition
routes and whether the small fixed listening panel has crossed a minimal
audibility threshold.  These are development thresholds, not paper metrics.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


DEFAULT_THRESHOLDS = {
    "teacher_explained_target_variance_min": 0.50,
    "caption_swap_high_noise_loss_penalty_min": 0.02,
    "structured_swap_high_noise_loss_penalty_min": 0.05,
    "music_clap_text_cosine_min": 0.12,
    "sound_clap_text_cosine_min": 0.20,
    "speech_wer_max": 0.95,
    "speech_utmos_min": 1.80,
}


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("quality gate selection is empty")
    return sum(values) / len(values)


def _teacher_value(rows: list[dict[str, Any]], variant: str, key: str) -> float:
    selected = [
        float(row[key])
        for row in rows
        if row["weights"] == "ema" and row["condition_variant"] == variant
    ]
    return _mean(selected)


def _teacher_high_noise(
    rows: list[dict[str, Any]], variant: str, key: str
) -> float:
    selected = [
        float(row[key])
        for row in rows
        if row["weights"] == "ema"
        and row["condition_variant"] == variant
        and math.isclose(float(row["timestep"]), 0.9, abs_tol=1.0e-6)
    ]
    return _mean(selected)


def _evaluation_row(path: Path, checkpoint_step: int) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = [
        row
        for row in payload["rows"]
        if int(row["checkpoint_step"]) == checkpoint_step
    ]
    if len(rows) != 1:
        raise ValueError(
            f"expected one evaluation row for step {checkpoint_step}, got {len(rows)}"
        )
    return rows[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path)
    parser.add_argument("--checkpoint-step", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="write a FAIL receipt without returning a non-zero status",
    )
    args = parser.parse_args()

    teacher = json.loads(args.teacher.resolve(strict=True).read_text(encoding="utf-8"))
    checkpoint_step = int(
        args.checkpoint_step
        if args.checkpoint_step is not None
        else teacher["checkpoint_step"]
    )
    aggregates = teacher["aggregates"]
    values = {
        "teacher_explained_target_variance": _teacher_value(
            aggregates, "correct", "explained_target_variance"
        ),
        "caption_swap_high_noise_loss_penalty": _teacher_high_noise(
            aggregates, "swap_caption", "mse_relative_to_correct"
        ),
        "structured_swap_high_noise_loss_penalty": _teacher_high_noise(
            aggregates, "swap_structured", "mse_relative_to_correct"
        ),
    }

    if args.evaluation is not None:
        evaluation = _evaluation_row(
            args.evaluation.resolve(strict=True), checkpoint_step
        )
        for key in (
            "music_clap_text_cosine",
            "sound_clap_text_cosine",
            "speech_wer",
            "speech_utmos",
        ):
            values[key] = float(evaluation[key])

    checks = {
        "teacher_explained_target_variance": (
            values["teacher_explained_target_variance"]
            >= DEFAULT_THRESHOLDS["teacher_explained_target_variance_min"]
        ),
        "caption_swap_high_noise_loss_penalty": (
            values["caption_swap_high_noise_loss_penalty"]
            >= DEFAULT_THRESHOLDS["caption_swap_high_noise_loss_penalty_min"]
        ),
        "structured_swap_high_noise_loss_penalty": (
            values["structured_swap_high_noise_loss_penalty"]
            >= DEFAULT_THRESHOLDS["structured_swap_high_noise_loss_penalty_min"]
        ),
    }
    if args.evaluation is not None:
        checks.update(
            {
                "music_clap_text_cosine": (
                    values["music_clap_text_cosine"]
                    >= DEFAULT_THRESHOLDS["music_clap_text_cosine_min"]
                ),
                "sound_clap_text_cosine": (
                    values["sound_clap_text_cosine"]
                    >= DEFAULT_THRESHOLDS["sound_clap_text_cosine_min"]
                ),
                "speech_wer": (
                    values["speech_wer"] <= DEFAULT_THRESHOLDS["speech_wer_max"]
                ),
                "speech_utmos": (
                    values["speech_utmos"] >= DEFAULT_THRESHOLDS["speech_utmos_min"]
                ),
            }
        )

    passed = all(checks.values())
    receipt = {
        "schema": "stable_audio_tools.sceneplan_44_development_quality_gate",
        "schema_version": 1,
        "status": "PASS" if passed else "FAIL",
        "checkpoint_step": checkpoint_step,
        "checkpoint": teacher["checkpoint"],
        "teacher_diagnostic": str(args.teacher.resolve()),
        "evaluation_summary": (
            str(args.evaluation.resolve()) if args.evaluation is not None else None
        ),
        "threshold_scope": (
            "minimal development audibility and condition sensitivity; not paper quality"
        ),
        "thresholds": DEFAULT_THRESHOLDS,
        "values": values,
        "checks": checks,
        "failed_checks": [key for key, value in checks.items() if not value],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0 if passed or args.report_only else 1


if __name__ == "__main__":
    raise SystemExit(main())
