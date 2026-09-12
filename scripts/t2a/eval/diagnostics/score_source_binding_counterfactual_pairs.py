#!/usr/bin/env python3
"""Score same-noise base/control-swapped Spatial-CoT generation pairs."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import torch
import torchaudio


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.t2a.eval.evaluate_spatial_cot_checkpoint import (  # noqa: E402
    _atomic_json,
    _spatial_alignment_metrics,
)


def _load_foa(path: str | Path) -> tuple[torch.Tensor, int]:
    audio, sample_rate = torchaudio.load(str(path))
    if audio.ndim != 2 or audio.shape[0] != 4 or not torch.isfinite(audio).all():
        raise ValueError(f"expected finite four-channel FOA: {path}")
    return audio.float(), int(sample_rate)


def _relative_delta(value: torch.Tensor, reference: torch.Tensor) -> float:
    count = min(value.shape[-1], reference.shape[-1])
    value, reference = value[..., :count], reference[..., :count]
    denominator = reference.square().mean().sqrt().clamp_min(1.0e-12)
    return float((value - reference).square().mean().sqrt() / denominator)


def _finite_angle(metrics: dict[str, Any], name: str) -> float:
    value = metrics.get("angular_error_mean_deg")
    if value is None or not math.isfinite(float(value)):
        raise ValueError(f"{name} has no finite angular error")
    return float(value)


def score(result_path: Path) -> dict[str, Any]:
    report = json.loads(result_path.read_text(encoding="utf-8"))
    settings = report.get("settings") or {}
    if settings.get("render_seed_policy") != "same_each_turn":
        raise ValueError("counterfactual scoring requires same_each_turn render seed")
    if settings.get("renderer_context") != "none" or settings.get(
        "renderer_plan"
    ) != "target":
        raise ValueError("counterfactual scoring requires target Plan and no context")
    turns = report.get("turn_results") or []
    if not turns or len(turns) % 2:
        raise ValueError("counterfactual result must contain complete base/swap pairs")

    pairs = []
    for pair_index in range(len(turns) // 2):
        base = turns[2 * pair_index]
        swapped = turns[2 * pair_index + 1]
        generated_base, sample_rate = _load_foa(base["audio_path"])
        generated_swapped, swapped_rate = _load_foa(swapped["audio_path"])
        target_base, target_rate = _load_foa(base["target_audio_path"])
        target_swapped, target_swapped_rate = _load_foa(
            swapped["target_audio_path"]
        )
        if len({sample_rate, swapped_rate, target_rate, target_swapped_rate}) != 1:
            raise ValueError("pair audio sample rates differ")
        hop = int((base.get("spatial_alignment") or {}).get("hop") or 1024)

        base_correct = _spatial_alignment_metrics(
            generated_base, target_base, hop=hop
        )
        base_wrong = _spatial_alignment_metrics(
            generated_base, target_swapped, hop=hop
        )
        swap_correct = _spatial_alignment_metrics(
            generated_swapped, target_swapped, hop=hop
        )
        swap_wrong = _spatial_alignment_metrics(
            generated_swapped, target_base, hop=hop
        )
        base_correct_angle = _finite_angle(base_correct, "base_correct")
        base_wrong_angle = _finite_angle(base_wrong, "base_wrong")
        swap_correct_angle = _finite_angle(swap_correct, "swap_correct")
        swap_wrong_angle = _finite_angle(swap_wrong, "swap_wrong")
        pairs.append(
            {
                "pair_index": pair_index,
                "base": {
                    "correct_target_angle_deg": base_correct_angle,
                    "wrong_target_angle_deg": base_wrong_angle,
                    "correct_target_margin_deg": (
                        base_wrong_angle - base_correct_angle
                    ),
                    "correct_preference": base_correct_angle < base_wrong_angle,
                },
                "swapped": {
                    "correct_target_angle_deg": swap_correct_angle,
                    "wrong_target_angle_deg": swap_wrong_angle,
                    "correct_target_margin_deg": (
                        swap_wrong_angle - swap_correct_angle
                    ),
                    "correct_preference": swap_correct_angle < swap_wrong_angle,
                },
                "target_pair_relative_rms": _relative_delta(
                    target_swapped, target_base
                ),
                "generated_pair_relative_rms": _relative_delta(
                    generated_swapped, generated_base
                ),
            }
        )

    preferences = [
        bool(pair[role]["correct_preference"])
        for pair in pairs
        for role in ("base", "swapped")
    ]
    margins = [
        float(pair[role]["correct_target_margin_deg"])
        for pair in pairs
        for role in ("base", "swapped")
    ]
    return {
        "schema": "stable_audio_tools.source_binding_counterfactual_score",
        "schema_version": 1,
        "result": str(result_path.resolve()),
        "family_id": report.get("family_id"),
        "family_rank": report.get("family_rank"),
        "pair_count": len(pairs),
        "condition_accuracy": sum(preferences) / len(preferences),
        "mean_correct_target_margin_deg": sum(margins) / len(margins),
        "pairs": pairs,
        "status": "PASS" if all(preferences) else "FAIL",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    value = score(args.result.expanduser().resolve())
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        _atomic_json(args.output, value)
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
