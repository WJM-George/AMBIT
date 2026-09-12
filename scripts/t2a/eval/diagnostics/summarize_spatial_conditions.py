#!/usr/bin/env python3
"""Summarize Plan/q and previous-FOA causal interventions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


CONTEXT_CAPTURE_THRESHOLD_DEG = 1.0


def _angle(value: dict[str, Any], target: str) -> float | None:
    result = value["alignment"][target].get("angular_error_mean_deg")
    return None if result is None else float(result)


def _preference(value: dict[str, Any]) -> float | None:
    correct = _angle(value, "current_plan")
    rotated = _angle(value, "rotated_plan")
    return None if correct is None or rotated is None else correct - rotated


def _difference(left: float | None, right: float | None) -> float | None:
    return None if left is None or right is None else left - right


def _mean_present(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return None if not present else sum(present) / len(present)


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(args.root.glob("family*/step*/RESULT.json"))
    if not paths:
        raise RuntimeError(f"no diagnostic results under {args.root}")

    cases = []
    for path in paths:
        report = json.loads(path.read_text(encoding="utf-8"))
        factorial = report["plan_q_factorial"]
        names = {
            "cc": "plan_correct__q_correct",
            "cr": "plan_correct__q_rotated",
            "rc": "plan_rotated__q_correct",
            "rr": "plan_rotated__q_rotated",
        }
        preferences = {
            short: _preference(factorial[name]) for short, name in names.items()
        }
        q_effect_under_correct_plan = _difference(
            preferences["cr"], preferences["cc"]
        )
        plan_effect_under_correct_q = _difference(
            preferences["rc"], preferences["cc"]
        )
        q_effect_under_rotated_plan = _difference(
            preferences["rr"], preferences["rc"]
        )
        plan_effect_under_rotated_q = _difference(
            preferences["rr"], preferences["cr"]
        )

        context_by_turn: dict[str, dict[str, Any]] = {}
        foreign_capture_turns = 0
        previous_capture_turns = 0
        foreign_context_effects: list[float | None] = []
        previous_context_effects: list[float | None] = []
        for item in report["context_interventions"]:
            turn = str(item["turn"])
            context_by_turn.setdefault(turn, {})[item["context"]] = item
        context_summary = {}
        for turn, values in sorted(context_by_turn.items(), key=lambda item: int(item[0])):
            compact = {}
            for name, item in values.items():
                current_angle = item["current_target"].get("angular_error_mean_deg")
                previous_angle = item["previous_target"].get("angular_error_mean_deg")
                foreign_angle = item["foreign_previous"].get("angular_error_mean_deg")
                compact[name] = {
                    "current_plan_error_deg": item["current_plan"].get(
                        "angular_error_mean_deg"
                    ),
                    "current_target_error_deg": current_angle,
                    "previous_target_error_deg": previous_angle,
                    "foreign_previous_error_deg": foreign_angle,
                    "latent_mae_to_current_target": item[
                        "latent_mae_to_current_target"
                    ],
                    "latent_mae_to_context": item["latent_mae_to_context"],
                }
            none = compact.get("none")
            foreign = compact.get("foreign_previous")
            previous = compact.get("target_previous")
            foreign_effect = (
                None
                if none is None or foreign is None
                else _difference(
                    none["foreign_previous_error_deg"],
                    foreign["foreign_previous_error_deg"],
                )
            )
            previous_effect = (
                None
                if none is None or previous is None
                else _difference(
                    none["previous_target_error_deg"],
                    previous["previous_target_error_deg"],
                )
            )
            foreign_context_effects.append(foreign_effect)
            previous_context_effects.append(previous_effect)
            if (
                foreign_effect is not None
                and foreign_effect >= CONTEXT_CAPTURE_THRESHOLD_DEG
            ):
                foreign_capture_turns += 1
            if (
                previous_effect is not None
                and previous_effect >= CONTEXT_CAPTURE_THRESHOLD_DEG
            ):
                previous_capture_turns += 1
            context_summary[turn] = {
                "conditions": compact,
                "foreign_toward_context_effect_deg": foreign_effect,
                "previous_toward_context_effect_deg": previous_effect,
            }

        step = path.parent.name.removeprefix("step")
        cases.append(
            {
                "family_rank": report["family_rank"],
                "family_id": report["family_id"],
                "step": int(step),
                "q_rotation_consistency_mae": report[
                    "q_rotation_consistency_mae"
                ],
                "creation": {
                    "preference_deg": preferences,
                    "q_effect_under_correct_plan_deg": q_effect_under_correct_plan,
                    "plan_effect_under_correct_q_deg": plan_effect_under_correct_q,
                    "q_effect_under_rotated_plan_deg": q_effect_under_rotated_plan,
                    "plan_effect_under_rotated_q_deg": plan_effect_under_rotated_q,
                    "condition_metrics": {
                        short: {
                            "current_plan_error_deg": _angle(
                                factorial[name], "current_plan"
                            ),
                            "rotated_plan_error_deg": _angle(
                                factorial[name], "rotated_plan"
                            ),
                            "latent_mae_vs_baseline": factorial[name][
                                "latent_mae_vs_baseline"
                            ],
                            "field_error_vs_baseline_deg": factorial[name][
                                "audio_field_vs_baseline"
                            ].get("angular_error_mean_deg"),
                        }
                        for short, name in names.items()
                    },
                },
                "context": {
                    "capture_threshold_deg": CONTEXT_CAPTURE_THRESHOLD_DEG,
                    "foreign_causal_capture_turns": foreign_capture_turns,
                    "previous_causal_capture_turns": previous_capture_turns,
                    "mean_foreign_toward_context_effect_deg": _mean_present(
                        foreign_context_effects
                    ),
                    "mean_previous_toward_context_effect_deg": _mean_present(
                        previous_context_effects
                    ),
                    "turns": context_summary,
                },
                "result": str(path),
            }
        )

    summary = {
        "schema": "stable_audio_tools.spatial_condition_diagnostic_summary",
        "schema_version": 2,
        "cases": cases,
    }
    conclusion_dir = args.root / "conclusion"
    conclusion_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(conclusion_dir / "SUMMARY.json", summary)
    lines = [
        "# Spatial condition diagnostics",
        "",
        "Preference is `error-to-correct-Plan - error-to-rotated-Plan`; positive "
        "means the output is closer to the rotated geometry.",
        "",
        "Context capture is causal: supplying that context must move the output "
        f"at least {CONTEXT_CAPTURE_THRESHOLD_DEG:g} degree toward it relative to "
        "the same-noise `context=none` output.",
        "",
        "| family | step | q effect (correct Plan) | Plan effect (correct q) | foreign causal captures | previous causal captures |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for case in cases:
        creation = case["creation"]
        context = case["context"]
        lines.append(
            f"| {case['family_rank']} | {case['step']} | "
            f"{creation['q_effect_under_correct_plan_deg']} | "
            f"{creation['plan_effect_under_correct_q_deg']} | "
            f"{context['foreign_causal_capture_turns']}/3 | "
            f"{context['previous_causal_capture_turns']}/3 |"
        )
    (conclusion_dir / "SUMMARY.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
