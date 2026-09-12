from __future__ import annotations

import copy
import unittest
from pathlib import Path

from scripts.t2a.eval.spatial_cot_text_gate import (
    EVALUATOR_VERSION,
    RESULT_SCHEMA,
    compare_free_results,
    compare_teacher_results,
    load_text_panel,
    rotate,
)


def _result(mode: str) -> dict:
    common = {
        "schema": RESULT_SCHEMA,
        "schema_version": 1,
        "evaluator_version": EVALUATOR_VERSION,
        "status": "DIAGNOSTIC",
        "mode": mode,
        "checkpoint": f"/{mode}.ckpt",
        "panel_name": "fixed",
        "panel_sha256": "abc",
        "family_ranks": [1, 2],
        "case_count": 8,
    }
    if mode == "teacher_forced_counterfactual":
        common["metrics"] = {
            "state_planner": {
                "aligned_ce": 1.0,
                "token_accuracy": 0.80,
                "ambiguous_token_accuracy": 0.60,
                "exact_fraction": 0.25,
                "counterfactual": {
                    name: {"ce_gap": gap, "aligned_reference_ce": 1.0}
                    for name, gap in (
                        ("instruction", 0.5),
                        ("state", 0.4),
                        ("full", 0.8),
                    )
                },
            },
            "understanding": {
                "aligned_ce": 1.2,
                "token_accuracy": 0.75,
                "ambiguous_token_accuracy": 0.55,
                "exact_fraction": 0.20,
                "counterfactual": {
                    "audio": {"ce_gap": 0.6, "aligned_reference_ce": 1.2}
                },
            },
        }
    else:
        common["metrics"] = {
            objective: {
                "token_accuracy": 0.50,
                "source_component_accuracy": 0.60,
                "decoded_exact_fraction": 0.10,
                "failure_count": 1,
                **(
                    {
                        "edit_field_accuracy": 1.0,
                        "edit_exact_fraction": 1.0,
                    }
                    if objective == "state_planner"
                    else {}
                ),
            }
            for objective in ("state_planner", "understanding")
        }
    return common


class SpatialCotTextGateTests(unittest.TestCase):
    def setUp(self):
        repo = Path(__file__).resolve().parents[1]
        self.panel, self.ranks = load_text_panel(
            repo / "scripts/t2a/eval/panels/text_validation10k_panel16.json"
        )

    def test_panel_is_source_disjoint_and_has_fixed_free_cases(self):
        self.assertEqual(self.panel["split"], "validation")
        self.assertEqual(len(self.ranks), 16)
        self.assertEqual(len(self.panel["free_decode_cases"]), 4)
        self.assertEqual(self.panel["turns"], 4)

    def test_rotation_is_deterministic_and_rejects_identity(self):
        self.assertEqual(rotate([0, 1, 2, 3], 1), [1, 2, 3, 0])
        with self.assertRaises(ValueError):
            rotate([0, 1], 0)

    def test_teacher_comparison_passes_non_regression(self):
        baseline = _result("teacher_forced_counterfactual")
        candidate = copy.deepcopy(baseline)
        candidate["checkpoint"] = "/candidate.ckpt"
        report = compare_teacher_results(
            baseline,
            candidate,
            thresholds=self.panel["gate"]["teacher_forced"],
        )
        self.assertEqual(report["status"], "PASS")
        self.assertFalse(report["failures"])

    def test_teacher_comparison_fails_condition_collapse(self):
        baseline = _result("teacher_forced_counterfactual")
        candidate = copy.deepcopy(baseline)
        candidate["checkpoint"] = "/candidate.ckpt"
        candidate["metrics"]["understanding"]["counterfactual"]["audio"][
            "ce_gap"
        ] = -0.1
        report = compare_teacher_results(
            baseline,
            candidate,
            thresholds=self.panel["gate"]["teacher_forced"],
        )
        self.assertEqual(report["status"], "FAIL")
        self.assertTrue(
            any("audio_counterfactual_ce_gap" in item for item in report["failures"])
        )

    def test_teacher_comparison_is_invariant_to_ce_scale_improvement(self):
        baseline = _result("teacher_forced_counterfactual")
        candidate = copy.deepcopy(baseline)
        candidate["checkpoint"] = "/candidate.ckpt"
        for objective, scale in (("state_planner", 0.5), ("understanding", 0.5)):
            metrics = candidate["metrics"][objective]
            metrics["aligned_ce"] *= scale
            for counterfactual in metrics["counterfactual"].values():
                counterfactual["ce_gap"] *= scale
                counterfactual["aligned_reference_ce"] *= scale
        report = compare_teacher_results(
            baseline,
            candidate,
            thresholds=self.panel["gate"]["teacher_forced"],
        )
        self.assertEqual(report["status"], "PASS")

    def test_teacher_comparison_fails_relative_condition_collapse(self):
        baseline = _result("teacher_forced_counterfactual")
        candidate = copy.deepcopy(baseline)
        candidate["checkpoint"] = "/candidate.ckpt"
        candidate["metrics"]["state_planner"]["counterfactual"]["full"][
            "ce_gap"
        ] *= 0.25
        report = compare_teacher_results(
            baseline,
            candidate,
            thresholds=self.panel["gate"]["teacher_forced"],
        )
        self.assertEqual(report["status"], "FAIL")
        self.assertTrue(
            any("relative_ce_gap" in item for item in report["failures"])
        )

    def test_free_comparison_is_fail_closed_on_generation_errors(self):
        baseline = _result("free_decode")
        candidate = copy.deepcopy(baseline)
        candidate["checkpoint"] = "/candidate.ckpt"
        candidate["metrics"]["state_planner"]["failure_count"] = 2
        report = compare_free_results(
            baseline,
            candidate,
            thresholds=self.panel["gate"]["free_decode"],
        )
        self.assertEqual(report["status"], "FAIL")
        self.assertTrue(any("failure_count" in item for item in report["failures"]))


if __name__ == "__main__":
    unittest.main()
