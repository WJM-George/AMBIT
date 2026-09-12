from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from scripts.t2a.eval.summarize_spatial_cot_pilot import main


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class SpatialCotPilotSummaryTests(unittest.TestCase):
    def _fixture(self, root: Path, *, understanding_gap: float) -> list[str]:
        repo = Path(__file__).resolve().parents[1]
        panel = repo / "scripts/t2a/eval/panels/text_validation10k_panel16.json"
        batf = root / "BATF.json"
        trajectory = root / "trajectory"
        free = root / "free.json"
        closed = root / "closed"
        output = root / "PILOT_SUMMARY.json"
        _write(
            batf,
            {
                "pre_free_gate_passed": True,
                "winner": {
                    "validation_mean_loss_ratio": 0.90,
                    "state_planner_ce": 1.0,
                    "understanding_ce": 1.0,
                    "state_planner_full_counterfactual_ce_gap": 0.10,
                },
            },
        )
        for index, step in enumerate((1000, 2000, 3000, 4000, 5000)):
            checkpoint = f"/step_{step}.ckpt"
            step_root = trajectory / f"step_{step}"
            _write(
                step_root / "renderer" / "RESULT.json",
                {
                    "paired_metrics": {
                        "case_count": 192,
                        "mean_loss_ratio": 0.88 - 0.02 * index,
                    }
                },
            )
            _write(
                step_root / "text" / "RESULT.json",
                {
                    "mode": "teacher_forced_counterfactual",
                    "status": "DIAGNOSTIC",
                    "evaluator_version": 2,
                    "case_count": 64,
                    "checkpoint": checkpoint,
                    "metrics": {
                        "state_planner": {
                            "aligned_ce": 0.90 - 0.05 * index,
                            "ambiguous_token_accuracy": 0.7,
                            "counterfactual": {"full": {"ce_gap": 0.1}},
                        },
                        "understanding": {
                            "aligned_ce": 0.95 - 0.05 * index,
                            "ambiguous_token_accuracy": 0.7,
                            "counterfactual": {
                                "audio": {"ce_gap": understanding_gap}
                            },
                        },
                    },
                },
            )
            _write(
                step_root / "text" / "COMPARISON.json",
                {
                    "candidate_checkpoint": checkpoint,
                    "comparison_version": 2,
                    "status": "PASS",
                },
            )
        _write(free, {"status": "PASS"})
        for family_rank in (274, 8179):
            _write(
                closed / f"family_{family_rank}" / "RESULT.json",
                {"status": "PASS", "closed_loop": True},
            )
        return [
            "summarize_spatial_cot_pilot.py",
            "--batf-summary",
            str(batf),
            "--trajectory-root",
            str(trajectory),
            "--free-comparison",
            str(free),
            "--closed-loop-root",
            str(closed),
            "--panel",
            str(panel),
            "--output",
            str(output),
        ]

    def test_complete_improving_pilot_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            argv = self._fixture(Path(directory), understanding_gap=0.02)
            with patch.object(sys, "argv", argv):
                with redirect_stdout(io.StringIO()):
                    self.assertEqual(main(), 0)
            result = json.loads(Path(argv[-1]).read_text(encoding="utf-8"))
            self.assertEqual(result["status"], "PASS")

    def test_audio_condition_insensitivity_blocks_formal_promotion(self):
        with tempfile.TemporaryDirectory() as directory:
            argv = self._fixture(Path(directory), understanding_gap=0.001)
            with patch.object(sys, "argv", argv):
                with redirect_stdout(io.StringIO()):
                    self.assertEqual(main(), 1)
            result = json.loads(Path(argv[-1]).read_text(encoding="utf-8"))
            self.assertEqual(result["status"], "FAIL")
            self.assertTrue(
                any("understanding_audio" in item for item in result["failures"])
            )


if __name__ == "__main__":
    unittest.main()
