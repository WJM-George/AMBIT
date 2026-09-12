import importlib.util
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/t2a/eval/compare_p10_v12_upgrade_to_dense150k.py"
)
SPEC = importlib.util.spec_from_file_location("p10_v12_dense_compare", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class P10V12DensePairedComparisonTests(unittest.TestCase):
    @staticmethod
    def _rows(step, values, *, sample_suffix=""):
        return [
            {
                "checkpoint_step": step,
                "panel_id": panel_id,
                "sample_id": f"sample_{panel_id}{sample_suffix}",
                "value": value,
            }
            for panel_id, value in values.items()
        ]

    def test_candidate_minus_baseline_is_paired_by_id(self):
        baseline = self._rows(150000, {"b": 2.0, "a": 1.0})
        candidate = self._rows(2500, {"a": 1.5, "b": 3.0})
        result = MODULE.paired_delta(
            baseline,
            candidate,
            baseline_step=150000,
            candidate_step=2500,
            domain=None,
            getter=lambda row: row["value"],
        )
        self.assertEqual(result["count"], 2)
        self.assertAlmostEqual(result["mean"], 0.75)

    def test_rejects_panel_mismatch(self):
        baseline = self._rows(150000, {"a": 1.0, "b": 2.0})
        candidate = self._rows(2500, {"a": 1.5})
        with self.assertRaisesRegex(RuntimeError, "panels differ"):
            MODULE.paired_delta(
                baseline,
                candidate,
                baseline_step=150000,
                candidate_step=2500,
                domain=None,
                getter=lambda row: row["value"],
            )

    def test_rejects_sample_identity_mismatch(self):
        baseline = self._rows(150000, {"a": 1.0})
        candidate = self._rows(2500, {"a": 1.5}, sample_suffix="_other")
        with self.assertRaisesRegex(RuntimeError, "sample IDs differ"):
            MODULE.paired_delta(
                baseline,
                candidate,
                baseline_step=150000,
                candidate_step=2500,
                domain=None,
                getter=lambda row: row["value"],
            )


if __name__ == "__main__":
    unittest.main()
