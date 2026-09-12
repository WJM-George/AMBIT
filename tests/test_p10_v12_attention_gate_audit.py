import importlib.util
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/t2a/train/audit_p10_v12_attention_gates.py"
)
SPEC = importlib.util.spec_from_file_location("p10_v12_attention_gate_audit", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class P10V12AttentionGateAuditTests(unittest.TestCase):
    @staticmethod
    def _gate(**overrides):
        objectives = {
            "train/softblock_event_bias_abs_mean": 0.003,
            "train/softblock_speech_bias_abs_mean": 0.001,
            "train/softblock_bias_abs_max": 0.01,
            "train/softblock_saturation_fraction": 0.0,
            "train/softblock_active_layers": 15.0,
        }
        objectives.update(overrides)
        return {
            "status": "PASS",
            "metric_observations": 2500,
            "objectives": objectives,
            "metric_windows": {
                name: {"last_mean": value}
                for name, value in objectives.items()
            },
        }

    def test_accepts_bounded_nontrivial_gates(self):
        result = MODULE.make_audit(self._gate(), minimum_observations=100)
        self.assertEqual(result["status"], "PASS")
        self.assertTrue(all(result["checks"].values()))

    def test_rejects_zero_initialized_gate_that_never_learned(self):
        result = MODULE.make_audit(
            self._gate(**{"train/softblock_speech_bias_abs_mean": 0.0}),
            minimum_observations=100,
        )
        self.assertEqual(result["status"], "FAIL")
        self.assertFalse(result["checks"]["speech_gate_is_nontrivial"])

    def test_rejects_saturated_gate(self):
        result = MODULE.make_audit(
            self._gate(**{"train/softblock_saturation_fraction": 0.5}),
            minimum_observations=100,
        )
        self.assertEqual(result["status"], "FAIL")
        self.assertFalse(result["checks"]["gates_are_not_saturated"])

    def test_uses_final_window_instead_of_lucky_latest_batch(self):
        gate = self._gate()
        gate["metric_windows"]["train/softblock_speech_bias_abs_mean"][
            "last_mean"
        ] = 0.0
        result = MODULE.make_audit(gate, minimum_observations=100)
        self.assertEqual(result["status"], "FAIL")
        self.assertFalse(result["checks"]["speech_gate_is_nontrivial"])

    def test_rejects_missing_diagnostics(self):
        gate = self._gate()
        del gate["objectives"]["train/softblock_bias_abs_max"]
        result = MODULE.make_audit(gate, minimum_observations=100)
        self.assertEqual(result["status"], "FAIL")
        self.assertIn("train/softblock_bias_abs_max", result["missing_metrics"])


if __name__ == "__main__":
    unittest.main()
