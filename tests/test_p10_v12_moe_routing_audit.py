import importlib.util
import math
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/t2a/train/audit_p10_v12_moe_routing.py"
)
SPEC = importlib.util.spec_from_file_location("p10_v12_moe_routing_audit", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class P10V12MoERoutingAuditTests(unittest.TestCase):
    @staticmethod
    def _gate(**overrides):
        objectives = {
            "train/moe_router_entropy": 1.1,
            "train/moe_router_max_probability": 0.45,
            "train/moe_top1_weight_mean": 0.65,
            "train/moe_conflict_gate_saturation_fraction": 0.05,
            "train/moe_conflict_logit_l2": 1.0,
            "train/moe_prior_evidence_top1_agreement": 0.5,
            "train/moe_active_layers": 4.0,
            **{
                f"train/moe_expert_{index}_dispatch": 0.25
                for index in range(4)
            },
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

    def test_accepts_selective_balanced_routing(self):
        result = MODULE.make_audit(self._gate(), minimum_observations=100)
        self.assertEqual(result["status"], "PASS")
        self.assertTrue(all(result["checks"].values()))

    def test_rejects_observed_router_v1_maximum_entropy_collapse(self):
        result = MODULE.make_audit(
            self._gate(
                **{
                    "train/moe_router_entropy": math.log(4.0),
                    "train/moe_router_max_probability": 0.25,
                    "train/moe_top1_weight_mean": 0.5,
                    "train/moe_conflict_gate_saturation_fraction": 1.0,
                }
            ),
            minimum_observations=100,
        )
        self.assertEqual(result["status"], "FAIL")
        self.assertFalse(result["checks"]["not_maximum_entropy_averaging"])
        self.assertFalse(result["checks"]["conflict_gate_not_saturated"])

    def test_uses_final_window_instead_of_lucky_latest_batch(self):
        gate = self._gate()
        gate["metric_windows"]["train/moe_router_entropy"][
            "last_mean"
        ] = math.log(4.0)
        result = MODULE.make_audit(gate, minimum_observations=100)
        self.assertEqual(result["status"], "FAIL")
        self.assertFalse(result["checks"]["not_maximum_entropy_averaging"])

    def test_rejects_missing_new_diagnostics(self):
        gate = self._gate()
        del gate["objectives"]["train/moe_router_max_probability"]
        result = MODULE.make_audit(gate, minimum_observations=100)
        self.assertEqual(result["status"], "FAIL")
        self.assertIn(
            "train/moe_router_max_probability", result["missing_metrics"]
        )

    def test_rejects_effectively_top1_route(self):
        result = MODULE.make_audit(
            self._gate(
                **{
                    "train/moe_router_entropy": 0.36,
                    "train/moe_router_max_probability": 0.89,
                    "train/moe_top1_weight_mean": 0.95,
                }
            ),
            minimum_observations=100,
        )
        self.assertEqual(result["status"], "FAIL")
        self.assertFalse(result["checks"]["top2_route_remains_effective"])


if __name__ == "__main__":
    unittest.main()
