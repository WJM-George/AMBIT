from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.t2a.eval.summarize_spatial_cot_capacity_probe import summarize


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _case(
    root: Path,
    *,
    family: int,
    step: int,
    retention: float,
    location: float,
    target_error: float,
    plan_error: float,
    clap: float,
) -> None:
    directory = root / f"family{family}" / f"step{step}"
    family_id = f"family_{family}"
    _write(
        directory / "RESULT.json",
        {
            "status": "PASS",
            "family_id": family_id,
            "family_rank": family,
            "turn_results": [
                {
                    "content_alignment": {
                        "generated_target_audio_cosine": clap,
                    },
                    "spatial_alignment": {
                        "angular_error_mean_deg": target_error,
                    },
                    "plan_spatial_alignment": {
                        "angular_error_mean_deg": plan_error,
                    },
                }
            ],
        },
    )
    sources = []
    for index in range(2):
        target = 0.1 + 0.01 * index
        sources.append(
            {
                "source_id": f"source_{index}",
                "caption": f"source {index}",
                "target": {
                    "active": {"mean": target},
                    "activity_margin": target,
                },
                "generated": {
                    "active": {"mean": target * retention},
                    "activity_margin": target * retention,
                },
            }
        )
    _write(
        directory / "SOURCE_PRESENCE_OPENFLAM.json",
        {"family_id": family_id, "sources": sources},
    )
    _write(
        directory / "SOURCE_LOCATION_METRICS.json",
        {
            "family_id": family_id,
            "target_audio_assignment": {
                "accuracy_on_target_discriminable_sources": location,
                "valid_source_count": 2,
            },
        },
    )


class SpatialCotCapacitySummaryTests(unittest.TestCase):
    def test_requires_one_common_source_binding_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for family in (0, 17):
                _case(
                    root,
                    family=family,
                    step=0,
                    retention=0.05,
                    location=0.5,
                    target_error=70.0,
                    plan_error=65.0,
                    clap=0.5,
                )
                _case(
                    root,
                    family=family,
                    step=100,
                    retention=0.5,
                    location=1.0,
                    target_error=30.0,
                    plan_error=20.0,
                    clap=0.51,
                )
            result = summarize(root)
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["common_candidate_steps"], [100])
            self.assertEqual(result["recommended_step"], 100)

    def test_one_missing_weak_source_blocks_all_families(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for family in (0, 17):
                _case(
                    root,
                    family=family,
                    step=0,
                    retention=0.05,
                    location=0.5,
                    target_error=70.0,
                    plan_error=65.0,
                    clap=0.5,
                )
                _case(
                    root,
                    family=family,
                    step=100,
                    retention=0.1 if family == 17 else 0.5,
                    location=1.0,
                    target_error=30.0,
                    plan_error=20.0,
                    clap=0.51,
                )
            result = summarize(root)
            self.assertEqual(result["status"], "BLOCK")
            self.assertEqual(result["common_candidate_steps"], [])
            self.assertFalse(
                result["families"]["17"]["100"]["source_presence_pass"]
            )

    def test_missing_source_diagnostic_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _case(
                root,
                family=0,
                step=0,
                retention=0.1,
                location=0.5,
                target_error=70.0,
                plan_error=65.0,
                clap=0.5,
            )
            _case(
                root,
                family=0,
                step=100,
                retention=0.5,
                location=1.0,
                target_error=30.0,
                plan_error=20.0,
                clap=0.51,
            )
            (root / "family0/step100/SOURCE_LOCATION_METRICS.json").unlink()
            with self.assertRaisesRegex(RuntimeError, "required capacity diagnostic"):
                summarize(root)


if __name__ == "__main__":
    unittest.main()
