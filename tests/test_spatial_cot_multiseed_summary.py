from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.t2a.eval.summarize_spatial_cot_multiseed_probe import summarize


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _case(
    root: Path,
    *,
    seed: int,
    step: int,
    good: bool,
    clap: float,
    target_error: float,
    plan_error: float,
) -> None:
    directory = root / f"seed{seed}" / f"step{step}"
    family_id = "family_5"
    source_ids = ["source_0", "source_1"]
    _write(
        directory / "RESULT.json",
        {
            "status": "PASS",
            "family_id": family_id,
            "family_rank": 5,
            "turn_results": [
                {
                    "content_alignment": {"generated_target_audio_cosine": clap},
                    "spatial_alignment": {"angular_error_mean_deg": target_error},
                    "plan_spatial_alignment": {"angular_error_mean_deg": plan_error},
                }
            ],
        },
    )
    location_rows = [
        {
            "source_id": source_id,
            "target_valid": True,
            "generated_correct": good or index == 0,
            "generated_diagonal_margin": 0.1 if good or index == 0 else -0.1,
        }
        for index, source_id in enumerate(source_ids)
    ]
    _write(
        directory / "SOURCE_LOCATION_METRICS.json",
        {
            "schema": "stable_audio_tools.source_location_semantic_scores",
            "family_id": family_id,
            "family_rank": 5,
            "source_ids": source_ids,
            "target_audio_assignment": {
                "valid_source_count": 2,
                "accuracy_on_target_discriminable_sources": 1.0 if good else 0.5,
                "rows": location_rows,
            },
        },
    )
    ast_rows = [
        {
            "source_id": source_id,
            "anchor_label": label,
            "target_valid": True,
            "generated_correct": True,
            "generated_diagonal_margin": 0.2,
            "generated_to_reference_anchor_ratio": 0.5,
        }
        for source_id, label in zip(source_ids, ("Speech", "Music"))
    ]
    _write(
        directory / "SOURCE_SEMANTICS_AST.json",
        {
            "schema": "stable_audio_tools.source_semantics_ast",
            "family_id": family_id,
            "family_rank": 5,
            "source_ids": source_ids,
            "anchor_source": "independent_isolated_vae_target",
            "anchor_assignment": {
                "valid_source_count": 2,
                "accuracy_on_target_discriminable_sources": 1.0,
                "rows": ast_rows,
            },
            "sources": [
                {"source_id": source_id, "active_rms_ratio": 1.0}
                for source_id in source_ids
            ],
        },
    )


class SpatialCotMultiseedSummaryTest(unittest.TestCase):
    def test_passes_only_when_candidate_sources_pass_across_seeds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for seed in (42, 43):
                _case(
                    root,
                    seed=seed,
                    step=0,
                    good=False,
                    clap=0.6,
                    target_error=50.0,
                    plan_error=60.0,
                )
                _case(
                    root,
                    seed=seed,
                    step=100,
                    good=True,
                    clap=0.62,
                    target_error=30.0,
                    plan_error=35.0,
                )
            result = summarize(root, min_seed_pass_fraction=1.0)
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["candidate"]["semantic_pass_seed_count"], 2)

    def test_one_persistent_source_assignment_failure_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for seed in (42, 43):
                for step in (0, 100):
                    _case(
                        root,
                        seed=seed,
                        step=step,
                        good=step == 100 and seed == 42,
                        clap=0.62,
                        target_error=30.0 if step == 100 else 50.0,
                        plan_error=35.0 if step == 100 else 60.0,
                    )
            result = summarize(root, min_seed_pass_fraction=1.0)
            self.assertEqual(result["status"], "BLOCK")
            self.assertFalse(result["aggregate_gates"]["seed_semantics"])


if __name__ == "__main__":
    unittest.main()
