#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.t2a.eval.diagnostics.score_source_location_semantics import (
    _assignment_metrics,
    _clap_text_embeddings,
    _compile_active_source_tracks,
    _demix_foa_sources,
    _resolve_scoring_inputs,
)
from scripts.t2a.eval.diagnostics.score_source_semantics_ast import (
    _active_ast_waveform,
    _anchor_assignment,
    _anchors_from_spec,
    _select_target_anchors,
)
from scripts.t2a.eval.diagnostics.score_source_presence_openflam import (
    _presence_summary,
)
from scripts.t2a.eval.diagnostics.score_semantic_conditions import (
    CORRECT_CONDITION,
    _target_fidelity_summary,
)


class SourceLocationDemixerTest(unittest.TestCase):
    def test_recovers_two_opposite_point_sources(self) -> None:
        frames, hop = 4, 64
        sample_count = frames * hop
        phase = torch.linspace(0.0, 8.0 * math.pi, sample_count)
        sources = torch.stack((torch.sin(phase), torch.cos(0.7 * phase)))
        tracks = torch.zeros(2, 8, frames)
        tracks[:, 0] = 1.0
        tracks[0, 1] = 1.0  # front, +X
        tracks[1, 1] = -1.0  # behind, -X

        steering = torch.tensor(
            [
                [1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0)],
                [0.0, 0.0],
                [0.0, 0.0],
                [1.0, -1.0],
            ]
        )
        audio = steering @ sources
        recovered, metadata = _demix_foa_sources(audio, tracks, ridge=1.0e-6)
        correlations = torch.nn.functional.cosine_similarity(
            recovered, sources, dim=-1
        )
        self.assertTrue(torch.all(correlations > 0.9999), correlations)
        self.assertEqual(metadata["hop"], hop)
        self.assertEqual(metadata["active_frame_counts"], [frames, frames])

    def test_inactive_frames_are_exactly_zero(self) -> None:
        frames, hop = 2, 16
        tracks = torch.zeros(1, 8, frames)
        tracks[0, 0, 0] = 1.0
        tracks[0, 1, 0] = 1.0
        audio = torch.randn(4, frames * hop)
        recovered, _ = _demix_foa_sources(audio, tracks, ridge=0.05)
        self.assertGreater(float(recovered[:, :hop].abs().sum()), 0.0)
        self.assertEqual(float(recovered[:, hop:].abs().sum()), 0.0)


class TargetCalibratedAssignmentTest(unittest.TestCase):
    def test_abstains_on_ambiguous_target_and_scores_valid_rows(self) -> None:
        target = torch.tensor(
            [
                [1.0, 0.2, 0.1],
                [0.1, 1.0, 0.2],
                [0.4, 0.4, 0.41],
            ]
        )
        generated = torch.tensor(
            [
                [0.9, 0.1, 0.0],
                [0.8, 0.2, 0.1],
                [0.2, 0.2, 0.3],
            ]
        )
        metrics = _assignment_metrics(
            generated,
            target,
            source_ids=["source_0", "source_1", "source_2"],
            min_target_gap=0.05,
        )
        self.assertEqual(metrics["valid_source_count"], 2)
        self.assertEqual(metrics["accuracy_on_target_discriminable_sources"], 0.5)
        self.assertFalse(metrics["rows"][2]["target_valid"])


class AstSourceSemanticTest(unittest.TestCase):
    def test_active_waveform_excludes_inactive_samples_and_keeps_raw_rms(self) -> None:
        stem = torch.tensor([1.0, -1.0, 9.0, 9.0])
        waveform, signal = _active_ast_waveform(
            stem,
            torch.tensor([True, False]),
            hop=2,
            sample_rate=16_000,
            target_seconds=1,
        )
        self.assertEqual(tuple(waveform.shape), (16_000,))
        self.assertAlmostEqual(signal["active_rms"], 1.0)
        self.assertFalse(signal["silent"])
        self.assertLess(float(waveform.abs().max()), 1.0)

    def test_target_only_ast_anchors_are_source_specific(self) -> None:
        target = torch.tensor(
            [
                [0.8, 0.05, 0.01],
                [0.02, 0.6, 0.02],
                [0.01, 0.03, 0.9],
            ]
        )
        anchors = _select_target_anchors(
            target,
            ["Speech", "Vehicle", "Music"],
            min_probability=0.02,
            min_margin=0.01,
        )
        self.assertEqual([row["label"] for row in anchors], ["Speech", "Vehicle", "Music"])
        self.assertTrue(all(row["target_valid"] for row in anchors))
        assignment = _anchor_assignment(
            target,
            target,
            anchors=anchors,
            source_ids=["s0", "s1", "s2"],
            min_target_gap=0.01,
        )
        self.assertEqual(assignment["accuracy_on_target_discriminable_sources"], 1.0)

    def test_ambiguous_ast_target_fails_closed(self) -> None:
        target = torch.tensor([[0.4, 0.1], [0.4, 0.2]])
        anchors = _select_target_anchors(
            target,
            ["Music", "Sound"],
            min_probability=0.02,
            min_margin=0.05,
        )
        self.assertFalse(anchors[0]["target_valid"])

    def test_independent_ast_anchors_are_family_bound_and_source_ordered(self) -> None:
        spec = {
            "schema": "stable_audio_tools.independent_ast_source_anchors",
            "schema_version": 1,
            "family_rank": 5,
            "family_id": "family-five",
            "source_anchors": [
                {
                    "source_id": "source_1",
                    "ast_label": "Vehicle",
                    "pre_vae_probability": 0.2,
                    "post_vae_probability": 0.3,
                    "status": "PASS",
                },
                {
                    "source_id": "source_0",
                    "ast_label": "Speech",
                    "pre_vae_probability": 0.8,
                    "post_vae_probability": 0.7,
                    "status": "PASS",
                },
            ],
        }
        anchors = _anchors_from_spec(
            spec,
            family_rank=5,
            family_id="family-five",
            source_ids=["source_0", "source_1"],
            labels=["Speech", "Vehicle", "Music"],
        )
        self.assertEqual([row["label"] for row in anchors], ["Speech", "Vehicle"])
        self.assertTrue(all(row["target_valid"] for row in anchors))
        with self.assertRaises(ValueError):
            _anchors_from_spec(
                spec,
                family_rank=6,
                family_id="family-five",
                source_ids=["source_0", "source_1"],
                labels=["Speech", "Vehicle", "Music"],
            )


class ClapSingletonCaptionTest(unittest.TestCase):
    def test_duplicates_single_query_and_returns_one_embedding(self) -> None:
        class FakeClap:
            def __init__(self):
                self.queries = None

            def get_text_embedding(self, queries, *, use_tensor):
                self.queries = list(queries)
                self.assert_use_tensor = use_tensor
                if len(queries) == 1:
                    raise AssertionError("singleton query would trigger squeeze")
                return torch.tensor([[1.0, 0.0], [1.0, 0.0]])

        model = FakeClap()
        embeddings = _clap_text_embeddings(model, ["Car"])
        self.assertEqual(model.queries, ["Car", "Car"])
        self.assertTrue(model.assert_use_tensor)
        self.assertEqual(tuple(embeddings.shape), (1, 2))


class PersistentSourceSlotTest(unittest.TestCase):
    def test_sparse_source_id_returns_one_active_track(self) -> None:
        plan = {
            "audio": {"duration_sec": 1.0},
            "scene": {
                "sources": [
                    {
                        "source_id": "source_1",
                        "event": {"label": "Car"},
                        "activity": {"onset_sec": 0.0, "offset_sec": 1.0},
                    }
                ]
            },
        }
        tracks = _compile_active_source_tracks(
            plan, num_frames=4, source_ids=["source_1"]
        )
        self.assertEqual(tuple(tracks.shape), (1, 8, 4))
        self.assertTrue(torch.all(tracks[0, 0] > 0.5))


class SourceLocationReportInputTest(unittest.TestCase):
    def test_accepts_standard_one_turn_checkpoint_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan_path = root / "turn_00.target_scene_plan.json"
            plan = {
                "scene": {
                    "sources": [
                        {"source_id": "source_0", "event": {"label": "Bird"}}
                    ]
                }
            }
            plan_path.write_text(json.dumps(plan), encoding="utf-8")
            report_path = root / "RESULT.json"
            report = {
                "evaluator_version": 2,
                "family_rank": 0,
                "family_id": "family_0",
                "turn_results": [
                    {
                        "turn": 0,
                        "audio_path": "turn_00.wav",
                        "target_audio_path": "turn_00.target.wav",
                        "target_plan_path": plan_path.name,
                        "content_alignment": {"sample_rate": 48000},
                        "spatial_alignment": {"hop": 1024, "frame_count": 432},
                    }
                ],
            }
            resolved = _resolve_scoring_inputs(report, report_path)
            self.assertEqual(resolved["kind"], "checkpoint_evaluation")
            self.assertEqual(resolved["plan"], plan)
            self.assertEqual(resolved["hop"], 1024)
            self.assertEqual(resolved["frame_count"], 432)
            self.assertEqual(resolved["generated_path"], root / "turn_00.wav")

    def test_rejects_multi_turn_checkpoint_result(self) -> None:
        report = {"evaluator_version": 2, "turn_results": [{}, {}]}
        with self.assertRaisesRegex(ValueError, "exactly one"):
            _resolve_scoring_inputs(report, Path("RESULT.json"))


class OpenFlamPresenceSummaryTest(unittest.TestCase):
    def test_reports_active_presence_and_inactive_leakage(self) -> None:
        values = torch.tensor([0.9, 0.7, 0.2, 0.0])
        active = torch.tensor([True, True, False, False])
        summary = _presence_summary(values, active)
        self.assertAlmostEqual(summary["active"]["mean"], 0.8, places=6)
        self.assertAlmostEqual(summary["inactive"]["mean"], 0.1, places=6)
        self.assertAlmostEqual(summary["activity_margin"], 0.7, places=6)


class TargetFidelitySummaryTest(unittest.TestCase):
    def test_keeps_spectral_and_target_silence_failures_visible(self) -> None:
        hop = 16
        target = torch.zeros(4, hop * 8)
        target[:, : hop * 4] = 0.25
        generated = target.clone()
        generated[:, hop * 4 :] = 0.1
        report = {
            "settings": {"downsampling_ratio": hop},
            "conditions": {
                CORRECT_CONDITION: {
                    "vs_target": {
                        "latent": {"relative_rmse": 0.4},
                        "audio": {
                            "relative_rmse": 0.5,
                            "cosine": 0.6,
                            "channel_correlations_wyzx": [0.7, 0.6, 0.5, 0.4],
                            "multiresolution_log_spectral_mae": 0.3,
                        },
                        "field": {
                            "angular_error_mean_deg": 12.0,
                            "diffuseness_mae": 0.2,
                            "valid_direction_fraction": 0.8,
                        },
                    }
                }
            },
        }
        summary = _target_fidelity_summary(
            report,
            generated_audio=generated,
            target_audio=target,
        )
        self.assertAlmostEqual(summary["audio_channel_correlation_mean"], 0.55)
        self.assertEqual(summary["audio_multiresolution_log_spectral_mae"], 0.3)
        self.assertGreater(
            summary["silence_alignment"]["generated_silent_rms"], 0.09
        )


if __name__ == "__main__":
    unittest.main()
