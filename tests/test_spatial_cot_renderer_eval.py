from __future__ import annotations

import unittest

import torch

from scripts.t2a.eval.evaluate_spatial_cot_checkpoint import (
    _clap_content_metrics,
    _edit_field_metrics,
    _latent_alignment_metrics,
    _plan_spatial_alignment_metrics,
    _plan_field_metrics,
    _plan_gate_failures,
    _renderer_audio_gate_failures,
    _select_renderer_previous_latent,
    _silence_alignment_metrics,
    _spatial_alignment_metrics,
    _teacher_forced_token_metrics,
)
from stable_audio_tools.data.spatial_story import (
    compile_source_tracks,
    source_tracks_to_mixture_trajectory,
)


class SpatialCotRendererEvalTests(unittest.TestCase):
    def test_latent_alignment_reports_exact_and_orthogonal_fit(self):
        target = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        exact = _latent_alignment_metrics(target, target)
        orthogonal = _latent_alignment_metrics(
            torch.tensor([[0.0, 1.0], [-1.0, 0.0]]),
            target,
        )
        self.assertEqual(exact["mae"], 0.0)
        self.assertEqual(exact["rmse"], 0.0)
        self.assertAlmostEqual(exact["cosine"], 1.0, places=6)
        self.assertAlmostEqual(orthogonal["cosine"], 0.0, places=6)
        self.assertGreater(orthogonal["relative_rmse"], 1.0)

    def test_renderer_context_interventions_select_expected_previous_latent(self):
        family_latents = torch.arange(3 * 2 * 4, dtype=torch.float32).reshape(
            3, 2, 4
        )
        generated = torch.full((2, 4), -1.0)
        device = torch.device("cpu")

        self.assertIs(
            _select_renderer_previous_latent(
                "closed_loop",
                turn_index=2,
                closed_loop_previous=generated,
                family_latents=family_latents,
                device=device,
            ),
            generated,
        )
        self.assertIsNone(
            _select_renderer_previous_latent(
                "none",
                turn_index=2,
                closed_loop_previous=generated,
                family_latents=family_latents,
                device=device,
            )
        )
        self.assertIsNone(
            _select_renderer_previous_latent(
                "target_previous",
                turn_index=0,
                closed_loop_previous=generated,
                family_latents=family_latents,
                device=device,
            )
        )
        self.assertTrue(
            torch.equal(
                _select_renderer_previous_latent(
                    "target_previous",
                    turn_index=2,
                    closed_loop_previous=generated,
                    family_latents=family_latents,
                    device=device,
                ),
                family_latents[1],
            )
        )

    def test_clap_content_metric_uses_foa_w_channel(self):
        class DummyClap:
            def __init__(self):
                self.model = torch.nn.Linear(1, 1, bias=False)

            @staticmethod
            def get_audio_embedding_from_data(x, use_tensor=True):
                assert use_tensor
                return torch.stack((x.mean(dim=-1), x.square().mean(dim=-1)), dim=-1)

            @staticmethod
            def get_text_embedding(text, use_tensor=True):
                assert text == ["speech"] and use_tensor
                return torch.tensor([[1.0, 1.0]])

        generated = torch.zeros(4, 480)
        target = torch.zeros(4, 480)
        generated[0] = 0.5
        target[0] = 1.0
        # Directional channels differ completely but must not alter the
        # content embedding selected from ACN channel-zero W.
        generated[1:] = 10.0
        target[1:] = -10.0
        metrics = _clap_content_metrics(
            DummyClap(), generated, target, "speech", sample_rate=48_000
        )
        self.assertAlmostEqual(
            metrics["generated_target_audio_cosine"], 1.0, places=6
        )
        self.assertEqual(metrics["channel"], "W")

    def test_clap_content_metric_disables_outer_autocast(self):
        class AutocastAwareClap:
            def __init__(self):
                self.model = torch.nn.Linear(1, 1, bias=False)

            @staticmethod
            def get_audio_embedding_from_data(x, use_tensor=True):
                assert use_tensor
                assert not torch.is_autocast_enabled("cpu")
                assert x.dtype == torch.float32
                return torch.stack((x.mean(dim=-1), x.square().mean(dim=-1)), dim=-1)

            @staticmethod
            def get_text_embedding(text, use_tensor=True):
                assert text == ["speech"] and use_tensor
                assert not torch.is_autocast_enabled("cpu")
                return torch.tensor([[1.0, 1.0]])

        foa = torch.ones(4, 480)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            metrics = _clap_content_metrics(
                AutocastAwareClap(), foa, foa, "speech", sample_rate=48_000
            )
        self.assertAlmostEqual(
            metrics["generated_target_audio_cosine"], 1.0, places=6
        )

    def test_clap_single_caption_preserves_tokenizer_batch_dimension(self):
        class SingleCaptionClap:
            def __init__(self):
                self.model = torch.nn.Linear(1, 1, bias=False)

            @staticmethod
            def tokenize(texts, **kwargs):
                assert texts == ["speech"]
                assert kwargs["return_tensors"] == "pt"
                return {
                    "input_ids": torch.ones(1, 77, dtype=torch.long),
                    "attention_mask": torch.ones(1, 77, dtype=torch.long),
                }

            @staticmethod
            def get_audio_embedding_from_data(x, use_tensor=True):
                return torch.stack((x.mean(dim=-1), x.square().mean(dim=-1)), dim=-1)

            @staticmethod
            def get_text_embedding(texts, *, tokenizer, use_tensor=True):
                tokenized = tokenizer(texts)
                assert tokenized["input_ids"].shape == (1, 77)
                return torch.tensor([[1.0, 1.0]])

        foa = torch.ones(4, 480)
        metrics = _clap_content_metrics(
            SingleCaptionClap(), foa, foa, "speech", sample_rate=48_000
        )
        self.assertAlmostEqual(
            metrics["generated_target_audio_cosine"], 1.0, places=6
        )

    def test_source_tracks_are_silent_outside_activity(self):
        plan = {
            "audio": {"duration_sec": 1.0},
            "scene": {
                "sources": [
                    {
                        "source_id": "source_0",
                        "activity": {
                            "onset_sec": 0.25,
                            "offset_sec": 0.5,
                            "quality": "source_annotation",
                        },
                        "motion": {
                            "keyframes": [
                                {
                                    "t_norm": 0.0,
                                    "position": {
                                        "azimuth_deg": 0.0,
                                        "elevation_deg": 0.0,
                                        "distance_m": 2.0,
                                    },
                                }
                            ]
                        },
                        "acoustics": {"gain_db": -6.0},
                    }
                ]
            },
        }
        tracks = compile_source_tracks(plan, num_frames=4)["tracks"][0]
        self.assertEqual(tracks[0].tolist(), [0.0, 1.0, 0.0, 0.0])
        self.assertTrue(torch.equal(tracks[1:7, 0], torch.zeros(6)))
        self.assertGreater(float(tracks[1:7, 1].abs().sum()), 0.0)
        self.assertTrue(torch.equal(tracks[1:7, 2:], torch.zeros(6, 2)))
        self.assertTrue(torch.equal(tracks[7], torch.ones(4)))

    def test_mixture_trajectory_matches_energy_weighted_foa_convention(self):
        tracks = torch.zeros(2, 8, 3)
        tracks[0, 0, :2] = 1.0
        tracks[0, 1, :2] = 1.0
        tracks[0, 4:6, :2] = 1.0
        tracks[1, 0, 1] = 1.0
        tracks[1, 2, 1] = 1.0
        tracks[1, 4:6, 1] = 1.0

        trajectory = source_tracks_to_mixture_trajectory(tracks)
        self.assertTrue(
            torch.allclose(trajectory[0], torch.tensor([1.0, 0.0, 0.0, 0.0]))
        )
        expected_diffuseness = 1.0 - 2.0**-0.5
        self.assertTrue(
            torch.allclose(
                trajectory[1],
                torch.tensor([0.5, 0.5, 0.0, expected_diffuseness]),
                atol=1.0e-6,
            )
        )
        self.assertTrue(
            torch.allclose(trajectory[2], torch.tensor([0.0, 0.0, 0.0, 1.0]))
        )

    def test_received_level_is_nonredundant_and_marks_activity(self):
        tracks = torch.zeros(2, 8, 2)
        tracks[:, 0, 0] = 1.0
        tracks[0, 1, 0] = 1.0
        tracks[1, 1, 0] = -1.0
        tracks[:, 4:6, 0] = 1.0

        trajectory = source_tracks_to_mixture_trajectory(
            tracks, fourth_component="received_level"
        )
        self.assertTrue(torch.equal(trajectory[0, :3], torch.zeros(3)))
        self.assertEqual(float(trajectory[0, 3]), 1.0)
        self.assertTrue(torch.equal(trajectory[1], torch.zeros(4)))

    def test_edit_field_metrics_distinguish_wrong_change_from_exact_delta(self):
        previous = {
            "scene": {
                "sources": [
                    {
                        "source_id": "source_1",
                        "acoustics": {"gain_db": -2.5},
                    }
                ]
            }
        }
        target = {
            "scene": {
                "sources": [
                    {
                        "source_id": "source_1",
                        "acoustics": {"gain_db": -8.5},
                    }
                ]
            }
        }
        diff = {
            "changed": [
                {
                    "source_id": "source_1",
                    "fields": ["acoustics.gain_db", "content.source_audio_id"],
                }
            ]
        }
        wrong = {
            "scene": {
                "sources": [
                    {
                        "source_id": "source_1",
                        "acoustics": {"gain_db": -5.0},
                    }
                ]
            }
        }
        wrong_metrics = _edit_field_metrics(wrong, target, previous, diff)
        self.assertEqual(wrong_metrics["changed_field_count"], 1)
        self.assertEqual(wrong_metrics["changed_field_accuracy"], 0.0)
        self.assertEqual(wrong_metrics["applied_field_fraction"], 1.0)

        exact_metrics = _edit_field_metrics(target, target, previous, diff)
        self.assertTrue(exact_metrics["changed_fields_exact"])
        self.assertEqual(exact_metrics["changed_field_accuracy"], 1.0)

    @staticmethod
    def _plane_wave(direction, *, frames=4, hop=128):
        phase = torch.linspace(0, 8 * torch.pi, frames * hop)
        signal = torch.sin(phase)
        dx, dy, dz = direction
        return torch.stack(
            [signal / 2**0.5, signal * dy, signal * dz, signal * dx]
        )

    def test_spatial_alignment_reports_identical_and_orthogonal_directions(self):
        front = self._plane_wave((1.0, 0.0, 0.0))
        left = self._plane_wave((0.0, 1.0, 0.0))

        identical = _spatial_alignment_metrics(front, front, hop=128)
        orthogonal = _spatial_alignment_metrics(left, front, hop=128)

        self.assertAlmostEqual(identical["angular_error_mean_deg"], 0.0, places=3)
        self.assertAlmostEqual(identical["direction_cosine"], 1.0, places=5)
        self.assertAlmostEqual(identical["diffuseness_mae"], 0.0, places=6)
        self.assertAlmostEqual(orthogonal["angular_error_mean_deg"], 90.0, places=3)

    def test_plan_spatial_alignment_uses_unambiguous_single_source_frames(self):
        plan = {
            "audio": {"duration_sec": 1.0},
            "scene": {
                "sources": [
                    {
                        "source_id": "source_0",
                        "activity": {"onset_sec": 0.0, "offset_sec": 1.0},
                        "motion": {
                            "keyframes": [
                                {
                                    "t_norm": 0.0,
                                    "position": {
                                        "azimuth_deg": 0.0,
                                        "elevation_deg": 0.0,
                                        "distance_m": 1.0,
                                    },
                                }
                            ]
                        },
                    }
                ]
            },
        }
        front = self._plane_wave((1.0, 0.0, 0.0))
        left = self._plane_wave((0.0, 1.0, 0.0))
        aligned = _plan_spatial_alignment_metrics(front, plan, hop=128)
        wrong = _plan_spatial_alignment_metrics(left, plan, hop=128)
        self.assertEqual(aligned["single_source_frame_count"], 4)
        self.assertAlmostEqual(aligned["angular_error_mean_deg"], 0.0, places=3)
        self.assertAlmostEqual(wrong["angular_error_mean_deg"], 90.0, places=3)

    def test_audio_gate_catches_tail_leakage_and_plan_direction_regression(self):
        target = torch.zeros(4, 8 * 128)
        target[:, : 4 * 128] = self._plane_wave(
            (1.0, 0.0, 0.0), frames=4, hop=128
        )
        generated = target.clone()
        generated[:, 6 * 128 :] = 0.05
        silence = _silence_alignment_metrics(
            generated, target, frame_samples=128, settling_frames=1
        )
        failures = _renderer_audio_gate_failures(
            plan_spatial={
                "single_source_frame_count": 4,
                "valid_direction_fraction": 1.0,
                "angular_error_mean_deg": 80.0,
            },
            target_plan_spatial={"angular_error_mean_deg": 10.0},
            silence=silence,
            max_plan_spatial_excess_deg=30.0,
            min_plan_spatial_valid_fraction=0.5,
            max_settled_trailing_rms=1.0e-3,
        )
        self.assertTrue(any("angular excess" in item for item in failures))
        self.assertTrue(any("trailing-silence" in item for item in failures))

    def test_teacher_forced_metrics_use_the_free_decode_fsm_mask(self):
        class Codec:
            id_to_token = [f"token_{index}" for index in range(6)]

            @staticmethod
            def allowed_next_ids(prefix, **_kwargs):
                return ({1}, {2, 4}, {3, 5})[len(prefix)]

        logits = torch.zeros(3, 6)
        logits[1, 2], logits[1, 4] = 2.0, 1.0
        logits[2, 3], logits[2, 5] = 0.5, 2.0
        metrics = _teacher_forced_token_metrics(
            codec=Codec(),
            logits=logits,
            target_tokens=torch.tensor([1, 2, 3]),
            loss_group_ids=torch.tensor([1, 3, 6]),
            min_sources=1,
            max_sources=4,
            fixed_duration_sec=10.05,
        )
        self.assertEqual(metrics["ambiguous_token_count"], 2)
        self.assertEqual(metrics["error_count"], 1)
        self.assertFalse(metrics["exact"])
        self.assertEqual(metrics["errors"][0]["index"], 2)
        self.assertEqual(metrics["errors"][0]["group"], "motion")
        self.assertEqual(metrics["errors"][0]["target_rank"], 2)
        self.assertAlmostEqual(
            metrics["errors"][0]["target_minus_best_logit"], -1.5
        )

    def test_decoded_plan_gate_reports_field_mismatches(self):
        target = {
            "audio": {"duration_sec": 10.05},
            "mix": {"type": "single"},
            "scene": {
                "room": {"rt60_s": 0.3},
                "sources": [{"source_id": "s0", "event": {"label": "bird"}}],
            },
        }
        predicted = {
            **target,
            "scene": {
                "room": {"rt60_s": 0.5},
                "sources": [{"source_id": "s0", "event": {"label": "car"}}],
            },
        }
        metrics = _plan_field_metrics(predicted, target)
        self.assertFalse(metrics["decoded_plan_exact"])
        self.assertFalse(metrics["room_exact"])
        failures = _plan_gate_failures(
            plan_accuracy=0.5,
            field_metrics=metrics,
            min_plan_token_accuracy=1.0,
            min_source_component_accuracy=1.0,
            require_exact_plan=True,
        )
        self.assertEqual(len(failures), 3)

if __name__ == "__main__":
    unittest.main()
