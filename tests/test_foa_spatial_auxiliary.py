from __future__ import annotations

import unittest

import torch
from torch import nn

from stable_audio_tools.training.foa_spatial_auxiliary import (
    DecodedFOASpatialAuxiliaryLoss,
)


class _FrozenToyFOADecoder(nn.Module):
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return latent[:, :4].repeat_interleave(256, dim=-1)


class DecodedFOASpatialAuxiliaryTests(unittest.TestCase):
    def _loss(self) -> DecodedFOASpatialAuxiliaryLoss:
        return DecodedFOASpatialAuxiliaryLoss(
            {
                "target_modality": "foa_latent",
                "weight": 0.25,
                "samples_per_rank": 1,
                "crop_frames": 8,
                "require_activity_mask": True,
                "channel_order": "WYZX",
                "n_ffts": [256],
                "smooth_frames": 3,
            },
            sample_rate=44_100,
            fixed_window_frames=12,
        )

    def test_full_window_input_produces_finite_nonzero_gradient(self):
        generator = torch.Generator().manual_seed(7)
        omni = torch.randn(12, generator=generator)
        target = torch.stack(
            [omni, 0.4 * omni, 0.2 * omni, 0.7 * omni], dim=0
        )
        predicted_clean = target.clone()
        predicted_clean[1:] = -predicted_clean[1:]
        predicted_flow = predicted_clean.requires_grad_(True)
        activity = torch.tensor(
            [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0], dtype=torch.float32
        )

        auxiliary = self._loss()
        value = auxiliary(
            [predicted_flow],
            [torch.zeros_like(predicted_flow)],
            [torch.tensor(0.0)],
            target_latents=[target],
            activity_masks=[activity],
            metadata_rows=[{}],
            pretransform=_FrozenToyFOADecoder(),
            rank=0,
            step=0,
        )
        value.backward()

        self.assertTrue(torch.isfinite(value))
        self.assertGreater(float(value), 0.0)
        self.assertIsNotNone(predicted_flow.grad)
        self.assertTrue(torch.isfinite(predicted_flow.grad).all())
        self.assertGreater(float(predicted_flow.grad.norm()), 0.0)
        self.assertEqual(
            float(auxiliary.last_metrics["main_window_frames"]), 12.0
        )
        self.assertEqual(
            float(auxiliary.last_metrics["aux_crop_frames"]), 8.0
        )

    def test_rejects_shortened_primary_window(self):
        auxiliary = self._loss()
        shortened = torch.zeros(4, 8, requires_grad=True)
        with self.assertRaisesRegex(ValueError, "fixed-window contract"):
            auxiliary(
                [shortened],
                [torch.zeros_like(shortened)],
                [torch.tensor(0.0)],
                target_latents=[torch.zeros_like(shortened)],
                activity_masks=[torch.ones(8)],
                metadata_rows=[{}],
                pretransform=_FrozenToyFOADecoder(),
                rank=0,
                step=0,
            )

    def test_multirow_selection_is_rotating_and_batch_stratified(self):
        auxiliary = DecodedFOASpatialAuxiliaryLoss(
            {
                "samples_per_rank": 4,
                "crop_frames": 8,
                "n_ffts": [256],
            },
            sample_rate=44_100,
            fixed_window_frames=12,
        )
        self.assertEqual(
            auxiliary._selected_rows(24, rank=0, step=0),
            [0, 6, 12, 18],
        )
        self.assertEqual(
            auxiliary._selected_rows(24, rank=1, step=1),
            [2, 8, 14, 20],
        )

    def test_metadata_filter_selects_only_source_isolated_rows(self):
        auxiliary = DecodedFOASpatialAuxiliaryLoss(
            {
                "samples_per_rank": 2,
                "crop_frames": 8,
                "n_ffts": [256],
                "selection_metadata_key": "curriculum_kind",
                "selection_metadata_values": ["isolated_source_creation"],
            },
            sample_rate=44_100,
            fixed_window_frames=12,
        )
        metadata = [
            {"curriculum_kind": "mixture_creation"},
            {"curriculum_kind": "isolated_source_creation"},
            {"curriculum_kind": "isolated_source_creation"},
            {"curriculum_kind": "mixture_creation"},
        ]
        eligible = auxiliary._eligible_rows(4, metadata)
        self.assertEqual(eligible, [1, 2])
        self.assertEqual(
            auxiliary._selected_rows(
                4, rank=0, step=0, eligible_rows=eligible
            ),
            [1, 2],
        )

    def test_metadata_filter_empty_batch_is_differentiable_zero(self):
        auxiliary = DecodedFOASpatialAuxiliaryLoss(
            {
                "samples_per_rank": 1,
                "crop_frames": 8,
                "n_ffts": [256],
                "selection_metadata_key": "curriculum_kind",
                "selection_metadata_values": ["isolated_source_creation"],
            },
            sample_rate=44_100,
            fixed_window_frames=12,
        )
        predicted = torch.randn(4, 12, requires_grad=True)
        value = auxiliary(
            [predicted],
            [torch.zeros_like(predicted)],
            [torch.tensor(0.5)],
            target_latents=[torch.zeros_like(predicted)],
            activity_masks=[torch.ones(12)],
            metadata_rows=[{"curriculum_kind": "mixture_creation"}],
            pretransform=_FrozenToyFOADecoder(),
            rank=0,
            step=0,
        )
        value.backward()
        self.assertEqual(float(value), 0.0)
        self.assertIsNotNone(predicted.grad)
        self.assertEqual(float(predicted.grad.norm()), 0.0)
        self.assertEqual(float(auxiliary.last_metrics["eligible_fraction"]), 0.0)


if __name__ == "__main__":
    unittest.main()
