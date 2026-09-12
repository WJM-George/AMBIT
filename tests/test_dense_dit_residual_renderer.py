from __future__ import annotations

import copy
import unittest
from unittest.mock import patch

import torch
from torch import nn

from stable_audio_tools.models.dense_dit_renderer import FrozenDenseDiTRenderer
from stable_audio_tools.models.dit import DiffusionTransformer
from stable_audio_tools.models.transfusion_spatial import TransfusionSpatialWrapper
from stable_audio_tools.models.transfusion_spatial import _SourceRegionalFrameAdapter


class _FakeDenseRuntime:
    pad_channel_first = staticmethod(
        lambda values: (
            torch.stack(list(values)),
            torch.ones(
                len(values),
                values[0].shape[-1],
                dtype=torch.bool,
                device=values[0].device,
            ),
        )
    )

    def __init__(self, dense_velocity):
        self.dense_velocity = dense_velocity

    def predict_velocity(
        self,
        noised,
        times,
        conditioning,
        valid,
        *,
        prepared_conditioning=None,
        dit_times=None,
        external_cross_attn_residual=None,
        external_cross_attn_kv_lora_down=None,
        external_cross_attn_kv_lora_up=None,
        external_cross_attn_kv_lora_scale=1.0,
        external_cross_attn_kv_lora_start_index=0,
        external_layer_replacements=None,
        external_layer_replacement_start_index=0,
    ):
        self.external_cross_attn_residual = external_cross_attn_residual
        self.external_cross_attn_kv_lora_down = (
            external_cross_attn_kv_lora_down
        )
        self.external_cross_attn_kv_lora_up = external_cross_attn_kv_lora_up
        self.external_cross_attn_kv_lora_scale = (
            external_cross_attn_kv_lora_scale
        )
        self.external_cross_attn_kv_lora_start_index = (
            external_cross_attn_kv_lora_start_index
        )
        self.external_layer_replacements = external_layer_replacements
        self.external_layer_replacement_start_index = (
            external_layer_replacement_start_index
        )
        return self.dense_velocity.expand_as(noised)

    def source_region_membership(
        self,
        conditioning,
        *,
        prepared_conditioning,
        max_sources,
        device,
    ):
        membership = getattr(self, "membership", None)
        if membership is None:
            raise RuntimeError("test runtime has no source membership")
        return membership.to(device=device)

    @staticmethod
    def prepare_conditioning(conditioning, *, device):
        return {"conditioning": conditioning, "device": device}

    @staticmethod
    def compose_source_condition_residual(
        source_condition_codes,
        conditioning,
        *,
        prepared_conditioning,
    ):
        del conditioning, prepared_conditioning
        return torch.stack(list(source_condition_codes))


class _RngConsumingDenseRuntime(FrozenDenseDiTRenderer):
    def __init__(self):
        self._device = None
        self._renderer = None

    def _load_uncached(self, device):
        torch.randn(4096, device=device)
        self._device = device
        self._renderer = object()
        return self._renderer


class DenseDiTResidualRendererTests(unittest.TestCase):
    def test_native_layer_replacement_is_exact_then_trainable(self):
        torch.manual_seed(71)
        model = DiffusionTransformer(
            io_channels=2,
            embed_dim=64,
            cond_token_dim=16,
            depth=3,
            num_heads=1,
            diffusion_objective="rectified_flow",
            activation_checkpointing=False,
        ).eval().requires_grad_(False)
        for layer in model.transformer.layers[-2:]:
            nn.init.xavier_uniform_(layer.self_attn.to_out.weight)
            nn.init.xavier_uniform_(layer.cross_attn.to_out.weight)
            nn.init.xavier_uniform_(layer.ff.ff[2].weight)
        replacements = nn.ModuleList(
            [copy.deepcopy(layer) for layer in model.transformer.layers[-2:]]
        ).requires_grad_(True)
        latent = torch.randn(2, 2, 5)
        times = torch.tensor([0.25, 0.75])
        condition = torch.randn(2, 4, 16)
        with patch(
            "stable_audio_tools.models.transformer.flash_attn_func", None
        ), patch(
            "stable_audio_tools.models.transformer.flash_attn_varlen_func", None
        ):
            baseline = model(latent, times, cross_attn_cond=condition)
            exact = model(
                latent,
                times,
                cross_attn_cond=condition,
                external_layer_replacements=replacements,
                external_layer_replacement_start_index=1,
            )
        torch.testing.assert_close(exact, baseline, rtol=0.0, atol=0.0)

        exact.square().mean().backward()
        replacement_grad = sum(
            float(parameter.grad.abs().sum())
            for parameter in replacements.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(replacement_grad, 0.0)
        self.assertTrue(
            all(parameter.grad is None for parameter in model.parameters())
        )

        with torch.no_grad():
            replacements[-1].ff.ff[2].weight.add_(0.01)
        with patch(
            "stable_audio_tools.models.transformer.flash_attn_func", None
        ), patch(
            "stable_audio_tools.models.transformer.flash_attn_varlen_func", None
        ):
            changed = model(
                latent,
                times,
                cross_attn_cond=condition,
                external_layer_replacements=replacements,
                external_layer_replacement_start_index=1,
            )
        self.assertGreater(float((changed - baseline).abs().max()), 0.0)

        with patch(
            "stable_audio_tools.models.transformer.flash_attn_func", None
        ), patch(
            "stable_audio_tools.models.transformer.flash_attn_varlen_func", None
        ), self.assertRaisesRegex(ValueError, "replacement range"):
            model(
                latent,
                times,
                cross_attn_cond=condition,
                external_layer_replacements=replacements,
                external_layer_replacement_start_index=2,
            )

    def test_native_kv_lora_adapter_is_function_preserving_at_init(self):
        adapter = _SourceRegionalFrameAdapter(
            max_sources=2,
            num_features=3,
            output_dim=8,
            semantic_rank=4,
            condition_token_dim=6,
            condition_temporal_bins=2,
            dense_kv_lora_layer_count=2,
            dense_kv_lora_context_dim=8,
            dense_kv_lora_rank=3,
            source_aggregation="permutation_equivariant",
        )
        self.assertEqual(tuple(adapter.dense_kv_lora_down.shape), (2, 8, 3))
        self.assertEqual(tuple(adapter.dense_kv_lora_up.shape), (2, 3, 16))
        self.assertGreater(float(adapter.dense_kv_lora_down.abs().sum()), 0.0)
        torch.testing.assert_close(
            adapter.dense_kv_lora_up,
            torch.zeros_like(adapter.dense_kv_lora_up),
        )

    def test_zero_condition_regional_input_is_exact_and_differentiable(self):
        torch.manual_seed(37)
        model = DiffusionTransformer(
            io_channels=2,
            embed_dim=64,
            cond_token_dim=16,
            depth=1,
            num_heads=1,
            diffusion_objective="rectified_flow",
            activation_checkpointing=False,
        ).eval()
        # Fresh blocks intentionally zero-init branch outputs; the real Dense
        # checkpoint has trained non-zero cross-attention projections.
        nn.init.xavier_uniform_(
            model.transformer.layers[0].cross_attn.to_out.weight
        )
        latent = torch.randn(2, 2, 5)
        times = torch.tensor([0.25, 0.75])
        condition = torch.randn(2, 4, 16)
        regional = torch.zeros_like(condition, requires_grad=True)
        mask = torch.ones(2, 4, dtype=torch.bool)
        with patch(
            "stable_audio_tools.models.transformer.flash_attn_func", None
        ), patch(
            "stable_audio_tools.models.transformer.flash_attn_varlen_func", None
        ):
            baseline = model(
                latent,
                times,
                cross_attn_cond=condition,
                cross_attn_cond_mask=mask,
            )
            controlled = model(
                latent,
                times,
                cross_attn_cond=condition,
                cross_attn_cond_mask=mask,
                external_cross_attn_residual=regional,
            )
        torch.testing.assert_close(controlled, baseline, rtol=0.0, atol=0.0)
        controlled.square().mean().backward()
        self.assertIsNotNone(regional.grad)
        self.assertTrue(torch.isfinite(regional.grad).all())
        self.assertGreater(float(regional.grad.abs().sum()), 0.0)

        with self.assertRaises(ValueError):
            model(
                latent,
                times,
                cross_attn_cond=condition,
                external_cross_attn_residual=torch.zeros(2, 3, 16),
            )

    def test_zero_native_cross_attention_kv_lora_is_exact_and_differentiable(self):
        torch.manual_seed(39)
        model = DiffusionTransformer(
            io_channels=2,
            embed_dim=64,
            cond_token_dim=16,
            depth=2,
            num_heads=1,
            diffusion_objective="rectified_flow",
            activation_checkpointing=False,
        ).eval()
        for layer in model.transformer.layers:
            nn.init.xavier_uniform_(layer.cross_attn.to_out.weight)
        latent = torch.randn(2, 2, 5)
        times = torch.tensor([0.25, 0.75])
        condition = torch.randn(2, 4, 16)
        mask = torch.ones(2, 4, dtype=torch.bool)
        down = torch.randn(1, 64, 4, requires_grad=True)
        up = torch.zeros(1, 4, 128, requires_grad=True)
        attention_patches = (
            patch("stable_audio_tools.models.transformer.flash_attn_func", None),
            patch(
                "stable_audio_tools.models.transformer.flash_attn_varlen_func",
                None,
            ),
        )
        with attention_patches[0], attention_patches[1]:
            baseline = model(
                latent,
                times,
                cross_attn_cond=condition,
                cross_attn_cond_mask=mask,
            )
            controlled = model(
                latent,
                times,
                cross_attn_cond=condition,
                cross_attn_cond_mask=mask,
                external_cross_attn_kv_lora_down=down,
                external_cross_attn_kv_lora_up=up,
                external_cross_attn_kv_lora_start_index=1,
            )
        torch.testing.assert_close(controlled, baseline, rtol=0.0, atol=0.0)
        controlled.square().mean().backward()
        self.assertIsNotNone(up.grad)
        self.assertTrue(torch.isfinite(up.grad).all())
        self.assertGreater(float(up.grad.abs().sum()), 0.0)
        self.assertIsNotNone(down.grad)
        torch.testing.assert_close(down.grad, torch.zeros_like(down.grad))

        with self.assertRaises(ValueError):
            model(
                latent,
                times,
                cross_attn_cond=condition,
                cross_attn_cond_mask=mask,
                external_cross_attn_kv_lora_down=down,
                external_cross_attn_kv_lora_up=torch.zeros(1, 3, 128),
                external_cross_attn_kv_lora_start_index=1,
            )

    def test_lazy_frozen_renderer_loading_is_rng_transparent(self):
        torch.manual_seed(29)
        expected = torch.randn(8)
        torch.manual_seed(29)
        _RngConsumingDenseRuntime().ensure_loaded(torch.device("cpu"))
        actual = torch.randn(8)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def test_native_sampling_contract_uses_interval_count_and_channel_first_rng(self):
        self.assertEqual(FrozenDenseDiTRenderer.native_integration_points(30), 31)
        with self.assertRaises(ValueError):
            FrozenDenseDiTRenderer.native_integration_points(0)

        torch.manual_seed(17)
        expected = torch.randn(1, 2, 3)[0]
        torch.manual_seed(17)
        actual = FrozenDenseDiTRenderer.draw_native_noise(
            2,
            3,
            device=torch.device("cpu"),
        )
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

        torch.manual_seed(17)
        flattened_time_first = torch.randn(3, 2).movedim(-1, 0)
        self.assertFalse(torch.equal(actual, flattened_time_first))

    def test_dense_conditioning_uses_exact_fixed_tensor_duration(self):
        wrapper = TransfusionSpatialWrapper.__new__(TransfusionSpatialWrapper)
        nn.Module.__init__(wrapper)
        wrapper.model_config = {
            "fixed_audio_window": {
                "duration_sec": 442368 / 44100,
                "downsampling_ratio": 1024,
                "sample_rate": 44100,
            }
        }
        condition = wrapper.dense_dit_conditioning(
            "a fixed ten-second scene",
        )
        self.assertEqual(condition["seconds_total"], 442368 / 44100)
