from __future__ import annotations

import copy
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from stable_audio_tools.models.autoencoders import (
    AudioAutoencoder,
    create_autoencoder_from_config,
)
from stable_audio_tools.models.foa_spar import FOALatentSPARDecoder
from stable_audio_tools.training.losses.foa_spar import (
    SCMSupervisedLatentSPARObjective,
)
from stable_audio_tools.training.autoencoders import AutoencoderTrainingWrapper


class _ToyDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Conv1d(8, 4, kernel_size=1)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        decoded = self.projection(latents)
        return F.interpolate(decoded, scale_factor=16, mode="linear")


def _spar() -> FOALatentSPARDecoder:
    return FOALatentSPARDecoder(
        latent_dim=8,
        spatial_latent_dim=3,
        sample_rate=8_000,
        n_bands=4,
        n_fft=128,
        hop_length=32,
        smooth_frames=3,
        max_gain=1.5,
    )


class FOALatentSPARDecoderTests(unittest.TestCase):
    def test_zero_initialized_head_is_function_preserving(self):
        generator = torch.Generator().manual_seed(1)
        decoded_base = torch.randn(2, 4, 997, generator=generator)
        latents = torch.randn(2, 8, 17, generator=generator)

        decoded, info = _spar()(decoded_base, latents, return_info=True)

        self.assertTrue(torch.equal(decoded, decoded_base))
        self.assertEqual(tuple(info["latent_band_gains"].shape), (2, 3, 4, 17))
        self.assertEqual(float(info["latent_band_gains"].abs().max()), 0.0)

    def test_spatial_branch_reads_only_zs_and_preserves_w(self):
        generator = torch.Generator().manual_seed(2)
        module = _spar()
        with torch.no_grad():
            module.gain_head.weight.normal_(mean=0.0, std=0.05, generator=generator)

        decoded_base = torch.randn(1, 4, 997, generator=generator)
        latents = torch.randn(1, 8, 17, generator=generator)
        gains = module.predict_band_gains(latents)

        changed_transport = latents.clone()
        changed_transport[:, : module.transport_latent_dim] += 100.0
        self.assertTrue(
            torch.equal(gains, module.predict_band_gains(changed_transport))
        )

        zero_spatial = latents.clone()
        zero_spatial[:, module.transport_latent_dim :] = 0.0
        decoded = module(decoded_base, zero_spatial)
        self.assertTrue(torch.equal(decoded, decoded_base))

        decoded = module(decoded_base, latents)
        self.assertTrue(torch.equal(decoded[:, :1], decoded_base[:, :1]))
        self.assertGreater(float((decoded[:, 1:] - decoded_base[:, 1:]).abs().mean()), 0.0)
        self.assertLessEqual(float(gains.abs().max()), module.max_gain + 1e-6)

    def test_overlap_add_reconstructs_constant_gain(self):
        generator = torch.Generator().manual_seed(3)
        module = _spar()
        omni = torch.randn(2, 1, 997, generator=generator)
        unit_gains = torch.ones(2, 3, 4, 1, dtype=torch.complex64)

        spatial, _ = module.synthesize_spatial(omni, unit_gains)

        expected = omni.expand(-1, 3, -1)
        self.assertTrue(torch.allclose(spatial, expected, atol=2e-6, rtol=1e-5))

    def test_scm_teacher_recovers_known_wiener_gain_and_residual(self):
        generator = torch.Generator().manual_seed(4)
        module = _spar()
        omni = torch.randn(2, 1, 997, generator=generator)
        known_gain = torch.tensor([0.5, -0.25, 1.2]).view(1, 3, 1)
        target = torch.cat((omni, known_gain * omni), dim=1)

        teacher = module.scm_teacher(target)

        expected_gain = known_gain.unsqueeze(2).expand(2, 3, 4, -1)
        self.assertTrue(
            torch.allclose(
                teacher["band_gains"].real,
                expected_gain,
                atol=2e-5,
                rtol=1e-5,
            )
        )
        self.assertLess(float(teacher["band_gains"].imag.abs().max()), 2e-5)
        self.assertLess(float(teacher["residual"].abs().max()), 2e-5)

    def test_supervision_backpropagates_to_zero_initialized_head(self):
        generator = torch.Generator().manual_seed(5)
        module = _spar()
        decoded_base = torch.randn(2, 4, 997, generator=generator)
        latents = torch.randn(2, 8, 17, generator=generator)
        target = torch.randn(2, 4, 997, generator=generator)
        _, info = module(decoded_base, latents, return_info=True)

        terms = SCMSupervisedLatentSPARObjective()(module, info, target)
        terms["latent_spar_gain"].backward()

        self.assertIsNotNone(module.gain_head.weight.grad)
        self.assertTrue(torch.isfinite(module.gain_head.weight.grad).all())
        self.assertGreater(float(module.gain_head.weight.grad.norm()), 0.0)
        self.assertTrue(torch.isfinite(terms["latent_spar_orthogonality"]))

    def test_old_autoencoder_state_loads_strictly_as_zero_head_warm_start(self):
        old_model = AudioAutoencoder(
            encoder=None,
            decoder=_ToyDecoder(),
            latent_dim=8,
            downsampling_ratio=16,
            sample_rate=8_000,
            io_channels=4,
        )
        spar_model = AudioAutoencoder(
            encoder=None,
            decoder=_ToyDecoder(),
            latent_dim=8,
            downsampling_ratio=16,
            sample_rate=8_000,
            io_channels=4,
            latent_spar=_spar(),
        )
        result = spar_model.load_state_dict(old_model.state_dict(), strict=True)
        self.assertEqual(result.missing_keys, [])
        self.assertEqual(result.unexpected_keys, [])

        latents = torch.randn(2, 8, 17)
        expected = old_model.decode(latents)
        actual, info = spar_model.decode(latents, return_spar_info=True)
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(float(info["latent_band_gains"].abs().max()), 0.0)

    def test_factory_wiring_is_explicitly_opt_in(self):
        config = {
            "sample_rate": 8_000,
            "model": {
                "encoder": {
                    "type": "oobleck",
                    "config": {
                        "in_channels": 4,
                        "channels": 4,
                        "c_mults": [1],
                        "strides": [2],
                        "latent_dim": 8,
                        "use_snake": False,
                    },
                },
                "decoder": {
                    "type": "oobleck",
                    "config": {
                        "out_channels": 4,
                        "channels": 4,
                        "c_mults": [1],
                        "strides": [2],
                        "latent_dim": 8,
                        "use_snake": False,
                        "final_tanh": False,
                    },
                },
                "latent_dim": 8,
                "downsampling_ratio": 2,
                "io_channels": 4,
                "latent_spar": {
                    "enabled": True,
                    "type": "foa_latent_spar",
                    "config": {
                        "spatial_latent_dim": 3,
                        "n_bands": 4,
                        "n_fft": 128,
                        "hop_length": 32,
                        "smooth_frames": 3,
                    },
                },
            },
        }

        model = create_autoencoder_from_config(config)

        self.assertIsInstance(model.latent_spar, FOALatentSPARDecoder)
        self.assertEqual(model.latent_spar.transport_latent_dim, 5)
        disabled_loss_wrapper = AutoencoderTrainingWrapper(
            model,
            sample_rate=8_000,
            loss_config={
                "time": {"type": "l1", "weights": {"l1": 0.0}},
                "latent_spar": {"enabled": False},
            },
            use_ema=False,
        )
        self.assertFalse(hasattr(disabled_loss_wrapper, "latent_spar_objective"))

        baseline_config = copy.deepcopy(config)
        del baseline_config["model"]["latent_spar"]
        baseline = create_autoencoder_from_config(baseline_config)
        self.assertIsNone(baseline.latent_spar)
        self.assertFalse(
            any(key.startswith("latent_spar.") for key in baseline.state_dict())
        )
        latents = torch.randn(1, 8, 16)
        self.assertTrue(torch.equal(baseline.decode(latents), baseline.decoder(latents)))

        baseline_wrapper = AutoencoderTrainingWrapper(
            baseline,
            sample_rate=8_000,
            loss_config={"time": {"type": "l1", "weights": {"l1": 0.0}}},
            use_ema=False,
        )
        self.assertFalse(hasattr(baseline_wrapper, "latent_spar_objective"))

        disabled_config = copy.deepcopy(config)
        disabled_config["model"]["latent_spar"]["enabled"] = False
        disabled = create_autoencoder_from_config(disabled_config)
        self.assertIsNone(disabled.latent_spar)

    def test_training_wrapper_installs_and_evaluates_all_opt_in_losses(self):
        model = AudioAutoencoder(
            encoder=None,
            decoder=_ToyDecoder(),
            latent_dim=8,
            downsampling_ratio=16,
            sample_rate=8_000,
            io_channels=4,
            latent_spar=_spar(),
        )
        loss_config = {
            "spectral": {
                "type": "mrstft",
                "config": {
                    "fft_sizes": [128],
                    "hop_sizes": [32],
                    "win_lengths": [128],
                    "perceptual_weighting": False,
                },
                "weights": {"mrstft": 1.0},
            },
            "time": {"type": "l1", "weights": {"l1": 0.0}},
            "latent_spar": {
                "enabled": True,
                "type": "scm_supervised",
                "crop_frames": 4,
                "weights": {
                    "gain": 0.05,
                    "residual_stft": 0.1,
                    "orthogonality": 0.01,
                    "leakage": 0.01,
                },
            },
        }
        wrapper = AutoencoderTrainingWrapper(
            model,
            sample_rate=8_000,
            loss_config=loss_config,
            use_ema=False,
        )
        latents = torch.randn(2, 8, 17)
        decoded, spar_info = model.decode(latents, return_spar_info=True)
        target = torch.randn_like(decoded)
        loss_info = {"decoded": decoded, "reals": target}
        loss_info.update(
            wrapper.latent_spar_objective(model.latent_spar, spar_info, target)
        )
        transport_latents = latents.clone()
        transport_latents[:, model.latent_spar.transport_latent_dim :] = 0.0
        loss_info["latent_spar_leakage"] = (
            model.decode(transport_latents)[:, 1:].abs().mean()
        )

        loss, terms = wrapper.losses_gen(loss_info)
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(
            set(terms),
            {
                "mrstft_loss",
                "latent_spar_gain_loss",
                "latent_spar_residual_mrstft",
                "latent_spar_orthogonality_loss",
                "latent_spar_leakage_loss",
            },
        )
        self.assertGreater(float(model.latent_spar.gain_head.weight.grad.norm()), 0.0)


if __name__ == "__main__":
    unittest.main()
