"""Training safety/geometry checks for the isolated decoded-audio prototype."""
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from stable_audio_tools.training.losses.sceneplan_editing_audio import (
    EditingAudioAuxiliaryConfig,
    EditingDecodedAudioAuxiliary,
    select_audio_auxiliary_rows,
)

torch.set_num_threads(2)


class ToyFrozenVAE(nn.Module):
    def __init__(self):
        super().__init__()
        generator = torch.Generator().manual_seed(29)
        self.register_buffer("kernel", torch.randn(4, 64, 3, generator=generator) / 20)
        self.eval()

    def decode(self, latent):
        return F.conv1d(latent, self.kernel, padding=1).tanh().repeat_interleave(1024, dim=-1)


def fixture_inputs():
    generator = torch.Generator().manual_seed(7)
    decoder = ToyFrozenVAE()
    noised = torch.randn(2, 64, 6, generator=generator)
    velocity = torch.randn(2, 64, 6, generator=generator).requires_grad_()
    target = decoder.decode(torch.randn(1, 64, 6, generator=generator))[0].detach().requires_grad_()
    source = decoder.decode(torch.randn(1, 64, 6, generator=generator))[0].detach().requires_grad_()
    kwargs = dict(
        decoder=decoder, operations=["static_to_linear", "event_addition"],
        model_num_samples=[4500, 4500], valid_latent_frames=[5, 5],
        source_audio={0: source}, target_audio={0: target},
    )
    return noised, velocity, torch.tensor([0.2, 0.2]), kwargs


def test_backward_reaches_velocity_without_supervision_or_padding_gradients():
    noised, velocity, times, kwargs = fixture_inputs()
    result = EditingDecodedAudioAuxiliary()(noised, velocity, times, **kwargs)
    result.loss.backward()
    assert result.selected_rows == (0,)
    assert torch.isfinite(velocity.grad).all() and velocity.grad[0, :, :5].norm() > 0
    assert torch.count_nonzero(velocity.grad[0, :, 5:]) == 0
    assert torch.count_nonzero(velocity.grad[1]) == 0
    assert kwargs["source_audio"][0].grad is None
    assert kwargs["target_audio"][0].grad is None


def test_latent_and_waveform_padding_cannot_change_auxiliary():
    noised, velocity, times, kwargs = fixture_inputs()
    auxiliary = EditingDecodedAudioAuxiliary()
    baseline = auxiliary(noised, velocity, times, **kwargs).loss.detach()
    changed_noised, changed_velocity = noised.clone(), velocity.detach().clone().requires_grad_()
    changed_noised[:, :, 5:] = 1000
    with torch.no_grad():
        changed_velocity[:, :, 5:] = -1000
    changed = dict(kwargs)
    for name in ("source_audio", "target_audio"):
        audio = kwargs[name][0].detach().clone()
        audio[:, 4500:] = 1000
        changed[name] = {0: audio}
    actual = auxiliary(changed_noised, changed_velocity, times, **changed)
    torch.testing.assert_close(actual.loss.detach(), baseline, rtol=0, atol=0)
    actual.loss.backward()
    assert torch.count_nonzero(changed_velocity.grad[:, :, 5:]) == 0


def test_out_of_scope_rows_require_no_audio_or_decoder_work():
    class MustNotDecode(nn.Module):
        def decode(self, value):
            raise AssertionError("out-of-scope auxiliary tried to decode")

    velocity = torch.randn(3, 64, 4, requires_grad=True)
    result = EditingDecodedAudioAuxiliary()(
        torch.zeros_like(velocity), velocity, torch.tensor([0.2, 0.9, 0.0]),
        decoder=MustNotDecode(), operations=["event_addition", "static_to_linear", "linear_to_static"],
        model_num_samples=[4096] * 3, valid_latent_frames=[4] * 3,
        source_audio={}, target_audio={},
    )
    assert result.selected_rows == () and result.loss.item() == 0
    result.loss.backward()
    assert torch.count_nonzero(velocity.grad) == 0


def test_selection_includes_boundary_rotates_and_preserves_rng():
    times = torch.tensor([0.3, 0.2, 0.31, 0.1])
    operations = ["static_to_linear"] * 3 + ["event_removal"]
    rng = torch.get_rng_state().clone()
    config = EditingAudioAuxiliaryConfig()
    assert select_audio_auxiliary_rows(times, operations, config) == (0,)
    assert select_audio_auxiliary_rows(times, operations, config, selection_offset=1) == (1,)
    assert torch.equal(rng, torch.get_rng_state())


def test_vae_must_be_frozen_and_in_eval_mode():
    noised, velocity, times, kwargs = fixture_inputs()
    kwargs["decoder"].train()
    with pytest.raises(ValueError, match="frozen eval-mode"):
        EditingDecodedAudioAuxiliary()(noised, velocity, times, **kwargs)
    kwargs["decoder"].eval().register_parameter("unsafe_weight", nn.Parameter(torch.ones(())))
    with pytest.raises(ValueError, match="frozen eval-mode"):
        EditingDecodedAudioAuxiliary()(noised, velocity, times, **kwargs)


def test_zero_spatial_contrast_is_reported_as_unavailable():
    noised, velocity, times, kwargs = fixture_inputs()
    kwargs["source_audio"] = {0: kwargs["target_audio"][0]}
    result = EditingDecodedAudioAuxiliary()(noised, velocity, times, **kwargs)
    assert result.metrics["edit_scm_available_resolutions"] == 0
    assert result.metrics["scm_edit"] == 0
    assert result.metrics["scm_full"] > 0
    result.loss.backward()
    assert torch.isfinite(velocity.grad).all()


def test_invalid_geometry_fails_before_decoding():
    noised, velocity, times, kwargs = fixture_inputs()
    kwargs["valid_latent_frames"] = [4, 5]
    with pytest.raises(ValueError, match="geometry"):
        EditingDecodedAudioAuxiliary()(noised, velocity, times, **kwargs)


@pytest.mark.parametrize("weight", [-1, float("nan"), float("inf")])
def test_invalid_loss_weights_fail(weight):
    with pytest.raises(ValueError, match="weights"):
        EditingAudioAuxiliaryConfig(w_spectral_weight=weight)
