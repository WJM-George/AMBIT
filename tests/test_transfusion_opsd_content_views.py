import pytest
import torch
import torch.nn.functional as F

from stable_audio_tools.training.transfusion_opsd.event_content_views import (
    ContentViewConfig, event_relative_content_view,
)


def event():
    x = torch.zeros(1, 4, 22050)
    t = torch.arange(9000) / 44100.
    x[0, 0, 1000:10000] = .2 * torch.cos(2 * torch.pi * 440 * t)
    x[0, 1] = x[0, 0] * .3
    return x


def test_arbitrary_silence_placement_leaves_event_view_unchanged():
    x = event()
    original, _ = event_relative_content_view(x)
    shifted, info = event_relative_content_view(F.pad(x, (3219, 6571)))
    assert torch.equal(original, shifted)
    assert info['admissible'] and info['discarded_w_energy_fraction'] == 0


def test_interior_silence_and_quiet_word_are_not_removed():
    x = event(); x[..., 3000:6000] = 0; x[..., 6000:9000] *= .002
    result, _ = event_relative_content_view(x)
    assert torch.equal(result[..., 4410:13410], x[..., 1000:10000])


def test_cumulative_quiet_content_vetoes_trimming():
    x = torch.zeros(1, 4, 80000)
    x[:, 0] = .9e-4; x[:, 0, 40000] = 1
    result, info = event_relative_content_view(x, config=ContentViewConfig(context_samples=0))
    assert not info['admissible'] and info['discarded_w_energy_fraction'] > 1e-6
    assert result is x


def test_retained_samples_keep_gradient_without_changing_input():
    x = event().requires_grad_(); before = x.detach().clone()
    result, info = event_relative_content_view(x)
    result.square().sum().backward()
    assert torch.equal(x.detach(), before) and x.grad.abs().sum() > 0
    assert torch.equal(x.grad[..., info['start_sample']:info['end_sample']],
                       2 * x.detach()[..., info['start_sample']:info['end_sample']])


def test_silence_is_explicitly_not_present():
    _, info = event_relative_content_view(torch.zeros(1, 4, 10000))
    assert not info['present']


@pytest.mark.parametrize('kwargs', [dict(relative_amplitude_floor=float('nan')),
    dict(context_samples=-1), dict(max_discarded_w_energy_fraction=-1), dict(silence_peak=0)])
def test_invalid_calibration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        ContentViewConfig(**kwargs)
