from __future__ import annotations

import torch

from stable_audio_tools.training.diffusion import DiffusionCondTrainingWrapper


def _metadata(kind: str) -> dict:
    return {
        "model_sceneplan": {
            "sources": [{"kind": kind}],
        }
    }


def test_sound_transient_weight_is_scoped_and_mean_preserving() -> None:
    wrapper = object.__new__(DiffusionCondTrainingWrapper)
    wrapper.sceneplan_sound_transient_enabled = True
    wrapper.sceneplan_sound_transient_scope = "sound_only"
    wrapper.sceneplan_sound_transient_quantile = 0.8
    wrapper.sceneplan_sound_transient_weight = 3.0
    wrapper.sceneplan_sound_transient_dilation_frames = 1

    clean = torch.zeros((2, 2, 8), dtype=torch.float32)
    clean[:, :, 4:] = 10.0
    loss = torch.ones_like(clean)
    valid = torch.ones((2, 8), dtype=torch.bool)
    weighted, metrics = wrapper._apply_sceneplan_sound_transient_weight(
        loss,
        clean,
        valid,
        [_metadata("sound"), _metadata("music")],
    )

    # The sound attack plus one neighboring frame on either side is emphasized.
    assert torch.all(weighted[0, :, 3:6] > 1.0)
    assert torch.all(weighted[0, :, :3] < 1.0)
    assert torch.all(weighted[0, :, 6:] < 1.0)
    # A music row with an identical latent derivative is outside the causal arm.
    torch.testing.assert_close(weighted[1], torch.ones_like(weighted[1]))
    # Per-row valid-frame normalization preserves the optimizer loss scale.
    torch.testing.assert_close(weighted.mean(dim=(1, 2)), torch.ones(2))
    torch.testing.assert_close(
        metrics["train/sound_transient_eligible_fraction"], torch.tensor(0.5)
    )
    torch.testing.assert_close(
        metrics["train/sound_transient_frame_fraction"], torch.tensor(3.0 / 8.0)
    )


def test_flat_sound_latent_does_not_mark_every_frame_transient() -> None:
    wrapper = object.__new__(DiffusionCondTrainingWrapper)
    wrapper.sceneplan_sound_transient_enabled = True
    wrapper.sceneplan_sound_transient_scope = "sound_only"
    wrapper.sceneplan_sound_transient_quantile = 0.8
    wrapper.sceneplan_sound_transient_weight = 3.0
    wrapper.sceneplan_sound_transient_dilation_frames = 2

    clean = torch.zeros((1, 4, 12), dtype=torch.float32)
    loss = torch.arange(48, dtype=torch.float32).reshape(1, 4, 12)
    valid = torch.ones((1, 12), dtype=torch.bool)
    weighted, metrics = wrapper._apply_sceneplan_sound_transient_weight(
        loss,
        clean,
        valid,
        [_metadata("sound")],
    )

    torch.testing.assert_close(weighted, loss)
    torch.testing.assert_close(
        metrics["train/sound_transient_frame_fraction"], torch.tensor(0.0)
    )
