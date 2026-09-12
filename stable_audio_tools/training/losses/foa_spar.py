"""Training objective for decoder-side FOA Latent-SPAR."""

from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from ...models.foa_spar import FOALatentSPARDecoder


class SCMSupervisedLatentSPARObjective(nn.Module):
    """Prepare unweighted Latent-SPAR losses and residual waveforms.

    The objective deliberately does not own the SPAR decoder, so registering it
    inside the Lightning wrapper cannot duplicate model parameters or state.
    Loss weights and schedules remain the responsibility of the existing
    ``MultiLoss`` machinery.
    """

    def forward(
        self,
        spar_decoder: FOALatentSPARDecoder,
        spar_info: Dict[str, torch.Tensor],
        target_foa: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        required = {"omni", "residual", "latent_band_gains"}
        missing = required.difference(spar_info)
        if missing:
            raise KeyError(f"Missing Latent-SPAR decode info: {sorted(missing)}")
        if target_foa.ndim != 3 or target_foa.shape[1] != 4:
            raise ValueError("target_foa must have shape [batch, 4, samples]")

        residual = spar_info["residual"]
        omni = spar_info["omni"]
        samples = min(target_foa.shape[-1], residual.shape[-1], omni.shape[-1])
        if samples <= 0:
            raise ValueError("Latent-SPAR supervision received an empty waveform")

        target = target_foa[..., :samples]
        residual = residual[..., :samples].contiguous()
        omni = omni[..., :samples]
        teacher = spar_decoder.scm_teacher(target)

        teacher_frame_gains = teacher["frame_band_gains"]
        predicted_frame_gains = spar_decoder.interpolate_band_gains(
            spar_info["latent_band_gains"], teacher_frame_gains.shape[-1]
        )
        gain_error = (predicted_frame_gains - teacher_frame_gains).abs()
        gain_weight = teacher["gain_weight"].unsqueeze(1)
        gain_loss = (gain_error * gain_weight).sum() / (
            gain_weight.sum() * gain_error.shape[1] + spar_decoder.eps
        )

        orthogonality = spar_decoder.residual_coherence(residual, omni)
        return {
            "latent_spar_gain": gain_loss,
            "latent_spar_orthogonality": orthogonality,
            "latent_spar_residual": residual,
            "latent_spar_residual_target": teacher["residual"].contiguous(),
        }
