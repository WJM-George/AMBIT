"""Decoder-side latent SPAR synthesis for first-order Ambisonics.

The base waveform decoder is interpreted as producing ``[W, rY, rZ, rX]``.
A small, bias-free head reads only the spatial latent channels and predicts a
complex Wiener gain for each directional channel and ERB band.  The official
FOA reconstruction is then

    YZX = rYZX + iSTFT(H(z_S) * STFT(W)).

The head is zero-initialized, so enabling this module on an existing decoder is
a function-preserving warm start.  SCM-derived Wiener targets are exposed for
training, but are never required by the inference path.
"""

from __future__ import annotations

from typing import Dict

import torch
from torch import nn
from torch.nn import functional as F


def _erb_rate(frequency_hz: torch.Tensor) -> torch.Tensor:
    """Glasberg-Moore ERB-rate scale."""

    return 21.4 * torch.log10(1.0 + 0.00437 * frequency_hz)


class FOALatentSPARDecoder(nn.Module):
    """Add SCM-supervisable latent SPAR synthesis after a 4-channel decoder.

    Args:
        latent_dim: Total number of continuous latent channels.
        spatial_latent_dim: Trailing latent channels assigned to ``z_S``.
        sample_rate: Waveform sample rate.
        n_bands: Number of ERB-spaced complex gains per directional channel.
        n_fft: Analysis/synthesis FFT size.
        hop_length: Analysis/synthesis hop size.
        smooth_frames: Temporal SCM smoothing width used by the teacher.
        max_gain: Radial bound applied to every complex Wiener gain.
        energy_floor_ratio: Relative floor used only in normalized diagnostics.
        eps: Numerical epsilon.

    Input channel order is fixed to native FOA ``[W, Y, Z, X]``.  The module
    intentionally has no learned bias: ``z_S = 0`` must imply ``H = 0``.
    """

    def __init__(
        self,
        latent_dim: int,
        spatial_latent_dim: int = 24,
        sample_rate: int = 44100,
        n_bands: int = 8,
        n_fft: int = 2048,
        hop_length: int = 512,
        smooth_frames: int = 5,
        max_gain: float = 1.5,
        energy_floor_ratio: float = 1e-3,
        eps: float = 1e-8,
    ):
        super().__init__()

        latent_dim = int(latent_dim)
        spatial_latent_dim = int(spatial_latent_dim)
        n_bands = int(n_bands)
        n_fft = int(n_fft)
        hop_length = int(hop_length)
        smooth_frames = int(smooth_frames)

        if latent_dim <= 1:
            raise ValueError("latent_dim must be greater than one")
        if not 0 < spatial_latent_dim < latent_dim:
            raise ValueError("spatial_latent_dim must split the latent channels")
        if n_bands <= 0:
            raise ValueError("n_bands must be positive")
        if n_fft <= 1 or hop_length <= 0 or hop_length > n_fft:
            raise ValueError("Require n_fft > 1 and 0 < hop_length <= n_fft")
        if smooth_frames <= 0:
            raise ValueError("smooth_frames must be positive")
        if smooth_frames % 2 == 0:
            smooth_frames += 1
        if max_gain <= 0:
            raise ValueError("max_gain must be positive")
        if energy_floor_ratio <= 0:
            raise ValueError("energy_floor_ratio must be positive")

        self.latent_dim = latent_dim
        self.spatial_latent_dim = spatial_latent_dim
        self.transport_latent_dim = latent_dim - spatial_latent_dim
        self.sample_rate = int(sample_rate)
        self.n_bands = n_bands
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.smooth_frames = smooth_frames
        self.max_gain = float(max_gain)
        self.energy_floor_ratio = float(energy_floor_ratio)
        self.eps = float(eps)

        # Three FOA directional channels x ERB bands x (real, imaginary).
        # No bias is an architectural guarantee that H(z_S=0) is exactly zero.
        self.gain_head = nn.Conv1d(
            spatial_latent_dim,
            3 * n_bands * 2,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        nn.init.zeros_(self.gain_head.weight)

        frequencies = torch.fft.rfftfreq(n_fft, d=1.0 / self.sample_rate)
        erb_frequencies = _erb_rate(frequencies)
        erb_edges = torch.linspace(
            float(erb_frequencies[0]),
            float(erb_frequencies[-1]),
            n_bands + 1,
        )
        # bucketize against interior edges gives IDs in [0, n_bands - 1].
        bin_to_band = torch.bucketize(erb_frequencies, erb_edges[1:-1])
        band_membership = F.one_hot(bin_to_band, num_classes=n_bands).T.float()

        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)
        self.register_buffer("bin_to_band", bin_to_band.long(), persistent=False)
        self.register_buffer("band_membership", band_membership, persistent=False)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        # A checkpoint predating Latent-SPAR has no head tensor.  Supplying the
        # known zero initialization here makes strict model-weight warm starts
        # function preserving.  A present tensor is still validated normally.
        head_key = prefix + "gain_head.weight"
        if head_key not in state_dict:
            state_dict[head_key] = torch.zeros_like(self.gain_head.weight)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _limit_complex_gain(self, gain: torch.Tensor) -> torch.Tensor:
        magnitude = gain.abs()
        scale = torch.clamp(self.max_gain / (magnitude + self.eps), max=1.0)
        return gain * scale

    def predict_band_gains(self, latents: torch.Tensor) -> torch.Tensor:
        """Predict bounded complex gains shaped ``[B, 3, bands, latent_T]``."""

        if latents.ndim != 3:
            raise ValueError("latents must have shape [batch, channels, frames]")
        if latents.shape[1] != self.latent_dim:
            raise ValueError(
                f"Expected {self.latent_dim} latent channels, got {latents.shape[1]}"
            )

        spatial_latents = latents[:, -self.spatial_latent_dim :]
        # Keep the small complex-valued synthesis path in float32 under AMP.
        with torch.autocast(device_type=latents.device.type, enabled=False):
            raw = self.gain_head(spatial_latents.to(self.gain_head.weight.dtype))
            raw = raw.float()
            batch, _, frames = raw.shape
            raw = raw.reshape(batch, 3, self.n_bands, 2, frames)
            raw = raw.permute(0, 1, 2, 4, 3).contiguous()
            gain = torch.view_as_complex(raw)
            return self._limit_complex_gain(gain)

    @staticmethod
    def _interpolate_complex(gain: torch.Tensor, frames: int) -> torch.Tensor:
        if gain.shape[-1] == frames:
            return gain
        batch, channels, bands, _ = gain.shape
        real = F.interpolate(
            gain.real.reshape(batch, channels * bands, -1),
            size=frames,
            mode="linear",
            align_corners=False,
        ).reshape(batch, channels, bands, frames)
        imag = F.interpolate(
            gain.imag.reshape(batch, channels * bands, -1),
            size=frames,
            mode="linear",
            align_corners=False,
        ).reshape(batch, channels, bands, frames)
        return torch.complex(real, imag)

    def interpolate_band_gains(self, gain: torch.Tensor, frames: int) -> torch.Tensor:
        """Public time interpolation used by synthesis and teacher matching."""

        return self._interpolate_complex(gain, int(frames))

    def _padding_for_length(self, samples: int) -> tuple[int, int]:
        # Explicit context padding keeps center=False while avoiding the Hann
        # endpoints becoming uncovered during overlap-add.
        left = self.n_fft - self.hop_length
        right = left
        padded = samples + left + right
        remainder = (padded - self.n_fft) % self.hop_length
        right += (self.hop_length - remainder) % self.hop_length
        return left, right

    def _stft(self, audio: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        if audio.ndim != 3:
            raise ValueError("audio must have shape [batch, channels, samples]")
        batch, channels, samples = audio.shape
        left, right = self._padding_for_length(samples)
        audio = F.pad(audio.float(), (left, right))
        spectrum = torch.stft(
            audio.reshape(batch * channels, audio.shape[-1]),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window.float(),
            center=False,
            return_complex=True,
        )
        spectrum = spectrum.reshape(
            batch, channels, spectrum.shape[-2], spectrum.shape[-1]
        )
        return spectrum, left, samples

    def _istft(self, spectrum: torch.Tensor, left: int, samples: int) -> torch.Tensor:
        """Differentiable center=False inverse STFT using explicit overlap-add."""

        if spectrum.ndim != 4:
            raise ValueError("spectrum must have shape [batch, channels, bins, frames]")
        batch, channels, _, frames = spectrum.shape
        output_length = self.n_fft + self.hop_length * (frames - 1)

        time_frames = torch.fft.irfft(spectrum, n=self.n_fft, dim=-2)
        window = self.window.float()
        time_frames = time_frames * window.view(1, 1, -1, 1)
        folded = F.fold(
            time_frames.reshape(batch * channels, self.n_fft, frames),
            output_size=(1, output_length),
            kernel_size=(1, self.n_fft),
            stride=(1, self.hop_length),
        ).reshape(batch, channels, output_length)

        window_sq = window.square().view(1, self.n_fft, 1).expand(1, -1, frames)
        denominator = F.fold(
            window_sq,
            output_size=(1, output_length),
            kernel_size=(1, self.n_fft),
            stride=(1, self.hop_length),
        ).reshape(1, 1, output_length)
        folded = folded / denominator.clamp_min(self.eps)
        return folded[..., left : left + samples]

    def synthesize_spatial(
        self,
        omni: torch.Tensor,
        band_gains: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Synthesize three directional channels from W and complex band gains.

        Returns the waveform contribution and gains interpolated to STFT frames.
        """

        if omni.ndim != 3 or omni.shape[1] != 1:
            raise ValueError("omni must have shape [batch, 1, samples]")
        if band_gains.ndim != 4 or band_gains.shape[1:3] != (3, self.n_bands):
            raise ValueError(
                f"band_gains must have shape [batch, 3, {self.n_bands}, frames]"
            )

        output_dtype = omni.dtype
        with torch.autocast(device_type=omni.device.type, enabled=False):
            omni_spectrum, left, samples = self._stft(omni)
            frame_gains = self._interpolate_complex(
                band_gains.to(torch.complex64), omni_spectrum.shape[-1]
            )
            bin_gains = frame_gains[:, :, self.bin_to_band, :]
            spatial_spectrum = omni_spectrum * bin_gains
            spatial = self._istft(spatial_spectrum, left, samples)
        return spatial.to(output_dtype), frame_gains

    def forward(
        self,
        decoded_base: torch.Tensor,
        latents: torch.Tensor,
        return_info: bool = False,
    ):
        if decoded_base.ndim != 3 or decoded_base.shape[1] != 4:
            raise ValueError(
                "FOALatentSPARDecoder expects [batch, 4, samples] in [W,Y,Z,X] order"
            )
        if decoded_base.shape[0] != latents.shape[0]:
            raise ValueError("decoded_base and latents must have the same batch size")

        latent_gains = self.predict_band_gains(latents)
        omni = decoded_base[:, :1]
        residual = decoded_base[:, 1:]
        spatial_prediction, frame_gains = self.synthesize_spatial(omni, latent_gains)
        output = torch.cat((omni, residual + spatial_prediction), dim=1)

        if not return_info:
            return output
        return output, {
            "omni": omni,
            "residual": residual,
            "spatial_prediction": spatial_prediction,
            "latent_band_gains": latent_gains,
            "frame_band_gains": frame_gains,
        }

    def _smooth_time(self, value: torch.Tensor) -> torch.Tensor:
        if self.smooth_frames == 1:
            return value
        if value.is_complex():
            stacked = torch.view_as_real(value).movedim(-1, -2)
            smoothed = self._smooth_time(stacked)
            return torch.view_as_complex(smoothed.movedim(-2, -1).contiguous())
        batch = value.shape[0]
        frames = value.shape[-1]
        flat = value.reshape(batch, -1, frames)
        smoothed = F.avg_pool1d(
            flat,
            kernel_size=self.smooth_frames,
            stride=1,
            padding=self.smooth_frames // 2,
            count_include_pad=False,
        )
        return smoothed.reshape_as(value)

    @torch.no_grad()
    def scm_teacher(self, target_foa: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Build complex Wiener gain and residual teachers from target SCMs."""

        if target_foa.ndim != 3 or target_foa.shape[1] != 4:
            raise ValueError("target_foa must have shape [batch, 4, samples]")

        target_foa = target_foa.float()
        spectrum, _, _ = self._stft(target_foa)
        omni_spectrum = spectrum[:, :1]
        directional_spectrum = spectrum[:, 1:]

        c_cw = self._smooth_time(
            directional_spectrum * torch.conj(omni_spectrum)
        )
        c_ww = self._smooth_time(omni_spectrum.abs().square())[:, 0]

        band_cross = torch.einsum(
            "kf,bcft->bckt", self.band_membership.to(c_cw.dtype), c_cw
        )
        band_omni_energy = torch.einsum(
            "kf,bft->bkt", self.band_membership, c_ww
        )
        teacher_gain = band_cross / band_omni_energy.unsqueeze(1).clamp_min(self.eps)
        teacher_gain = self._limit_complex_gain(teacher_gain)

        predicted_from_omni, frame_gains = self.synthesize_spatial(
            target_foa[:, :1], teacher_gain
        )
        target_residual = target_foa[:, 1:] - predicted_from_omni.float()

        gain_weight = band_omni_energy / (
            band_omni_energy.sum(dim=(-2, -1), keepdim=True) + self.eps
        )
        return {
            "band_gains": teacher_gain,
            "frame_band_gains": frame_gains,
            "gain_weight": gain_weight,
            "residual": target_residual,
        }

    def residual_coherence(
        self,
        residual: torch.Tensor,
        omni: torch.Tensor,
    ) -> torch.Tensor:
        """Energy-weighted residual/W coherence used for identifiability."""

        if residual.ndim != 3 or residual.shape[1] != 3:
            raise ValueError("residual must have shape [batch, 3, samples]")
        if omni.ndim != 3 or omni.shape[1] != 1:
            raise ValueError("omni must have shape [batch, 1, samples]")
        samples = min(residual.shape[-1], omni.shape[-1])

        with torch.autocast(device_type=residual.device.type, enabled=False):
            residual_spectrum, _, _ = self._stft(residual[..., :samples])
            omni_spectrum, _, _ = self._stft(omni[..., :samples])
            cross = self._smooth_time(
                residual_spectrum * torch.conj(omni_spectrum)
            )
            residual_energy = self._smooth_time(residual_spectrum.abs().square())
            omni_energy = self._smooth_time(omni_spectrum.abs().square())

            residual_floor = (
                self.energy_floor_ratio
                * residual_energy.mean(dim=(-2, -1), keepdim=True)
            ).clamp_min(self.eps)
            omni_floor = (
                self.energy_floor_ratio
                * omni_energy.mean(dim=(-2, -1), keepdim=True)
            ).clamp_min(self.eps)
            coherence = cross.abs() / (
                (residual_energy + residual_floor)
                * (omni_energy + omni_floor)
            ).sqrt()
            energy_weight = omni_energy / (
                omni_energy.sum(dim=(-2, -1), keepdim=True) + self.eps
            )
            numerator = (energy_weight * coherence).sum()
            denominator = energy_weight.sum() * residual.shape[1] + self.eps
            return numerator / denominator
