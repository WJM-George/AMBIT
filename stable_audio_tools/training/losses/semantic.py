import math
import typing as tp

import audiotools
import torch
import torchaudio

from einops import rearrange
from torch.nn import functional as F
from torch import nn

def fold_channels_into_batch(x):
    x = rearrange(x, 'b c ... -> (b c) ...')
    return x

class HubertLoss(nn.Module):
    def __init__(self,
        feature_ids: tp.Optional[tp.List[int]] = None,
        weight: float = 1.0,
        model_name: str = "HUBERT_LARGE"
    ):
        super().__init__()

        self.weight = weight
        self.feature_ids = feature_ids
        self.model_name = model_name
        
        # Load model based on the specified model name
        if self.model_name == "WAVLM_LARGE":
            bundle = torchaudio.pipelines.WAVLM_LARGE
        elif self.model_name == "HUBERT_LARGE":
            bundle = torchaudio.pipelines.HUBERT_LARGE
        elif self.model_name == "WAV2VEC2_LARGE_LV60K":
            bundle = torchaudio.pipelines.WAV2VEC2_LARGE_LV60K
        else:
            raise ValueError(f"Unsupported model_name: {self.model_name}")

        self.model = bundle.get_model()

        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        x = fold_channels_into_batch(x)
        y = fold_channels_into_batch(y)

        conv_features = (
            self.feature_ids is not None and
            len(self.feature_ids) == 1 and
            self.feature_ids[0] == -1)

        # Extract features from conv layers only.
        if conv_features:
            if self.model.normalize_waveform:
                x = nn.functional.layer_norm(x, x.shape)
                y = nn.functional.layer_norm(y, y.shape)
            x_list, _ = self.model.model.feature_extractor(x, None)
            y_list, _ = self.model.model.feature_extractor(y, None)
            x_list = [x_list]
            y_list = [y_list]
        else:
            x_list, _ = self.model.extract_features(x)
            y_list, _ = self.model.extract_features(y)

        loss = 0
        denom = 0
        for i, (x, y) in enumerate(zip(x_list, y_list)):
            if self.feature_ids is None or i in self.feature_ids or conv_features:
                loss += F.l1_loss(x, y) / (y.std() + 1e-5)
                denom += 1

        loss = loss / denom
        return self.weight * loss

class CLAPLoss(nn.Module):
    """Feature matching loss using CLAP (HTSAT) intermediate layer features.

    Extracts features from the HTSAT Swin Transformer BasicLayer stages
    in CLAP's audio branch and computes L1 loss normalized by target
    feature standard deviation, following the same pattern as HubertLoss.

    HTSAT-base has 4 BasicLayer stages with output dimensions
    [256, 512, 1024, 1024]. Use feature_ids to select specific layers
    (0-3), or None for all layers.

    Only non-fusion CLAP models are supported (fusion models require
    precomputed mel spectrograms with a non-differentiable pipeline).
    """

    def __init__(
        self,
        sample_rate: int = 48000,
        clap_model: str = 'music_audioset_epoch_15_esc_90.14.pt',
        feature_ids: tp.Optional[tp.List[int]] = None,
        weight: float = 1.0,
    ):
        super().__init__()
        self.weight = weight
        self.feature_ids = feature_ids
        self.source_sample_rate = sample_rate
        self.clap_sample_rate = 48000
        self.clap_max_len = 480000  # 10 seconds at 48kHz

        # Load CLAP, keep only the audio branch, discard text encoder
        from stable_audio_tools.training.metrics.fad_metrics import load_clap_model
        clap_module = load_clap_model(clap_model=clap_model, device='cpu')

        audio_branch = clap_module.model.audio_branch
        assert not audio_branch.enable_fusion, (
            "CLAPLoss only supports non-fusion CLAP models. "
            "Use e.g. 'music_audioset_epoch_15_esc_90.14.pt'."
        )

        # Keep only the audio branch — text encoder is not needed
        self.audio_branch = audio_branch
        del clap_module

        for param in self.audio_branch.parameters():
            param.requires_grad = False

        # Resampler for converting source sample rate to 48kHz
        if self.source_sample_rate != self.clap_sample_rate:
            self.resampler = torchaudio.transforms.Resample(
                self.source_sample_rate, self.clap_sample_rate
            )
        else:
            self.resampler = None

    def _preprocess(self, audio):
        """Resample and pad/truncate to CLAP's expected 10s at 48kHz."""
        if self.resampler is not None:
            audio = self.resampler(audio)

        T = audio.shape[-1]
        if T > self.clap_max_len:
            audio = audio[..., :self.clap_max_len]
        elif T < self.clap_max_len:
            audio = F.pad(audio, (0, self.clap_max_len - T))

        return audio

    def _extract_features(self, audio):
        """Run audio through HTSAT's spectrogram frontend and transformer
        layers, returning intermediate features from each BasicLayer."""
        ab = self.audio_branch

        # Non-fusion spectrogram path
        x = ab.spectrogram_extractor(audio)  # (B, 1, T, freq_bins)
        x = ab.logmel_extractor(x)           # (B, 1, T, mel_bins)
        x = x.transpose(1, 3)
        x = ab.bn0(x)
        x = x.transpose(1, 3)
        # spec_augmenter is skipped since model is in eval mode
        x = ab.reshape_wav2img(x)

        # Patch embedding + Swin Transformer layers
        x = ab.patch_embed(x)
        if ab.ape:
            x = x + ab.absolute_pos_embed
        x = ab.pos_drop(x)

        features = []
        for layer in ab.layers:
            x, _ = layer(x)
            features.append(x)

        return features

    @torch.autocast(device_type='cuda', enabled=False)
    def forward(self, x, y):
        # Force fp32 — CLAP's STFT and logmel are numerically fragile in fp16
        x = fold_channels_into_batch(x).float()
        y = fold_channels_into_batch(y).float()

        x = self._preprocess(x)
        y = self._preprocess(y)

        x_features = self._extract_features(x)
        y_features = self._extract_features(y)

        loss = 0
        denom = 0
        for i, (xf, yf) in enumerate(zip(x_features, y_features)):
            if self.feature_ids is None or i in self.feature_ids:
                loss += F.l1_loss(xf, yf) / (yf.std() + 1e-5)
                denom += 1

        loss = loss / denom
        return self.weight * loss

# Implementation taken from:
# https://github.com/descriptinc/descript-audio-codec/blob/c7cfc5d2647e26471dc394f95846a0830e7bec34/dac/nn/loss.py#L231
class MelSpectrogramLoss(nn.Module):
    """Compute distance between mel spectrograms. Can be used
    in a multi-scale way.

    Parameters
    ----------
    n_mels : List[int]
        Number of mels per STFT, by default [150, 80],
    window_lengths : List[int], optional
        Length of each window of each STFT, by default [2048, 512]
    loss_fn : typing.Callable, optional
        How to compare each loss, by default nn.L1Loss()
    clamp_eps : float, optional
        Clamp on the log magnitude, below, by default 1e-5
    mag_weight : float, optional
        Weight of raw magnitude portion of loss, by default 1.0
    log_weight : float, optional
        Weight of log magnitude portion of loss, by default 1.0
    pow : float, optional
        Power to raise magnitude to before taking log, by default 2.0
    weight : float, optional
        Weight of this loss, by default 1.0
    """

    def __init__(self, sample_rate: int,
        n_mels: tp.List[int],
        window_lengths: tp.List[int],
        loss_fn: tp.Callable = nn.L1Loss(),
        clamp_eps: float = 1e-5,
        mag_weight: float = 1.0,
        log_weight: float = 1.0,
        pow: float = 2.0,
        weight: float = 1.0,
        mel_fmin: tp.Optional[tp.List[float]] = None,
        mel_fmax: tp.Optional[tp.List[float]] = None,
        window_type: tp.Optional[str] = None,
    ):
        super().__init__()
        self.stft_params = [{
            "window_length": w,
            "hop_length": w // 4,
            "window_type": window_type,
        } for w in window_lengths]

        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.loss_fn = loss_fn
        self.clamp_eps = clamp_eps
        self.log_weight = log_weight
        self.mag_weight = mag_weight
        self.weight = weight
        self.pow = pow

        self.mel_fmin = (
            mel_fmin
            if mel_fmin is not None else
            [0.0 for _ in range(len(window_lengths))]
        )
        self.mel_fmax = (
            mel_fmax
            if mel_fmax is not None else
            [None for _ in range(len(window_lengths))]
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        x = audiotools.AudioSignal(x, self.sample_rate)
        y = audiotools.AudioSignal(y, self.sample_rate)

        loss = 0.0
        for n_mels, fmin, fmax, params in zip(
            self.n_mels, self.mel_fmin, self.mel_fmax, self.stft_params,
        ):
            x_mels = x.mel_spectrogram(n_mels, mel_fmin=fmin, mel_fmax=fmax, **params)
            y_mels = y.mel_spectrogram(n_mels, mel_fmin=fmin, mel_fmax=fmax, **params)

            loss += self.log_weight * self.loss_fn(
                x_mels.clamp(self.clamp_eps).pow(self.pow).log10(),
                y_mels.clamp(self.clamp_eps).pow(self.pow).log10(),
            )
            loss += self.mag_weight * self.loss_fn(x_mels, y_mels)
        return loss


class BandWeightedMelSpectrogramLoss(nn.Module):
    """Parallel frequency-band mel reconstruction loss with one shared STFT.

    Each configured band owns an independent mel filterbank and scalar weight.
    The filterbanks are concatenated so all band mel features are produced by one
    matrix multiplication after a single STFT per waveform. Per-band losses are
    averaged over channels, mel bins, and time, then combined using normalized
    band weights so changing the number of bands does not change the global scale.
    """

    def __init__(
        self,
        sample_rate: int,
        bands: tp.List[tp.Dict[str, tp.Any]],
        n_fft: int = 2048,
        hop_length: int = 512,
        win_length: tp.Optional[int] = None,
        clamp_eps: float = 1e-5,
        mag_weight: float = 0.1,
        log_weight: float = 1.0,
        power: float = 2.0,
        mel_scale: str = "slaney",
        norm: tp.Optional[str] = "slaney",
    ):
        super().__init__()
        if not bands:
            raise ValueError("BandWeightedMelSpectrogramLoss requires at least one band")
        if n_fft <= 0 or hop_length <= 0:
            raise ValueError("n_fft and hop_length must be positive")

        self.sample_rate = int(sample_rate)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.win_length = int(win_length or n_fft)
        self.clamp_eps = float(clamp_eps)
        self.mag_weight = float(mag_weight)
        self.log_weight = float(log_weight)
        self.power = float(power)
        self.mel_scale = mel_scale
        self.norm = norm

        if self.win_length > self.n_fft:
            raise ValueError("win_length must not exceed n_fft")
        if self.log_weight < 0 or self.mag_weight < 0:
            raise ValueError("log_weight and mag_weight must be non-negative")
        if self.log_weight == 0 and self.mag_weight == 0:
            raise ValueError("At least one of log_weight or mag_weight must be positive")

        nyquist = self.sample_rate / 2
        filterbanks = []
        band_slices = []
        band_names = []
        raw_weights = []
        offset = 0

        for index, band in enumerate(bands):
            name = str(band.get("name", f"band_{index}"))
            fmin = float(band["fmin"])
            fmax = float(band["fmax"])
            n_mels = int(band["n_mels"])
            weight = float(band["weight"])

            if not 0 <= fmin < fmax <= nyquist:
                raise ValueError(
                    f"{name}: expected 0 <= fmin < fmax <= {nyquist}, "
                    f"got fmin={fmin}, fmax={fmax}"
                )
            if n_mels <= 0:
                raise ValueError(f"{name}: n_mels must be positive")
            if weight < 0:
                raise ValueError(f"{name}: weight must be non-negative")

            filterbank = torchaudio.functional.melscale_fbanks(
                n_freqs=self.n_fft // 2 + 1,
                f_min=fmin,
                f_max=fmax,
                n_mels=n_mels,
                sample_rate=self.sample_rate,
                norm=self.norm,
                mel_scale=self.mel_scale,
            )
            if torch.any(filterbank.sum(dim=0) == 0):
                raise ValueError(
                    f"{name}: mel filterbank contains empty filters; "
                    "reduce n_mels or increase n_fft"
                )

            filterbanks.append(filterbank)
            band_slices.append((offset, offset + n_mels))
            band_names.append(name)
            raw_weights.append(weight)
            offset += n_mels

        raw_weights_tensor = torch.tensor(raw_weights, dtype=torch.float32)
        if raw_weights_tensor.sum() <= 0:
            raise ValueError("At least one band weight must be positive")

        self.band_names = tuple(band_names)
        self.band_slices = tuple(band_slices)
        self.last_band_losses: tp.Dict[str, torch.Tensor] = {}
        self.last_weighted_band_losses: tp.Dict[str, torch.Tensor] = {}
        self.register_buffer(
            "mel_filterbank",
            torch.cat(filterbanks, dim=1).to(torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "band_weights",
            raw_weights_tensor / raw_weights_tensor.sum(),
            persistent=True,
        )
        self.register_buffer(
            "window",
            torch.hann_window(self.win_length),
            persistent=False,
        )

    def _mel_spectrogram(self, audio: torch.Tensor) -> torch.Tensor:
        batch_size, channels, samples = audio.shape
        audio = audio.reshape(batch_size * channels, samples)
        spectrogram = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=False,
            return_complex=True,
        ).abs().pow(self.power)
        mel = torch.einsum("bft,fm->bmt", spectrogram, self.mel_filterbank)
        return mel.reshape(batch_size, channels, mel.shape[-2], mel.shape[-1])

    @torch.amp.autocast("cuda", enabled=False)
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or y.ndim != 3:
            raise ValueError(
                "BandWeightedMelSpectrogramLoss expects [batch, channels, time] tensors"
            )
        if x.shape != y.shape:
            raise ValueError(f"Input and target shapes must match, got {x.shape} and {y.shape}")
        if x.shape[-1] < self.n_fft:
            raise ValueError(
                f"Audio length {x.shape[-1]} must be at least n_fft={self.n_fft}"
            )

        x_mel = self._mel_spectrogram(x.float())
        y_mel = self._mel_spectrogram(y.float())

        band_losses = []
        for start, end in self.band_slices:
            x_band = x_mel[:, :, start:end]
            y_band = y_mel[:, :, start:end]
            band_loss = x_band.new_zeros(())

            if self.log_weight > 0:
                band_loss = band_loss + self.log_weight * F.l1_loss(
                    x_band.clamp_min(self.clamp_eps).log10(),
                    y_band.clamp_min(self.clamp_eps).log10(),
                )
            if self.mag_weight > 0:
                band_loss = band_loss + self.mag_weight * F.l1_loss(
                    x_band,
                    y_band,
                )
            band_losses.append(band_loss)

        stacked_losses = torch.stack(band_losses)
        weighted_losses = stacked_losses * self.band_weights
        self.last_band_losses = {
            name: loss.detach()
            for name, loss in zip(self.band_names, stacked_losses)
        }
        self.last_weighted_band_losses = {
            name: loss.detach()
            for name, loss in zip(self.band_names, weighted_losses)
        }
        return torch.sum(weighted_losses)


class HighFrequencySpectralOvershootLoss(nn.Module):
    """Penalize only spectral energy that exceeds the target in a high band.

    A symmetric reconstruction loss can encourage bandwidth extension on
    band-limited inputs. This loss leaves under-prediction to the regular MRSTFT
    objective and adds pressure only where the reconstruction is louder than the
    target by more than ``margin_db``.
    """

    def __init__(
        self,
        sample_rate: int,
        fmin: float = 8_000.0,
        fmax: tp.Optional[float] = None,
        n_fft: int = 2_048,
        hop_length: int = 512,
        win_length: tp.Optional[int] = None,
        margin_db: float = 1.0,
        floor_db: float = -80.0,
        log_excess_scale_db: float = 20.0,
        linear_weight: float = 0.1,
    ):
        super().__init__()
        self.sample_rate = int(sample_rate)
        self.fmin = float(fmin)
        self.fmax = float(fmax if fmax is not None else sample_rate / 2)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.win_length = int(win_length or n_fft)
        self.margin_db = float(margin_db)
        self.floor_ratio = 10.0 ** (float(floor_db) / 20.0)
        self.log_excess_scale_db = float(log_excess_scale_db)
        self.linear_weight = float(linear_weight)

        nyquist = self.sample_rate / 2
        if not 0 <= self.fmin < self.fmax <= nyquist:
            raise ValueError(
                f"Expected 0 <= fmin < fmax <= {nyquist}, "
                f"got fmin={self.fmin}, fmax={self.fmax}"
            )
        if self.n_fft <= 0 or self.hop_length <= 0:
            raise ValueError("n_fft and hop_length must be positive")
        if not 0 < self.win_length <= self.n_fft:
            raise ValueError("win_length must be in (0, n_fft]")
        if self.margin_db < 0 or self.log_excess_scale_db <= 0:
            raise ValueError("margin_db must be non-negative and log_excess_scale_db positive")
        if self.linear_weight < 0:
            raise ValueError("linear_weight must be non-negative")

        frequencies = torch.fft.rfftfreq(self.n_fft, d=1.0 / self.sample_rate)
        frequency_mask = (frequencies >= self.fmin) & (frequencies <= self.fmax)
        if not torch.any(frequency_mask):
            raise ValueError("The configured high-frequency band contains no STFT bins")

        self.register_buffer("frequency_mask", frequency_mask, persistent=False)
        self.register_buffer(
            "window",
            torch.hann_window(self.win_length),
            persistent=False,
        )
        self.last_log_excess = torch.tensor(0.0)
        self.last_linear_excess = torch.tensor(0.0)
        self.last_active_fraction = torch.tensor(0.0)

    def _magnitude(self, audio: torch.Tensor) -> torch.Tensor:
        batch_size, channels, samples = audio.shape
        audio = audio.reshape(batch_size * channels, samples)
        spectrum = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=False,
            return_complex=True,
        )
        magnitude = spectrum.abs() / self.window.sum().clamp_min(1e-8)
        magnitude = magnitude[:, self.frequency_mask]
        return magnitude.reshape(batch_size, channels, magnitude.shape[-2], magnitude.shape[-1])

    @torch.amp.autocast("cuda", enabled=False)
    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if prediction.ndim != 3 or target.ndim != 3:
            raise ValueError(
                "HighFrequencySpectralOvershootLoss expects [batch, channels, time] tensors"
            )
        if prediction.shape != target.shape:
            raise ValueError(
                f"Prediction and target shapes must match, got "
                f"{prediction.shape} and {target.shape}"
            )
        if prediction.shape[-1] < self.n_fft:
            raise ValueError(
                f"Audio length {prediction.shape[-1]} must be at least n_fft={self.n_fft}"
            )

        prediction = prediction.float()
        target = target.float()
        prediction_mag = self._magnitude(prediction)
        target_mag = self._magnitude(target)

        target_rms = target.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-4)
        target_rms = target_rms.unsqueeze(-1)
        magnitude_floor = target_rms * self.floor_ratio

        ratio_db = (20.0 / math.log(10.0)) * (
            torch.log(prediction_mag + magnitude_floor)
            - torch.log(target_mag + magnitude_floor)
        )
        log_excess = F.relu(ratio_db - self.margin_db) / self.log_excess_scale_db
        linear_excess = F.relu(prediction_mag - target_mag) / target_rms

        self.last_log_excess = log_excess.mean().detach()
        self.last_linear_excess = linear_excess.mean().detach()
        self.last_active_fraction = (log_excess > 0).float().mean().detach()
        return log_excess.mean() + self.linear_weight * linear_excess.mean()


def _logistic_frequency_gate(
    frequencies: torch.Tensor, fc: float, beta: float, floor: float
) -> torch.Tensor:
    """Monotone log-frequency gate g(f) = floor + (1 - floor) / (1 + (f/fc)^beta).

    Smooth, bounded in [floor, 1], equals 1 at DC and decays to `floor` above the
    corner frequency `fc` with steepness `beta`. Shared by the phase and SCM
    losses so every frequency-dependent weight in the objective is one curve
    family with three interpretable parameters.
    """
    if fc <= 0 or beta <= 0:
        raise ValueError("fc and beta must be positive")
    if not 0.0 <= floor <= 1.0:
        raise ValueError("floor must be in [0, 1]")
    ratio = (frequencies / fc).clamp_min(0.0)
    return floor + (1.0 - floor) / (1.0 + ratio.pow(beta))


class FrequencyGatedIFGDPhaseLoss(nn.Module):
    """Frequency-gated instantaneous-frequency / group-delay phase loss.

    Phasor-form IF/GD loss as used by SAME (arXiv:2605.18613, Sec. 3.1.3):
    products of complex STFT values at adjacent frames (IF) / adjacent bins (GD)
    are normalized to unit phasors, and the loss is the cosine distance between
    predicted and reference phasors, weighted by a detached geometric-mean
    magnitude factor. Working on phasors avoids explicit phase unwrapping.

    On top of that formulation this module adds psychoacoustic frequency gates:

      * lambda_IF(f) = g(f; if_fc, if_beta, if_floor): auditory-nerve phase
        locking to the stimulus fine structure degrades above ~1.5-4 kHz
        (Palmer & Russell 1986), so fine-structure phase supervision is
        concentrated below `if_fc` (default 2 kHz).
      * lambda_GD(f) = g(f; gd_fc, gd_beta, gd_floor): group-delay distortion
        stays audible up to ~8 kHz (Blauert & Laws 1978), so the GD gate has a
        higher corner and a larger floor - transient alignment needs wideband
        supervision.

    Gates multiply the per-bin magnitude weights and the loss is a weighted
    mean, so enabling the gates does not change the loss scale. Phase is
    supervised per physical channel (channels folded into the batch), which for
    FOA follows the eps-ar-VAE finding that phase must be supervised on
    physical channels only. A K-weighting prefilter is deliberately omitted:
    any linear filter applied to both signals cancels inside the phasor
    products, so it would only reshape the magnitude weights, which the gates
    already do explicitly.

    An optional normalized complex-distance term (SAME's L_cd) is included via
    `cd_weight`; it is not frequency-gated.
    """

    def __init__(
        self,
        sample_rate: int,
        n_ffts: tp.Sequence[int] = (2048, 1024, 512, 256),
        hop_ratio: int = 4,
        if_fc: float = 2000.0,
        if_beta: float = 2.0,
        if_floor: float = 0.1,
        gd_fc: float = 8000.0,
        gd_beta: float = 2.0,
        gd_floor: float = 0.25,
        if_weight: float = 1.0,
        gd_weight: float = 1.0,
        cd_weight: float = 1.0,
        eps: float = 1e-3,
    ):
        super().__init__()
        self.sample_rate = int(sample_rate)

        if not n_ffts:
            raise ValueError("n_ffts must contain at least one FFT size")
        self.n_ffts = [int(n) for n in n_ffts]
        if any(n <= 1 for n in self.n_ffts):
            raise ValueError("All n_ffts must be > 1")
        if hop_ratio <= 0:
            raise ValueError("hop_ratio must be positive")
        self.hop_lengths = [max(1, n // int(hop_ratio)) for n in self.n_ffts]

        if if_weight < 0 or gd_weight < 0 or cd_weight < 0:
            raise ValueError("if_weight, gd_weight and cd_weight must be non-negative")
        if if_weight == 0 and gd_weight == 0 and cd_weight == 0:
            raise ValueError("At least one of if_weight, gd_weight, cd_weight must be positive")
        self.if_weight = float(if_weight)
        self.gd_weight = float(gd_weight)
        self.cd_weight = float(cd_weight)
        self.eps = float(eps)

        for resolution_index, n_fft in enumerate(self.n_ffts):
            frequencies = torch.fft.rfftfreq(n_fft, d=1.0 / self.sample_rate)
            # GD compares adjacent bins; gate at the midpoint frequency.
            gd_frequencies = 0.5 * (frequencies[1:] + frequencies[:-1])
            self.register_buffer(
                f"window_{resolution_index}",
                torch.hann_window(n_fft),
                persistent=False,
            )
            self.register_buffer(
                f"if_gate_{resolution_index}",
                _logistic_frequency_gate(frequencies, if_fc, if_beta, if_floor),
                persistent=False,
            )
            self.register_buffer(
                f"gd_gate_{resolution_index}",
                _logistic_frequency_gate(gd_frequencies, gd_fc, gd_beta, gd_floor),
                persistent=False,
            )

        self.last_if_loss = torch.tensor(0.0)
        self.last_gd_loss = torch.tensor(0.0)
        self.last_cd_loss = torch.tensor(0.0)

    def _stft(self, audio: torch.Tensor, resolution_index: int) -> torch.Tensor:
        n_fft = self.n_ffts[resolution_index]
        window = getattr(self, f"window_{resolution_index}")
        return torch.stft(
            audio,
            n_fft=n_fft,
            hop_length=self.hop_lengths[resolution_index],
            win_length=n_fft,
            window=window,
            center=False,
            return_complex=True,
        )

    def _gated_phasor_term(
        self,
        pred: torch.Tensor,
        ref: torch.Tensor,
        gate: torch.Tensor,
        dim: int,
    ) -> torch.Tensor:
        """Weighted-mean cosine distance between adjacent-element phasors.

        `dim` = -1 compares adjacent time frames (IF), `dim` = -2 adjacent
        frequency bins (GD). `gate` broadcasts over the frequency axis.
        """
        if dim == -1:
            pred_hi, pred_lo = pred[..., :, 1:], pred[..., :, :-1]
            ref_hi, ref_lo = ref[..., :, 1:], ref[..., :, :-1]
            gate = gate.unsqueeze(-1)  # [F, 1] over [.., F, T-1]
        elif dim == -2:
            pred_hi, pred_lo = pred[..., 1:, :], pred[..., :-1, :]
            ref_hi, ref_lo = ref[..., 1:, :], ref[..., :-1, :]
            gate = gate.unsqueeze(-1)  # [F-1, 1] over [.., F-1, T]
        else:
            raise ValueError("dim must be -1 (IF) or -2 (GD)")

        pred_prod = pred_hi * torch.conj(pred_lo)
        ref_prod = ref_hi * torch.conj(ref_lo)
        pred_denom = (pred_hi.abs() * pred_lo.abs()).clamp_min(self.eps)
        ref_denom = (ref_hi.abs() * ref_lo.abs()).clamp_min(self.eps)
        pred_phasor = pred_prod / pred_denom
        ref_phasor = ref_prod / ref_denom

        # Unclamped magnitudes: bins whose phasor denominators were clamped
        # (near-silence, |U| < 1) get a proportionally tiny weight instead of
        # polluting the weighted mean.
        magnitude_weight = torch.sqrt(
            pred_hi.abs() * pred_lo.abs() * ref_hi.abs() * ref_lo.abs()
        ).detach()
        weight = magnitude_weight * gate
        cosine_distance = 1.0 - (pred_phasor * torch.conj(ref_phasor)).real
        return (weight * cosine_distance).sum() / weight.sum().clamp_min(1e-12)

    @torch.amp.autocast("cuda", enabled=False)
    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if prediction.ndim != 3 or target.ndim != 3:
            raise ValueError(
                "FrequencyGatedIFGDPhaseLoss expects [batch, channels, time] tensors"
            )
        if prediction.shape != target.shape:
            raise ValueError(
                f"Prediction and target shapes must match, got "
                f"{prediction.shape} and {target.shape}"
            )
        if prediction.shape[-1] < max(self.n_ffts):
            raise ValueError(
                f"Audio length {prediction.shape[-1]} must be at least max(n_ffts)={max(self.n_ffts)}"
            )

        batch_size, channels, samples = prediction.shape
        prediction = prediction.reshape(batch_size * channels, samples).float()
        target = target.reshape(batch_size * channels, samples).float()

        total_if = prediction.new_zeros(())
        total_gd = prediction.new_zeros(())
        total_cd = prediction.new_zeros(())

        for resolution_index in range(len(self.n_ffts)):
            pred_spectrum = self._stft(prediction, resolution_index)
            ref_spectrum = self._stft(target, resolution_index)

            if self.if_weight > 0:
                total_if = total_if + self._gated_phasor_term(
                    pred_spectrum, ref_spectrum,
                    getattr(self, f"if_gate_{resolution_index}"), dim=-1,
                )
            if self.gd_weight > 0:
                total_gd = total_gd + self._gated_phasor_term(
                    pred_spectrum, ref_spectrum,
                    getattr(self, f"gd_gate_{resolution_index}"), dim=-2,
                )
            if self.cd_weight > 0:
                squared_error = (pred_spectrum - ref_spectrum).abs().square()
                sigma = squared_error.std(dim=[-1, -2], keepdim=True).detach().clamp_min(1e-5)
                total_cd = total_cd + torch.log(squared_error / sigma + 1.0).mean()

        num_resolutions = len(self.n_ffts)
        if_loss = total_if / num_resolutions
        gd_loss = total_gd / num_resolutions
        cd_loss = total_cd / num_resolutions

        self.last_if_loss = if_loss.detach()
        self.last_gd_loss = gd_loss.detach()
        self.last_cd_loss = cd_loss.detach()

        return (
            self.if_weight * if_loss
            + self.gd_weight * gd_loss
            + self.cd_weight * cd_loss
        )


class FOASpatialCovarianceLoss(nn.Module):
    """Time-frequency spatial covariance matrix (SCM) matching loss for FOA.

    For a 4-channel FOA frame vector s(f, t) in C^4, every linear spatial
    rendering y = A s (binaural, loudspeaker, rotation) has second-order
    statistics that depend on the signal only through the SCM
    C(f, t) = E_tau[s s^H] (C_y = A C A^H). Matching the trace-normalized SCM
    on perceptual time-frequency tiles therefore matches direction, diffuseness,
    channel level structure and inter-channel phase for any downstream linear
    renderer. The active-intensity direction loss (FOASpatialConsistencyLoss,
    after FOA Tokenizer arXiv:2510.22241) is the special case that compares
    Re{conj(W) * [X, Y, Z]}, i.e. the real part of the first SCM row; this loss
    additionally constrains reactive intensity, the XYZ block and diffuseness.

    Closest published prior art is the broadband *time-domain* Pearson
    correlation-matrix L1 of Hirvonen & Namazi (arXiv:2411.12008), which has no
    time-frequency resolution and no phase information; this loss is the
    time-frequency complex generalization they left unexplored, with a polar
    split of the off-diagonal terms:

      * level: L1 between trace-normalized diagonal energy fractions
        (generalizes directional/omni energy-ratio matching);
      * coherence: L1 between pairwise coherence magnitudes
        gamma_ij = |C_ij| / sqrt(C_ii C_jj) in [0, 1] (diffuseness /
        envelopment statistics);
      * ipd: cosine distance between off-diagonal unit phasors
        U_ij = C_ij / |C_ij| (inter-channel phase), weighted by the reference
        coherence gamma_ref (phase of nearly-incoherent pairs is noise and is
        softly masked out) and by a duplex-theory frequency gate
        lambda_IPD(f) = g(f; ipd_fc, ipd_beta, ipd_floor): fine-structure
        inter-channel phase cues are perceptually usable only below ~1.5 kHz
        (Rayleigh's duplex theory), above which level/coherence cues dominate.

    All terms are weighted by the reference trace energy (per-item normalized)
    and reduced as weighted means, so the loss is bounded, scale-invariant and
    its magnitude is insensitive to the number of resolutions. Unlike the
    intensity-vector loss, the SCM loss is invariant to the FOA channel
    ordering convention as long as prediction and target use the same one, so
    no `channel_order` argument is needed.

    Expects [batch, 4, time] waveforms; an all-FOA batch is assumed.
    """

    def __init__(
        self,
        sample_rate: int,
        n_ffts: tp.Sequence[int] = (2048, 512),
        hop_ratio: int = 4,
        smooth_frames: int = 5,
        level_weight: float = 1.0,
        coherence_weight: float = 1.0,
        ipd_weight: float = 1.0,
        ipd_fc: float = 1500.0,
        ipd_beta: float = 2.0,
        ipd_floor: float = 0.1,
        energy_floor_ratio: float = 1e-3,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.sample_rate = int(sample_rate)

        if not n_ffts:
            raise ValueError("n_ffts must contain at least one FFT size")
        self.n_ffts = [int(n) for n in n_ffts]
        if any(n <= 1 for n in self.n_ffts):
            raise ValueError("All n_ffts must be > 1")
        if hop_ratio <= 0:
            raise ValueError("hop_ratio must be positive")
        self.hop_lengths = [max(1, n // int(hop_ratio)) for n in self.n_ffts]

        smooth_frames = int(smooth_frames)
        if smooth_frames < 1:
            raise ValueError("smooth_frames must be >= 1")
        if smooth_frames % 2 == 0:
            smooth_frames += 1  # keep output length identical to input
        self.smooth_frames = smooth_frames

        if level_weight < 0 or coherence_weight < 0 or ipd_weight < 0:
            raise ValueError("level_weight, coherence_weight and ipd_weight must be non-negative")
        if level_weight == 0 and coherence_weight == 0 and ipd_weight == 0:
            raise ValueError(
                "At least one of level_weight, coherence_weight, ipd_weight must be positive"
            )
        if energy_floor_ratio <= 0:
            raise ValueError("energy_floor_ratio must be positive")

        self.level_weight = float(level_weight)
        self.coherence_weight = float(coherence_weight)
        self.ipd_weight = float(ipd_weight)
        self.energy_floor_ratio = float(energy_floor_ratio)
        self.eps = float(eps)

        # Upper-triangular off-diagonal channel pairs of the 4x4 SCM.
        pair_i, pair_j = torch.triu_indices(4, 4, offset=1)
        self.register_buffer("pair_i", pair_i, persistent=False)
        self.register_buffer("pair_j", pair_j, persistent=False)

        for resolution_index, n_fft in enumerate(self.n_ffts):
            frequencies = torch.fft.rfftfreq(n_fft, d=1.0 / self.sample_rate)
            self.register_buffer(
                f"window_{resolution_index}",
                torch.hann_window(n_fft),
                persistent=False,
            )
            self.register_buffer(
                f"ipd_gate_{resolution_index}",
                _logistic_frequency_gate(frequencies, ipd_fc, ipd_beta, ipd_floor),
                persistent=False,
            )

        self.last_level_loss = torch.tensor(0.0)
        self.last_coherence_loss = torch.tensor(0.0)
        self.last_ipd_loss = torch.tensor(0.0)
        self.last_mean_ipd_cos = torch.tensor(0.0)

    def _stft(self, audio: torch.Tensor, resolution_index: int) -> torch.Tensor:
        batch_size, channels, samples = audio.shape
        n_fft = self.n_ffts[resolution_index]
        window = getattr(self, f"window_{resolution_index}")
        spectrum = torch.stft(
            audio.reshape(batch_size * channels, samples),
            n_fft=n_fft,
            hop_length=self.hop_lengths[resolution_index],
            win_length=n_fft,
            window=window,
            center=False,
            return_complex=True,
        )
        return spectrum.reshape(batch_size, channels, spectrum.shape[-2], spectrum.shape[-1])

    def _smooth_time(self, x: torch.Tensor) -> torch.Tensor:
        """Average over self.smooth_frames along the last (time) axis, same length."""
        if self.smooth_frames == 1:
            return x
        if x.is_complex():
            # view_as_real appends a trailing (re, im) axis: [..., F, T, 2].
            # Move it off the time axis, smooth real/imag jointly, and restore.
            stacked = torch.view_as_real(x).movedim(-1, -2)  # [..., F, 2, T]
            smoothed = self._smooth_time(stacked)
            return torch.view_as_complex(smoothed.movedim(-2, -1).contiguous())
        batch_size = x.shape[0]
        frames = x.shape[-1]
        flat = x.reshape(batch_size, -1, frames)
        smoothed = F.avg_pool1d(
            flat,
            kernel_size=self.smooth_frames,
            stride=1,
            padding=self.smooth_frames // 2,
            count_include_pad=False,
        )
        return smoothed.reshape(*x.shape)

    def _covariance_stats(self, spectrum: torch.Tensor):
        """Smoothed diagonal energies [B,4,F,T] and off-diagonal SCM entries [B,6,F,T]."""
        diag_energy = self._smooth_time(spectrum.abs().square())
        cross = self._smooth_time(
            spectrum[:, self.pair_i] * torch.conj(spectrum[:, self.pair_j])
        )
        return diag_energy, cross

    @torch.amp.autocast("cuda", enabled=False)
    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if prediction.ndim != 3 or target.ndim != 3:
            raise ValueError("FOASpatialCovarianceLoss expects [batch, channels, time] tensors")
        if prediction.shape != target.shape:
            raise ValueError(
                f"Prediction and target shapes must match, got {prediction.shape} and {target.shape}"
            )
        if prediction.shape[1] != 4:
            raise ValueError(
                f"FOASpatialCovarianceLoss requires 4 channels (FOA), got {prediction.shape[1]}"
            )
        if prediction.shape[-1] < max(self.n_ffts):
            raise ValueError(
                f"Audio length {prediction.shape[-1]} must be at least max(n_ffts)={max(self.n_ffts)}"
            )

        prediction = prediction.float()
        target = target.float()

        level_num = prediction.new_zeros(())
        level_den = prediction.new_zeros(())
        coherence_num = prediction.new_zeros(())
        coherence_den = prediction.new_zeros(())
        ipd_num = prediction.new_zeros(())
        ipd_den = prediction.new_zeros(())
        ipd_cos_num = prediction.new_zeros(())

        for resolution_index in range(len(self.n_ffts)):
            pred_spectrum = self._stft(prediction, resolution_index)
            ref_spectrum = self._stft(target, resolution_index)

            pred_diag, pred_cross = self._covariance_stats(pred_spectrum)
            ref_diag, ref_cross = self._covariance_stats(ref_spectrum)

            pred_trace = pred_diag.sum(dim=1)  # [B, F, T]
            ref_trace = ref_diag.sum(dim=1)

            # Reference-energy weights, normalized per item so items and
            # resolutions contribute comparably.
            energy_weight = ref_trace / (
                ref_trace.sum(dim=(-2, -1), keepdim=True) + self.eps
            )
            # Energy-scaled floor keeps ratios bounded on near-silent bins
            # without a dataset-specific absolute constant.
            energy_floor = (
                self.energy_floor_ratio * ref_trace.mean(dim=(-2, -1), keepdim=True)
            ).clamp_min(1e-12)

            if self.level_weight > 0:
                pred_fraction = pred_diag / (pred_trace + energy_floor).unsqueeze(1)
                ref_fraction = ref_diag / (ref_trace + energy_floor).unsqueeze(1)
                level_error = (pred_fraction - ref_fraction).abs().sum(dim=1)
                level_num = level_num + (energy_weight * level_error).sum()
                level_den = level_den + energy_weight.sum()

            pair_floor = energy_floor.unsqueeze(1)
            pred_pair_energy = (
                (pred_diag[:, self.pair_i] + pair_floor)
                * (pred_diag[:, self.pair_j] + pair_floor)
            ).sqrt()
            ref_pair_energy = (
                (ref_diag[:, self.pair_i] + pair_floor)
                * (ref_diag[:, self.pair_j] + pair_floor)
            ).sqrt()
            pred_coherence = pred_cross.abs() / pred_pair_energy
            ref_coherence = ref_cross.abs() / ref_pair_energy

            if self.coherence_weight > 0:
                coherence_error = (pred_coherence - ref_coherence).abs().sum(dim=1)
                coherence_num = coherence_num + (energy_weight * coherence_error).sum()
                coherence_den = coherence_den + energy_weight.sum()

            if self.ipd_weight > 0:
                pred_phasor = pred_cross / (pred_cross.abs() + pair_floor)
                ref_phasor = ref_cross / (ref_cross.abs() + pair_floor)
                ipd_cosine = (pred_phasor * torch.conj(ref_phasor)).real

                ipd_gate = getattr(self, f"ipd_gate_{resolution_index}")
                ipd_bin_weight = (
                    energy_weight.unsqueeze(1)
                    * ref_coherence.detach().clamp(0.0, 1.0)
                    * ipd_gate.view(1, 1, -1, 1)
                )
                ipd_num = ipd_num + (ipd_bin_weight * (1.0 - ipd_cosine)).sum()
                ipd_cos_num = ipd_cos_num + (ipd_bin_weight * ipd_cosine).sum()
                ipd_den = ipd_den + ipd_bin_weight.sum()

        level_loss = level_num / level_den.clamp_min(self.eps)
        coherence_loss = coherence_num / coherence_den.clamp_min(self.eps)
        ipd_loss = ipd_num / ipd_den.clamp_min(self.eps)

        self.last_level_loss = level_loss.detach()
        self.last_coherence_loss = coherence_loss.detach()
        self.last_ipd_loss = ipd_loss.detach()
        self.last_mean_ipd_cos = (ipd_cos_num / ipd_den.clamp_min(self.eps)).detach()

        return (
            self.level_weight * level_loss
            + self.coherence_weight * coherence_loss
            + self.ipd_weight * ipd_loss
        )


class FOASpatialConsistencyLoss(nn.Module):
    """DirAC-style spatial consistency loss for first-order ambisonics (FOA).

    Ports the spatial consistency loss of "FOA Tokenizer" (arXiv:2510.22241,
    Sudarsanam & Gamper) from their discrete FOA codec to this continuous 4ch
    VAE. The loss compares time-frequency *active intensity vectors*

        I(t, f) = Re{ conj(W(t, f)) * [X(t, f), Y(t, f), Z(t, f)] }

    of the reconstruction against the target and penalizes directional
    misalignment ``1 - cos(I_rec, I_ref)`` on bins where the target carries a
    reliable directional cue. Following the paper, bins are masked and weighted
    by target energy and (1 - diffuseness), because spatial direction is only
    perceptually meaningful in energetic, non-diffuse regions.

    In addition to the directional term, an optional ``ratio_weight`` term
    matches the log directional-to-omni energy ratio ``log(|XYZ|^2 / |W|^2)``
    on energetic bins. This preserves the diffuseness/spaciousness statistics
    that the cosine term deliberately masks out (this mirrors the
    ``dir_energy_ratio_err`` metric already used by this repo's VAE eval).

    Notes:
      * Expects ``[batch, 4, time]`` waveforms in the repo's FOA channel
        layout ``[W, Y, Z, X]`` (configurable via ``channel_order``).
      * Assumes an all-FOA batch (true for the vae_v2 dataset). For corpora
        mixing binaural/mono rows into the 4-slot layout, gate this loss off
        or filter the dataset, since [L, R, 0, 0] rows have no FOA meaning.
      * Diffuseness needs temporal averaging to be defined, so intensity and
        energies are smoothed over ``smooth_frames`` STFT frames before the
        coherence ``r = |mean(I)| / sqrt(mean|W|^2 * mean|XYZ|^2)`` and
        diffuseness ``D = 1 - r`` are computed.
    """

    def __init__(
        self,
        sample_rate: int,
        channel_order: str = "WYZX",
        n_ffts: tp.Sequence[int] = (2048, 512),
        hop_ratio: int = 4,
        smooth_frames: int = 5,
        energy_rel_threshold: float = 1e-3,
        diffuseness_threshold: float = 0.95,
        direction_weight: float = 1.0,
        ratio_weight: float = 0.25,
        ratio_floor_ratio: float = 1e-3,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.sample_rate = int(sample_rate)

        order = channel_order.upper().strip()
        if sorted(order) != sorted("WXYZ"):
            raise ValueError(
                f"channel_order must be a permutation of 'WXYZ', got '{channel_order}'"
            )
        self.channel_order = order
        self.w_index = order.index("W")
        self.dir_indices = [order.index("X"), order.index("Y"), order.index("Z")]

        if not n_ffts:
            raise ValueError("n_ffts must contain at least one FFT size")
        self.n_ffts = [int(n) for n in n_ffts]
        if any(n <= 0 for n in self.n_ffts):
            raise ValueError("All n_ffts must be positive")
        if hop_ratio <= 0:
            raise ValueError("hop_ratio must be positive")
        self.hop_lengths = [max(1, n // int(hop_ratio)) for n in self.n_ffts]

        smooth_frames = int(smooth_frames)
        if smooth_frames < 1:
            raise ValueError("smooth_frames must be >= 1")
        if smooth_frames % 2 == 0:
            smooth_frames += 1  # keep output length identical to input
        self.smooth_frames = smooth_frames

        if not 0 < diffuseness_threshold <= 1:
            raise ValueError("diffuseness_threshold must be in (0, 1]")
        if energy_rel_threshold < 0:
            raise ValueError("energy_rel_threshold must be non-negative")
        if direction_weight < 0 or ratio_weight < 0:
            raise ValueError("direction_weight and ratio_weight must be non-negative")
        if direction_weight == 0 and ratio_weight == 0:
            raise ValueError("At least one of direction_weight or ratio_weight must be positive")

        if ratio_floor_ratio <= 0:
            raise ValueError("ratio_floor_ratio must be positive")

        self.energy_rel_threshold = float(energy_rel_threshold)
        self.diffuseness_threshold = float(diffuseness_threshold)
        self.direction_weight = float(direction_weight)
        self.ratio_weight = float(ratio_weight)
        self.ratio_floor_ratio = float(ratio_floor_ratio)
        self.eps = float(eps)

        for resolution_index, n_fft in enumerate(self.n_ffts):
            self.register_buffer(
                f"window_{resolution_index}",
                torch.hann_window(n_fft),
                persistent=False,  # keep state_dict unchanged so old ckpts resume strictly
            )

        self.last_direction_loss = torch.tensor(0.0)
        self.last_ratio_loss = torch.tensor(0.0)
        self.last_active_fraction = torch.tensor(0.0)
        self.last_mean_cos = torch.tensor(0.0)

    def _stft(self, audio: torch.Tensor, resolution_index: int) -> torch.Tensor:
        batch_size, channels, samples = audio.shape
        n_fft = self.n_ffts[resolution_index]
        window = getattr(self, f"window_{resolution_index}")
        spectrum = torch.stft(
            audio.reshape(batch_size * channels, samples),
            n_fft=n_fft,
            hop_length=self.hop_lengths[resolution_index],
            win_length=n_fft,
            window=window,
            center=False,
            return_complex=True,
        )
        return spectrum.reshape(batch_size, channels, spectrum.shape[-2], spectrum.shape[-1])

    def _smooth_time(self, x: torch.Tensor) -> torch.Tensor:
        """Average over self.smooth_frames along the last (time) axis, same length."""
        if self.smooth_frames == 1:
            return x
        batch_size = x.shape[0]
        frames = x.shape[-1]
        flat = x.reshape(batch_size, -1, frames)
        smoothed = F.avg_pool1d(
            flat,
            kernel_size=self.smooth_frames,
            stride=1,
            padding=self.smooth_frames // 2,
            count_include_pad=False,
        )
        return smoothed.reshape(*x.shape)

    def _intensity_and_energies(self, spectrum: torch.Tensor):
        """Smoothed intensity [B,3,F,T], omni energy [B,F,T], directional energy [B,F,T]."""
        w = spectrum[:, self.w_index]
        directional = spectrum[:, self.dir_indices]

        intensity = (w.conj().unsqueeze(1) * directional).real
        omni_energy = w.abs().square()
        dir_energy = directional.abs().square().sum(dim=1)

        intensity = self._smooth_time(intensity)
        omni_energy = self._smooth_time(omni_energy)
        dir_energy = self._smooth_time(dir_energy)
        return intensity, omni_energy, dir_energy

    @torch.amp.autocast("cuda", enabled=False)
    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if prediction.ndim != 3 or target.ndim != 3:
            raise ValueError("FOASpatialConsistencyLoss expects [batch, channels, time] tensors")
        if prediction.shape != target.shape:
            raise ValueError(
                f"Prediction and target shapes must match, got {prediction.shape} and {target.shape}"
            )
        if prediction.shape[1] != 4:
            raise ValueError(
                f"FOASpatialConsistencyLoss requires 4 channels (FOA), got {prediction.shape[1]}"
            )
        if prediction.shape[-1] < max(self.n_ffts):
            raise ValueError(
                f"Audio length {prediction.shape[-1]} must be at least max(n_ffts)={max(self.n_ffts)}"
            )

        prediction = prediction.float()
        target = target.float()

        direction_num = prediction.new_zeros(())
        ratio_num = prediction.new_zeros(())
        direction_den = prediction.new_zeros(())
        ratio_den = prediction.new_zeros(())
        cos_num = prediction.new_zeros(())
        active_bins = prediction.new_zeros(())
        total_bins = prediction.new_zeros(())

        for resolution_index in range(len(self.n_ffts)):
            pred_spectrum = self._stft(prediction, resolution_index)
            target_spectrum = self._stft(target, resolution_index)

            pred_intensity, pred_omni, pred_dir = self._intensity_and_energies(pred_spectrum)
            target_intensity, target_omni, target_dir = self._intensity_and_energies(target_spectrum)

            # Coherence r in [0, 1]; diffuseness D = 1 - r (DirAC). Needs the
            # temporal smoothing applied above to be meaningful.
            target_intensity_norm = target_intensity.norm(dim=1)
            coherence = target_intensity_norm / (
                (target_omni * target_dir).clamp_min(0).sqrt() + self.eps
            )
            diffuseness = (1.0 - coherence).clamp(0.0, 1.0)

            energy = target_omni + target_dir
            mean_energy = energy.mean(dim=(-2, -1), keepdim=True)
            energy_mask = energy > (self.energy_rel_threshold * mean_energy)

            # Per-item normalized energy weights make items and resolutions comparable.
            energy_weight = energy / (energy.sum(dim=(-2, -1), keepdim=True) + self.eps)

            directional_mask = energy_mask & (diffuseness < self.diffuseness_threshold)
            direction_bin_weight = directional_mask.float() * energy_weight * (1.0 - diffuseness)

            cosine = F.cosine_similarity(pred_intensity, target_intensity, dim=1, eps=self.eps)

            direction_num = direction_num + (direction_bin_weight * (1.0 - cosine)).sum()
            direction_den = direction_den + direction_bin_weight.sum()
            cos_num = cos_num + (direction_bin_weight * cosine).sum()
            active_bins = active_bins + directional_mask.float().sum()
            total_bins = total_bins + directional_mask.numel()

            if self.ratio_weight > 0:
                # Floor relative to the batch-item energy scale keeps the log
                # ratio bounded on near-silent bins without a dataset-specific
                # absolute constant.
                ratio_floor = (self.ratio_floor_ratio * mean_energy).clamp_min(1e-12)
                target_ratio = torch.log((target_dir + ratio_floor) / (target_omni + ratio_floor))
                pred_ratio = torch.log((pred_dir + ratio_floor) / (pred_omni + ratio_floor))
                ratio_bin_weight = energy_mask.float() * energy_weight
                ratio_num = ratio_num + (ratio_bin_weight * (pred_ratio - target_ratio).abs()).sum()
                ratio_den = ratio_den + ratio_bin_weight.sum()

        direction_loss = direction_num / direction_den.clamp_min(self.eps)
        ratio_loss = ratio_num / ratio_den.clamp_min(self.eps)

        self.last_direction_loss = direction_loss.detach()
        self.last_ratio_loss = ratio_loss.detach()
        self.last_active_fraction = (active_bins / total_bins.clamp_min(1.0)).detach()
        self.last_mean_cos = (cos_num / direction_den.clamp_min(self.eps)).detach()

        return self.direction_weight * direction_loss + self.ratio_weight * ratio_loss
