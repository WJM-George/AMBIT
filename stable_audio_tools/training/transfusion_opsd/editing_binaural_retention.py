"""Frozen FOA-to-binaural rendering and smooth paired spectral/spatial losses.

These are training surrogates, not renamed LSD/GCC/CRW evaluation scores.
Use only identified paired targets; never equate a removal output to its input.
"""
from pathlib import Path
import sys

import torch
from torch import nn


class FrozenKemarRenderer(nn.Module):
    def __init__(self):
        super().__init__()
        from .editing_nine_metrics import BENCH
        if not Path(BENCH).exists():
            raise FileNotFoundError(f'Editing benchmark suite is not installed: {BENCH}')
        sys.path.insert(0, str(BENCH))
        from benchmark_audio_v1 import KemarFoaDecoder
        decoder = KemarFoaDecoder()
        self.register_buffer('filters', torch.from_numpy(decoder.filters.copy()).float())
        self.identity = decoder.description()

    def forward(self, foa):
        if foa.ndim != 2 or foa.shape[0] != 4:
            raise ValueError('Expected channel-first four-channel FOA.')
        n = foa.shape[-1]
        nfft = 1 << (n + self.filters.shape[-1] - 2).bit_length()
        spectrum = torch.fft.rfft(foa.float(), n=nfft)
        filters = torch.fft.rfft(self.filters.float(), n=nfft)
        return torch.fft.irfft(torch.einsum('cf,ecf->ef', spectrum, filters), n=nfft)[..., :n]


def binaural_features(stereo):
    if stereo.ndim != 2 or stereo.shape[0] != 2:
        raise ValueError('Expected left/right stereo.')
    features = []
    for fft in (512, 2048):
        spec = torch.stft(stereo.float(), n_fft=fft, hop_length=fft // 4,
            window=torch.hann_window(fft, device=stereo.device), center=False, return_complex=True)
        power = spec.abs().square()
        cross = spec[0] * spec[1].conj()
        features.append((power, cross))
    return features


def binaural_distances(predicted, target):
    spectral, phase, level = [], [], []
    for (pp, pc), (tp, tc) in zip(predicted, target):
        floor = (tp.mean().detach() * 1e-5).clamp_min(1e-10)
        spectral.append((pp.clamp_min(floor).log() - tp.clamp_min(floor).log()).abs().mean())
        # A fixed target energy mask prevents suppressing the prediction from
        # hiding its spatial error. Compare cross spectra before argmax.
        weight = tp.sum(0).detach()
        weight = weight / weight.sum().clamp_min(1e-10)
        # Clamp before sqrt: sqrt(0) has an infinite derivative, even when a
        # later clamp masks its output. Silent bins must have finite gradients.
        pphase = pc / (pp[0] * pp[1]).clamp_min(floor.square()).sqrt()
        tphase = tc / (tp[0] * tp[1]).clamp_min(floor.square()).sqrt()
        phase.append((weight * (pphase - tphase).abs().square()).sum())
        pild = (pp[0].clamp_min(floor).log() - pp[1].clamp_min(floor).log())
        tild = (tp[0].clamp_min(floor).log() - tp[1].clamp_min(floor).log())
        level.append((weight * torch.nn.functional.smooth_l1_loss(pild, tild, reduction='none')).sum())
    return dict(log_spectral=torch.stack(spectral).mean(),
                cross_phase=torch.stack(phase).mean(), interaural_level=torch.stack(level).mean())


def select_operation_rows(metadata_batches, step, rank, count=2):
    """Rotate operation priorities within the already sampled paired batch."""
    groups = {}
    for batch_index, metadata in enumerate(metadata_batches):
        for row_index, row in enumerate(metadata):
            groups.setdefault(row['operation'], []).append((batch_index, row_index))
    operations = sorted(groups)
    if not operations:
        return []
    start = (step + rank) % len(operations)
    operations = operations[start:] + operations[:start]
    selected = []
    for operation in operations:
        group = groups[operation]
        selected.append(group[(step + rank) % len(group)])
        if len(selected) == count:
            return selected
    remaining = [row for group in groups.values() for row in group if row not in selected]
    return selected + remaining[:max(0, count - len(selected))]
