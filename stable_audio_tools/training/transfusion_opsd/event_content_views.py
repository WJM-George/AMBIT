"""Experimental content views independent of request timing and AR plans.

Only the content observer sees these views. Generation and spatial/time scoring
retain the original audio. Historical EventAudioProtection remains unchanged.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ContentViewConfig:
    relative_amplitude_floor: float = 1e-4
    context_samples: int = 4410
    max_discarded_w_energy_fraction: float = 1e-6
    silence_peak: float = 1e-8

    def __post_init__(self):
        if not (math.isfinite(self.relative_amplitude_floor) and 0 < self.relative_amplitude_floor < 1):
            raise ValueError('Declare a finite relative amplitude floor in (0, 1).')
        if type(self.context_samples) is not int or self.context_samples < 0:
            raise ValueError('Content context must be a nonnegative sample count.')
        if not (math.isfinite(self.max_discarded_w_energy_fraction) and 0 <= self.max_discarded_w_energy_fraction < 1):
            raise ValueError('Declare a finite bound on removed W energy.')
        if not (math.isfinite(self.silence_peak) and self.silence_peak > 0):
            raise ValueError('Declare a positive finite silence threshold.')


def event_relative_content_view(waveform, *, config=ContentViewConfig()):
    """Anchor a single foreground event; keep interior gaps and quiet words.

    Locate the first/last sample above a peak-relative W amplitude threshold.
    Preserve all audio between those points, plus 100 ms context at each end.
    Missing exterior context is zero padded. Veto trimming when the actual
    discarded W energy exceeds the declared fraction, even if all discarded
    samples are individually quiet. The support selection is detached; retained
    samples preserve their autograd path for a future differentiable observer.

    This is a narrowly scoped content view, not a source separator, perceptual
    equivalence proof, or permission to ignore quiet sources in a mixture.
    """
    if (waveform.ndim != 3 or waveform.shape[:2] != (1, 4) or waveform.shape[-1] == 0
            or not waveform.is_floating_point() or not torch.isfinite(waveform).all()):
        raise ValueError('Expected finite floating WYZX audio [1, 4, samples].')
    with torch.no_grad():
        w = waveform[0, 0].detach().double()
        peak = float(w.abs().max())
        if peak <= config.silence_peak:
            # Silence remains a missing-content negative, never a good event.
            view = waveform.new_zeros(1, 4, 44100)
            return view, dict(config=asdict(config), present=False, admissible=True,
                input_samples=waveform.shape[-1], output_samples=44100, start_sample=None,
                end_sample=None, discarded_w_energy_fraction=0., silence_peak=peak)
        support = (w.abs() >= peak * config.relative_amplitude_floor).nonzero().flatten()
        first, end = int(support[0]), int(support[-1]) + 1
        wanted_start, wanted_end = first - config.context_samples, end + config.context_samples
        start, stop = max(0, wanted_start), min(waveform.shape[-1], wanted_end)
        total = w.square().sum()
        discarded = w[:start].square().sum() + w[stop:].square().sum()
        fraction = float(discarded / total)
        admissible = fraction <= config.max_discarded_w_energy_fraction
    if not admissible:
        return waveform, dict(config=asdict(config), present=True, admissible=False,
            reason='Exterior quiet energy exceeds the declared trimming budget.',
            input_samples=waveform.shape[-1], output_samples=waveform.shape[-1],
            start_sample=0, end_sample=waveform.shape[-1], proposed_start_sample=start,
            proposed_end_sample=stop, discarded_w_energy_fraction=fraction)
    view = F.pad(waveform[..., start:stop], (max(0, -wanted_start), max(0, wanted_end - waveform.shape[-1])))
    return view, dict(config=asdict(config), present=True, admissible=True,
        input_samples=waveform.shape[-1], output_samples=view.shape[-1], start_sample=start,
        end_sample=stop, support_start_sample=first, support_end_sample=end,
        discarded_w_energy_fraction=fraction)


class EventRelativeAudioProtection:
    """Frozen CLAP on fixed event placements; ASR on one canonical event view.

    No best-ASR selection or minimum-over-views content score. All CLAP views
    have fixed equal weight. The base scorer and its receipt are preserved.
    """
    def __init__(self, scorer, *, config=ContentViewConfig(), placements=(0,)):
        if (not placements or placements[0] != 0 or len(set(placements)) != len(placements)
                or any(type(x) is not int or x < 0 for x in placements)):
            raise ValueError('Declare unique nonnegative integer placements beginning with zero.')
        self.scorer, self.config, self.placements = scorer, config, tuple(placements)
        self.receipt = dict(contract='event_relative_content_observer_v1_experimental',
            base_observer=scorer.receipt, config=asdict(config), placements=list(placements),
            asr_view='One canonical event-relative view, no prompt, no best-transcript selection.',
            scope='Single foreground source content only; original waveform required for time and spatial scoring.')

    @torch.no_grad()
    def measure(self, waveform, requirements):
        if len(requirements['sources']) != 1 or requirements.get('relations'):
            raise ValueError('Event-relative content view requires a single foreground source.')
        view, receipt = event_relative_content_view(waveform, config=self.config)
        result = self.scorer.measure(view, requirements)
        scores = [result['clap_by_requested_source']]
        windows = [result['clap_audio_windows']]
        for placement in self.placements[1:]:
            extra, count, _ = self.scorer._clap_source_scores(F.pad(view, (placement, 0)), requirements)
            scores.append({key: float(value) for key, value in extra.items()})
            windows.append(count)
        keys = scores[0].keys()
        averaged = {key: sum(row[key] for row in scores) / len(scores) for key in keys}
        return dict(result, clap_by_requested_source=averaged,
            clap_source_mean=sum(averaged.values()) / len(averaged), clap_audio_windows=sum(windows),
            content_view=receipt, clap_by_placement=scores, placements=list(self.placements))
