"""Scoped presence, semantic and reference-blind ASR evidence for generation.

This experimental observer does not change generated audio or replace the
original request-relative spatial/time observer. Consensus is confidence
evidence, not a proof of transcript correctness.
"""
from collections import Counter
import re

import torch
import torch.nn.functional as F
import torchaudio

from .event_activity_compass_reward import RequestActivityCompassReward
from .event_content_views import ContentViewConfig, event_relative_content_view
from .event_rewards import EventRequestReward


def transcript_consensus(texts):
    """A strict majority of three normalized recognitions, without a reference.

    Keep word repetitions. Only case and non-word punctuation are normalized,
    using the same English token convention as the existing WER scorer.
    """
    if len(texts) != 3 or any(not isinstance(text, str) for text in texts):
        raise ValueError('Exactly three independently recognized strings are required.')
    words = [tuple(re.findall(r"[a-z0-9']+", text.lower())) for text in texts]
    counts = Counter(words)
    winner, count = counts.most_common(1)[0]
    agreed = count >= 2
    representative = words.index(winner) if agreed else None
    return dict(status='agreed' if agreed else 'ambiguous',
        normalized_text=' '.join(winner) if agreed else None,
        representative_index=representative, support_count=count,
        raw_texts=list(texts), votes=[dict(text=' '.join(key), count=value) for key, value in counts.items()])


def requested_transcript(requirements):
    values = [c['value'] for source in requirements['sources'] for c in source['constraints'] if c['op'] == 'transcript']
    if len(values) > 1:
        raise ValueError('Current content observer supports at most one formal speech source.')
    return values[0] if values else None


class ComposedEventContentObserver:
    """Existing W-activity presence, single-view PCM16 CLAP, three-view ASR."""
    asr_placements = (0, 4410, 8820)

    def __init__(self, scorer, *, config=ContentViewConfig()):
        if scorer.receipt['contract'] != 'event_clap_pcm16_observation_v1_experimental':
            raise ValueError('Bind the separately declared PCM16 CLAP observer.')
        self.scorer, self.config = scorer, config
        self.receipt = dict(contract='event_composed_content_evidence_v1_experimental',
            semantic_observer=scorer.receipt,
            semantic_view='Single canonical event-relative view; original .005 paired-drop guard is unchanged.',
            presence='Existing RequestActivityCompassReward source_presence_failure on original, untrimmed audio.',
            asr_placements=list(self.asr_placements),
            asr_selection='Three English, no-prompt/no-VAD/no-previous-text recognitions. The reference never enters majority selection of normalized word sequences; no majority means ambiguous.',
            scope='Single foreground source. Presence is not source identity, and consensus is not ground-truth certification.')

    @torch.no_grad()
    def asr_evidence(self, canonical, requirements):
        reference = requested_transcript(requirements)
        if reference is None:
            return dict(required=False, consensus=None, requested_transcript_error=None, decodes=[])
        from scripts.t2a.eval.score_sceneplan_dit_p10_speech import _transcribe, _error_rates
        decodes = []
        for placement in self.asr_placements:
            mono = F.pad(canonical[:, 0].float(), (placement, 0))
            mono = mono / mono.abs().max().clamp_min(1e-8) * (10 ** (-1 / 20))
            values = torchaudio.functional.resample(mono, 44100, 16000)[0].cpu().numpy()
            decodes.append(_transcribe(self.scorer.whisper, values))
        consensus = transcript_consensus([row['text'] for row in decodes])
        error = (_error_rates(consensus['normalized_text'], reference)
                 if consensus['status'] == 'agreed' else None)
        return dict(required=True, consensus=consensus, requested_transcript_error=error, decodes=decodes)

    @torch.no_grad()
    def presence_evidence(self, waveform, request, requirements):
        measured = RequestActivityCompassReward(EventRequestReward(request, requirements)).measure(waveform)
        return dict(source_presence_failure=measured['costs']['source_presence_failure'],
            active_frames=measured['active_frames'], duration_sec=measured['duration_sec'],
            profile=measured['profile'])

    @torch.no_grad()
    def measure(self, waveform, request, requirements):
        if len(requirements['sources']) != 1 or requirements.get('relations'):
            raise ValueError('Declare an observable single foreground source.')
        canonical, view = event_relative_content_view(waveform, config=self.config)
        scores, windows, _ = self.scorer._clap_source_scores(canonical, requirements)
        scores = {key: float(value) for key, value in scores.items()}
        return dict(content_view=view, presence=self.presence_evidence(waveform, request, requirements),
            clap_by_requested_source=scores, clap_source_mean=sum(scores.values()) / len(scores),
            clap_audio_windows=windows, asr=self.asr_evidence(canonical, requirements))


def compare_content_evidence(after, before, *, clap_drop=.005):
    """Missing/ambiguous ASR evidence never silently becomes zero WER."""
    failures, uncertain = [], []
    if after['presence']['source_presence_failure'] > 0:
        failures.append('source_presence')
    if not after['content_view']['admissible']:
        failures.append('content_view_energy')
    clap_delta = after['clap_source_mean'] - before['clap_source_mean']
    if clap_delta < -clap_drop - 1e-9:
        failures.append('clap_similarity')
    if after['asr']['required'] != before['asr']['required']:
        raise ValueError('Content comparison cannot change the requested transcript scope.')
    wer_delta = None
    if after['asr']['required']:
        a, b = after['asr']['requested_transcript_error'], before['asr']['requested_transcript_error']
        if a is None or b is None:
            uncertain.append('asr_consensus')
        else:
            wer_delta = a['wer'] - b['wer']
            if wer_delta > 1e-9:
                failures.append('asr_wer')
    return dict(passed=not failures and not uncertain, failures=failures, uncertain=uncertain,
                clap_delta=clap_delta, wer_delta=wer_delta)
