"""Actual-output content checks using our frozen FOA VAE-latent CLAP.

The whole FOA waveform is encoded by the trained native frontend. Only the
independent speech recognizer uses the established event-relative W view.
There is no LAION load, score, or default borrowed semantic threshold.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import torch

from .event_asr_bounds import BoundedASREventContentObserver, compare_transcript_bounds
from .event_content_views import ContentViewConfig, event_relative_content_view
from .native_latent_clap import FrozenNativeLatentCLAP, requested_semantic_text


class NativeCLAPEventContentObserver(BoundedASREventContentObserver):
    """Reuse the verified ASR procedure with an explicit native CLAP frontend."""

    def __init__(self, *, observer: FrozenNativeLatentCLAP, text_encoder, vae, whisper,
                 checkpoint_identity, posterior_seeds, config=ContentViewConfig()):
        if not isinstance(observer, FrozenNativeLatentCLAP):
            raise TypeError('Use our native FOA-latent CLAP')
        seeds = tuple(posterior_seeds)
        if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or s < 0 for s in seeds):
            raise ValueError('Declare distinct nonnegative posterior seeds')
        if not checkpoint_identity.get('sha256') or not checkpoint_identity.get('path'):
            raise ValueError('Bind the exact frozen native CLAP checkpoint')
        for model in (observer, text_encoder, vae):
            if model.training or any(p.requires_grad for p in model.parameters()):
                raise ValueError('Native content evaluation requires frozen models')
        self.observer, self.text_encoder, self.vae = observer, text_encoder, vae
        self.device = next(observer.parameters()).device
        self.scorer = SimpleNamespace(whisper=whisper)
        self.config = config
        self.posterior_seeds = seeds
        self.text_cache = {}
        # Do not call the historical PCM16 observer constructor or relabel its
        # contract. The inherited methods used below concern ASR/presence only.
        self.receipt = dict(contract='native_foa_clap_content_observer_v1',
            checkpoint=dict(checkpoint_identity), posterior_seeds=list(seeds),
            semantic_view='Full WYZX/SN3D FOA -> native FP32 VAE -> per-row posterior -> FP16 latent -> frozen semantic head.',
            text_view='Native semantic description populated only from requested content; no hidden geometry/time fields.',
            asr_view='Original3 no-prompt recognitions on the canonical event-relative W channel.',
            asr_placements=list(self.asr_placements),
            scope='Single foreground source; CLAP does not certify exact words or spatial geometry. No LAION score or implicit semantic threshold.')

    def _require_audio_autoencoder(self):
        return self.vae

    @torch.no_grad()
    def measure(self, waveform, request, requirements):
        if len(requirements['sources']) != 1 or requirements.get('relations'):
            raise ValueError('Declare an observable single foreground source')
        if waveform.ndim != 3 or waveform.shape[:2] != (1,4) or not torch.isfinite(waveform).all():
            raise ValueError('Actual full FOA must be finite [1,4,samples] at44.1kHz')
        from stable_audio_tools.models.sceneplan_transfusion_editing_pipeline import ScenePlanTransfusionEditingDiTPipeline
        text = requested_semantic_text(requirements)
        if text not in self.text_cache:
            hidden = self.text_encoder([text], self.device)
            self.text_cache[text] = self.observer.text_features(hidden, hidden)['semantic']
        text_feature = self.text_cache[text]
        observations = []
        for seed in self.posterior_seeds:
            latent, mask = ScenePlanTransfusionEditingDiTPipeline.encode_source_foa(self, waveform,
                model_num_samples=[waveform.shape[-1]], vae_seeds=[seed])
            features = self.observer(latent, mask)
            score = float((features['semantic']*text_feature).sum(-1)[0])
            observations.append(dict(posterior_seed=seed, score=score))
        canonical, content_view = event_relative_content_view(waveform, config=self.config)
        return dict(contract=self.receipt['contract'], checkpoint=self.receipt['checkpoint'],
            native_semantic=dict(text=text, observations=observations,
                mean=sum(x['score'] for x in observations)/len(observations)),
            content_view=content_view, presence=self.presence_evidence(waveform,request,requirements),
            asr=self.asr_evidence(canonical,requirements))


def compare_native_content_evidence(after, before, *, maximum_semantic_drop: float):
    """Require a new declared native budget; missing evidence stays unresolved."""
    if not math.isfinite(maximum_semantic_drop) or maximum_semantic_drop < 0:
        raise ValueError('Declare a finite nonnegative native semantic drop budget')
    if (after.get('contract') != 'native_foa_clap_content_observer_v1'
            or before.get('contract') != after['contract'] or before['checkpoint'] != after['checkpoint']):
        raise ValueError('Compare the same native CLAP identity and observation contract')
    a,b = after['native_semantic'],before['native_semantic']
    seeds = [r['posterior_seed'] for r in a['observations']]
    if not seeds or seeds != [r['posterior_seed'] for r in b['observations']] or a['text'] != b['text']:
        raise ValueError('Pair the same posterior noise and requested semantic text')
    deltas = [x['score']-y['score'] for x,y in zip(a['observations'],b['observations'])]
    if not all(math.isfinite(x) for x in deltas):
        raise ValueError('Nonfinite native semantic observations')
    failures,uncertain = [],[]
    if min(deltas) < -maximum_semantic_drop-1e-9:
        failures.append('native_clap_semantic')
    if after['presence']['source_presence_failure'] > 0:
        failures.append('source_presence')
    if not after['content_view']['admissible']:
        failures.append('content_view_energy')
    if after['asr']['required'] != before['asr']['required']:
        raise ValueError('Requested speech scope changed')
    asr = None
    if after['asr']['required']:
        asr = compare_transcript_bounds(after['asr']['observed_error_bounds'],before['asr']['observed_error_bounds'])
        if asr['status'] == 'observed_regression': failures.append('asr_observed_bounds')
        elif asr['status'] == 'uncertain': uncertain.append('asr_observed_bounds')
    return dict(passed=not failures and not uncertain,failures=failures,uncertain=uncertain,
        native_semantic_deltas=deltas,posterior_seeds=seeds,
        native_semantic_mean_delta=sum(deltas)/len(deltas),asr_bounds_comparison=asr)
