"""Native FOA CLAP observations and stopped-teacher latent distillation.

The frozen CLAP44 tower reads the student's clean [B,64,T] VAE latent.
Its ordinary FP16 input cast is the trained model's boundary. Audio features
remain differentiable; the no-grad AR source-feature convenience API is not
used here. Exact speech words and request geometry need separate validation.
"""
from __future__ import annotations

import math
from typing import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from stable_audio_tools.data.sceneplan_transfusion_editing_clap44 import _content_text
from stable_audio_tools.models.sceneplan_transfusion_editing_clap44 import EditingCLAP44


def requested_semantic_text(requirements: Mapping) -> str:
    """Use native content wording, populated only from the requested sources.

    Spatial coordinates, times, gain and room are deliberately absent from
    the semantic head. Missing speech transcripts are not invented.
    """
    sources = requirements.get('sources', [])
    if not sources:
        raise ValueError('Native CLAP needs explicit requested sources')
    texts = []
    for source in sources:
        core = source.get('core')
        if not isinstance(core, str) or not core.strip():
            raise ValueError('Requested source content is missing')
        kind = source.get('kind')
        if kind == 'speech':
            words = [c['value'] for c in source.get('constraints', []) if c.get('op') == 'transcript']
            if len(words) != 1 or not isinstance(words[0], str) or not words[0].strip():
                raise ValueError('This semantic text adapter requires one explicit speech transcript')
            fields = dict(kind=kind, speaker_description=core, transcript=words[0])
        elif kind in {'sound', 'music'}:
            fields = dict(kind=kind, description=core)
        else:
            raise ValueError(f'Unsupported requested source kind: {kind}')
        texts.append(_content_text(fields) + '.')
    return f'{len(sources)} audible sources. ' + ' '.join(sorted(texts))


class FrozenNativeLatentCLAP(nn.Module):
    """Freeze observer parameters while retaining the student latent graph."""

    def __init__(self, encoder: EditingCLAP44):
        super().__init__()
        if not isinstance(encoder, EditingCLAP44):
            raise TypeError('Expected the native trained FOA CLAP44 encoder')
        self.encoder = encoder.eval().requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True):
        super().train(False)
        self.encoder.eval()
        return self

    def forward(self, clean_latent: Tensor, mask: Tensor) -> dict[str, Tensor]:
        if self.encoder.training or any(p.requires_grad for p in self.encoder.parameters()):
            raise RuntimeError('The native CLAP observer must remain frozen')
        return self.encoder.encode_audio(clean_latent, mask)

    def text_features(self, semantic_features: Tensor, scene_features: Tensor) -> dict[str, Tensor]:
        # Requests/teacher text are training targets, not an audio input.
        with torch.no_grad():
            return self.encoder.encode_text_features(semantic_features, scene_features)


def native_feature_distillation(student: Mapping[str, Tensor], teacher: Mapping[str, Tensor],
                                *, semantic_weight: float, scene_weight: float) -> Tensor:
    """Per-example cosine distillation; teacher features never receive gradients.

    A scene teacher should come from an independently validated execution
    repair. Preserving the entire scene embedding of a spatially wrong audio
    would oppose the requested repair, so both weights are explicit.
    """
    weights = dict(semantic=semantic_weight, scene=scene_weight)
    if any(not math.isfinite(w) or w < 0 for w in weights.values()) or sum(weights.values()) <= 0:
        raise ValueError('Provide finite nonnegative weights with a positive total')
    values = []
    batch = None
    for head, weight in weights.items():
        if weight == 0:
            continue
        left, right = student[head], teacher[head].detach()
        if left.ndim != 2 or left.shape != right.shape or left.device != right.device:
            raise ValueError('Each native CLAP teacher must match its student head')
        if batch is not None and left.shape[0] != batch:
            raise ValueError('Native CLAP heads refer to different batches')
        batch = left.shape[0]
        if not bool(torch.isfinite(left).all() and torch.isfinite(right).all()):
            raise ValueError('Nonfinite native CLAP features')
        if bool((left.float().norm(dim=-1) < 1e-8).any() or (right.float().norm(dim=-1) < 1e-8).any()):
            raise ValueError('Zero native CLAP feature')
        values.append(weight * (1 - F.cosine_similarity(left.float(), right.float(), dim=-1).clamp(-1, 1)))
    return torch.stack(values).sum(0)


def native_text_scores(audio: Mapping[str, Tensor], text: Mapping[str, Tensor]) -> dict[str, Tensor]:
    """Separate cosine matrices; never compare their scale with LAION scores."""
    result = {}
    for head in ('semantic', 'scene'):
        left, right = audio[head], text[head].detach()
        if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1]:
            raise ValueError('Native audio/text head geometry differs')
        result[head] = F.normalize(left.float(), dim=-1) @ F.normalize(right.float(), dim=-1).T
    return result
