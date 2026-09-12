"""Cross-condition self-reference targets on genuine diffusion query states.

An improved candidate-conditioned prediction may still be worse than the
original plan's content. Both references therefore participate in target
qualification. This local target construction is not a learned-policy claim.
"""
from __future__ import annotations

from dataclasses import asdict
import math

import torch

from .objectives import DiffusionTargets, valid_mask
from .shared_step import project_shared_displacement


def protected_against_references(score, references, *, min_gain):
    if not references or not math.isfinite(min_gain) or min_gain <= 0:
        raise ValueError('reference repair requires references and a positive gain threshold')
    if any(score.costs.keys() != ref.costs.keys() for ref in references):
        raise ValueError('cross-condition references require identical protection metrics')
    return (score.utility > max(ref.utility for ref in references) + min_gain
        and all(score.costs[key] <= min(ref.costs[key] for ref in references)
            for key in score.costs))


def build_reference_repair(anchor, mask, *, objective, score_clean, score_suffix,
        reference_clean, reference_suffix, radius=.02, target_steps=2, min_gain=1e-6,
        protect_candidate=True):
    if (anchor.shape[0] != 1 or not math.isfinite(radius) or not 0 < radius <= 1
            or not isinstance(target_steps, int) or target_steps < 1):
        raise ValueError('one valid query and a bounded positive target-search budget are required')
    full = valid_mask(anchor, mask)
    base = anchor.detach().float().masked_fill(~full, 0)
    with torch.no_grad():
        own_clean, own_suffix = score_clean(base), score_suffix(base)
    clean_refs = [reference_clean, own_clean] if protect_candidate else [reference_clean]
    suffix_refs = [reference_suffix, own_suffix] if protect_candidate else [reference_suffix]
    budget = radius * base.norm()
    evidence = {'contract': 'cross_condition_reference_repair_v1', 'radius': radius,
        'target_steps': target_steps, 'protect_candidate_as_well': protect_candidate,
        'reference_clean': asdict(reference_clean), 'reference_suffix': asdict(reference_suffix),
        'candidate_clean': asdict(own_clean), 'candidate_suffix': asdict(own_suffix), 'trials': []}
    if not bool(torch.isfinite(base).all()) or not budget > 0:
        return None, {**evidence, 'qualified': False, 'reason': 'invalid_anchor'}
    value = base.clone()
    for _ in range(target_steps):
        with torch.enable_grad():
            leaf = value.detach().requires_grad_(True)
            scalar = objective(leaf)
            if scalar.ndim != 0 or not torch.isfinite(scalar):
                raise ValueError('reference repair objective must be finite and scalar')
            gradient, = torch.autograd.grad(scalar, leaf)
        gradient = gradient.masked_fill(~full, 0)
        if not torch.isfinite(gradient).all():
            raise ValueError('reference repair gradient must be finite')
        value = value.detach() + (budget / target_steps) * gradient / gradient.norm().clamp_min(1e-12)
        delta = value - base
        value = (base + delta * (budget / delta.norm().clamp_min(1e-12)).clamp(max=1)).detach()
    for alpha in (1., .5, .25):
        candidate = base + alpha * (value - base)
        with torch.no_grad():
            clean, suffix = score_clean(candidate), score_suffix(candidate)
        okay = (protected_against_references(clean, clean_refs, min_gain=min_gain)
            and protected_against_references(suffix, suffix_refs, min_gain=min_gain))
        evidence['trials'].append({'scale': alpha, 'clean': asdict(clean), 'suffix': asdict(suffix),
            'qualified': okay})
        if okay:
            return DiffusionTargets(base, candidate.detach(), mask.detach()), {**evidence, 'qualified': True}
    return None, {**evidence, 'qualified': False, 'reason': 'cross_condition_protection_failed'}


def normalized_spectral_reference(waveform, reference, *, fft_sizes=(1024, 4096)):
    """W-channel spectral content reference; phase/absolute gain not targets.

The reference is the model's own paired generation, not a dataset waveform.
This proxy is used only to propose a latent target; CLAP/ASR still verify it.
"""
    if waveform.shape != reference.shape or waveform.ndim != 3 or waveform.shape[:2] != (1, 4):
        raise ValueError('paired audio reference needs equal-length single WYZX waveforms')
    signal, target = waveform[:, 0].float(), reference[:, 0].detach().float().to(waveform.device)
    signal = signal / signal.square().mean(-1, keepdim=True).clamp_min(1e-8).sqrt()
    target = target / target.square().mean(-1, keepdim=True).clamp_min(1e-8).sqrt()
    terms = []
    for size in fft_sizes:
        if size > signal.shape[-1] or size < 4:
            raise ValueError('spectral window must fit inside the actual waveform')
        window = torch.hann_window(size, device=waveform.device)
        def spectrum(value):
            return torch.stft(value, size, hop_length=size // 4, window=window,
                center=False, return_complex=True).abs()
        a, b = spectrum(signal), spectrum(target)
        terms.append((torch.log1p(a) - torch.log1p(b)).square().mean())
    return torch.stack(terms).mean()


def reference_repair_objective(latent, *, decode, spatial_reward, content_cost,
        reference_waveform, reference_weight=1., diagnostics=None):
    """Propose reference consistency while protecting spatial/CLAP gradients.

Projection is local and homogeneous. Decoded clean and exact native suffix
comparisons decide qualification; no finite-output guarantee is inferred.
"""
    if not math.isfinite(reference_weight) or reference_weight < 0:
        raise ValueError('reference weight must be finite and nonnegative')
    audio = decode(latent)
    utility, content = spatial_reward(audio), content_cost(audio)
    reference_cost = normalized_spectral_reference(audio, reference_waveform)
    gu, = torch.autograd.grad(utility, latent, retain_graph=True)
    gc, = torch.autograd.grad(content, latent, retain_graph=True)
    gr, = torch.autograd.grad(reference_cost, latent)
    normalize = lambda value: value / value.norm().clamp_min(1e-12)
    proposed = normalize(gu) - normalize(gc) - reference_weight * normalize(gr)
    direction, projection = project_shared_displacement([proposed.detach()],
        [-gu.detach()], [gc.detach()], [torch.ones_like(latent)])
    if diagnostics is not None:
        diagnostics.append({'spatial': float(utility.detach()), 'content_cost': float(content.detach()),
            'spectral_reference_cost': float(reference_cost.detach()), 'reference_weight': reference_weight,
            'gradient_norms': [float(value.norm()) for value in (gu, gc, gr)], 'projection': projection})
    return (latent * direction[0].detach()).sum()
