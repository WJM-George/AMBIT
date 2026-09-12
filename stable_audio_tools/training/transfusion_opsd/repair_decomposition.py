"""Geometry of legal condition responses at one frozen execution state.

This is a controllability diagnostic, not a teacher acceptance rule. A span or
interpolated response need not be an available action, and latent distance does
not certify decoded quality or a finite student update.
"""
from __future__ import annotations

import math

import torch

from .objectives import valid_mask


@torch.no_grad()
def repair_response_geometry(clean_predictions, repair, mask, legal, *, baseline=0,
        relative_eigenvalue_cutoff=1e-8):
    """Compare [A,B,C,T] predictions to a [B,C,T] desired repair from baseline.

    All predictions must use the same state, time and frozen executor. The
    caller verifies that provenance; tensors alone cannot establish it.
    """
    if clean_predictions.ndim != 4 or repair.shape != clean_predictions.shape[1:]:
        raise ValueError('expected predictions [A,B,C,T] and repair [B,C,T]')
    actions = clean_predictions.shape[0]
    if legal.dtype != torch.bool or legal.shape != (actions,):
        raise ValueError('one boolean legality value is required per action')
    if not 0 <= baseline < actions or not bool(legal[baseline]):
        raise ValueError('baseline must be a legal action')
    if not math.isfinite(relative_eigenvalue_cutoff) or not 0 < relative_eigenvalue_cutoff < 1:
        raise ValueError('declare a relative eigenvalue cutoff in (0,1)')
    if not torch.isfinite(clean_predictions).all() or not torch.isfinite(repair).all():
        raise ValueError('non-finite clean prediction or repair')
    full = valid_mask(repair, mask)
    r = repair.detach().double().masked_fill(~full, 0).flatten()
    d = (clean_predictions.double() - clean_predictions[baseline].double()).masked_fill(
        ~full.unsqueeze(0), 0).flatten(1)
    rr = r.square().sum()
    indices = [i for i in range(actions) if bool(legal[i]) and i != baseline]
    result = dict(repair_norm=float(rr.sqrt()), baseline_action=baseline,
        nonzero_repair=bool(rr > 1e-24), legal_actions=legal.cpu().tolist(),
        relative_eigenvalue_cutoff=relative_eigenvalue_cutoff,
        interpolation_is_available_action=False, span_is_available_action=False,
        decoded_quality_verified=False, actions=[])
    if rr <= 1e-24:
        result.update(best_discrete_action=baseline, best_discrete_explained_fraction=None,
            span_explained_fraction=None, effective_response_rank=None,
            interpretation='No nonzero local repair direction; control is not assessed.')
        return result
    for i in range(actions):
        if not bool(legal[i]):
            result['actions'].append(None)
            continue
        dd, dot = d[i].square().sum(), d[i] @ r
        alpha = (dot / dd).clamp(0, 1) if dd > 1e-24 else dd.new_zeros(())
        result['actions'].append(dict(action=i, response_norm=float(dd.sqrt()),
            response_norm_over_repair=float((dd / rr).sqrt()),
            cosine=float(dot / (dd * rr).sqrt()) if dd > 1e-24 else None,
            discrete_explained_fraction=float((2 * dot - dd) / rr),
            residual_norm_over_repair=float(((rr - 2 * dot + dd).clamp_min(0) / rr).sqrt()),
            optimistic_segment_alpha=float(alpha),
            optimistic_segment_explained_fraction=float((2 * alpha * dot - alpha.square() * dd) / rr)))
    best = max((x for x in result['actions'] if x is not None),
        key=lambda x: (x['discrete_explained_fraction'], x['action'] == baseline, -x['action']))
    result.update(best_discrete_action=best['action'],
        best_discrete_explained_fraction=best['discrete_explained_fraction'])
    if indices:
        directions = d[indices]
        gram = directions @ directions.T
        b = directions @ r
        eigenvalues, vectors = torch.linalg.eigh(gram)
        keep = eigenvalues > max(float(eigenvalues[-1]) * relative_eigenvalue_cutoff, 1e-24)
        coordinates = vectors.T @ b
        energy = (coordinates[keep].square() / eigenvalues[keep]).sum()
        result.update(span_explained_fraction=float((energy / rr).clamp(0, 1)),
            effective_response_rank=int(keep.sum()), response_gram=gram.cpu().tolist(),
            repair_response_dot=b.cpu().tolist(), gram_eigenvalues=eigenvalues.cpu().tolist(),
            span_action_indices=indices)
    else:
        result.update(span_explained_fraction=0., effective_response_rank=0,
            response_gram=[], repair_response_dot=[], gram_eigenvalues=[], span_action_indices=[])
    return result
