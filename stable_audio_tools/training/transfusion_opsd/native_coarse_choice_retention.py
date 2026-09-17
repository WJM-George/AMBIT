"""Retain probability mass on compatible native choices, without numeric GT.

This is a common capability-retention loss, not an execution-improvement
teacher. Native choices remain discrete at inference. A reference-centered
cone permits small changes; trusted source-input evidence can additionally
permit an improvement already authorized by the source-direction guard.
"""
from collections import Counter, defaultdict
import math

import torch

from .native_token_alignment import validate_native_plan_tokens


def _distance(a, b):
    return abs((a - b + 180.) % 360. - 180.)


def native_azimuth_cone_targets(codec, native_tokens, native_plan, allowed_fn, *,
                                 radius_deg, source_evidence=None, source_tolerance_deg=32.5):
    """Build stopped legal sets on the actually visited structured prefixes.

    The caller supplies a current native plan, never an unspecified hidden GT.
    Coarse retention alone does not certify its correctness. Source evidence
    is only applicable to one uniquely bound static source and must have been
    independently qualified before this call. Other fields are not labeled.
    """
    if not math.isfinite(radius_deg) or not 0 < radius_deg < 180:
        raise ValueError('Use a finite, nontrivial coarse azimuth radius.')
    if not math.isfinite(source_tolerance_deg) or not 0 < source_tolerance_deg <= 180:
        raise ValueError('Invalid source direction tolerance.')
    ids = list(map(int, native_tokens))
    validate_native_plan_tokens(codec, ids, native_plan)
    from ...data.model_sceneplan_codec import SOURCE_SLOT_TOKENS
    slots = {codec._tid(v): f'source_{i}' for i, v in enumerate(SOURCE_SLOT_TOKENS)}
    sources = {s['source_id']: s for s in native_plan['sources']}
    kinds = Counter(s['kind'] for s in native_plan['sources'])
    angles = {token: float(i - 180) for i, token in enumerate(codec.azimuth_ids)}
    evidence = source_evidence or {}
    observed = []
    if evidence.get('available'):
        windows = evidence['windows']
        if (evidence['source_count'] != 1 or len(windows) < 3 or
                not all(w['observable'] and math.isfinite(w['azimuth_deg']) for w in windows)):
            raise ValueError('Available source evidence requires a single qualified source.')
        observed = [w['azimuth_deg'] for w in windows]
    text_begin, text_end = codec._tid('<text_begin>'), codec._tid('<text_end>')
    source_begin, source_end = codec._tid('<source_begin>'), codec._tid('<source_end>')
    azimuth = codec._tid('<azimuth_bin>')
    source, inside_text = None, False
    occurrences = Counter()
    targets = []
    for j, token in enumerate(ids):
        if token == text_begin:
            inside_text = True
            continue
        if token == text_end:
            inside_text = False
            continue
        if inside_text:
            continue
        if token == source_begin:
            source = slots[ids[j + 1]]
        elif token == source_end:
            source = None
        elif token == azimuth:
            if source not in sources or ids[j + 1] not in angles:
                raise ValueError('Azimuth lacks a valid source-bound atomic choice.')
            pos, current = j + 1, angles[ids[j + 1]]
            allowed = sorted(allowed_fn(ids[:pos]))
            if (len(allowed) != len(set(allowed)) or not set(allowed) <= angles.keys()
                    or ids[pos] not in allowed):
                raise ValueError('Invalid native azimuth grammar support.')
            s = sources[source]
            use_source = bool(observed and evidence.get('source_kind') == s['kind']
                              and kinds[s['kind']] == 1 and s['trajectory']['type'] == 'static')
            nearby = [i for i in allowed if _distance(angles[i], current) <= radius_deg + 1e-6]
            extra = [i for i in allowed if use_source and
                     all(_distance(angles[i], a) <= min(source_tolerance_deg, _distance(current, a)) + 1e-6
                         for a in observed)]
            acceptable = sorted(set(nearby) | set(extra))
            assert ids[pos] in acceptable
            targets.append(dict(position=pos, observed_token_id=ids[pos], allowed_ids=allowed,
                                acceptable_ids=acceptable, source_id=source, kind=s['kind'],
                                field=source + '/azimuth', waypoint_index=occurrences[source],
                                current_azimuth_deg=current, radius_deg=radius_deg,
                                source_evidence_applied=use_source,
                                source_supported_extra_ids=sorted(set(extra) - set(nearby))))
            occurrences[source] += 1
    return targets


def field_balanced_native_set_ce(logits, targets):
    """Negative log of compatible legal mass, averaged within/across fields.

    If probability lies outside the compatible set this supplies a gradient
    already at the reference model, unlike a KL to that same distribution.
    It does not prescribe how mass is divided among acceptable choices.
    """
    if logits.ndim != 2 or not torch.isfinite(logits).all():
        raise ValueError('Expected finite [prefix positions, vocabulary] logits.')
    groups = defaultdict(list)
    for t in targets:
        pos, allowed, accepted = t['position'] - 1, t['allowed_ids'], t['acceptable_ids']
        if (not 0 <= pos < logits.shape[0] or not allowed or not accepted or
                len(allowed) != len(set(allowed)) or len(accepted) != len(set(accepted)) or
                not set(accepted) <= set(allowed) or min(allowed) < 0 or max(allowed) >= logits.shape[-1]):
            raise ValueError('Invalid legal support or compatible target set.')
        row = logits[pos].float()
        groups[t['field']].append(row[allowed].logsumexp(0) - row[accepted].logsumexp(0))
    if not groups:
        return logits.sum() * 0
    return torch.stack([torch.stack(v).mean() for v in groups.values()]).mean()
