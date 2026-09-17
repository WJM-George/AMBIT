"""Instruction-owned constraints at student prefixes, separate from OPSD rewards.

Bindings use visible request/plan text only. Ambiguity is explicit; there is
no target-plan access, audio source separation or inferred hidden geometry.
"""
from collections import defaultdict
import math
import re

import torch

from .native_token_alignment import validate_native_plan_tokens

_STOP = set('a an the is are of and or with in on at to from its it this that sound sounds '
            'music voice described as plays playing has have featuring features heard'.split())


def words(text):
    return set(re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", text.lower())) - _STOP


def parse_edit_request(request, operation):
    from .editing_spatial_retention import request_facts
    result = request_facts(request, 'event_addition' if operation == 'event_removal' else operation)
    if result is not None:
        result.update(operation=operation, removal=operation == 'event_removal')
    return result


def bind_edit_target(plan, facts, *, minimum_overlap=.60, minimum_margin=.15):
    if facts is None:
        return dict(available=False, reason='unparsed_request')
    field = 'transcript' if facts['kind'] == 'speech' else 'description'
    wanted = facts.get('fields', {}).get(field, '')
    target_words = words(wanted)
    candidates = []
    for source in plan['sources']:
        # Identify the content before checking its predicted category. A
        # violin mislabeled as sound must still receive its music/angle labels.
        text = source.get(field) or source.get('description') or source.get('transcript', '')
        observed = words(text)
        intersection = len(target_words & observed)
        score = intersection / max(1, min(len(target_words), len(observed)))
        exact = ' '.join(wanted.split()).casefold() == ' '.join(text.split()).casefold()
        candidates.append(dict(source_id=source['source_id'], score=score, exact=exact,
                               shared_words=intersection, text=text, predicted_kind=source['kind']))
    candidates.sort(key=lambda x: (x['exact'], x['score']), reverse=True)
    if not candidates:
        return dict(available=False, reason='no_matching_content', candidates=[])
    if 'fields' not in facts and len(candidates) == 1 and candidates[0]['predicted_kind'] == facts['kind']:
        # Compatibility for callers supplying an already isolated source.
        return dict(available=True, source_id=candidates[0]['source_id'],
                    evidence='caller_isolated_kind', candidates=candidates,
                    acoustic_identity_certified=False)
    best = candidates[0]
    second = candidates[1] if len(candidates) > 1 else None
    # Exact or distinctive phrase agreement supports a source identity. Being
    # the only source of a broad class is insufficient if the text disagrees.
    reliable = best['exact'] or (best['score'] >= minimum_overlap and best['shared_words'] >= 3)
    unique = second is None or (best['exact'] and not second['exact']) or (
        not second['exact'] and best['score'] - second['score'] >= minimum_margin)
    if not reliable or not unique:
        return dict(available=False, reason='ambiguous_or_unsupported_text_binding', candidates=candidates)
    return dict(available=True, source_id=best['source_id'], evidence='request_plan_text',
                candidates=candidates, requested_kind=facts['kind'], predicted_kind=best['predicted_kind'],
                kind_matches=best['predicted_kind'] == facts['kind'], acoustic_identity_certified=False)


def native_field_sites(codec, tokens):
    """Source-bound numeric and text positions in the actual emitted sequence."""
    from ...data.model_sceneplan_codec import SOURCE_SLOT_TOKENS
    ids = list(map(int, tokens))
    slots = {codec._tid(t): f'source_{i}' for i, t in enumerate(SOURCE_SLOT_TOKENS)}
    names = ['<num_sources>', '<source_begin>', '<source_end>', '<kind>', '<text_begin>', '<text_end>',
             '<onset_frame>', '<offset_frame>', '<azimuth_bin>', '<elevation_bin>', '<distance_bin>',
             '<position>', '<start>', '<end>', '<description>', '<speaker_description>', '<transcript>',
             '<trajectory_begin>']
    markers = {codec._tid(t): t for t in names}
    source, point, text_field, inside = None, None, None, False
    numeric, texts = {}, defaultdict(list)
    for index, token in enumerate(ids):
        marker = markers.get(token)
        if inside:
            texts[(source, text_field)].append(index)
            if marker == '<text_end>':
                inside = False
            continue
        if marker == '<source_begin>':
            source, point = slots[ids[index + 1]], None
        elif marker == '<source_end>':
            source = None
        elif marker in ('<description>', '<speaker_description>', '<transcript>'):
            text_field = marker
        elif marker == '<text_begin>':
            inside = True
        elif marker in ('<position>', '<start>', '<end>'):
            point = marker[1:-1]
        elif marker == '<num_sources>':
            numeric[('scene', 'num_sources')] = index + 1
        elif marker == '<kind>':
            numeric[(source, 'kind')] = index + 1
        elif marker in ('<onset_frame>', '<offset_frame>'):
            numeric[(source, marker[1:-7] + '_sec')] = index + 1
        elif marker == '<trajectory_begin>':
            numeric[(source, 'motion')] = index + 1
        elif marker in ('<azimuth_bin>', '<elevation_bin>', '<distance_bin>') and point in ('position', 'start', 'end'):
            numeric[(source, point + '/' + marker[1:-5])] = index + 1
    return numeric, dict(texts)


def request_field_targets(codec, tokens, plan, facts, binding, allowed_fn, *,
                          angle_tolerance=15., elevation_tolerance=10., time_tolerance=.35,
                          distance_relative_tolerance=.25, distance_absolute_tolerance=.25):
    """Use permissible sets for explicitly requested values, never hidden GT.

    When an earlier native choice makes the requested support unreachable,
    record it instead of relabeling an illegal continuation as correct.
    """
    ids = list(map(int, tokens))
    validate_native_plan_tokens(codec, ids, plan)
    if not binding['available'] or facts is None:
        return dict(targets=[], unavailable=[binding.get('reason', 'unparsed')])
    numeric, _ = native_field_sites(codec, ids)
    source_id = binding['source_id']
    source = next(s for s in plan['sources'] if s['source_id'] == source_id)
    targets, unavailable = [], []

    def add(key, values, accept, role):
        pos = numeric.get(key)
        if pos is None:
            unavailable.append('/'.join(key) + ':missing_native_site')
            return
        allowed = sorted(allowed_fn(ids[:pos]))
        accepted = [i for i in allowed if i in values and accept(values[i])]
        if not accepted:
            unavailable.append('/'.join(key) + ':incompatible_earlier_prefix')
            return
        targets.append(dict(position=pos, allowed_ids=allowed, acceptable_ids=accepted,
                            observed_token_id=ids[pos], field='/'.join(key), role=role))

    if facts.get('removal'):
        # A current output may already omit a retained source. Its count minus
        # one is therefore NOT a trusted target count, even when the unwanted
        # source is clearly present. Keep negative scope explicit without
        # fabricating a count label or positively copying the removal quote.
        unavailable.append('removal_requires_retained_source_count_evidence')
        return dict(targets=targets, unavailable=unavailable)

    from ...data.model_sceneplan_codec_v3 import KIND_TOKENS
    add((source_id, 'kind'), {codec._tid(v): k for k, v in KIND_TOKENS.items()},
        lambda kind: kind == facts['kind'], 'explicit_request_source_kind')
    points = ['position'] if len(facts['azimuths']) == 1 else (
        ['start', 'end'] if len(facts['azimuths']) == 2 else [])
    if points:
        from ...data.model_sceneplan_codec import MOTION_TOKENS
        motion = 'static' if len(points) == 1 else 'linear'
        add((source_id, 'motion'), {codec._tid(v): k for k, v in MOTION_TOKENS.items()},
            lambda value: value == motion, 'request_motion')
    if source['trajectory']['type'] != ('static' if len(points) == 1 else 'linear'):
        points = []
    for label, key, choices, tolerance in [
        ('azimuth', 'azimuths', {i: n - 180 for n, i in enumerate(codec.azimuth_ids)}, angle_tolerance),
        ('elevation', 'elevations', {i: n - 90 for n, i in enumerate(codec.elevation_ids)}, elevation_tolerance),
        ('distance', 'distances', {i: codec._distance_value(n) for n, i in enumerate(codec.distance_ids)}, None),
    ]:
        if len(facts.get(key, [])) != len(points):
            continue
        for point, desired in zip(points, facts[key]):
            if label == 'azimuth':
                acceptable = lambda v, d=desired: abs((v - d + 180) % 360 - 180) <= angle_tolerance
            elif label == 'elevation':
                acceptable = lambda v, d=desired: abs(v - d) <= elevation_tolerance
            else:
                acceptable = lambda v, d=desired: abs(v - d) <= max(distance_absolute_tolerance, abs(d) * distance_relative_tolerance)
            add((source_id, point + '/' + label), choices, acceptable, 'explicit_request_coarse_range')
    if facts.get('activity'):
        frame_values = {i: n * 1024 / 44100 for n, i in enumerate(codec.frame_ids)}
        for name, desired in zip(('onset_sec', 'offset_sec'), facts['activity']):
            add((source_id, name), frame_values, lambda v, d=desired: abs(v - d) <= time_tolerance,
                'explicit_request_coarse_interval')
    return dict(targets=targets, unavailable=unavailable)


def frozen_text_targets(codec, tokens, reference_logits, allowed_fn, *, excluded_sources=()):
    _, fields = native_field_sites(codec, tokens)
    targets = []
    for (source, field), positions in fields.items():
        if source in excluded_sources:
            continue
        for pos in positions:
            allowed = sorted(allowed_fn(tokens[:pos]))
            if len(allowed) > 1:
                targets.append(dict(position=pos, ids=allowed, field=source + '/' + field,
                    p=reference_logits[pos - 1, allowed].float().softmax(-1).detach()))
    return targets


def field_balanced_reference_kl(logits, holds):
    groups = defaultdict(list)
    for hold in holds:
        lp = logits[hold['position'] - 1, hold['ids']].float().log_softmax(-1)
        p = hold['p']
        groups[hold['field']].append((p * (p.clamp_min(1e-30).log() - lp)).sum())
    return torch.stack([torch.stack(v).mean() for v in groups.values()]).mean() if groups else logits.sum() * 0
