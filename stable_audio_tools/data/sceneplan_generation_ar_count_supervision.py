"""Training-only count prefixes; production inference receives raw text only."""
from __future__ import annotations

import hashlib

from .sceneplan_generation_ar_natural_constraints import FRAME_SECONDS, validate_requirements


def count_prefix_candidates(codec, tokens, request, requirements):
    """Vary only unrequested header values for an auxiliary count loss.

    These are six-token prefixes, not full alternative witness plans. Explicit
    source intervals bound a free clip duration. Existing N2 request-first
    annotations have numeric duration and categorical room constraints only;
    unsupported scene constraints fail closed instead of being dropped.
    """
    validate_requirements(request, requirements)
    tokens = [int(t) for t in tokens]
    if len(tokens) < 7 or tokens[5] != codec.token_to_id['<num_sources>']:
        raise ValueError('Expected the existing codec header before source count')
    count = len(requirements['sources'])
    if tokens[6] != codec.token_to_id[f'<num_sources_{count}>']:
        raise ValueError('Count label disagrees with the request annotation')
    duration_fixed = room_fixed = False
    for c in requirements['scene']:
        if c['op'] == 'numeric' and c['field'] == 'duration_sec':
            duration_fixed = True
            seconds = codec.frame_ids.index(tokens[2]) * FRAME_SECONDS
            if abs(seconds - float(c['value'])) > .25:
                raise ValueError('Witness header contradicts the requested duration')
        elif c['op'] == 'room':
            room_fixed = True
            if tokens[4] != codec.token_to_id[f"<room_{c['value']}>"]:
                raise ValueError('Witness header contradicts the requested room')
        else:
            raise ValueError('Unsupported scene constraint for header augmentation')
    min_duration = FRAME_SECONDS
    for source in requirements['sources']:
        for c in source['constraints']:
            if c['op'] != 'numeric':
                continue
            if c['field'] == 'event_duration_sec':
                raise ValueError('Event-duration augmentation needs a joint feasibility implementation')
            if c['field'] == 'offset_sec':
                min_duration = max(min_duration, float(c['value']))
            elif c['field'] == 'onset_sec':
                min_duration = max(min_duration, float(c['value']) + FRAME_SECONDS)
    frame_ids = [tokens[2]] if duration_fixed else [
        codec.frame_ids[codec._frame_from_seconds(x, mode='nearest')]
        for x in (8., 10., 12., 14.) if x >= min_duration
    ]
    # Preserve a legal original duration as an option, without declaring it the
    # unique correct completion. Its full witness was checked during data QA.
    frame_ids = sorted(set(frame_ids + [tokens[2]]))
    rooms = [tokens[4]] if room_fixed else [codec.token_to_id[f'<room_{name}>'] for name in ('dry', 'moderate', 'reverberant', 'outdoor')]
    values = []
    for frame in frame_ids:
        for room in rooms:
            prefix = tokens[:6].copy(); prefix[2] = frame; prefix[4] = room
            if tokens[6] not in codec.allowed_next_ids(prefix):
                raise ValueError('Auxiliary prefix does not permit the requested count')
            values.append(prefix)
    return values


def choose_count_prefix(candidates, *, step, row, parent_index, view):
    """Stateless sampling reproduces exactly after checkpoint resume."""
    if not candidates:
        raise ValueError('No count-prefix candidates')
    key = f'N3-count-prefix-42/{step}/{row}/{parent_index}/{view}'
    index = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'big') % len(candidates)
    return candidates[index]
