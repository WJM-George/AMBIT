"""Limit reference distillation to certified parts of native transcripts.

This changes the scope of paired-data-informed retention, not the self-teacher
or the inference policy. A divergent suffix is unverified; individual tokens
inside it are not all asserted to be wrong and are not given replacement labels.
"""
from __future__ import annotations

from collections.abc import Sequence

from .native_prefix_supervision import _sites


def unverified_transcript_logit_positions(codec, native_tokens: Sequence[int], frontiers):
    """Return positions to exempt from an old-prefix reference KL.

    Frontiers must come from ``build_native_prefix_targets`` on these exact
    tokens with verified paired source/instruction provenance. Exempt the first
    divergence through that transcript's text_end. Preserve the matched prefix
    and every other field. Returned indices address next-token logits, hence
    token position p maps to logit position p-1.

    The caller should preserve the previous loss denominator so that removing
    unverified positions does not increase the weight of remaining positions.
    """
    tokens = list(map(int, native_tokens))
    _, text_fields = _sites(codec, tokens)
    excluded = set()
    seen_fields = set()
    for frontier in frontiers:
        field = tuple(frontier['field'].split('/'))
        if len(field) != 2 or field[1] != '<transcript>':
            raise ValueError('Only source-bound transcript frontiers can exempt a suffix.')
        if frontier.get('role') != 'paired_transcript_first_error':
            raise ValueError('A verified paired transcript frontier is required.')
        if field not in text_fields or field in seen_fields:
            raise ValueError('Frontier source is absent or has duplicate corrections.')
        seen_fields.add(field)
        position = int(frontier['position'])
        positions = text_fields[field]
        if position not in positions or position < 1:
            raise ValueError('Frontier does not belong to this native transcript.')
        if tokens[position] != int(frontier['observed_token_id']):
            raise ValueError('Frontier does not match the supplied native sequence.')
        if tokens[position] == int(frontier['token_id']):
            raise ValueError('A matching token is not a transcript divergence.')
        excluded.update(p-1 for p in positions if p >= position)
    return sorted(excluded)
