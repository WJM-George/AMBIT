"""Trusted field anchors and first-error text targets on native prefixes.

These are paired supervision, not execution-derived self-teachers. Only the
matching prefix of a text field is retained; a transcript contributes at most
one corrective next token. Nothing after a transcript divergence is relabeled.
Numeric fields are left to coarse constraints and distribution retention.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence

import torch


def _sites(codec, tokens: Sequence[int]):
    from ...data.model_sceneplan_codec import SOURCE_SLOT_TOKENS

    ids = list(map(int, tokens))
    slot_names = {codec._tid(x): f'source_{i}' for i, x in enumerate(SOURCE_SLOT_TOKENS)}
    markers = {codec._tid(x): x for x in (
        '<room>', '<num_sources>', '<source_begin>', '<source_end>', '<kind>',
        '<trajectory_begin>', '<speaker_description>', '<description>', '<transcript>')}
    atomic, texts, source = {}, {}, None
    j = 0
    while j < len(ids):
        marker = markers.get(ids[j])
        if marker == '<source_end>':
            source = None
        elif marker == '<source_begin>':
            source = slot_names[ids[j+1]]
            atomic[(source, 'identity')] = j+1
        elif marker in ('<room>', '<num_sources>'):
            atomic[('scene', marker)] = j+1
        elif marker in ('<kind>', '<trajectory_begin>'):
            if source is None:
                raise ValueError('A native source field has no source identity.')
            atomic[(source, marker)] = j+1
        elif marker in ('<speaker_description>', '<description>', '<transcript>'):
            if source is None or ids[j+1] != codec._tid('<text_begin>'):
                raise ValueError('Expected a source-bound native text field.')
            end = ids.index(codec._tid('<text_end>'), j+2)
            texts[(source, marker)] = list(range(j+2, end+1))
            j = end
        j += 1
    return atomic, texts


def build_native_prefix_targets(codec, native_tokens, paired_plan, allowed_fn: Callable):
    """Build serializable targets using actual tokens and the paired source IDs.

    All retained atomic values equal their paired targets. Matching text
    prefixes are retained without asserting that the rest of the field is
    correct. Correction is limited to transcript, whose words are specified.
    The caller must establish paired sample/instruction/latent provenance.
    """
    old = list(map(int, native_tokens))
    truth = codec.encode(paired_plan)['input_ids'].tolist()
    old_atoms, old_texts = _sites(codec, old)
    new_atoms, new_texts = _sites(codec, truth)
    anchors, frontiers = [], []

    def entry(pos, target, field, role):
        allowed = sorted(allowed_fn(old[:pos]))
        if target not in allowed or old[pos] not in allowed:
            raise ValueError('Paired correction is incompatible with the visited native prefix.')
        if len(allowed) < 2:
            return None
        return dict(position=pos, token_id=int(target), observed_token_id=old[pos],
                    field='/'.join(field), role=role, allowed_ids=allowed)

    for field, pos in old_atoms.items():
        other = new_atoms.get(field)
        if other is not None and old[pos] == truth[other]:
            value = entry(pos, old[pos], field, 'paired_correct_choice_retention')
            if value is not None:
                anchors.append(value)
    for field, positions in old_texts.items():
        expected = new_texts.get(field)
        if expected is None:
            continue
        for pos, other in zip(positions, expected):
            target = truth[other]
            if old[pos] != target:
                if field[1] == '<transcript>':
                    value = entry(pos, target, field, 'paired_transcript_first_error')
                    if value is not None:
                        frontiers.append(value)
                break
            value = entry(pos, target, field, 'paired_matching_text_prefix_retention')
            if value is not None:
                anchors.append(value)
    return dict(anchors=anchors, frontiers=frontiers,
                scope='Paired supervision at student-visited prefixes; no labels after text divergence; no numeric equality target.')


def field_balanced_native_ce(logits: torch.Tensor, targets):
    """Average legal next-token CE within fields, then equally across fields.

    Correct choices receive a nonzero initial reinforcement gradient. This
    differs from a zero-at-reference KL/barrier, and does not guarantee their
    retention after a shared-parameter update.
    """
    if logits.ndim != 2 or not torch.isfinite(logits).all():
        raise ValueError('Expected finite [prefix positions, vocabulary] logits.')
    groups = defaultdict(list)
    for target in targets:
        pos, allowed = target['position']-1, target['allowed_ids']
        if not 0 <= pos < logits.shape[0] or len(allowed) != len(set(allowed)):
            raise ValueError('Invalid native supervision position or legal support.')
        index = allowed.index(target['token_id'])
        groups[target['field']].append(-logits[pos, allowed].float().log_softmax(0)[index])
    if not groups:
        return logits.sum()*0
    return torch.stack([torch.stack(v).mean() for v in groups.values()]).mean()
