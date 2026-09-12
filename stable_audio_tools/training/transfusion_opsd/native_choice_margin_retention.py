"""Preserve native transcript decisions without asserting reference correctness.

This is behavior retention at frozen native prefixes, not a positive teacher.
The caller must separately authorize desired text corrections and judge actual
decoded plans/audio. No gradient through an argmax is claimed.
"""
from collections import defaultdict
import math

import torch
from torch.nn import functional as F

from .native_prefix_supervision import _sites


def transcript_choice_references(codec, tokens, reference_logits, allowed_fn):
    ids = list(map(int, tokens))
    if reference_logits.ndim != 2 or reference_logits.shape[0] != len(ids) - 1:
        raise ValueError('Reference logits must match the complete native prefix sequence.')
    if not torch.isfinite(reference_logits).all():
        raise ValueError('Nonfinite reference logits.')
    _, texts = _sites(codec, ids)
    records = []
    for field, positions in texts.items():
        if field[1] != '<transcript>':
            continue
        for pos in positions:
            allowed = sorted(allowed_fn(ids[:pos]))
            if ids[pos] not in allowed or len(allowed) != len(set(allowed)):
                raise ValueError('Observed native choice must belong to unique legal support.')
            others = [i for i in allowed if i != ids[pos]]
            if not others:
                continue
            row = reference_logits[pos - 1].detach().float()
            gap = float(row[ids[pos]] - row[others].max())
            if gap < -1e-5:
                raise ValueError('Observed token is not a greedy choice of this reference forward.')
            records.append(dict(field='/'.join(field), position=pos, token_id=ids[pos],
                                other_ids=others, reference_gap=max(gap, 0.0),
                                role='uncertified_C0_behavior_retention'))
    return records


def native_choice_margin_loss(logits, references, temperature=1.0):
    """Field-balanced soft gap penalty, including a gradient at the reference.

L = T softplus((gap_ref - gap_student)/T). The competitor is the strongest
currently legal alternative, not a frozen runner-up. At equality L=T log(2),
so the value is not an error count. Reference gaps/IDs are stopped data.
"""
    if logits.ndim != 2 or not torch.isfinite(logits).all():
        raise ValueError('Expected finite next-token logits.')
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('Temperature must be positive and finite.')
    fields = defaultdict(list)
    for item in references:
        pos, target, others = item['position'] - 1, item['token_id'], item['other_ids']
        if not 0 <= pos < logits.shape[0] or target in others or not others or len(others) != len(set(others)):
            raise ValueError('Invalid retained native decision.')
        gap = logits[pos, target].float() - logits[pos, others].float().max()
        fields[item['field']].append(temperature * F.softplus((item['reference_gap'] - gap) / temperature))
    if not fields:
        return logits.sum() * 0
    return torch.stack([torch.stack(items).mean() for items in fields.values()]).mean()
