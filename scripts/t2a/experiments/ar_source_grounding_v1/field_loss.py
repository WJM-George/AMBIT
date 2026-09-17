"""Label-only source-role masks and a conditional source-preference objective."""
from collections.abc import Mapping, Sequence

import torch
from torch.nn import functional as F

METHOD_CONTRACT = 'editing_ar_unchanged_source_field_preference_probe_v1'
CONTENT_FIELDS = ('semantics', 'transcript', 'azimuth')


def target_field_masks(codec, encoded: Mapping, *, edited_ids: Sequence[str], unchanged_ids: Sequence[str]):
    """Classify full target tokens; callers shift by one exactly like native labels."""
    ids = encoded['input_ids'].detach().cpu()
    groups = encoded['loss_group_ids'].detach().cpu()
    if ids.ndim != 1 or ids.shape != groups.shape:
        raise ValueError('Expected aligned full target token and native loss-group vectors')
    edited, unchanged = set(edited_ids), set(unchanged_ids)
    if edited & unchanged:
        raise ValueError('Edited and unchanged target source IDs overlap')
    slots = {codec._tid(f'<source_slot_{i}>'): f'source_{i}' for i in range(4)}
    begin, end = codec._tid('<source_begin>'), codec._tid('<source_end>')
    role_ids = torch.zeros_like(ids)
    seen, current, awaiting_slot = set(), None, False
    for i, token in enumerate(ids.tolist()):
        if token == begin:
            if current is not None or awaiting_slot:
                raise ValueError('Nested or incomplete target source span')
            awaiting_slot = True
        elif awaiting_slot:
            if token not in slots or slots[token] in seen:
                raise ValueError('Target source slot is invalid or repeated')
            current = slots[token]
            if current not in edited | unchanged:
                raise ValueError('Target source has no declared label role')
            seen.add(current)
            awaiting_slot = False
        elif token == end:
            if current is None:
                raise ValueError('Target source span closes without a source')
            current = None
        if current is not None:
            role_ids[i] = 1 if current in unchanged else 2
    if current is not None or awaiting_slot or seen != edited | unchanged:
        raise ValueError('Target source roles do not match complete canonical target spans')
    text = torch.isin(ids, torch.tensor(sorted(codec.text_ids)))
    fields = {'semantics': text & groups.eq(2), 'transcript': text & groups.eq(7),
        'azimuth': torch.isin(ids, torch.tensor(codec.azimuth_ids)) & groups.eq(5)}
    return {f'{role}/{name}': role_ids.eq(value) & mask
        for role, value in [('unchanged', 1), ('edited', 2)] for name, mask in fields.items()}


def source_preference_terms(clean_token_ce, donor_token_ce, masks, *, margin=0.1):
    """Return per-row/field terms and eligibility, leaving global reduction explicit."""
    if clean_token_ce.shape != donor_token_ce.shape or clean_token_ce.ndim != 2 or margin < 0:
        raise ValueError('Source preference requires aligned token losses and a nonnegative margin')
    terms, eligible, differences = [], [], []
    for field in CONTENT_FIELDS:
        mask = masks[f'unchanged/{field}'].to(device=clean_token_ce.device, dtype=torch.bool)
        if mask.shape != clean_token_ce.shape:
            raise ValueError('Source-role mask is not aligned to native shifted labels')
        count = mask.sum(1)
        difference = ((donor_token_ce - clean_token_ce) * mask).sum(1) / count.clamp_min(1)
        valid = count > 0
        terms.append(F.softplus(float(margin) - difference) * valid)
        eligible.append(valid)
        differences.append(difference)
    return torch.stack(terms, 1), torch.stack(eligible, 1), torch.stack(differences, 1)
