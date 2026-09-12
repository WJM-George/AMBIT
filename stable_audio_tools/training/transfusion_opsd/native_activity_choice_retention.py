"""Native timing behavior retention until actual evidence authorizes a change.

References come from actual greedy prefixes. They preserve the old policy's
choices, not hidden paired timestamps or a claim that those choices are correct.
The loss is the existing native_choice_margin_loss; inference stays discrete.
"""
import torch


def activity_choice_references(codec, tokens, reference_logits, allowed_fn, *, exempt_fields=()):
    from ...data.model_sceneplan_codec_v3 import SOURCE_SLOT_TOKENS

    ids = list(map(int, tokens))
    if reference_logits.ndim != 2 or reference_logits.shape[0] != len(ids) - 1:
        raise ValueError('Reference logits must match the actual native prefix sequence.')
    if not torch.isfinite(reference_logits).all():
        raise ValueError('Nonfinite reference logits.')
    exemptions = set(exempt_fields)
    slots = {codec._tid(x): f'source_{i}' for i, x in enumerate(SOURCE_SLOT_TOKENS)}
    source, records, j = None, [], 0
    if not hasattr(codec, 'frame_ids'):
        raise ValueError('Activity retention requires the native V3/V4 atomic-frame codec.')
    frame_ids = set(codec.frame_ids)
    while j < len(ids):
        token = ids[j]
        if token == codec._tid('<text_begin>'):
            j = ids.index(codec._tid('<text_end>'), j+1)
        elif token == codec._tid('<source_begin>'):
            if source is not None or j+1 >= len(ids) or ids[j+1] not in slots:
                raise ValueError('Expected a unique current source slot.')
            source = slots[ids[j+1]]
        elif token == codec._tid('<source_end>'):
            source = None
        elif token == codec._tid('<activity_begin>'):
            if source is None:
                raise ValueError('Activity requires a bound native source.')
            cursor = j+1
            for name in ('<onset_frame>', '<offset_frame>'):
                if cursor >= len(ids) or ids[cursor] != codec._tid(name):
                    raise ValueError('Malformed native activity block.')
                field = source + '/' + name
                positions = [cursor+1]
                if cursor+1 >= len(ids) or ids[cursor+1] not in frame_ids:
                    raise ValueError('Activity must use the native atomic frame vocabulary.')
                if field not in exemptions:
                    for pos in positions:
                        allowed = sorted(allowed_fn(ids[:pos]))
                        if len(allowed) != len(set(allowed)) or ids[pos] not in allowed:
                            raise ValueError('Observed timing choice must belong to unique legal support.')
                        others = [v for v in allowed if v != ids[pos]]
                        if not others:
                            continue
                        row = reference_logits[pos-1].detach().float()
                        gap = float(row[ids[pos]] - row[others].max())
                        if gap < -1e-5:
                            raise ValueError('Observed timing token was not greedy in this reference forward.')
                        records.append(dict(field=field, position=pos, token_id=ids[pos], other_ids=others,
                            reference_gap=max(gap, 0.0), role='uncertified_C0_activity_behavior_retention'))
                cursor += 2
            if cursor >= len(ids) or ids[cursor] != codec._tid('<activity_end>'):
                raise ValueError('Unterminated native activity block.')
            j = cursor
        j += 1
    return records
