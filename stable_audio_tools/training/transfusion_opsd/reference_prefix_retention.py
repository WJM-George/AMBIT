"""Soft capability retention on the frozen reference's own native prefixes.

This complements retention on student prefixes after the two decodes diverge.
Reference predictions are anchors, not ground truth. Explicit request fields
and execution-authorized changes take precedence over these anchors.
"""
from .editing_request_constraints import native_field_sites, frozen_text_targets
from .native_prefix_supervision import _sites
from .native_token_alignment import validate_native_plan_tokens
from .removal_retention import removal_policy


def reference_prefix_targets(codec, tokens, plan, facts, binding, logits, allowed_fn, *, operation=None):
    tokens = list(map(int, tokens))
    validate_native_plan_tokens(codec, tokens, plan)
    numeric, _ = native_field_sites(codec, tokens)
    atoms, _ = _sites(codec, tokens)
    if ('scene', '<room>') in atoms:
        numeric[('scene', 'room')] = atoms[('scene', '<room>')]
    policy = removal_policy(plan, facts, binding, operation=operation)
    editable, removed = set(), set(policy['excluded_removed_sources'])
    all_sources = {s['source_id'] for s in plan['sources']}
    if policy['release_count']:
        editable.add(('scene', 'num_sources'))
    if facts and not policy['removal']:
        target_sources = {binding['source_id']} if binding.get('available') else all_sources
        # An ambiguous binding removes anchors; it never fabricates a
        # request label for a guessed source.
        fields = {'kind'}
        if len(facts.get('azimuths', [])) in (1, 2):
            fields.add('motion')
        for point in ('position', 'start', 'end'):
            for field, key in [('azimuth', 'azimuths'), ('elevation', 'elevations'), ('distance', 'distances')]:
                if facts.get(key):
                    fields.add(point + '/' + field)
        if facts.get('activity'):
            fields.update(('onset_sec', 'offset_sec'))
        fields.update('<' + field + '>' for field in facts.get('fields', {}))
        editable = {(source, field) for source in target_sources for field in fields}
        if facts.get('operation') == 'event_addition' and not binding.get('available'):
            editable.add(('scene', 'num_sources'))
    holds = []
    for key, position in numeric.items():
        if key[0] in removed or key in editable:
            continue
        ids = sorted(allowed_fn(tokens[:position]))
        if len(ids) > 1:
            holds.append(dict(position=position, ids=ids, field='/'.join(key),
                              p=logits[position - 1, ids].float().softmax(-1).detach()))
    texts = frozen_text_targets(codec, tokens, logits, allowed_fn, excluded_sources=removed)
    texts = [h for h in texts if tuple(h['field'].split('/', 1)) not in editable]
    return dict(structure=holds, text=texts, excluded_request_fields=sorted('/'.join(k) for k in editable),
                excluded_removed_sources=sorted(removed), binding_available=bool(binding.get('available')),
                role='Soft reference behavior retention; not execution-derived improvement or GT')
