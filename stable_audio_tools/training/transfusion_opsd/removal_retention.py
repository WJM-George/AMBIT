"""One removal mask for both student-prefix and reference-prefix retention.

Releasing an anchor is not a deletion label. In particular, the emitted
source count minus one is never used as a target count.
"""
from .editing_request_constraints import native_field_sites
from .native_prefix_supervision import _sites


def removal_policy(plan, facts, binding, *, operation=None):
    removal = operation == 'event_removal' or bool(facts and facts.get('removal'))
    removed = set()
    if removal:
        if binding.get('available'):
            removed.add(binding['source_id'])
        elif facts is None:
            # Parsing failure must not positively anchor an unknown target.
            removed.update(s['source_id'] for s in plan['sources'])
        else:
            removed.update(s['source_id'] for s in plan['sources'] if s['kind'] == facts['kind'])
            removed.update(c['source_id'] for c in binding.get('candidates', []) if c.get('shared_words', 0))
    return dict(removal=removal, excluded_removed_sources=sorted(removed),
                release_count=removal, reference_velocity_allowed=not removal)


def removal_positions(codec, tokens, policy):
    """Include source identity and every source token, not only its angles."""
    if not policy['removal']:
        return set()
    tokens = list(map(int, tokens))
    numeric, _ = native_field_sites(codec, tokens)
    atoms, _ = _sites(codec, tokens)
    excluded = {numeric[('scene', 'num_sources')]}
    identities = sorted((pos, source) for (source, field), pos in atoms.items() if field == 'identity')
    for index, (start, source) in enumerate(identities):
        if source not in policy['excluded_removed_sources']:
            continue
        # End before the next source_begin. Text may contain marker-like
        # tokens, so boundaries come from the codec-aware site parser.
        stop = identities[index + 1][0] - 1 if index + 1 < len(identities) else len(tokens)
        excluded.update(range(start - 1, stop))
    return excluded


def apply_removal_retention(codec, item, facts, binding):
    policy = removal_policy(item['plan'], facts, binding, operation=item['row']['operation'])
    excluded = removal_positions(codec, item['tokens'].tolist(), policy)
    before = {key: len(item[key]) for key in ('holds', 'coarse', 'text_reference')}
    for key in before:
        item[key] = [h for h in item[key] if h['position'] not in excluded]
    item['removal_retention'] = dict(policy, released_positions={
        key: before[key] - len(item[key]) for key in before})
    return policy
