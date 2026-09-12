"""Validate fixed query coverage before an expensive teacher collection starts."""


def teacher_panel_assignment(protocol, shard=None):
    sample_ids, seeds = protocol['sample_ids'], protocol['seeds']
    if (not sample_ids or any(not isinstance(key, str) or not key for key in sample_ids)
            or len(set(sample_ids)) != len(sample_ids) or set(seeds) != set(sample_ids)):
        raise ValueError('declare unique requests and exactly their seed panels')
    for values in seeds.values():
        if (not values or any(type(seed) is not int or not 0 <= seed < 2**63 for seed in values)
                or len(set(values)) != len(values)):
            raise ValueError('each request requires explicit unique integer noise seeds')
    total = sum(len(seeds[key]) for key in sample_ids)
    if type(protocol['expected_queries']) is not int or protocol['expected_queries'] != total:
        raise ValueError('global query count must match the explicit panel before model loading')
    shards = protocol.get('shards')
    if shards is None:
        if shard is not None:
            raise ValueError('a shard was requested without a declared partition')
        assigned = tuple(sample_ids)
    else:
        if shard is None or str(shard) not in shards or any(not ids for ids in shards.values()):
            raise ValueError('select a nonempty declared teacher shard')
        flat = [key for ids in shards.values() for key in ids]
        if len(set(flat)) != len(flat) or set(flat) != set(sample_ids):
            raise ValueError('teacher shards must partition the whole request panel exactly once')
        assigned = tuple(shards[str(shard)])
    return assigned, sum(len(seeds[key]) for key in assigned)
