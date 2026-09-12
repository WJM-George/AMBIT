import copy

import pytest

from stable_audio_tools.training.transfusion_opsd.teacher_panel import teacher_panel_assignment


def panel():
    return dict(sample_ids=['a', 'b', 'c', 'd'], seeds={key: [10, 11] for key in 'abcd'},
        expected_queries=8, shards={'0': ['a', 'c'], '1': ['b', 'd']})


def test_local_count_uses_the_assigned_requests():
    p = panel()
    assert teacher_panel_assignment(p, 0) == (('a', 'c'), 4)
    assert teacher_panel_assignment(p, 1) == (('b', 'd'), 4)
    p['seeds']['a'].append(12)
    p['expected_queries'] = 9
    assert teacher_panel_assignment(p, 0) == (('a', 'c'), 5)
    assert teacher_panel_assignment(p, 1) == (('b', 'd'), 4)


def test_stale_two_query_probe_count_is_rejected_before_collection():
    p = panel()
    p['expected_queries'] = 2
    with pytest.raises(ValueError, match='before model loading'):
        teacher_panel_assignment(p, 0)


def test_missing_or_overlapping_shards_cannot_silently_drop_queries():
    for partition in ({'0': ['a', 'b'], '1': ['b', 'd']}, {'0': ['a', 'b']}):
        p = panel()
        p['shards'] = partition
        with pytest.raises(ValueError, match='whole request panel'):
            teacher_panel_assignment(p, 0)


def test_duplicate_or_undeclared_noise_entries_are_rejected():
    p = panel()
    for seeds in (dict(p['seeds'], a=[10, 10]), dict(p['seeds'], extra=[12]),
            dict(p['seeds'], a=[True, 11])):
        invalid = copy.deepcopy(p)
        invalid['seeds'] = seeds
        with pytest.raises(ValueError):
            teacher_panel_assignment(invalid, 0)


def test_unsharded_collection_keeps_the_full_panel():
    p = panel()
    del p['shards']
    assert teacher_panel_assignment(p) == (('a', 'b', 'c', 'd'), 8)
    with pytest.raises(ValueError, match='without a declared partition'):
        teacher_panel_assignment(p, 0)
