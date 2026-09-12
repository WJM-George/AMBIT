import copy
from types import SimpleNamespace

import pytest

from stable_audio_tools.training.transfusion_opsd.event_teacher_identity import (
    restore_teacher_parent, teacher_model_version, validate_student_feedback_action,
    validate_teacher_binding, validate_teacher_query,
)


def refreshed():
    parent = dict(path='/research/current.pt', sha256='a' * 64,
        model_fingerprint='b' * 64, dit_fingerprint='c' * 64, model_version=1)
    reference = dict(path='/reference/original.json', sha256='d' * 64)
    return dict(teacher_model_version=1, teacher_parent=parent,
        initial_model_fingerprint=parent['model_fingerprint'],
        initialization_model_fingerprint='e' * 64,
        original_native_reference=reference, teacher_policy_kind='candidate_response',
        fresh_native_prefix_queries=True,
        input_files=[dict(path=x['path'], sha256=x['sha256']) for x in (parent, reference)])


def bind(p, **changes):
    kwargs = dict(model_version=1, model_fingerprint=p['initial_model_fingerprint'],
        parent_candidate=p['teacher_parent'], original_native_reference=p['original_native_reference'])
    kwargs.update(changes)
    return validate_teacher_binding(p, **kwargs)


def test_refreshed_teacher_matches_exact_parent_and_original_reference():
    p = refreshed()
    assert bind(p) == 1
    validate_teacher_query(dict(model_version=1, initial_model_fingerprint=p['initial_model_fingerprint']), p)


@pytest.mark.parametrize('changes', [dict(model_version=2), dict(model_fingerprint='f' * 64)])
def test_old_teacher_cannot_follow_an_update_even_if_version_or_hash_was_reused(changes):
    with pytest.raises(ValueError, match='different student'):
        bind(refreshed(), **changes)


def test_same_weights_from_a_different_optimizer_parent_are_not_interchangeable():
    p = refreshed()
    other = dict(p['teacher_parent'], sha256='f' * 64)
    with pytest.raises(ValueError, match='parent identity'):
        bind(p, parent_candidate=other)


def test_teacher_refresh_cannot_reset_the_capability_baseline():
    with pytest.raises(ValueError, match='original capability reference'):
        bind(refreshed(), original_native_reference=dict(path='/new/reference.json', sha256='a' * 64))


@pytest.mark.parametrize('missing', ['teacher_parent', 'original_native_reference'])
def test_incomplete_lineage_is_rejected_before_collection(missing):
    p = refreshed()
    del p[missing]
    with pytest.raises(ValueError, match='parent candidate|immutable original'):
        teacher_model_version(p)


def test_refreshed_prefix_cannot_be_relabelled_as_current():
    p = refreshed()
    with pytest.raises(ValueError, match='executor version'):
        validate_teacher_query(dict(model_version=0, initial_model_fingerprint=p['initial_model_fingerprint']), p)


def test_actual_nonzero_feedback_action_is_supported_without_future_scores():
    validate_student_feedback_action(dict(student_feedback_action=3), observed_action=3, model_version=1)
    with pytest.raises(ValueError, match='recorded prefix'):
        validate_student_feedback_action(dict(student_feedback_action=3), observed_action=0, model_version=1)
    with pytest.raises(ValueError, match='recorded prefix'):
        validate_student_feedback_action({}, observed_action=0, model_version=1)
    validate_student_feedback_action({}, observed_action=0, model_version=0)


def test_version_zero_pools_remain_usable_only_at_their_original_student():
    p = dict(initial_model_fingerprint='a' * 64)
    assert validate_teacher_binding(p, model_version=0, model_fingerprint='a' * 64) == 0
    with pytest.raises(ValueError, match='different student'):
        validate_teacher_binding(p, model_version=1, model_fingerprint='a' * 64)


def test_parent_reference_mismatch_fails_before_any_model_or_optimizer_restore(monkeypatch):
    p = refreshed()
    payload = dict(commits=1, collection_state=dict(next_collection_version=1,
        original_native_reference=dict(path='/different.json', sha256='a' * 64)))
    monkeypatch.setattr('torch.load', lambda *args, **kwargs: payload)
    monkeypatch.setattr('stable_audio_tools.training.transfusion_opsd.provenance.sha256_file',
        lambda path: p['teacher_parent']['sha256'])
    monkeypatch.setattr('stable_audio_tools.training.transfusion_opsd.event_experiment.state_fingerprint',
        lambda model: p['initialization_model_fingerprint'])
    def forbidden(*args, **kwargs):
        raise AssertionError('must reject before constructing optimizer or mutating the model')
    monkeypatch.setattr('stable_audio_tools.training.transfusion_opsd.event_trainer.EventRefinementTrainer', forbidden)
    with pytest.raises(ValueError, match='before restore'):
        restore_teacher_parent(SimpleNamespace(), p)


def test_parent_and_reference_must_be_pinned_inputs():
    p = refreshed()
    p['input_files'] = copy.deepcopy(p['input_files'][:1])
    with pytest.raises(ValueError, match='pin'):
        teacher_model_version(p)
