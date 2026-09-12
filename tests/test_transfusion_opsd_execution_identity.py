import copy
import hashlib

import pytest

from stable_audio_tools.training.transfusion_opsd.native_execution_identity import identity_aware_execution_protection


def evidence(tmp_path, name='audio', payload=b'unchanged execution bytes'):
    path = tmp_path / name
    path.write_bytes(payload)
    return dict(sample_id='request-a', plan={'duration_sec': 3}, plan_admissible=True,
        audio=dict(path=str(path), sha256=hashlib.sha256(payload).hexdigest()),
        costs={'requested_time_failure': 0.}, coarse={'observable': True},
        content={'all_requested_words_supported': False, 'hypotheses': ['a word', 'other words']},
        independent_ctc=[{'wer': .5}])


def ambiguous():
    return dict(passed=False, failures=[], uncertain=['asr_observed_bounds'])


def test_identity_is_nonregression_without_certifying_incorrect_content(tmp_path):
    row = evidence(tmp_path)
    original = ambiguous()
    result = identity_aware_execution_protection(row, row, original)
    assert result['passed']
    assert result['original_guard'] == original
    assert not result['identity_nonregression']['content_correctness_certified']
    assert not row['content']['all_requested_words_supported']
    assert original == ambiguous()


def test_changed_audio_with_same_uncertain_scores_is_not_exempt(tmp_path):
    before = evidence(tmp_path, 'before', b'before')
    after = evidence(tmp_path, 'after', b'after')
    assert identity_aware_execution_protection(after, before, ambiguous()) == ambiguous()


def test_forged_or_stale_file_identity_is_rejected(tmp_path):
    row = evidence(tmp_path)
    (tmp_path / 'audio').write_bytes(b'changed after scoring')
    with pytest.raises(ValueError, match='digest mismatch'):
        identity_aware_execution_protection(row, row, ambiguous())


@pytest.mark.parametrize('field,value', [('sample_id', 'another-request'),
    ('plan', {'duration_sec': 4}), ('content', {'all_requested_words_supported': True})])
def test_same_audio_does_not_override_different_request_or_observations(tmp_path, field, value):
    before = evidence(tmp_path)
    after = copy.deepcopy(before)
    after[field] = value
    assert identity_aware_execution_protection(after, before, ambiguous()) == ambiguous()


def test_identity_does_not_override_a_real_failure(tmp_path):
    row = evidence(tmp_path)
    guard = dict(passed=False, failures=['source_presence_failure'], uncertain=['asr_observed_bounds'])
    assert identity_aware_execution_protection(row, row, guard) == guard
