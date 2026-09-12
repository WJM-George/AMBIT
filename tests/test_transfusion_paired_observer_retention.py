import pytest

from stable_audio_tools.training.transfusion_opsd.paired_observer_retention import paired_error_retention


def test_identity_passes_even_when_observer_placements_disagree():
    values={'whisper@0':1/3,'whisper@4410':0.,'whisper@8820':0.,
            'ctc@whole_w':.2,'ctc@canonical_event_w':.1}
    result=paired_error_retention(values,dict(reversed(list(values.items()))))
    assert result['passed'] and all(row['change']==0 for row in result['rows'])


def test_better_average_cannot_hide_a_worse_whisper_view():
    before={'whisper@0':.8,'whisper@4410':.1}
    after={'whisper@0':0.,'whisper@4410':.2}
    assert not paired_error_retention(before,after)['passed']


def test_independent_ctc_regression_rejects_despite_whisper_gain():
    assert not paired_error_retention(
        {'whisper@0':.2,'ctc@whole_w':.1},
        {'whisper@0':0.,'ctc@whole_w':.2})['passed']


@pytest.mark.parametrize('before,after', [
    ({'a':0.},{'b':0.}), ({'a':0.},{'a':float('nan')}),
    ({'a':0.},{'a':-1.}), ({},{})])
def test_invalid_or_unmatched_views_are_rejected(before,after):
    with pytest.raises(ValueError):
        paired_error_retention(before,after)
