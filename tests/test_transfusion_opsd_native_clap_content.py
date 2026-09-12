import copy

import pytest

from stable_audio_tools.training.transfusion_opsd.native_clap_event_content import compare_native_content_evidence


def evidence():
    return dict(contract='native_foa_clap_content_observer_v1',checkpoint={'path':'selected.pt','sha256':'abc'},
        native_semantic=dict(text='requested content',observations=[{'posterior_seed':42,'score':.7},{'posterior_seed':43,'score':.6}]),
        presence={'source_presence_failure':0.},content_view={'admissible':True},asr={'required':False})


def test_one_bad_paired_posterior_is_not_hidden_by_the_mean():
    before=evidence(); after=copy.deepcopy(before)
    after['native_semantic']['observations'][0]['score']-=.04
    after['native_semantic']['observations'][1]['score']+=.04
    result=compare_native_content_evidence(after,before,maximum_semantic_drop=.02)
    assert not result['passed'] and result['failures']==['native_clap_semantic']
    assert result['native_semantic_mean_delta']==pytest.approx(0.)


def test_native_comparison_rejects_cross_checkpoint_and_noise_unpairing():
    before=evidence(); after=copy.deepcopy(before)
    after['checkpoint']['sha256']='changed'
    with pytest.raises(ValueError): compare_native_content_evidence(after,before,maximum_semantic_drop=.02)
    after=copy.deepcopy(before);after['native_semantic']['observations'][0]['posterior_seed']=44
    with pytest.raises(ValueError): compare_native_content_evidence(after,before,maximum_semantic_drop=.02)


def test_native_clap_does_not_override_unresolved_speech():
    before=evidence(); after=copy.deepcopy(before)
    for item in [after,before]:
        item['asr']=dict(required=True,observed_error_bounds=dict(reference_words=['the','priest','hesitated'],
            lower_wer=0.,upper_wer=1/3,all_requested_words_supported=False))
    result=compare_native_content_evidence(after,before,maximum_semantic_drop=.02)
    assert not result['passed'] and result['uncertain']==['asr_observed_bounds']
