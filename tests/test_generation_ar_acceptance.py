import copy
from stable_audio_tools.data.sceneplan_generation_ar_acceptance import score_scene, summarize_scenes


def plan():
    return {'sample_id':'acceptance-fixture','duration_sec':2.,'room':{'type':'dry'},
            'sources':[{'source_id':'source_0','kind':'sound','description':'A bell rings.','gain_db':0.,
                        'activity':{'onset_sec':0.,'offset_sec':2.},
                        'trajectory':{'type':'linear','start':{'azimuth_deg':160.,'elevation_deg':0.,'distance_m':1.},
                                      'end':{'azimuth_deg':-160.,'elevation_deg':0.,'distance_m':1.}}}]}


def test_unknown_semantics_cannot_establish_acceptance():
    target=plan();row=score_scene(target,copy.deepcopy(target))
    result=summarize_scenes([row],bootstrap_replicates=10)
    assert row['sources'][0]['motion']
    assert result['semantic_pending']==1
    assert not result['metrics']['joint']['complete']
    assert not result['acceptance_pass']


def test_opposite_direction_fails_even_with_tolerable_endpoints():
    target=plan();hyp=copy.deepcopy(target)
    target['sources'][0]['trajectory']['start']['azimuth_deg']=0.
    target['sources'][0]['trajectory']['end']['azimuth_deg']=179.
    hyp['sources'][0]['trajectory']['start']['azimuth_deg']=0.
    hyp['sources'][0]['trajectory']['end']['azimuth_deg']=-179.
    row=score_scene(target,hyp,{'source_0':True})['sources'][0]
    assert row['start'] and row['end']
    assert not row['motion'] and not row['joint']


def test_missing_and_extra_sources_affect_semantic_denominators():
    target=plan();hyp=copy.deepcopy(target)
    additional=copy.deepcopy(hyp['sources'][0]);additional['source_id']='source_1';hyp['sources'].append(additional)
    extra=score_scene(target,hyp,{'source_0':True})
    result=summarize_scenes([extra],bootstrap_replicates=10)
    assert result['metrics']['semantic_precision']['rate']==.5
    assert result['metrics']['semantic_recall']['rate']==1.
    assert result['metrics']['joint']['rate']==0.
    missing=summarize_scenes([score_scene(target,None)],bootstrap_replicates=10)
    assert missing['metrics']['semantic_recall']['rate']==0.
    assert missing['missing_sources']==1 and not missing['acceptance_pass']


def test_correct_source_and_semantics_pass():
    target=plan();row=score_scene(target,copy.deepcopy(target),{'source_0':True})
    assert row['sources'][0]['direction_checks']=={'azimuth_deg':True}
    assert summarize_scenes([row],bootstrap_replicates=10)['acceptance_pass']
