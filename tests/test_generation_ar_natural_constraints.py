from copy import deepcopy

import pytest

from stable_audio_tools.data.sceneplan_generation_ar_natural_constraints import (
    SCHEMA, evaluate_natural_request, validate_requirements,
)


REQUEST = 'Rain behind me, then a dog barking on my left. Let them overlap.'


def requirements():
    return {'schema': SCHEMA, 'output_order': None, 'scene': [], 'sources': [
        {'key': 'rain', 'kind': 'sound', 'core': 'Rainfall', 'evidence': 'Rain behind me',
         'constraints': [{'op': 'sector', 'point': 'both', 'value': 'behind', 'evidence': 'behind me'}]},
        {'key': 'dog', 'kind': 'sound', 'core': 'A dog barking', 'evidence': 'a dog barking on my left',
         'constraints': [{'op': 'sector', 'point': 'both', 'value': 'left', 'evidence': 'on my left'}]}],
        'relations': [{'op': 'starts_after', 'a': 'dog', 'b': 'rain', 'evidence': 'then a dog'},
                      {'op': 'overlaps', 'a': 'dog', 'b': 'rain', 'evidence': 'Let them overlap'}]}


def source(sid, text, az, onset, offset, distance=2):
    return {'source_id': sid, 'kind': 'sound', 'description': text, 'gain_db': 0,
            'activity': {'onset_sec': onset, 'offset_sec': offset},
            'trajectory': {'type': 'static', 'position': {'azimuth_deg': az, 'elevation_deg': 0, 'distance_m': distance}}}


def plan():
    # Output order differs from mention order; wording is not verbatim.
    return {'sample_id': 'test', 'duration_sec': 12, 'room': {'type': 'outdoor'}, 'sources': [
        source('source_0', 'A hound gives short barks.', 70, 3, 7),
        source('source_1', 'The steady patter of falling rain.', 175, 0, 10)]}


LABELS = {('rain', 'source_0'): False, ('rain', 'source_1'): True,
          ('dog', 'source_0'): True, ('dog', 'source_1'): False}


def evaluate(p, labels=LABELS):
    return evaluate_natural_request(REQUEST, requirements(), p, labels, completion_reasonable=True)


def test_unordered_sources_and_synonyms_pass():
    result = evaluate(plan())
    assert result['acceptance_joint']
    assert result['assignment'] == {'rain': 'source_1', 'dog': 'source_0'}


def test_unspecified_coordinates_and_times_have_multiple_valid_answers():
    a = plan(); b = deepcopy(a)
    b['sources'][0]['activity'] = {'onset_sec': 1, 'offset_sec': 11}
    b['sources'][0]['trajectory']['position'].update(azimuth_deg=110, distance_m=5)
    b['sources'][1]['trajectory']['position'].update(azimuth_deg=-155, distance_m=8)
    assert evaluate(a)['acceptance_joint'] and evaluate(b)['acceptance_joint']
    assert not evaluate(b)['hidden_target_fields_compared']


def test_assignment_cannot_swap_coordinates_independently_of_sound_identity():
    p = plan()
    p['sources'][0]['trajectory'], p['sources'][1]['trajectory'] = p['sources'][1]['trajectory'], p['sources'][0]['trajectory']
    result = evaluate(p)
    assert result['assignment']['dog'] == 'source_0'
    assert not result['acceptance_joint']
    assert all(not row['constraints'][0]['pass'] for row in result['sources'])


def test_relative_relation_is_checked_even_when_both_sounds_and_positions_match():
    p = plan(); p['sources'][0]['activity']['onset_sec'] = 0
    assert not evaluate(p)['acceptance_joint']


def test_missing_and_extra_sources_fail():
    p = plan(); p['sources'].append(source('source_2', 'An unrequested bell.', 0, 1, 2))
    assert evaluate(p)['extra'] == 1 and not evaluate(p)['acceptance_joint']
    p = plan(); p['sources'].pop()
    assert evaluate(p)['missing'] == 1 and not evaluate(p)['acceptance_joint']


def test_pending_semantics_or_completion_cannot_pass():
    assert evaluate(plan(), {})['semantic_pending']
    assert not evaluate(plan(), {})['acceptance_joint']
    result = evaluate_natural_request(REQUEST, requirements(), plan(), LABELS)
    assert result['request_constraints_joint'] and not result['acceptance_joint']


def test_evidence_must_come_from_raw_request():
    req = requirements(); req['sources'][0]['constraints'][0]['evidence'] = 'at exactly 180 degrees'
    with pytest.raises(ValueError, match='verbatim'):
        validate_requirements(REQUEST, req)


def test_after_finishing_compares_to_offset_not_onset():
    p = plan(); req = requirements()
    request = REQUEST.replace('Let them overlap.', 'The dog must start after the rain stops.')
    req['relations'] = [{'op': 'starts_after_end', 'a': 'dog', 'b': 'rain', 'evidence': 'The dog must start after the rain stops'}]
    assert not evaluate_natural_request(request, req, p, LABELS, completion_reasonable=True)['acceptance_joint']
    p['sources'][0]['activity'] = {'onset_sec': 10.1, 'offset_sec': 11.5}
    assert evaluate_natural_request(request, req, p, LABELS, completion_reasonable=True)['acceptance_joint']


def test_full_scene_uses_generated_duration_not_a_hidden_reference_duration():
    request = REQUEST + ' Keep the rain going throughout the clip.'
    req = requirements(); req['sources'][0]['constraints'].append({'op': 'full_scene', 'evidence': 'Keep the rain going throughout the clip'})
    p = plan()
    assert not evaluate_natural_request(request, req, p, LABELS, completion_reasonable=True)['acceptance_joint']
    p['sources'][1]['activity']['offset_sec'] = p['duration_sec']
    assert evaluate_natural_request(request, req, p, LABELS, completion_reasonable=True)['acceptance_joint']


def test_crossing_in_front_can_start_and_end_directly_to_the_sides():
    request = 'A motorcycle crosses in front of me from left to right.'
    req = {'schema': SCHEMA, 'output_order': None, 'sources': [{'key': 'bike', 'kind': 'sound',
           'core': 'A motorcycle engine', 'evidence': 'A motorcycle', 'constraints': [
               {'op': 'sector', 'point': 'start', 'value': 'left', 'evidence': 'from left'},
               {'op': 'sector', 'point': 'end', 'value': 'right', 'evidence': 'to right'},
               {'op': 'sector', 'point': 'path', 'value': 'front', 'evidence': 'crosses in front'}]}], 'relations': [], 'scene': []}
    p = {'sample_id': 'crossing', 'duration_sec': 10, 'room': {'type': 'outdoor'}, 'sources': [source('source_0', 'A motorcycle engine.', 90, 1, 8)]}
    p['sources'][0]['trajectory'] = {'type': 'linear', 'start': {'azimuth_deg': 90, 'elevation_deg': 0, 'distance_m': 5},
                                   'end': {'azimuth_deg': -90, 'elevation_deg': 0, 'distance_m': 5}}
    labels = {('bike', 'source_0'): True}
    assert evaluate_natural_request(request, req, p, labels, completion_reasonable=True)['acceptance_joint']
    p['sources'][0]['trajectory']['start']['azimuth_deg'] = 130
    p['sources'][0]['trajectory']['end']['azimuth_deg'] = -130
    assert not evaluate_natural_request(request, req, p, labels, completion_reasonable=True)['acceptance_joint']


def test_requested_event_duration_does_not_require_start_at_zero():
    from stable_audio_tools.data.sceneplan_generation_ar_natural_constraints import source_constraint_pass
    s = source('source_0', 'A clarinet playing.', 0, 2, 10)
    assert source_constraint_pass(s, {'op':'numeric','field':'event_duration_sec','value':8})
    s['activity']['offset_sec']=9
    assert not source_constraint_pass(s, {'op':'numeric','field':'event_duration_sec','value':8})


def test_direct_direction_checks_elevation_and_toward_allows_partial_travel():
    from stable_audio_tools.data.sceneplan_generation_ar_natural_constraints import source_constraint_pass
    s=source('source_0','Footsteps.',0,0,4)
    c={'op':'direct','point':'both','value':'front'}
    assert source_constraint_pass(s,c)
    s['trajectory']['position']['elevation_deg']=45
    assert not source_constraint_pass(s,c)
    s['trajectory']={'type':'linear','start':{'azimuth_deg':180,'elevation_deg':0,'distance_m':3},'end':{'azimuth_deg':120,'elevation_deg':0,'distance_m':3}}
    assert source_constraint_pass(s,{'op':'angular_direction','value':'front'})
    s['trajectory']['start'],s['trajectory']['end']=s['trajectory']['end'],s['trajectory']['start']
    assert not source_constraint_pass(s,{'op':'angular_direction','value':'front'})


def test_end_with_event_and_strict_upper_bound():
    from stable_audio_tools.data.sceneplan_generation_ar_natural_constraints import source_constraint_pass
    s=source('source_0','A gong.',180,9,10)
    assert source_constraint_pass(s,{'op':'ends_scene'},scene_duration=10)
    assert not source_constraint_pass(s,{'op':'ends_scene'},scene_duration=12)
    p=plan();req=requirements();request=REQUEST+' Keep it under twelve seconds.'
    req['scene']=[{'op':'duration_range','min':0,'max':12,'max_inclusive':False,'evidence':'under twelve seconds'}]
    assert not evaluate_natural_request(request,req,p,LABELS,completion_reasonable=True)['acceptance_joint']
    p['duration_sec']=11.9
    assert evaluate_natural_request(request,req,p,LABELS,completion_reasonable=True)['acceptance_joint']
