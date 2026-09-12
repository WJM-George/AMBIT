import copy
import re

from stable_audio_tools.data.sceneplan_generation_template100 import catalog, render_pair, template_for
from stable_audio_tools.data.sceneplan_generation_ar_natural_constraints import validate_requirements, source_constraint_pass


def plan():
    p = {'azimuth_deg': 140., 'elevation_deg': 7., 'distance_m': 3.}
    return {'sample_id': 'private-id-not-a-request', 'duration_sec': 10., 'room': {'type': 'dry'},
            'sources': [{'source_id': 'source_0', 'kind': 'speech', 'speaker_description': 'a calm adult woman',
                         'transcript': 'Meet me at 7.', 'gain_db': 0.,
                         'activity': {'onset_sec': 1., 'offset_sec': 6.},
                         'trajectory': {'type': 'linear', 'start': p,
                             'end': {'azimuth_deg': -140., 'elevation_deg': -7., 'distance_m': 1.}}},
                        {'source_id': 'source_1', 'kind': 'sound', 'description': 'A dog barking.', 'gain_db': 0.,
                         'activity': {'onset_sec': 0., 'offset_sec': 9.8},
                         'trajectory': {'type': 'static', 'position': {**p, 'azimuth_deg': 90.}}}]}


def test_catalog_counts_and_split_surfaces_are_disjoint():
    records = catalog()
    assert [sum(r['split'] == s for r in records) for s in ('train', 'validation', 'test')] == [100, 10, 5]
    assert len({render_pair(plan(), r['id'])[0] for r in records}) == 115
    for split, size in [('train', 100), ('validation', 10), ('test', 5)]:
        assert len({template_for(split, i) for i in range(size)}) == size
        assert [template_for(split, i) for i in range(size)] == [template_for(split, i) for i in range(size)]


def test_every_template_preserves_speech_and_gt_satisfies_its_qualitative_constraints():
    value = plan()
    for recipe in catalog():
        request, req = render_pair(value, recipe['id'])
        assert 'Meet me at 7.' in request and 'a calm adult woman' in request
        assert not re.search(r'\d|azimuth|elevation|seconds|meters|source_\d|ScenePlan|private-id', request.replace('Meet me at 7.', ''))
        validate_requirements(request, req)
        for ref, gt in zip(req['sources'], value['sources']):
            assert all(source_constraint_pass(gt, c, scene_duration=value['duration_sec']) for c in ref['constraints'])


def test_reasonable_changed_numbers_pass_but_wrong_source_direction_and_phase_fail():
    value = plan(); request, req = render_pair(value, 'train_000')
    speech = copy.deepcopy(value['sources'][0])
    speech['trajectory']['start'].update(azimuth_deg=130., distance_m=4., elevation_deg=0.)
    speech['trajectory']['end'].update(azimuth_deg=-130., distance_m=2., elevation_deg=0.)
    speech['activity'] = {'onset_sec': .5, 'offset_sec': 3.}
    # A shorter scene and different free coordinates still satisfy the text.
    assert all(source_constraint_pass(speech, c, scene_duration=5.) for c in req['sources'][0]['constraints'])
    speech['trajectory']['end']['azimuth_deg'] = 0.
    end = next(c for c in req['sources'][0]['constraints'] if c['op'] == 'compass' and c['point'] == 'end')
    assert not source_constraint_pass(speech, end, scene_duration=5.)
    onset = next(c for c in req['sources'][0]['constraints'] if c['op'] == 'time_phase' and c['field'] == 'onset_sec')
    speech['activity']['onset_sec'] = 4.
    assert not source_constraint_pass(speech, onset, scene_duration=5.)
