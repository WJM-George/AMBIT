import copy
import itertools

from stable_audio_tools.data.sceneplan_generation_ar_exact_proof import (
    exact_core_labels,
    prove_exact_satisfaction,
)
from stable_audio_tools.data.sceneplan_generation_ar_natural_constraints import evaluate_natural_request
from stable_audio_tools.data.sceneplan_generation_template100 import render_pair


def fixture():
    position = {'azimuth_deg': 90., 'elevation_deg': 0., 'distance_m': 3.}
    plan = {'sample_id': 'proof_fixture', 'duration_sec': 10., 'room': {'type': 'dry'},
        'sources': [{'source_id': 'source_0', 'kind': 'sound', 'description': 'A dog barking.',
            'gain_db': 0., 'activity': {'onset_sec': 0., 'offset_sec': 10.},
            'trajectory': {'type': 'static', 'position': position}},
        {'source_id': 'source_1', 'kind': 'sound', 'description': 'Rain falling steadily.',
            'gain_db': 0., 'activity': {'onset_sec': 1., 'offset_sec': 8.5},
            'trajectory': {'type': 'linear', 'start': {**position, 'azimuth_deg': 135.},
                'end': {**position, 'azimuth_deg': -135., 'distance_m': 1.}}}]}
    request, requirements = render_pair(plan, 'train_000')
    return request, requirements, plan


def test_complete_proof_allows_free_numbers_and_unordered_sources():
    request, requirements, prediction = fixture()
    prediction['duration_sec'] = 5.
    for source in prediction['sources']:
        source['activity'] = {key: value / 2 for key, value in source['activity'].items()}
    prediction['sources'][1]['trajectory']['start'].update(azimuth_deg=130., distance_m=4.)
    prediction['sources'][1]['trajectory']['end'].update(azimuth_deg=-130., distance_m=2.)
    prediction['sources'].reverse()
    for index, source in enumerate(prediction['sources']):
        source['source_id'] = f'source_{index}'
    proof = prove_exact_satisfaction(request, requirements, prediction)
    assert proof and list(proof['assignment'].values()) == ['source_1', 'source_0']
    assert proof['completion_review'] == 'SEPARATE'


def test_paraphrase_is_left_unknown_for_the_judge():
    request, requirements, prediction = fixture()
    prediction['sources'][0]['description'] = 'Barking from a dog.'
    assert prove_exact_satisfaction(request, requirements, prediction) is None
    labels = exact_core_labels(requirements, prediction)
    assert (requirements['sources'][0]['key'], 'source_0') not in labels
    labels[requirements['sources'][0]['key'], 'source_0'] = True
    assert evaluate_natural_request(request, requirements, prediction, labels)['request_constraints_joint']


def test_incomplete_or_wrongly_bound_fields_do_not_shortcut_semantic_review():
    request, requirements, original = fixture()
    for mutation in ('missing', 'extra', 'motion', 'direction', 'time', 'misbinding'):
        prediction = copy.deepcopy(original)
        if mutation == 'missing': prediction['sources'].pop()
        elif mutation == 'extra':
            extra = copy.deepcopy(prediction['sources'][0]); extra['source_id'] = 'source_2'
            prediction['sources'].append(extra)
        elif mutation == 'motion':
            trajectory = prediction['sources'][1]['trajectory']
            prediction['sources'][1]['trajectory'] = {'type': 'static', 'position': trajectory['start']}
        elif mutation == 'direction': prediction['sources'][1]['trajectory']['end']['azimuth_deg'] = 0.
        elif mutation == 'time': prediction['sources'][1]['activity']['onset_sec'] = 8.
        else:
            a, b = prediction['sources']
            a['description'], b['description'] = b['description'], a['description']
        assert prove_exact_satisfaction(request, requirements, prediction) is None, mutation
    assert prove_exact_satisfaction(request, requirements, None) is None


def test_all_unknown_semantic_completions_preserve_proven_boolean_metrics():
    request, requirements, prediction = fixture()
    assert prove_exact_satisfaction(request, requirements, prediction)
    fixed = exact_core_labels(requirements, prediction)
    unknown = [(ref['key'], source['source_id']) for ref in requirements['sources']
        for source in prediction['sources'] if (ref['key'], source['source_id']) not in fixed]
    for values in itertools.product((False, True), repeat=len(unknown)):
        scored = evaluate_natural_request(request, requirements, prediction,
            {**fixed, **dict(zip(unknown, values))}, completion_reasonable=True)
        assert scored['request_constraints_joint'] and scored['acceptance_joint']
        assert not scored['semantic_pending']
        assert all(source['all_requested_pass'] for source in scored['sources'])
        assert all(check['pass'] for source in scored['sources'] for check in source['constraints'])
