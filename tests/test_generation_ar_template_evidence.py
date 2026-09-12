from stable_audio_tools.data.sceneplan_generation_ar_template_evidence import template_evidence_spans, focused_motion_evidence
from stable_audio_tools.data.sceneplan_generation_template100 import catalog, render_pair


def test_producer_trace_handles_all_forms_repeated_controls_and_control_text_inside_content():
    description = 'A voice repeats "Have it begin near the beginning and stop near the end." quietly.'
    plan = {'sample_id': 'synthetic_training_trace', 'duration_sec': 10., 'room': {'type': 'dry'}, 'sources': []}
    for index in range(4):
        plan['sources'].append({'source_id': f'source_{index}', 'kind': 'sound', 'description': description,
            'gain_db': 0., 'activity': {'onset_sec': 0., 'offset_sec': 10.},
            'trajectory': {'type': 'static', 'position': {'azimuth_deg': 90., 'elevation_deg': 0., 'distance_m': 2.}}})
    for recipe in catalog():
        request, requirements = render_pair(plan, recipe['id']); traces = template_evidence_spans(request, requirements, recipe)
        assert len(traces) == 4 and len({row['activity'] for row in traces}) == 4
        for trace in traces:
            for field in ('activity', 'motion', 'identity'):
                assert trace['block'][0] <= trace[field][0] <= trace[field][1] <= trace['block'][1]
            for field in ('activity', 'motion'):
                assert trace[field][1] < trace['identity'][0] or trace[field][0] > trace['identity'][1]


def test_focused_radial_evidence_stays_with_its_source_despite_quoted_duplicate():
    cue = 'It should move farther away from me as it goes.'
    plan = {'sample_id': 'synthetic_radial_trace', 'duration_sec': 10., 'room': {'type': 'dry'}, 'sources': []}
    for index in range(2):
        plan['sources'].append({'source_id': f'source_{index}', 'kind': 'sound', 'description': 'A voice says ' + cue,
            'gain_db': 0., 'activity': {'onset_sec': 0., 'offset_sec': 10.},
            'trajectory': {'type': 'linear',
                'start': {'azimuth_deg': 90., 'elevation_deg': 0., 'distance_m': 2.},
                'end': {'azimuth_deg': -90., 'elevation_deg': 0., 'distance_m': 4.}}})
    for recipe in catalog():
        request, requirements = render_pair(plan, recipe['id'])
        traces = template_evidence_spans(request, requirements, recipe)
        focused = [focused_motion_evidence(request, source, trace)
            for source, trace in zip(requirements['sources'], traces)]
        assert focused[0]['radial'] != focused[1]['radial']
        for trace, spans in zip(traces, focused):
            assert request[spans['radial'][0]:spans['radial'][1] + 1] == cue
            assert spans['start'][1] < spans['radial'][0]
            assert trace['motion'][0] <= spans['radial'][0] <= spans['radial'][1] <= trace['motion'][1]
