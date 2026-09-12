from collections import Counter
from copy import deepcopy
from pathlib import Path

import pytest

from stable_audio_tools.data.sceneplan_generation_ar_supervision import annotate_target_tokens


@pytest.fixture(scope='module')
def codec():
    path = Path('/mnt/sdb/audio_dataset/sceneplan_v2_1p124m/p11_single_turn_15s_v2/model_sceneplan_codec_v4')
    if not path.is_dir():
        pytest.skip('existing project codec artifact required')
    from stable_audio_tools.data.model_sceneplan_codec_v4 import ModelScenePlanCodecV4
    return ModelScenePlanCodecV4(path)


def plan():
    return {'sample_id': 'supervision', 'duration_sec': 10., 'room': {'type': 'outdoor'}, 'sources': [
        {'source_id': 'source_0', 'kind': 'sound', 'description': 'A dog barking.', 'gain_db': 0.,
         'activity': {'onset_sec': 1., 'offset_sec': 5.},
         'trajectory': {'type': 'static', 'position': {'azimuth_deg': 70., 'elevation_deg': 0., 'distance_m': 3.}}},
        {'source_id': 'source_1', 'kind': 'speech', 'speaker_description': 'A male voice.', 'transcript': 'Please come here.', 'gain_db': 0.,
         'activity': {'onset_sec': 6., 'offset_sec': 9.},
         'trajectory': {'type': 'linear', 'start': {'azimuth_deg': 0., 'elevation_deg': 0., 'distance_m': 8.},
                        'end': {'azimuth_deg': 0., 'elevation_deg': 0., 'distance_m': 2.}}},
    ]}


def provenance():
    fields = ['onset_sec', 'offset_sec', 'motion', 'core_text']
    fields += [f'{p}.{c}' for p in ('start', 'end') for c in ('azimuth_deg', 'elevation_deg', 'distance_m')]
    rows = []
    for key in ('dog', 'voice'):
        for field in fields + (['transcript'] if key == 'voice' else []):
            cs = [{'op': 'core_semantics' if field == 'core_text' else 'transcript'}] if field in ('core_text', 'transcript') else []
            rows.append({'source': key, 'field': field, 'origin': 'request_constrained' if cs else 'free_completion', 'constraints': cs})
    return rows + [{'source': None, 'field': field, 'origin': 'free_completion', 'constraints': []} for field in ('duration_sec', 'room')]


def constrain(rows, key, field, constraint):
    row = next(r for r in rows if r['source'] == key and r['field'] == field)
    row.update(origin='request_constrained', constraints=[constraint])


def annotate(codec, p, rows=None, binding=None, precise=False):
    tokens = codec.encode(p)['input_ids'].tolist()
    return tokens, annotate_target_tokens(codec, tokens, fully_specified=precise,
                                         provenance=provenance() if rows is None else rows,
                                         source_bindings={'dog': 'source_0', 'voice': 'source_1'} if binding is None else binding)


def test_free_witness_values_are_not_marked_as_requested_numbers(codec):
    tokens, labels = annotate(codec, plan())
    assert len(tokens) == len(labels)
    assert labels[6].category == 'count/requested'
    assert all(x.origin == 'free' for x in labels if x.family in ('time', 'space', 'room', 'motion'))
    assert all(x.origin == 'requested' for x in labels if x.family in ('core_text', 'kind', 'transcript'))
    assert labels[-1].category == 'structure/schema'
    assert labels[tokens.index(codec.token_to_id['<text_end>'])].category == 'structure/schema'


def test_precise_rendering_of_same_witness_keeps_full_numeric_supervision(codec):
    _, natural = annotate(codec, plan())
    _, precise = annotate(codec, plan(), precise=True)
    for a, b in zip(natural, precise):
        if a.family in ('time', 'space'):
            assert a.origin == 'free' and b.origin == 'explicit_numeric'


def test_event_duration_and_after_relation_do_not_fix_absolute_times(codec):
    rows = provenance()
    for field in ('onset_sec', 'offset_sec'):
        constrain(rows, 'dog', field, {'op': 'numeric', 'field': 'event_duration_sec', 'value': 4.})
    constrain(rows, 'voice', 'onset_sec', {'op': 'starts_after_end', 'a': 'voice', 'b': 'dog'})
    _, labels = annotate(codec, plan(), rows)
    assert all(x.origin == 'relative' for x in labels if x.source_id == 'source_0' and x.family == 'time')
    onset = next(x for x in labels if x.source_id == 'source_1' and x.fields == ('onset_sec',))
    assert onset.origin == 'relative'


def test_static_position_honors_constraint_on_either_endpoint(codec):
    rows = provenance()
    constrain(rows, 'dog', 'end.azimuth_deg', {'op': 'numeric', 'field': 'end.azimuth_deg', 'value': 70.})
    constrain(rows, 'dog', 'start.distance_m', {'op': 'distance_range', 'min': 1., 'max': 5.})
    _, labels = annotate(codec, plan(), rows)
    spatial = [x for x in labels if x.source_id == 'source_0' and x.family == 'space']
    assert [x.origin for x in spatial] == ['explicit_numeric', 'free', 'relative']
    assert spatial[0].fields == ('start.azimuth_deg', 'end.azimuth_deg')


def test_source_permutation_moves_all_supervision_with_its_content(codec):
    rows = provenance()
    constrain(rows, 'dog', 'onset_sec', {'op': 'numeric', 'field': 'onset_sec', 'value': 1.})
    original = plan(); swapped = deepcopy(original)
    swapped['sources'].reverse()
    for i, source in enumerate(swapped['sources']):
        source['source_id'] = f'source_{i}'
    _, a = annotate(codec, original, rows)
    _, b = annotate(codec, swapped, rows, {'dog': 'source_1', 'voice': 'source_0'})
    for sid_a, sid_b in [('source_0', 'source_1'), ('source_1', 'source_0')]:
        assert Counter((x.category, x.fields) for x in a if x.source_id == sid_a) == Counter((x.category, x.fields) for x in b if x.source_id == sid_b)


def test_missing_provenance_or_nonbijective_binding_fails_closed(codec):
    with pytest.raises(ValueError, match='Missing provenance'):
        annotate(codec, plan(), provenance()[1:])
    with pytest.raises(ValueError, match='bijection'):
        annotate(codec, plan(), binding={'dog': 'source_0', 'voice': 'source_0'})
    tokens = codec.encode(plan())['input_ids'].tolist()
    with pytest.raises(ValueError, match='require provenance'):
        annotate_target_tokens(codec, tokens, fully_specified=False)
