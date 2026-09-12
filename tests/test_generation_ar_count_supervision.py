import os
from pathlib import Path

import pytest

from stable_audio_tools.data.sceneplan_generation_ar_count_supervision import count_prefix_candidates, choose_count_prefix
from stable_audio_tools.data.sceneplan_generation_ar_natural_constraints import SCHEMA, FRAME_SECONDS


@pytest.fixture(scope='module')
def codec():
    path = Path(os.environ.get("AMBIT_DATA_ROOT", "data") + "/sceneplan_v2_1p124m/p11_single_turn_15s_v2/model_sceneplan_codec_v4")
    if not path.is_dir():
        pytest.skip('existing project codec artifact required')
    from stable_audio_tools.data.model_sceneplan_codec_v4 import ModelScenePlanCodecV4
    return ModelScenePlanCodecV4(path)


REQUEST = 'Only rain, outdoors for 14 seconds, audible until 13 seconds.'


def data(codec):
    plan = {'sample_id': 'count', 'duration_sec': 14., 'room': {'type': 'outdoor'}, 'sources': [
        {'source_id': 'source_0', 'kind': 'sound', 'description': 'Rain falling.', 'gain_db': 0.,
         'activity': {'onset_sec': 0., 'offset_sec': 13.},
         'trajectory': {'type': 'static', 'position': {'azimuth_deg': 0., 'elevation_deg': 0., 'distance_m': 2.}}}]}
    req = {'schema': SCHEMA, 'output_order': None, 'scene': [], 'relations': [],
           'sources': [{'key': 'rain', 'kind': 'sound', 'core': 'Rain falling.', 'evidence': 'rain', 'constraints': []}]}
    return codec.encode(plan)['input_ids'].tolist(), req


def test_free_prefixes_do_not_contain_the_count_label_or_change_target(codec):
    tokens, req = data(codec); before = tokens.copy()
    prefixes = count_prefix_candidates(codec, tokens, REQUEST, req)
    assert len(prefixes) == 16 and tokens == before
    assert all(len(p) == 6 and p[-1] == codec.token_to_id['<num_sources>'] for p in prefixes)
    assert all(p[i] == tokens[i] for p in prefixes for i in (0, 1, 3, 5))
    assert all(tokens[6] not in p for p in prefixes)


def test_explicit_headers_remain_exactly_the_same(codec):
    tokens, req = data(codec)
    req['scene'] = [{'op': 'numeric', 'field': 'duration_sec', 'value': 14., 'evidence': '14 seconds'},
                    {'op': 'room', 'value': 'outdoor', 'evidence': 'outdoors'}]
    assert count_prefix_candidates(codec, tokens, REQUEST, req) == [tokens[:6]]


def test_explicit_event_end_bounds_an_otherwise_free_clip_duration(codec):
    tokens, req = data(codec)
    req['sources'][0]['constraints'] = [{'op': 'numeric', 'field': 'offset_sec', 'value': 13., 'evidence': 'until 13 seconds'}]
    prefixes = count_prefix_candidates(codec, tokens, REQUEST, req)
    assert len(prefixes) == 4
    assert all(codec.frame_ids.index(p[2]) * FRAME_SECONDS >= 13. for p in prefixes)


def test_mislabeled_count_or_contradictory_header_fails(codec):
    tokens, req = data(codec); tokens[6] = codec.token_to_id['<num_sources_2>']
    with pytest.raises(ValueError, match='Count label'):
        count_prefix_candidates(codec, tokens, REQUEST, req)
    tokens, req = data(codec); req['scene'] = [{'op': 'room', 'value': 'dry', 'evidence': 'outdoors'}]
    with pytest.raises(ValueError, match='contradicts'):
        count_prefix_candidates(codec, tokens, REQUEST, req)


def test_sampling_is_stateless_and_resume_reproducible(codec):
    tokens, req = data(codec); prefixes = count_prefix_candidates(codec, tokens, REQUEST, req)
    a = [choose_count_prefix(prefixes, step=i, row=2, parent_index=10, view=3) for i in range(100)]
    resumed = [choose_count_prefix(prefixes, step=i, row=2, parent_index=10, view=3) for i in range(37, 100)]
    assert a[37:] == resumed and len({tuple(x) for x in a}) > 1
