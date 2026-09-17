import copy
from types import SimpleNamespace

import pytest

from stable_audio_tools.paths import data_path
from stable_audio_tools.training.transfusion_opsd.native_token_alignment import validate_native_plan_tokens
from stable_audio_tools.training.transfusion_opsd.native_coarse_choice_retention import native_azimuth_cone_targets
from stable_audio_tools.training.transfusion_opsd.request_grounded_text_retention import request_quoted_text_targets
from stable_audio_tools.training.transfusion_opsd.editing_spatial_retention import propose_current_decision


@pytest.fixture(scope='module')
def codec():
    from stable_audio_tools.data.model_sceneplan_codec_v4 import ModelScenePlanCodecV4
    path = data_path('sceneplan_v2_1p124m/p11_single_turn_15s_v2/model_sceneplan_codec_v4')
    if not path.exists():
        pytest.skip('Native codec artifact is not installed.')
    return ModelScenePlanCodecV4(path)


def native_plan(codec):
    from stable_audio_tools.models.sceneplan_transfusion_editing_pipeline import _align_decoded_sceneplan_to_audio_duration
    plan = dict(sample_id='alignment', duration_sec=3., room=dict(type='dry'), sources=[
        dict(source_id='source_0', kind='music', description='a quiet accordion melody',
             activity=dict(onset_sec=0., offset_sec=3.), gain_db=0.,
             trajectory=dict(type='static', position=dict(azimuth_deg=0., elevation_deg=0., distance_m=1.)))])
    ids = codec.encode(plan)['input_ids'].tolist()
    decoded = codec.decode(ids, sample_id=plan['sample_id'])
    return _align_decoded_sceneplan_to_audio_duration(decoded, 3.), ids


def alternative_segmentation(codec, ids):
    processor = codec.text_processor
    start = ids.index(codec._tid('<text_begin>')) + 1
    end = ids.index(codec._tid('<text_end>'), start)
    for position in range(start, end):
        piece = processor.id_to_piece(ids[position] - codec.text_offset)
        for split in range(1, len(piece)):
            strings = (piece[:split], piece[split:])
            pieces = [processor.piece_to_id(s) for s in strings]
            if any(processor.id_to_piece(i) != s for i, s in zip(pieces, strings)):
                continue
            replacement = [codec.text_offset + i for i in pieces]
            if not set(replacement) <= set(codec.text_ids):
                continue
            alternative = ids[:position] + replacement + ids[position + 1:]
            if codec.decode(alternative) == codec.decode(ids):
                return alternative
    raise AssertionError('Expected a legal alternative text segmentation in the native codec.')


def test_aliases_keep_actual_cone_and_quoted_prefix_positions(codec):
    plan, canonical = native_plan(codec)
    ids = alternative_segmentation(codec, canonical)
    assert ids != codec.encode(plan)['input_ids'].tolist()
    assert validate_native_plan_tokens(codec, ids, plan) == canonical
    cones = native_azimuth_cone_targets(codec, ids, plan, codec.allowed_next_ids, radius_deg=12.)
    assert cones[0]['position'] == ids.index(codec._tid('<azimuth_bin>')) + 1
    assert cones[0]['position'] != canonical.index(codec._tid('<azimuth_bin>')) + 1
    quoted = request_quoted_text_targets(codec, ids, plan,
        'Add the music described as "a quiet accordion melody".', codec.allowed_next_ids)
    assert quoted['targets']
    for target in quoted['targets']:
        assert target['token_id'] == ids[target['position']]
        assert target['token_id'] in codec.allowed_next_ids(ids[:target['position']])


@pytest.mark.parametrize('field', ['direction', 'content', 'source'])
def test_mismatched_plan_is_still_rejected(codec, field):
    plan, canonical = native_plan(codec)
    ids = alternative_segmentation(codec, canonical)
    wrong = copy.deepcopy(plan)
    if field == 'direction':
        wrong['sources'][0]['trajectory']['position']['azimuth_deg'] = 30.
    elif field == 'content':
        wrong['sources'][0]['description'] = 'a barking dog'
    else:
        wrong['sources'][0]['source_id'] = 'source_1'
    with pytest.raises(ValueError, match='different executable fields'):
        validate_native_plan_tokens(codec, ids, wrong)


def test_malformed_native_sequence_is_not_accepted(codec):
    plan, ids = native_plan(codec)
    with pytest.raises(ValueError):
        validate_native_plan_tokens(codec, ids[:-1], plan)


def test_proposal_changes_one_native_choice_without_reencoding_prefix(codec):
    plan, canonical = native_plan(codec)
    ids = alternative_segmentation(codec, canonical)
    adapter = SimpleNamespace(codec=codec, allowed_next_ids=lambda obs, prefix: codec.allowed_next_ids(prefix))
    observation = SimpleNamespace(model_num_samples=3 * 44100)
    facts = dict(kind='music', azimuths=[10.], elevations=[], distances=[], activity=None)
    result = propose_current_decision(adapter, observation, plan, ids, facts, step=0)
    assert result is not None
    assert result['position'] == ids.index(codec._tid('<azimuth_bin>')) + 1
    assert result['prefix'] == ids[:result['position']]
    changed = ids.copy()
    changed[result['position']] = result['choice_ids'][1]
    assert sum(a != b for a, b in zip(ids, changed)) == 1
    validate_native_plan_tokens(codec, changed, result['plans'][1])
    assert result['plans'][1]['sources'][0]['description'] == plan['sources'][0]['description']
    assert result['plans'][1]['sources'][0]['trajectory']['position']['azimuth_deg'] == 10.


def test_canonical_proposal_keeps_previous_choice_and_prefix(codec):
    plan, ids = native_plan(codec)
    adapter = SimpleNamespace(codec=codec, allowed_next_ids=lambda obs, prefix: codec.allowed_next_ids(prefix))
    result = propose_current_decision(adapter, SimpleNamespace(model_num_samples=3 * 44100), plan, ids,
        dict(kind='music', azimuths=[10.], elevations=[], distances=[], activity=None), step=0)
    assert result['prefix'] == ids[:result['position']]
    expected = copy.deepcopy(plan)
    expected['sources'][0]['trajectory']['position']['azimuth_deg'] = 10.
    encoded = codec.encode(expected)['input_ids'].tolist()
    differences = [i for i, (a, b) in enumerate(zip(ids, encoded)) if a != b]
    assert differences == [result['position']]
    assert result['choice_ids'] == [ids[differences[0]], encoded[differences[0]]]
