import copy
import pytest
import torch

from stable_audio_tools.paths import data_path
from stable_audio_tools.training.transfusion_opsd.execution_teacher_selection import same_plan_improvement_weights
from stable_audio_tools.training.transfusion_opsd.reference_prefix_retention import reference_prefix_targets
from stable_audio_tools.training.transfusion_opsd.editing_request_constraints import (
    parse_edit_request, bind_edit_target, field_balanced_reference_kl,
)


def execution(plan, seed, reward, qualified=True, semantic=.7):
    return dict(plan_index=plan, seed=seed, reward=reward, qualified_terminal=qualified,
                anchored_semantic=semantic, fixed_reference_semantic=.7,
                unchanged_windows=dict(available=False))


def test_teachers_are_selected_within_plan_without_borrowing_another_plans_target():
    rows = [execution(0, 1, -.1), execution(0, 2, -.5),
            execution(1, 1, -.8), execution(1, 2, -.9)]
    weights, report = same_plan_improvement_weights(rows, [.2, .8])
    assert weights == pytest.approx([.2, 0., .8, 0.])
    # Plan1's good sample is worse than both plan0 samples, yet its own plan
    # keeps its matching target. Cross-plan ranking must not relabel it.
    assert report['selected'] == 2


def test_qualification_and_fixed_semantic_floor_cannot_be_bought_by_direction():
    rows = [execution(0, 1, 1., semantic=.66), execution(0, 2, -.5)]
    assert same_plan_improvement_weights(rows, [1.])[0] == [0., 0.]
    rows[0] = execution(0, 1, 1., qualified=False)
    weights, _ = same_plan_improvement_weights(rows, [1.])
    assert weights[0] == 0.
    rows[0].pop('fixed_reference_semantic')
    assert same_plan_improvement_weights(rows, [1.])[0][0] == 0.


def test_selected_terminal_mass_does_not_inflate_sparse_teacher_coverage():
    rows = [execution(0, 1, -.1), execution(0, 2, -.5, qualified=False)]
    weights, report = same_plan_improvement_weights(rows, [.3])
    assert weights == pytest.approx([.15, 0.])
    assert report['groups'][0]['actual_mass'] <= report['groups'][0]['mass_ceiling']


@pytest.mark.parametrize('rows', [
    [execution(0, 1, -.1), execution(0, 2, -.1)],
    [execution(0, 1, -.1), execution(0, 1, -.3)],
    [execution(0, 1, -.1)],
])
def test_no_teacher_from_ties_or_repeated_or_single_noise(rows):
    assert sum(same_plan_improvement_weights(rows, [1.])[0]) == 0.


def test_nonfinite_execution_is_an_error():
    with pytest.raises(ValueError, match='Nonfinite'):
        same_plan_improvement_weights([execution(0, 1, float('nan'))], [1.])


@pytest.fixture(scope='module')
def codec():
    from stable_audio_tools.data.model_sceneplan_codec_v4 import ModelScenePlanCodecV4
    path = data_path('sceneplan_v2_1p124m/p11_single_turn_15s_v2/model_sceneplan_codec_v4')
    if not path.exists():
        pytest.skip('Native codec artifact is not installed.')
    return ModelScenePlanCodecV4(path)


def scene():
    def source(sid, text):
        return dict(source_id=sid, kind='sound', description=text, gain_db=0.,
                    activity=dict(onset_sec=.5, offset_sec=3.5),
                    trajectory=dict(type='static', position=dict(azimuth_deg=10., elevation_deg=0., distance_m=1.)))
    return dict(sample_id='test', duration_sec=4., room=dict(type='dry'), sources=[
        source('source_0', 'A sharp electronic alarm rings continuously.'),
        source('source_1', 'A deep resonant didgeridoo plays a low drone.')])


def targets(codec, request, operation):
    p = scene(); ids = codec.encode(p)['input_ids'].tolist(); p = codec.decode(ids)
    facts = parse_edit_request(request, operation)
    logits = torch.randn(len(ids) - 1, codec.vocab_size, requires_grad=True)
    result = reference_prefix_targets(codec, ids, p, facts, bind_edit_target(p, facts), logits, codec.allowed_next_ids)
    return result, logits


def test_reference_prefix_preserves_unedited_fields_and_allows_requested_changes(codec):
    request = ('Move the sound described as "A deep resonant didgeridoo plays a low drone." '
               'to azimuth -90 degrees, elevation 10 degrees and distance 2 meters.')
    result, teacher = targets(codec, request, 'stationary_spatial_relocation')
    fields = {h['field'] for h in result['structure'] + result['text']}
    assert 'source_0/kind' in fields and 'source_0/<description>' in fields
    assert 'source_1/kind' not in fields and 'source_1/<description>' not in fields
    assert 'source_1/position/azimuth' not in fields
    assert 'source_1/onset_sec' in fields
    student = torch.zeros_like(teacher, requires_grad=True)
    loss = field_balanced_reference_kl(student, result['structure'] + result['text'])
    loss.backward()
    assert teacher.grad is None and float(student.grad.norm()) > 0


def test_reference_does_not_reinforce_explicitly_removed_content_or_count(codec):
    request = 'Remove the sound described as "A deep resonant didgeridoo plays a low drone." entirely.'
    result, _ = targets(codec, request, 'event_removal')
    assert result['excluded_removed_sources'] == ['source_1']
    fields = {h['field'] for h in result['structure'] + result['text']}
    assert all(not f.startswith('source_1/') for f in fields)
    assert 'scene/num_sources' not in fields and 'source_0/<description>' in fields


def test_ambiguous_addition_is_not_a_fabricated_preserve_count_label(codec):
    request = ('Add the music described as "Bright rhythmic drums and a jazz trumpet." '
               'at azimuth 90 degrees.')
    result, _ = targets(codec, request, 'event_addition')
    assert not result['binding_available']
    assert 'scene/num_sources' not in {h['field'] for h in result['structure']}
