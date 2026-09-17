import copy
import pytest
import torch

from stable_audio_tools.paths import data_path
from stable_audio_tools.training.transfusion_opsd.editing_request_constraints import (
    bind_edit_target, parse_edit_request, request_field_targets, native_field_sites,
    frozen_text_targets, field_balanced_reference_kl,
)
from stable_audio_tools.training.transfusion_opsd.native_coarse_choice_retention import field_balanced_native_set_ce
from stable_audio_tools.training.transfusion_opsd.editing_binaural_retention import (
    FrozenKemarRenderer, binaural_features, binaural_distances, select_operation_rows,
)


@pytest.fixture(scope='module')
def codec():
    from stable_audio_tools.data.model_sceneplan_codec_v4 import ModelScenePlanCodecV4
    path = data_path('sceneplan_v2_1p124m/p11_single_turn_15s_v2/model_sceneplan_codec_v4')
    if not path.exists():
        pytest.skip('Native codec artifact not installed.')
    return ModelScenePlanCodecV4(path)


def scene():
    def source(sid, description, angle):
        return dict(source_id=sid, kind='sound', description=description, gain_db=0.,
            activity=dict(onset_sec=.5, offset_sec=3.5),
            trajectory=dict(type='linear', start=dict(azimuth_deg=angle, elevation_deg=5., distance_m=1.1),
                            end=dict(azimuth_deg=-35., elevation_deg=15., distance_m=2.1)))
    return dict(sample_id='binding', duration_sec=4., room=dict(type='dry'), sources=[
        source('source_0', 'A high-pitched warbling electronic tone plays a melody.', 70.),
        source('source_1', 'A deep resonant didgeridoo plays a continuous low-pitched drone.', 67.)])


def request():
    return ('Add the sound described as "A deep resonant didgeridoo plays a continuous low-pitched drone '
            'with a guttural vibrating timbre." moving from azimuth -168 degrees, elevation 5 degrees '
            'and distance 1.1 meters to azimuth -35 degrees, elevation 15 degrees and distance 2.1 meters '
            'between 0.5 and 3.5 seconds.')


def test_two_sounds_can_bind_by_distinctive_text_without_geometry():
    p = scene(); facts = parse_edit_request(request(), 'event_addition')
    bound = bind_edit_target(p, facts)
    assert bound['available'] and bound['source_id'] == 'source_1'
    assert not bound['acoustic_identity_certified']
    # Geometry is deliberately wrong; using it to bind would conceal the error.
    assert p['sources'][1]['trajectory']['start']['azimuth_deg'] == 67.
    p['sources'][0]['description'] = p['sources'][1]['description']
    assert not bind_edit_target(p, facts)['available']


def test_single_wrong_class_member_is_not_a_confident_identity():
    p = scene();p['sources'] = p['sources'][:1]
    assert not bind_edit_target(p, parse_edit_request(request(), 'event_addition'))['available']


def test_content_binding_survives_wrong_kind_and_corrects_native_kind(codec):
    p = scene()
    text = request().replace('Add the sound described as', 'Add the music described as')
    facts = parse_edit_request(text, 'event_addition')
    binding = bind_edit_target(p, facts)
    assert binding['available'] and not binding['kind_matches']
    assert binding['source_id'] == 'source_1'
    ids = codec.encode(p)['input_ids'].tolist(); p = codec.decode(ids)
    targets = request_field_targets(codec, ids, p, facts, binding, codec.allowed_next_ids)['targets']
    kind = next(t for t in targets if t['field'] == 'source_1/kind')
    assert ids[kind['position']] not in kind['acceptable_ids']
    from stable_audio_tools.data.model_sceneplan_codec_v3 import KIND_TOKENS
    assert kind['acceptable_ids'] == [codec._tid(KIND_TOKENS['music'])]
    assert any(t['field'] == 'source_1/start/azimuth' for t in targets)


def test_ambiguous_content_across_kinds_cannot_be_disambiguated_by_predicted_kind():
    p = scene();p['sources'][0]['description'] = p['sources'][1]['description']
    p['sources'][0]['kind'] = 'music'
    assert not bind_edit_target(p, parse_edit_request(request(), 'event_addition'))['available']


def test_wrong_requested_angle_gets_direct_legal_gradient(codec):
    p = scene();ids = codec.encode(p)['input_ids'].tolist();p = codec.decode(ids)
    facts = parse_edit_request(request(), 'event_addition')
    result = request_field_targets(codec, ids, p, facts, bind_edit_target(p, facts), codec.allowed_next_ids)
    target = next(t for t in result['targets'] if t['field'] == 'source_1/start/azimuth')
    assert ids[target['position']] not in target['acceptable_ids']
    assert codec.azimuth_ids[12] in target['acceptable_ids']  # -168 degrees
    assert not any(t['field'].startswith('source_0/') for t in result['targets'])
    logits = torch.zeros(len(ids) - 1, codec.vocab_size, requires_grad=True)
    field_balanced_native_set_ce(logits, [target]).backward()
    assert logits.grad[target['position'] - 1, codec.azimuth_ids[12]] < 0
    assert logits.grad[target['position'] - 1, ids[target['position']]] > 0
    assert any(t['field'].endswith('/onset_sec') for t in result['targets'])


def test_request_ranges_allow_small_error_and_wrap(codec):
    p = scene();p['sources'][1]['trajectory']['start']['azimuth_deg'] = -179.
    ids = codec.encode(p)['input_ids'].tolist();p = codec.decode(ids)
    facts = parse_edit_request(request().replace('-168 degrees', '179 degrees'), 'event_addition')
    targets = request_field_targets(codec, ids, p, facts, bind_edit_target(p, facts), codec.allowed_next_ids)['targets']
    t = next(t for t in targets if t['field'] == 'source_1/start/azimuth')
    assert ids[t['position']] in t['acceptable_ids']


def test_removal_has_negative_scope_without_inventing_a_source_count(codec):
    text = 'Eliminate the sound described as "A deep resonant didgeridoo plays a continuous low-pitched drone." for its entire active interval.'
    facts = parse_edit_request(text, 'event_removal');assert facts['removal']
    p = scene();ids = codec.encode(p)['input_ids'].tolist();p = codec.decode(ids)
    result = request_field_targets(codec, ids, p, facts, bind_edit_target(p, facts), codec.allowed_next_ids)
    assert result['targets'] == []
    assert result['unavailable'] == ['removal_requires_retained_source_count_evidence']


def test_text_reference_preserves_only_unedited_sources(codec):
    p = scene();ids = codec.encode(p)['input_ids'].tolist()
    teacher = torch.randn(len(ids) - 1, codec.vocab_size, requires_grad=True)
    targets = frozen_text_targets(codec, ids, teacher, codec.allowed_next_ids, excluded_sources=['source_1'])
    assert targets and all(t['field'].startswith('source_0/') for t in targets)
    student = torch.zeros_like(teacher, requires_grad=True)
    loss = field_balanced_reference_kl(student, targets);loss.backward()
    assert teacher.grad is None and float(student.grad.norm()) > 0


def test_binaural_mirror_is_detected_and_gradient_is_finite():
    try:
        FrozenKemarRenderer()
    except (FileNotFoundError, ModuleNotFoundError):
        pytest.skip('KEMAR benchmark suite is not installed.')
    t = torch.arange(8192) / 44100
    w = torch.sin(2 * torch.pi * 440 * t) + .3 * torch.sin(2 * torch.pi * 853 * t)
    truth = torch.stack([w, .7 * w, .1 * w, .5 * w])
    renderer = FrozenKemarRenderer()
    with torch.no_grad():target = binaural_features(renderer(truth))
    correct = binaural_distances(binaural_features(renderer(truth)), target)
    assert max(float(v) for v in correct.values()) < 1e-6
    mirror = truth.clone();mirror[1].neg_();mirror.requires_grad_(True)
    distances = binaural_distances(binaural_features(renderer(mirror)), target)
    loss = sum(distances.values());assert float(loss) > .01
    loss.backward();assert torch.isfinite(mirror.grad).all() and float(mirror.grad.norm()) > 0
    assert not list(renderer.parameters())


def test_renderer_matches_frozen_benchmark():
    try:
        renderer = FrozenKemarRenderer()
    except (FileNotFoundError, ModuleNotFoundError):
        pytest.skip('KEMAR benchmark suite is not installed.')
    from benchmark_audio_v1 import KemarFoaDecoder
    x = torch.randn(4, 8192) * .01
    expected = torch.from_numpy(KemarFoaDecoder().decode(x.numpy()))
    actual = renderer(x)
    torch.testing.assert_close(actual.double(), expected, atol=1e-7, rtol=1e-4)


@pytest.mark.parametrize('silent_target', [False, True])
def test_binaural_silent_bins_do_not_produce_nan_gradients(silent_target):
    prediction = torch.zeros(2, 4096, requires_grad=True)
    target = torch.zeros_like(prediction) if silent_target else torch.randn_like(prediction) * .01
    loss = sum(binaural_distances(binaural_features(prediction), binaural_features(target)).values())
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(prediction.grad).all()


def test_auxiliary_selection_keeps_distinct_operations_and_all_rows_eligible():
    metadata = [[dict(operation='event_addition') for _ in range(4)],
                [dict(operation='event_removal'), dict(operation='static_to_linear')]]
    seen = set()
    for step in range(12):
        selected = select_operation_rows(metadata, step, rank=0, count=2)
        assert len(selected) == 2
        assert len({metadata[a][b]['operation'] for a,b in selected}) == 2
        seen.update(selected)
    assert seen == {(0,i) for i in range(4)} | {(1,0),(1,1)}
