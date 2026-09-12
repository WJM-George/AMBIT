import copy
import itertools

import torch
import pytest

from stable_audio_tools.models.sceneplan_generation_ar_qualitative_head import (
    ATTRIBUTES, CENTERS, PHASES, QualitativeExecutionHead, annotation_targets, complete_from_logits,
)


def logits(**preferred):
    return {name: [10. if index == preferred.get(name, 0) else 0. for index in range(len(values))]
        for name, values in ATTRIBUTES.items()}


def test_time_grid_is_valid_and_selected_categories_are_satisfied_even_for_short_scenes():
    for duration, onset, offset, speech in itertools.product((1, 5, 32, 432, 648), range(5), range(5), (False, True)):
        result = complete_from_logits(logits(onset=onset, offset=offset), duration, seed_key='time-test', speech=speech)
        on, off = result['onset_frame'], result['offset_frame']
        assert 0 <= on < off <= duration
        for field, frame in [('onset', on), ('offset', off)]:
            low, high = PHASES[ATTRIBUTES[field].index(result['labels'][field])]
            assert low - 1e-8 <= frame / duration <= high + 1e-8


def test_seeded_free_completions_preserve_compass_radial_direction_and_nonzero_motion():
    for start, end, radial in itertools.product(range(8), range(8), range(3)):
        scores = logits(motion=1, start=start, end=end, onset=1, offset=3, radial=radial)
        result = complete_from_logits(scores, 432, seed_key='first')
        assert result == complete_from_logits(scores, 432, seed_key='first')
        assert result != complete_from_logits(scores, 432, seed_key='second')
        trajectory = result['trajectory']; assert trajectory['start'] != trajectory['end']
        for point, index in [('start', start), ('end', end)]:
            assert abs((trajectory[point]['azimuth_deg'] - CENTERS[index] + 180.) % 360. - 180.) <= 14.
        delta = trajectory['end']['distance_m'] - trajectory['start']['distance_m']
        if radial == 1: assert delta < -.25
        elif radial == 2: assert delta > .25
        else: assert delta == 0.
    static = complete_from_logits(logits(motion=0, radial=2), 432, seed_key='static')
    assert static['trajectory']['type'] == 'static' and set(static['trajectory']) == {'type', 'position'}


def test_training_labels_depend_on_requested_categories_and_never_free_witness_fields():
    req = {'sources': [{'constraints': [
        {'op': 'motion', 'value': 'linear'},
        {'op': 'compass', 'point': 'start', 'value': 'left'},
        {'op': 'compass', 'point': 'end', 'value': 'right'},
        {'op': 'time_phase', 'field': 'onset_sec', 'value': 'early'},
        {'op': 'time_phase', 'field': 'offset_sec', 'value': 'ending'},
        {'op': 'distance_change', 'value': 'approaching'}]}]}
    assert annotation_targets(req) == [{'motion': 1, 'start': 2, 'end': 6, 'onset': 1, 'offset': 4, 'radial': 1}]
    changed = copy.deepcopy(req); changed['unused_witness'] = {'distance_m': 40., 'onset_sec': 2.3}
    assert annotation_targets(changed) == annotation_targets(req)


@pytest.mark.parametrize('use_ar_query', [False, True])
@pytest.mark.parametrize('attribute_queries', [False, True])
def test_source_conditioned_head_is_invariant_to_request_padding(use_ar_query, attribute_queries):
    torch.manual_seed(42)
    model = QualitativeExecutionHead(hidden_dim=8, width=8, heads=2, layers=2, use_ar_query=use_ar_query, attribute_queries=attribute_queries).eval()
    context = torch.randn(1, 5, 8); hidden = torch.randn(1, 2, 8)
    first = model(hidden, torch.tensor([[0, 1]]), torch.tensor([[0, 3]]), torch.tensor([[1, 4]]),
        context, torch.ones(1, 5).bool())
    padded = torch.cat([torch.randn(1, 2, 8), context, torch.randn(1, 3, 8)], 1)
    second = model(hidden, torch.tensor([[0, 1]]), torch.tensor([[2, 5]]), torch.tensor([[3, 6]]),
        padded, torch.tensor([[0, 0, 1, 1, 1, 1, 1, 0, 0, 0]]).bool())
    for name in ATTRIBUTES: torch.testing.assert_close(first[name], second[name], atol=1e-6, rtol=1e-6)
    if not use_ar_query:
        changed = model(torch.randn_like(hidden) * 50, torch.tensor([[0, 1]]), torch.tensor([[0, 3]]), torch.tensor([[1, 4]]),
            context, torch.ones(1, 5).bool())
        for name in ATTRIBUTES: torch.testing.assert_close(first[name], changed[name], atol=0., rtol=0.)


def test_attention_targets_are_outside_the_model_and_cannot_change_predictions():
    torch.manual_seed(9)
    model = QualitativeExecutionHead(hidden_dim=8, width=8, heads=2, layers=2, attribute_queries=True)
    inputs = (torch.randn(1, 2, 8), torch.tensor([[0, 1]]), torch.tensor([[0, 3]]), torch.tensor([[1, 4]]),
        torch.randn(1, 5, 8), torch.ones(1, 5).bool())
    plain = model(*inputs); scored, attention = model(*inputs, return_attention=True)
    for name in ATTRIBUTES: torch.testing.assert_close(plain[name], scored[name], atol=0., rtol=0.)
    assert attention.shape == (1, 2, 6, 5)
    torch.testing.assert_close(attention.sum(-1), torch.ones(1, 2, 6))
    target = torch.zeros_like(attention); target[..., 0] = 1
    loss = -(attention * target).sum(-1).log().mean(); loss.backward()
    assert model.blocks[-1].query.weight.grad.abs().sum() > 0
def test_source_local_attention_masks_other_source_blocks():
    import torch
    from stable_audio_tools.models.sceneplan_generation_ar_qualitative_head import QualitativeExecutionHead
    torch.manual_seed(11)
    head = QualitativeExecutionHead(hidden_dim=16, width=16, heads=4, layers=2,
        use_ar_query=False, attribute_queries=True, source_local_attention=True).eval()
    context = torch.randn(1, 20, 16)
    hidden = torch.randn(1, 2, 16)
    start = torch.tensor([[3, 13]])
    end = torch.tensor([[5, 15]])
    spans = torch.tensor([[[1, 9], [10, 18]]])
    _, attention = head(hidden, torch.zeros(1, 2).long(), start, end, context,
        torch.ones(1, 20).bool(), source_spans=spans, return_attention=True)
    assert not attention[0, 0, :, :1].any() and not attention[0, 0, :, 10:].any()
    assert not attention[0, 1, :, :10].any() and not attention[0, 1, :, 19:].any()
    torch.testing.assert_close(attention.sum(-1), torch.ones(1, 2, 6))
