import copy
import torch

from stable_audio_tools.training.transfusion_opsd.event_refinement import refinement_candidates, REFINEMENT_ACTIONS
from test_transfusion_opsd_event import codec, example


def test_teacher_queries_preserve_actual_proposal_and_requested_non_numeric_semantics(codec):
    plan, traces, verifier = example(codec)
    candidates, legal = refinement_candidates(codec, plan, traces, 0)
    assert len(candidates) == len(REFINEMENT_ACTIONS) == 7 and legal.all()
    assert candidates[0] == plan and all(verifier.admissible(value) for value in candidates)
    for candidate in candidates:
        before, after = copy.deepcopy(plan), copy.deepcopy(candidate)
        before['sources'][0].pop('trajectory')
        after['sources'][0].pop('trajectory')
        assert before == after
    for index in [5, 6]:
        assert candidates[index]['sources'][0]['trajectory']['position']['azimuth_deg'] == plan['sources'][0]['trajectory']['position']['azimuth_deg']
    assert candidates[5]['sources'][0]['trajectory']['position']['distance_m'] < plan['sources'][0]['trajectory']['position']['distance_m']
    assert candidates[6]['sources'][0]['trajectory']['position']['distance_m'] > plan['sources'][0]['trajectory']['position']['distance_m']


def test_expanded_output_keeps_greedy_baseline_and_reaches_actual_source_states():
    from stable_audio_tools.training.transfusion_opsd.event_completion import EventCompletionHead
    from stable_audio_tools.training.transfusion_opsd.event_refinement_policy import install_refinement_output
    from stable_audio_tools.models.sceneplan_generation_ar_qualitative_head import ATTRIBUTES
    head = EventCompletionHead(hidden_dim=16, width=12)
    state = torch.get_rng_state().clone()
    install_refinement_output(head)
    assert torch.equal(state, torch.get_rng_state())
    source = torch.randn(2, 1, 16, requires_grad=True)
    attributes = {key: torch.randn(2, 1, len(values), requires_grad=True) for key, values in ATTRIBUTES.items()}
    logits = head(source, attributes, torch.zeros(2, 1, 8))
    assert logits.shape == (2, 1, 7) and (logits.argmax(-1) == 0).all()
    loss = -logits.log_softmax(-1)[..., 5].mean()
    loss.backward()
    assert source.grad.norm() > 0 and all(value.grad.norm() > 0 for value in attributes.values())
