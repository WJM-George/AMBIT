import copy
import math

import pytest
import torch

from stable_audio_tools.data.model_sceneplan_codec_v4 import ModelScenePlanCodecV4, create_model_sceneplan_codec_v4_artifact
from stable_audio_tools.models.sceneplan_generation_ar_qualitative_head import ATTRIBUTES
from stable_audio_tools.training.transfusion_opsd.event_completion import (
    EventCompletionHead, completion_candidates, completion_geometry, apply_completion_actions)
from stable_audio_tools.training.transfusion_opsd.event_rewards import EventRequestReward, EventRequestFoaReward
from stable_audio_tools.training.transfusion_opsd.objectives import forward_kl
from stable_audio_tools.training.transfusion_opsd.execution_teacher import execution_teacher


@pytest.fixture(scope='module')
def codec(tmp_path_factory):
    root = tmp_path_factory.mktemp('event-opsd-codec') / 'codec'
    create_model_sceneplan_codec_v4_artifact(root)
    return ModelScenePlanCodecV4(root)


def example(codec):
    request = 'In dry acoustics, a dog barking stays static at my front left.'
    requirements = {'schema': 'generation_ar_natural_requirements_v2',
        'scene': [{'op': 'room', 'value': 'dry', 'evidence': 'dry acoustics'}],
        'sources': [{'key': 'requested_dog', 'kind': 'sound', 'core': 'a dog barking', 'evidence': 'a dog barking',
            'constraints': [{'op': 'motion', 'value': 'static', 'evidence': 'static'},
                {'op': 'compass', 'point': 'both', 'value': 'front_left', 'evidence': 'front left'}]}], 'relations': []}
    plan = codec.project_plan({'sample_id': 'example', 'duration_sec': 2., 'room': {'type': 'dry'},
        'sources': [{'source_id': 'source_0', 'kind': 'sound', 'description': 'a dog barking', 'gain_db': 0.,
            'activity': {'onset_sec': 0., 'offset_sec': 2.},
            'trajectory': {'type': 'static', 'position': {'azimuth_deg': 49., 'elevation_deg': 0., 'distance_m': 3.}}}]})
    traces = [{'field': 'description', 'generated_source_slot': 0, 'qualitative_control': {'labels':
        {'motion': 'static', 'start': 'front_left', 'end': 'front_left', 'onset': 'beginning', 'offset': 'ending', 'radial': 'free'}}}]
    return plan, traces, EventRequestReward(request, requirements)


def test_initial_completion_policy_has_exact_greedy_baseline_and_coupled_gradients(codec):
    plan, _, _ = example(codec)
    head = EventCompletionHead(hidden_dim=16, width=12)
    states = torch.randn(1, 1, 16, requires_grad=True)
    attributes = {key: torch.randn(1, 1, len(values), requires_grad=True) for key, values in ATTRIBUTES.items()}
    logits = head(states, attributes, completion_geometry(plan)[None])
    assert logits.argmax(-1).item() == 0
    # Current execution evidence changes the distribution at the *actual* new
    # decision. The loss reaches both AR states and learned attribute logits.
    legal = torch.ones(3, dtype=torch.bool)
    target, _, _, _ = execution_teacher(logits[0, 0], legal, [0, 1, 2], [0., 0., 1.],
        ar_temperature=1., temperature=.1, strength=.5)
    loss = forward_kl(logits, target[None, None], legal[None, None], temperature=1.)
    loss.backward()
    assert states.grad.norm() > 0
    assert all(value.grad.norm() > 0 for value in attributes.values())
    assert head.output.weight.grad.norm() > 0
    assert not target.requires_grad


def test_completion_alternatives_obey_request_and_preserve_all_other_fields(codec):
    plan, traces, reward = example(codec)
    alternatives, legal = completion_candidates(codec, plan, traces, 0)
    assert legal.all() and alternatives[0] == plan
    assert apply_completion_actions(codec, plan, traces, [0]) == plan
    for candidate in alternatives:
        assert reward.admissible(candidate)
        source = copy.deepcopy(candidate['sources'][0])
        source['trajectory'] = plan['sources'][0]['trajectory']
        assert source == plan['sources'][0]
    wrong = copy.deepcopy(plan)
    wrong['sources'][0]['trajectory']['position']['azimuth_deg'] = -90.
    assert not reward.admissible(wrong)


def test_unrequested_distance_and_timing_do_not_recover_hidden_numbers(codec):
    plan, _, reward = example(codec)
    candidate = copy.deepcopy(plan)
    candidate['sources'][0]['trajectory']['position']['distance_m'] = 8.
    candidate['sources'][0]['activity']['onset_sec'] = .2
    assert reward.admissible(candidate)
    assert reward.evaluate(candidate)['hidden_target_fields_compared'] is False


def test_unresolved_paraphrase_stays_unknown(codec):
    plan, _, reward = example(codec)
    plan['sources'][0]['description'] = 'a canine making barking noises'
    score = reward.evaluate(plan)
    assert score['semantic_pending']
    assert score['sources'][0]['core_semantics'] is None
    assert not reward.admissible(plan)


def test_spatial_feedback_targets_requested_sector_and_cannot_hide_silence(codec):
    plan, traces, request_reward = example(codec)
    alternatives, _ = completion_candidates(codec, plan, traces, 0)
    samples = round(plan['duration_sec'] * 44100)
    rewards = [EventRequestFoaReward(request_reward, p, model_num_samples=samples) for p in alternatives]
    time = torch.arange(samples) / 44100
    mono = .05 * torch.sin(2 * math.pi * 440 * time)
    angle = math.pi / 4
    audio = torch.stack((mono, mono * math.sin(angle), mono * 0, mono * math.cos(angle)))[None].requires_grad_(True)
    scores = [reward(audio) for reward in rewards]
    assert all(score == scores[0] for score in scores)  # Exact chosen angle is not the reward target.
    assert scores[0].utility > .9
    silence = rewards[0](torch.zeros_like(audio))
    assert silence.utility < scores[0].utility and silence.costs['silence'] > scores[0].costs['silence']
    rewards[0].differentiable(audio).backward()
    assert audio.grad is not None and torch.isfinite(audio.grad).all()


def test_task_teacher_only_strengthens_verified_requested_attributes(codec):
    from stable_audio_tools.training.transfusion_opsd.event_objectives import build_event_task_teacher, event_ar_loss
    from stable_audio_tools.training.transfusion_opsd.event_policy import EventProposal
    from stable_audio_tools.training.transfusion_opsd.adapters import GenerationObservation
    plan, traces, reward = example(codec)
    start = reward.request.index('a dog barking')
    traces[0].update(start=start, end=start + len('a dog barking') - 1)
    proposal = EventProposal(GenerationObservation('example', reward.request), (), traces, plan)
    output = {'inventory': {'count': torch.randn(1, 4, requires_grad=True),
        'kind': torch.randn(1, 4, 4, requires_grad=True),
        'start': torch.randn(1, 4, 3, len(reward.request), requires_grad=True),
        'end': torch.randn(1, 4, 3, len(reward.request), requires_grad=True)},
        'qualitative': {name: torch.randn(1, 1, len(values), requires_grad=True) for name, values in ATTRIBUTES.items()},
        'completion': torch.randn(1, 1, 3, requires_grad=True), 'completion_legal': torch.ones(1, 1, 3, dtype=torch.bool),
        'token_logits': torch.randn(1, 4, 12, requires_grad=True)}
    teacher = build_event_task_teacher(output, proposal, reward)
    # This request specified no relative radial movement or onset/offset phase.
    for name in ('radial', 'onset', 'offset'):
        assert torch.equal(teacher.heads['qualitative'][name], output['qualitative'][name])
    assert torch.equal(teacher.heads['inventory']['start'][:, :, 2], output['inventory']['start'][:, :, 2])
    loss = event_ar_loss(output, teacher)
    loss.backward()
    assert output['inventory']['count'].grad.norm() > 0
    assert output['qualitative']['start'].grad.norm() > 0
    assert not teacher.evidence['hidden_numeric_target']


def test_v2_does_not_turn_free_exact_onset_into_requested_boundary(codec):
    from stable_audio_tools.training.transfusion_opsd.event_rewards import EventRequestFoaRewardV2
    plan, _, request = example(codec)
    plan['sources'][0]['activity']['onset_sec'] = .3
    plan = codec.project_plan(plan)
    samples = round(plan['duration_sec'] * 44100)
    reward = EventRequestFoaRewardV2(request, plan, model_num_samples=samples)
    mono = .05 * torch.sin(torch.arange(samples) * .1)
    audio = torch.stack((mono, mono / 2 ** .5, mono * 0, mono / 2 ** .5))[None]
    score = reward(audio)
    assert score.costs['forbidden_activity_fraction'] == 0
    assert reward.diagnostics(audio)['activity_leakage'] > 0
    assert score.costs['source_presence_failure'] == 0
    assert reward(torch.zeros_like(audio)).costs['source_presence_failure'] == 1


def test_constrained_teacher_preserves_content_and_independent_validation_can_reject():
    from stable_audio_tools.training.transfusion_opsd.event_teachers import constrained_event_teacher
    from stable_audio_tools.training.transfusion_opsd.execution_teacher import qualify_execution_teacher
    from stable_audio_tools.training.transfusion_opsd.objectives import RewardScore
    construction = [[RewardScore(u, {'content': c}) for u, c in zip(values, [0., 0., .1])]
        for values in ([.8, .7, .81], [.9, .7, .95])]
    teacher, p, q, evidence = constrained_event_teacher(torch.zeros(3), torch.ones(3, dtype=torch.bool), construction)
    assert q is not None and not teacher.requires_grad
    assert q[0] > p[0] and q[1] < p[1] and q[2] <= p[2] + 1e-12
    assert torch.allclose(teacher.softmax(-1), q, atol=1e-15, rtol=0)
    assert not evidence['validation_data_used_in_solver']
    assert qualify_execution_teacher(p, q, construction, construction, min_gain=1e-6, min_utility=0., cost_limits={})[0]
    validation = [[RewardScore(u, {'content': c}) for u, c in zip([.8, .7, .81], [.1, 0., 0.])]]
    assert not qualify_execution_teacher(p, q, construction, validation, min_gain=1e-6, min_utility=0., cost_limits={})[0]


def test_constrained_teacher_does_not_lower_protection_to_make_infeasible_gain_pass():
    from stable_audio_tools.training.transfusion_opsd.event_teachers import constrained_event_teacher
    from stable_audio_tools.training.transfusion_opsd.objectives import RewardScore
    scores = [[RewardScore(value, {'content': value}) for value in (.2, .5, .8)]] * 2
    teacher, _, target, evidence = constrained_event_teacher(torch.zeros(3), torch.ones(3, dtype=torch.bool), scores)
    assert teacher is None and target is None
    assert evidence['reason'] == 'no_feasible_protected_construction_teacher'


def test_content_aware_target_ascends_spatial_reward_without_increasing_conflicting_content_cost():
    from stable_audio_tools.training.transfusion_opsd.event_latent_target import content_aware_latent_objective
    value = torch.tensor([[[1., 2.]]], requires_grad=True)
    utility = lambda x: x.sum()
    cost = lambda x: x[..., 0].sum()
    objective = content_aware_latent_objective(value, decode=lambda x: x, spatial_reward=utility, content_cost=cost)
    direction, = torch.autograd.grad(objective, value)
    candidate = value.detach() + .01 * direction
    assert utility(candidate) > utility(value) and cost(candidate) < cost(value)
