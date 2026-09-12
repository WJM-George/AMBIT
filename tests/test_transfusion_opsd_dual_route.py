from dataclasses import replace

import pytest
import torch
from torch import nn

from stable_audio_tools.training.transfusion_opsd.dual_route_batch import RouteLossTerm, dual_route_loss_closure
from stable_audio_tools.training.transfusion_opsd.expected_execution_teacher import expected_execution_teacher, qualify_expected_teacher
from stable_audio_tools.training.transfusion_opsd.objectives import RewardScore
from stable_audio_tools.training.transfusion_opsd.shared_step import joint_adam_step, _cpu_clone, checkpoint_rng_state


class TinyShared(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(.2, dtype=torch.float64))
        self.s = nn.Parameter(torch.tensor(.1, dtype=torch.float64))
        self.d = nn.Parameter(torch.tensor(.3, dtype=torch.float64))

    def partition(self):
        return {'ar_private': [('a', self.a)], 'shared': [('s', self.s)], 'dit_private': [('d', self.d)]}


def terms(model):
    return ([RouteLossTerm('ar-only', 0, lambda: (model.a + model.s - 1).square(), 'execution_teacher'),
             RouteLossTerm('ar-other', 0, lambda: (model.a + .5 * model.s - .9).square(), 'task_teacher')],
            [RouteLossTerm('dit-only', 0, lambda: (model.d + model.s - 1).square(), 'positive_target', 1.3),
             RouteLossTerm('dit-other', 0, lambda: (model.d + .4 * model.s - .8).square(), 'positive_target', .6)])


def equal(a, b):
    if isinstance(a, torch.Tensor):
        return torch.equal(a.cpu(), b.cpu())
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal(a[key], b[key]) for key in a)
    if isinstance(a, (tuple, list)):
        return type(a) is type(b) and len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


def test_checkpointed_disjoint_microbatches_match_full_objectives_and_gradients():
    model = TinyShared()
    ar, dit = terms(model)
    gradients = []
    for use_checkpoint in (False, True):
        closure = dual_route_loss_closure(ar, dit, model_version=0, checkpoint_terms=use_checkpoint)
        a, d, _ = closure()
        assert torch.allclose(d, (dit[0].loss() * 1.3 + dit[1].loss() * .6) / 2)
        ga = torch.autograd.grad(a, list(model.parameters()), retain_graph=True, allow_unused=True)
        gd = torch.autograd.grad(d, list(model.parameters()), allow_unused=True)
        gradients.append((ga, gd))
        with torch.no_grad():
            aa, dd, _ = closure()
        assert torch.equal(a.detach(), aa) and torch.equal(d.detach(), dd)
        assert not aa.requires_grad and not dd.requires_grad
    assert equal(gradients[0], gradients[1])


def test_independent_route_pools_commit_both_and_restore_nonempty_adam_on_rejection():
    model = TinyShared()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=0., foreach=False)
    ar, dit = terms(model)
    closure = dual_route_loss_closure(ar, dit, model_version=0)
    result = joint_adam_step(model, model.partition(), optimizer, closure, validate_execution=lambda: True)
    assert result.committed and result.ar_after < result.ar_before and result.dit_after < result.dit_before
    assert closure.evidence['execution_coupled']
    saved = _cpu_clone(model.state_dict()), _cpu_clone(optimizer.state_dict()), checkpoint_rng_state()
    assert saved[1]['state']
    result = joint_adam_step(model, model.partition(), optimizer, closure, validate_execution=lambda: False)
    assert not result.committed
    assert equal(saved, (model.state_dict(), optimizer.state_dict(), checkpoint_rng_state()))


def test_missing_route_stale_teacher_and_duplicate_evidence_fail_before_fitting():
    ar, dit = terms(TinyShared())
    with pytest.raises(ValueError, match='nonempty'):
        dual_route_loss_closure(ar, [], model_version=0)
    with pytest.raises(ValueError, match='stale'):
        dual_route_loss_closure(ar, dit, model_version=1)
    with pytest.raises(ValueError, match='duplicate'):
        dual_route_loss_closure([ar[0], ar[0]], dit, model_version=0)
    closure = dual_route_loss_closure([replace(x, kind='task_teacher') for x in ar], dit, model_version=0)
    assert not closure.evidence['execution_coupled']


def test_importance_estimator_keeps_zero_contributions_in_declared_sample_count():
    ar, dit = terms(TinyShared())
    closure = dual_route_loss_closure(ar, dit[:1], model_version=0, dit_denominator=4)
    _, loss, _ = closure()
    assert torch.allclose(loss, dit[0].loss() * dit[0].weight / 4)
    with pytest.raises(ValueError, match='denominators'):
        dual_route_loss_closure(ar, dit, model_version=0, dit_denominator=1)


def test_noise_marginal_teacher_reports_individual_cost_regressions_without_claiming_pointwise_safety():
    # Action 1 improves expected content while one noise has a content regression.
    construction = [[RewardScore(0., {'content': 1.}), RewardScore(1., {'content': value})]
        for value in (1.1, .7, .6, .8)]
    validation = [[RewardScore(0., {'content': 1.}), RewardScore(.8, {'content': value})]
        for value in (1.05, .8, .7, .8)]
    teacher, p, q, construction_evidence = expected_execution_teacher(torch.zeros(2), torch.ones(2, dtype=torch.bool), construction)
    assert q is not None and q[1] > p[1] and not teacher.requires_grad
    assert torch.allclose(teacher.softmax(-1), q, atol=1e-15, rtol=0)
    okay, evidence = qualify_expected_teacher(p, q, construction, validation)
    assert okay and evidence['splits']['validation']['per_noise_cost_regressions']['content'] == 1
    assert not evidence['population_guarantee_claimed'] and not evidence['every_noise_protection_claimed']
    assert construction_evidence['construction_noises'] == 4 and not construction_evidence['validation_used_in_solver']
    bad_validation = [[RewardScore(0., {'content': 0.}), RewardScore(.8, {'content': .1})]] * 4
    assert not qualify_expected_teacher(p, q, construction, bad_validation)[0]


def test_expected_teacher_cannot_trade_one_protection_metric_for_another():
    rows = [[RewardScore(0., {'content': 0., 'asr': 1.}), RewardScore(1., {'content': .1, 'asr': 0.})]] * 4
    teacher, _, target, evidence = expected_execution_teacher(torch.zeros(2), torch.ones(2, dtype=torch.bool), rows)
    assert teacher is None and target is None and evidence['reason'] == 'no_feasible_expected_teacher'


def test_expected_teacher_rejects_missing_validation_cost_and_illegal_mass():
    rows = [[RewardScore(0., {'content': 1.}), RewardScore(1., {'content': 0.}), RewardScore(100., {'content': 0.})]] * 4
    _, p, q, _ = expected_execution_teacher(torch.zeros(3), torch.tensor([True, True, False]), rows)
    assert p[2] == q[2] == 0
    with pytest.raises(ValueError, match='omitted'):
        qualify_expected_teacher(p, q, rows, [[RewardScore(x.utility, {}) for x in row] for row in rows])
