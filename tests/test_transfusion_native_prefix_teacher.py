import itertools

import pytest
import torch

from stable_audio_tools.training.transfusion_opsd.native_prefix_teacher import (
    DecisionDistribution, build_prefix_teacher, path_nodes, prefix_teacher_loss,
)


def paths(root, tails):
    return [[DecisionDistribution('root', root, first),
             DecisionDistribution('tail_'+str(first), tails[first], second)]
            for first, second in [(0, 0), (1, 1)]]


def probabilities(root, tails):
    return root.softmax(0)[:, None]*tails.softmax(1)


def fixed():
    return (torch.tensor([.6, -.2, .1], dtype=torch.float64),
            torch.tensor([[.1, -.3], [-.2, .4], [.7, -.1]], dtype=torch.float64))


def test_matches_independent_complete_joint_teacher_and_initial_gradient():
    r, t = fixed()
    q = torch.tensor([.2, .8], dtype=torch.float64)
    teacher = build_prefix_teacher(paths(r, t), q.log())
    reference = probabilities(r, t)
    mass = reference[0, 0]+reference[1, 1]
    full = reference.clone()
    full[0, 0], full[1, 1] = mass*q[0], mass*q[1]
    assert float(full.sum()) == pytest.approx(1.)
    assert torch.equal(full[2], reference[2])
    a, b = r.clone().requires_grad_(), t.clone().requires_grad_()
    loss, lp = prefix_teacher_loss(paths(a, b), teacher)
    joint = (full*(full.log()-probabilities(a, b).log())).sum()/mass
    conditional = (q*(q.log()-lp.log_softmax(0))).sum()
    assert float(loss) == pytest.approx(float(joint), abs=2e-7)
    assert float(loss) == pytest.approx(float(conditional), abs=2e-7)
    g = torch.autograd.grad(loss, (a, b), retain_graph=True)
    for comparison in (joint, conditional):
        h = torch.autograd.grad(comparison, (a, b), retain_graph=True)
        assert all(torch.allclose(x, y, atol=1e-12, rtol=1e-12) for x, y in zip(g, h))


def test_penalizes_unmeasured_mass_escape_that_conditional_kl_cannot_see():
    r, t = fixed()
    q = torch.tensor([.2, .8], dtype=torch.float64)
    teacher = build_prefix_teacher(paths(r, t), q.log())
    before, lp0 = prefix_teacher_loss(paths(r, t), teacher)
    moved = r.clone()
    moved[2] += 2.
    after, lp1 = prefix_teacher_loss(paths(moved, t), teacher)
    assert torch.allclose(lp0.log_softmax(0), lp1.log_softmax(0), atol=1e-14, rtol=0)
    assert float(after) > float(before)+.1


def test_unvisited_prefix_is_an_explicit_coverage_limit():
    r, t = fixed()
    teacher = build_prefix_teacher(paths(r, t))
    moved = t.clone()
    moved[2, 0] -= 5.
    loss, _ = prefix_teacher_loss(paths(r, moved), teacher)
    p, s = probabilities(r, t), probabilities(r, moved)
    assert abs(float(loss)) < 1e-12
    assert float((p*(p.log()-s.log())).sum()) > .1


def test_reference_fixed_point_and_teacher_stops_gradient():
    r, t = fixed()
    r.requires_grad_()
    t.requires_grad_()
    teacher = build_prefix_teacher(paths(r, t), normalization='prefix_occupancy')
    assert all(not x.probabilities.requires_grad for x in teacher.targets.values())
    assert teacher.normalizer == pytest.approx(sum(x.prefix_mass for x in teacher.targets.values()))
    loss, _ = prefix_teacher_loss(paths(r, t), teacher)
    loss.backward()
    assert abs(float(loss)) < 1e-12
    assert float(r.grad.abs().max()) < 1e-12
    assert float(t.grad.abs().max()) < 1e-12


def test_zero_teacher_mass_keeps_student_denominator():
    r = torch.tensor([.1, .3, -.2], dtype=torch.float64)
    observed = [[DecisionDistribution('r', r, i)] for i in range(3)]
    teacher = build_prefix_teacher(observed, torch.tensor([0., -torch.inf, -torch.inf]))
    student = r.clone().requires_grad_()
    loss, _ = prefix_teacher_loss([[DecisionDistribution('r', student, i)] for i in range(3)], teacher)
    gradient, = torch.autograd.grad(loss, student)
    assert torch.allclose(gradient, student.softmax(0)-student.new_tensor([1, 0, 0]), atol=1e-12, rtol=0)


def test_rejects_duplicate_incomplete_and_inconsistent_paths():
    r, t = fixed()
    p = paths(r, t)
    with pytest.raises(ValueError, match='distinct complete'):
        build_prefix_teacher([p[0], p[0]])
    with pytest.raises(ValueError, match='distinct complete'):
        build_prefix_teacher([p[0], p[0][:1]])
    p[1][0].site = 'different root support'
    with pytest.raises(ValueError, match='same native support'):
        build_prefix_teacher(p)
