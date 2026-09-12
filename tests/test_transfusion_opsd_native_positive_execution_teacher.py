import pytest
import torch

from stable_audio_tools.training.transfusion_opsd.native_execution_teacher import NativeDecisionSite
from stable_audio_tools.training.transfusion_opsd.native_positive_execution_teacher import (
    PositiveExecutionComparison as Comparison, build_native_positive_execution_teacher,
    native_positive_site_objective)


SITE = NativeDecisionSite.from_request('engine', 'An engine in front.', (1, 5), (470, 474, 478, 479, 480))
PRIOR = torch.tensor([-4., -6., -3., 0., -8.], dtype=torch.float64)


def build(**overrides):
    kwargs = dict(site=SITE, executor_fingerprint='a'*64, observer_fingerprint='b'*64,
        evidence_sha256='c'*64, reference_token=479, paired_seeds=(10, 20),
        comparisons=[Comparison(t, s, g, protection) for t, gains, protection in
            [(479, (0., 0.), 'passed'), (474, (.01, .02), 'passed'),
             (478, (.04, -.01), 'passed'), (470, (.03, .03), 'uncertain')]
            for s, g in zip((10, 20), gains)])
    kwargs.update(overrides)
    return build_native_positive_execution_teacher(PRIOR.requires_grad_(True), **kwargs)


def test_positive_credit_preserves_unknown_unqualified_and_unmeasured_mass():
    teacher = build()
    p, q = PRIOR.softmax(0), teacher.target_logits.softmax(0)
    assert teacher.qualified_gains == ((474, .015),)
    assert SITE.legal_ids[int(q.argmax())] == 474
    torch.testing.assert_close(p[[0, 2, 4]], q[[0, 2, 4]], rtol=1e-14, atol=1e-15)
    torch.testing.assert_close(p[[1, 3]].sum(), q[[1, 3]].sum())
    assert teacher.target_logits.grad_fn is None and not teacher.target_logits.requires_grad


def test_intended_change_has_no_original_argmax_penalty_or_retention_gate():
    teacher = build()
    student = teacher.target_logits.clone().requires_grad_()
    loss = native_positive_site_objective(student, PRIOR, site=SITE, selected_token=479,
        teacher=teacher, round_executor_fingerprint='a'*64, observer_fingerprint='b'*64)
    assert not loss['retained'] and loss['reference_kl'] > .5
    assert loss['retained_kl'] == loss['retained_margin'] == 0
    assert abs(float(loss['positive'])) < 1e-7
    untaught = native_positive_site_objective(student, PRIOR, site=SITE, selected_token=479)
    assert untaught['retained'] and untaught['retained_kl'] > .5 and untaught['retained_margin'] > 0


def test_student_receives_positive_gradient_and_teacher_stops():
    teacher = build()
    student = PRIOR.detach().clone().requires_grad_()
    loss = native_positive_site_objective(student, PRIOR, site=SITE, selected_token=479,
        teacher=teacher, round_executor_fingerprint='a'*64, observer_fingerprint='b'*64)
    loss['positive'].backward()
    assert torch.isfinite(student.grad).all() and student.grad[1] < 0 and student.grad[3] > 0
    assert PRIOR.grad is None


@pytest.mark.parametrize('field,value', [('round_executor_fingerprint','d'*64), ('observer_fingerprint','d'*64),
    ('site', NativeDecisionSite.from_request('engine', 'Different request.', (1,5), SITE.legal_ids))])
def test_stale_or_foreign_teacher_rejected(field, value):
    kwargs = dict(site=SITE, selected_token=479, teacher=build(), round_executor_fingerprint='a'*64,
                  observer_fingerprint='b'*64)
    kwargs[field] = value
    with pytest.raises(ValueError):
        native_positive_site_objective(PRIOR, PRIOR, **kwargs)


def test_partial_panel_and_all_uncertain_panels_rejected():
    with pytest.raises(ValueError, match='complete paired'):
        build(comparisons=[Comparison(479,10,0.,'passed'), Comparison(479,20,0.,'passed'), Comparison(474,10,.01,'passed')])
    with pytest.raises(ValueError, match='No protected improvement'):
        build(comparisons=[Comparison(t,s,g,status) for t,g,status in [(479,0.,'passed'),(474,.02,'uncertain')]
                           for s in (10,20)])


@pytest.mark.parametrize('rho', [0., 1., float('nan')])
def test_invalid_mixture_rejected(rho):
    with pytest.raises(ValueError):
        build(positive_mass_fraction=rho)
