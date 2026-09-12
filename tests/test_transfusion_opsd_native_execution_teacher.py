from dataclasses import replace
import math

import pytest
import torch
from types import SimpleNamespace

from stable_audio_tools.training.transfusion_opsd.native_execution_teacher import (
    NativeDecisionSite, PairedExecutionComparison, build_native_execution_teacher,
    native_execution_kl,
    native_free_distillation,
)


def fixture(*, risk=True):
    site = NativeDecisionSite.from_request('speech', 'Say the complete sentence.', (1, 5), (10, 20, 30, 40))
    rows = [PairedExecutionComparison(token, seed,
        'observed_regression' if risk and token == 20 and seed == 102 else 'observed_nonregression')
        for token in (10, 20, 30) for seed in (101, 102)]
    logits = torch.tensor([3., 2.9, 2., 1.], dtype=torch.double, requires_grad=True)
    args = dict(site=site, executor_fingerprint='a' * 64, reference_token=10,
        paired_seeds=(101, 102), comparisons=rows, evidence_scope='Transcript-only observed comparisons.')
    return logits, args


def test_no_observed_execution_failure_does_not_force_a_new_plan():
    logits, args = fixture(risk=False)
    teacher = build_native_execution_teacher(logits, **args)
    assert torch.equal(logits.detach(), teacher.target_logits)
    assert not teacher.target_logits.requires_grad


def test_risk_reduces_tested_odds_without_promoting_untested_native_choices():
    logits, args = fixture()
    teacher = build_native_execution_teacher(logits, **args)
    p, q = logits.detach().softmax(0), teacher.target_logits.softmax(0)
    assert q.sum().item() == pytest.approx(1., abs=1e-14)
    assert q[3].item() == pytest.approx(p[3].item(), abs=1e-14)
    assert q[1] < p[1]
    assert (q[1]/q[0] / (p[1]/p[0])).item() == pytest.approx(math.exp(-.5))
    assert (q[0]/q[2]).item() == pytest.approx((p[0]/p[2]).item())
    assert (teacher.target_logits-logits.detach()).abs().max() <= 1


def test_teacher_is_stopped_and_student_gradient_discourages_observed_failure():
    reference, args = fixture()
    teacher = build_native_execution_teacher(reference, **args)
    student = reference.detach().clone().requires_grad_()
    loss = native_execution_kl(student, teacher, site=args['site'], round_executor_fingerprint='a'*64)
    loss.backward()
    assert loss.item() > 0 and student.grad[1] > 0
    assert reference.grad is None and teacher.target_logits.grad is None
    assert student.grad[3].item() == pytest.approx(0., abs=1e-14)
    assert student.grad.sum().item() == pytest.approx(0., abs=1e-14)


def test_unknown_is_recorded_without_inventing_a_failure_label():
    logits, args = fixture(risk=False)
    args['comparisons'] = [replace(row, status='uncertain') if row.token == 20 else row
        for row in args['comparisons']]
    teacher = build_native_execution_teacher(logits, **args)
    assert (20, 0., 1.) in teacher.measured_risks
    assert torch.equal(teacher.target_logits, logits.detach())


@pytest.mark.parametrize('fault', ['missing_noise', 'duplicate', 'illegal', 'reference', 'new_noise'])
def test_incomplete_or_mismatched_execution_evidence_is_rejected(fault):
    logits, args = fixture()
    rows = args['comparisons']
    if fault == 'missing_noise': rows.pop()
    elif fault == 'duplicate': rows.append(rows[-1])
    elif fault == 'illegal': rows[-1] = replace(rows[-1], token=999)
    elif fault == 'reference': rows[0] = replace(rows[0], status='observed_regression')
    else: rows[-1] = replace(rows[-1], seed=999)
    with pytest.raises(ValueError): build_native_execution_teacher(logits, **args)


@pytest.mark.parametrize('fault', ['request', 'prefix', 'support_order', 'executor'])
def test_execution_feedback_cannot_silently_cross_decisions_or_executor_rounds(fault):
    logits, args = fixture()
    teacher = build_native_execution_teacher(logits, **args)
    site, executor = args['site'], 'a'*64
    if fault == 'request': site = replace(site, request_sha256='b'*64)
    elif fault == 'prefix': site = replace(site, prefix=(1, 6))
    elif fault == 'support_order': site = replace(site, legal_ids=(20, 10, 30, 40))
    else: executor = 'b'*64
    with pytest.raises(ValueError, match='identity differs'):
        native_execution_kl(logits, teacher, site=site, round_executor_fingerprint=executor)


def native_fixture():
    logits, args = fixture()
    table = torch.nn.Parameter(torch.stack([logits.detach(), logits.detach()]))
    def forward(tokens, mask, context, context_mask):
        return table[tokens[0, -1]-5][None, None].expand(1, tokens.shape[1], -1)
    bundle = SimpleNamespace(ar=forward, encode_event_requests=lambda *a, **k: (None, None))
    policy = SimpleNamespace(bundle=bundle, device=torch.device('cpu'))
    site = NativeDecisionSite.from_request('speech', 'Say the complete sentence.', (1, 5), (0, 1, 2, 3))
    choices = [SimpleNamespace(prefix=(1, value), legal_ids=site.legal_ids,
        selected_token=0, teacher_logits=logits.detach().clone()) for value in (5, 6)]
    proposal = SimpleNamespace(observation=SimpleNamespace(sample_id='speech', request='Say the complete sentence.'),
        free_decisions=choices)
    rows = [replace(row, token=args['site'].legal_ids.index(row.token)) for row in args['comparisons']]
    teacher = build_native_execution_teacher(logits, **dict(args, site=site, comparisons=rows, reference_token=0))
    return policy, proposal, table, teacher


def test_native_integration_preserves_uncredited_decisions_and_stops_the_teacher():
    policy, proposal, table, teacher = native_fixture()
    loss, retention, details = native_free_distillation(policy, proposal,
        execution_teachers=[teacher], round_executor_fingerprint='a'*64)
    assert loss > 0 and retention == 0
    assert [x['execution_credit'] for x in details] == [True, False]
    loss.backward()
    assert table.grad[0, 1] > 0 and table.grad[1].abs().max() < 1e-14
    assert teacher.target_logits.grad is None


def test_native_integration_without_credit_matches_full_original_distribution_kl():
    policy, proposal, table, teacher = native_fixture()
    with torch.no_grad(): table[1, 0] -= .3
    loss, retention, details = native_free_distillation(policy, proposal)
    expected = torch.stack([torch.nn.functional.kl_div(row.double().log_softmax(0),
        decision.teacher_logits.double().softmax(0), reduction='sum')
        for row, decision in zip(table, proposal.free_decisions)]).mean()
    assert float(loss) == pytest.approx(float(expected), abs=1e-9)
    assert loss == retention and all(not x['execution_credit'] for x in details)


@pytest.mark.parametrize('fault', ['unobserved_prefix', 'duplicate'])
def test_native_integration_rejects_unused_or_duplicated_supervision(fault):
    policy, proposal, table, teacher = native_fixture()
    teachers = ([replace(teacher, site=replace(teacher.site, prefix=(1, 7)))]
        if fault == 'unobserved_prefix' else [teacher, teacher])
    with pytest.raises(ValueError):
        native_free_distillation(policy, proposal, execution_teachers=teachers, round_executor_fingerprint='a'*64)
