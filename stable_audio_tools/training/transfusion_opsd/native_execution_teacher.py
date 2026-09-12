"""Bounded execution-risk distillation at an actual native planning decision.

Only paired, observed comparisons supply credit. Untested native choices keep
their original probability mass. This empirical teacher is scoped to one
request, decision site and frozen executor for a training round; it does not
certify a candidate or replace full output protection and fresh evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math

import torch


@dataclass(frozen=True)
class NativeDecisionSite:
    sample_id: str
    request_sha256: str
    prefix: tuple[int, ...]
    legal_ids: tuple[int, ...]

    @classmethod
    def from_request(cls, sample_id, request, prefix, legal_ids):
        return cls(sample_id, hashlib.sha256(request.encode('utf-8')).hexdigest(),
                   tuple(prefix), tuple(legal_ids))

    def __post_init__(self):
        if (not self.sample_id or len(self.request_sha256) != 64
                or not self.prefix or len(self.legal_ids) < 2
                or len(set(self.legal_ids)) != len(self.legal_ids)
                or any(type(x) is not int or x < 0 for x in (*self.prefix, *self.legal_ids))):
            raise ValueError('A native decision needs an identified request, prefix and unique legal support.')


@dataclass(frozen=True)
class PairedExecutionComparison:
    token: int
    seed: int
    status: str


@dataclass(frozen=True)
class NativeExecutionTeacher:
    site: NativeDecisionSite
    executor_fingerprint: str
    evidence_scope: str
    paired_seeds: tuple[int, ...]
    reference_token: int
    target_logits: torch.Tensor
    # These are empirical certain-failure and unknown fractions, not risk estimates
    # with population confidence guarantees. Zero failures is not proven safety.
    measured_risks: tuple[tuple[int, float, float], ...]
    max_logit_shift: float


def build_native_execution_teacher(reference_logits, *, site, executor_fingerprint,
        reference_token, paired_seeds, comparisons, evidence_scope, max_logit_shift=1.):
    """Reweight measured choices, keeping their total prior mass fixed.

    On measured support M, q(a) = p(M) softmax(log p(a) - lambda r(a));
    outside M, q(a) = p(a). Unknown comparisons remain explicitly unknown and
    contribute no certain-failure label. All choices need the same paired
    noises, including the reference. Teacher tensors always stop gradients.
    """
    if (reference_logits.ndim != 1 or reference_logits.numel() != len(site.legal_ids)
            or not reference_logits.is_floating_point() or not torch.isfinite(reference_logits).all()):
        raise ValueError('Teacher logits must span the complete native legal support.')
    if (not isinstance(executor_fingerprint, str) or len(executor_fingerprint) != 64
            or not evidence_scope or not math.isfinite(max_logit_shift) or not 0 < max_logit_shift <= 1):
        raise ValueError('Declare executor identity, evidence scope and a bounded positive tilt.')
    seeds = tuple(paired_seeds)
    if (len(seeds) < 2 or len(set(seeds)) != len(seeds)
            or any(type(seed) is not int or seed < 0 for seed in seeds)):
        raise ValueError('At least two unique paired noises are required.')
    rows = {}
    for item in comparisons:
        if (item.token not in site.legal_ids or item.seed not in seeds
                or item.status not in {'observed_regression', 'observed_nonregression', 'uncertain'}):
            raise ValueError('Execution comparison is outside the declared support or status contract.')
        key = (item.token, item.seed)
        if key in rows:
            raise ValueError('Duplicate paired execution comparison.')
        rows[key] = item.status
    measured = tuple(token for token in site.legal_ids if any(key[0] == token for key in rows))
    if len(measured) < 2 or reference_token not in measured:
        raise ValueError('Paired evidence needs the original reference and at least one alternative.')
    for token in measured:
        if {seed for action, seed in rows if action == token} != set(seeds):
            raise ValueError('Every measured choice needs the complete paired-noise panel.')
    if any(rows[reference_token, seed] != 'observed_nonregression' for seed in seeds):
        raise ValueError('The reference must be the same observed audio on each paired noise.')
    risks = tuple((token,
        sum(rows[token, seed] == 'observed_regression' for seed in seeds) / len(seeds),
        sum(rows[token, seed] == 'uncertain' for seed in seeds) / len(seeds)) for token in measured)
    fixed = reference_logits.detach().double().clone()
    target = fixed.clone()
    if any(risk > 0 for _, risk, _ in risks):
        ids = [site.legal_ids.index(token) for token in measured]
        penalty = fixed.new_tensor([risk for _, risk, _ in risks]) * max_logit_shift
        tilted = fixed[ids] - penalty
        # Equal partition mass on M also preserves the whole partition function.
        target[ids] = tilted + torch.logsumexp(fixed[ids], 0) - torch.logsumexp(tilted, 0)
    return NativeExecutionTeacher(site, executor_fingerprint, evidence_scope, seeds,
        reference_token, target, risks, max_logit_shift)


def native_execution_kl(student_logits, teacher, *, site, round_executor_fingerprint):
    """Train the real native logits; reject a teacher from a different round/site.

    The caller supplies the frozen executor identity declared for this round,
    not a stale global label. Within-round student updates need not keep the
    executor frozen. Refreshing supervision requires new execution evidence.
    """
    if site != teacher.site or round_executor_fingerprint != teacher.executor_fingerprint:
        raise ValueError('Execution teacher request, prefix, support or executor identity differs.')
    if (student_logits.ndim != 1 or student_logits.shape != teacher.target_logits.shape
            or not student_logits.is_floating_point() or not torch.isfinite(student_logits).all()):
        raise ValueError('Student must expose the same finite native legal logits.')
    target = teacher.target_logits.detach().to(student_logits.device).double()
    if not torch.isfinite(target).all():
        raise ValueError('Execution teacher logits must be finite.')
    lq, lp = target.log_softmax(-1), student_logits.double().log_softmax(-1)
    return (lq.exp() * (lq - lp)).sum().float()


def native_free_distillation(policy, proposal, *, execution_teachers=(), round_executor_fingerprint=None):
    """Replace retention only at observed native sites carrying execution credit.

    All other actual free decisions retain their full original distribution.
    The old argmax-margin penalty is deliberately absent: it would conflict
    with a teacher that legitimately changes the preferred native decision.
    """
    by_prefix = {}
    for teacher in execution_teachers:
        if teacher.site.prefix in by_prefix:
            raise ValueError('Duplicate execution teacher at the same native prefix.')
        by_prefix[teacher.site.prefix] = teacher
    observed = {decision.prefix for decision in proposal.free_decisions}
    if not observed or not set(by_prefix).issubset(observed):
        raise ValueError('Execution teacher must supervise an actual observed native decision.')
    context, context_mask = policy.bundle.encode_event_requests([proposal.observation.request], device=policy.device)
    losses, references, details = [], [], []
    for decision in proposal.free_decisions:
        tokens = torch.tensor([decision.prefix], dtype=torch.long, device=policy.device)
        output = policy.bundle.ar(tokens, torch.ones_like(tokens, dtype=torch.bool), context, context_mask)
        student = output[0, -1, list(decision.legal_ids)]
        reference = decision.teacher_logits.detach().to(student.device).double()
        lp, lq = student.double().log_softmax(-1), reference.log_softmax(-1)
        retention = (lq.exp() * (lq - lp)).sum().float()
        teacher = by_prefix.get(decision.prefix)
        if teacher is None:
            loss = retention
        else:
            site = NativeDecisionSite.from_request(proposal.observation.sample_id,
                proposal.observation.request, decision.prefix, decision.legal_ids)
            loss = native_execution_kl(student, teacher, site=site,
                round_executor_fingerprint=round_executor_fingerprint)
        losses.append(loss)
        references.append(retention)
        details.append(dict(prefix=decision.prefix, selected_token=decision.selected_token,
            support_size=len(decision.legal_ids), execution_credit=teacher is not None,
            objective_kl=float(loss.detach()), reference_kl=float(retention.detach())))
    return torch.stack(losses).mean(), torch.stack(references).mean(), details
