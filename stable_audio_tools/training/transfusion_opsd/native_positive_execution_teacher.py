"""Positive execution credit at a measured native AR decision.

Only protected improvements on every declared paired noise receive positive
credit. Unknown and unqualified choices retain their prior probability mass.
The teacher is empirical and round-specific, not a claim of generalization.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .native_decision_retention import selected_margin_penalty
from .native_execution_teacher import NativeDecisionSite, native_execution_kl


@dataclass(frozen=True)
class PositiveExecutionComparison:
    token: int
    seed: int
    gain: float
    protection: str  # passed, failed, or uncertain


@dataclass(frozen=True)
class NativePositiveExecutionTeacher:
    site: NativeDecisionSite
    executor_fingerprint: str
    observer_fingerprint: str
    evidence_sha256: str
    paired_seeds: tuple[int, ...]
    reference_token: int
    target_logits: torch.Tensor
    qualified_gains: tuple[tuple[int, float], ...]
    positive_mass_fraction: float
    minimum_mean_gain: float


def _identity(value):
    return (isinstance(value, str) and len(value) == 64
            and all(x in '0123456789abcdef' for x in value))


def build_native_positive_execution_teacher(reference_logits, *, site,
        executor_fingerprint, observer_fingerprint, evidence_sha256,
        reference_token, paired_seeds, comparisons, positive_mass_fraction=.75,
        minimum_mean_gain=.0025):
    """Move a fixed fraction of measured mass to demonstrated improvements.

    On M = {reference} union {qualified alternatives}, the conditional teacher
    is (1-rho)*p(.|M) + rho*normalized_positive_mean_gains. Outside M it equals
    p exactly. A finite rho below one leaves finite logits even at the reference.
    This is a declared distillation target, not an unbiased policy gradient.
    """
    if (reference_logits.ndim != 1 or reference_logits.numel() != len(site.legal_ids)
            or not reference_logits.is_floating_point() or not torch.isfinite(reference_logits).all()):
        raise ValueError('Finite logits must span the complete native legal support.')
    if not all(map(_identity, (executor_fingerprint, observer_fingerprint, evidence_sha256))):
        raise ValueError('Bind executor, observer and evidence identities.')
    if (not math.isfinite(positive_mass_fraction) or not 0 < positive_mass_fraction < 1
            or not math.isfinite(minimum_mean_gain) or minimum_mean_gain <= 0):
        raise ValueError('Declare a finite mixture strictly inside (0,1) and positive gain floor.')
    if (reference_token not in site.legal_ids
            or site.legal_ids[int(reference_logits.argmax())] != reference_token):
        raise ValueError('Reference must be the actual native greedy choice.')
    seeds = tuple(paired_seeds)
    if (len(seeds) < 2 or len(set(seeds)) != len(seeds)
            or any(type(x) is not int or x < 0 for x in seeds)):
        raise ValueError('Declare at least two unique paired noises.')
    rows = {}
    for x in comparisons:
        if (x.token not in site.legal_ids or x.seed not in seeds
                or x.protection not in {'passed', 'failed', 'uncertain'}
                or not math.isfinite(x.gain)):
            raise ValueError('Invalid paired execution comparison.')
        if (x.token, x.seed) in rows:
            raise ValueError('Duplicate paired execution comparison.')
        rows[x.token, x.seed] = x
    measured = {token for token, _ in rows}
    if reference_token not in measured or len(measured) < 2:
        raise ValueError('Need reference and alternative execution evidence.')
    for token in measured:
        if {seed for action, seed in rows if action == token} != set(seeds):
            raise ValueError('Every observed choice needs the complete paired-noise panel.')
    if any(rows[reference_token, s].gain != 0 or rows[reference_token, s].protection != 'passed' for s in seeds):
        raise ValueError('Reference comparisons must identify the same observed audio.')
    qualified = []
    for token in site.legal_ids:
        if token not in measured or token == reference_token:
            continue
        panel = [rows[token, s] for s in seeds]
        mean_gain = math.fsum(x.gain for x in panel) / len(panel)
        if (all(x.protection == 'passed' and x.gain > 0 for x in panel)
                and mean_gain >= minimum_mean_gain):
            qualified.append((token, mean_gain))
    if not qualified:
        raise ValueError('No protected improvement on every declared paired noise.')
    fixed = reference_logits.detach().double().clone()
    target = fixed.clone()
    tokens = [reference_token] + [token for token, _ in qualified]
    indices = [site.legal_ids.index(token) for token in tokens]
    # Work in log space; native priors can be sharply concentrated.
    log_mass = torch.logsumexp(fixed[indices], 0)
    log_prior = fixed[indices] - log_mass
    gains = fixed.new_tensor([0.] + [gain for _, gain in qualified])
    log_gain = gains.log() - gains.sum().log()
    mixed = torch.logaddexp(log_prior + math.log1p(-positive_mass_fraction),
                            log_gain + math.log(positive_mass_fraction))
    target[indices] = log_mass + mixed
    return NativePositiveExecutionTeacher(site, executor_fingerprint, observer_fingerprint,
        evidence_sha256, seeds, reference_token, target, tuple(qualified),
        positive_mass_fraction, minimum_mean_gain)


def native_positive_site_objective(student_logits, reference_logits, *, site,
        selected_token, teacher=None, round_executor_fingerprint=None,
        observer_fingerprint=None):
    """Exclude a taught positive site from original-choice retention and its gate.

    Reference KL is always reported for diagnosis. Only untaught sites receive
    reference KL and original argmax margin as training/retention constraints.
    """
    if (student_logits.ndim != 1 or student_logits.shape != reference_logits.shape
            or student_logits.numel() != len(site.legal_ids)
            or not torch.isfinite(student_logits).all() or not torch.isfinite(reference_logits).all()):
        raise ValueError('Aligned finite native logits are required.')
    reference = reference_logits.detach().to(student_logits.device).double()
    if selected_token != site.legal_ids[int(reference.argmax())]:
        raise ValueError('Original selected token does not match its native reference.')
    lp, lr = student_logits.double().log_softmax(-1), reference.log_softmax(-1)
    reference_kl = (lr.exp() * (lr - lp)).sum().float()
    zero = student_logits.new_zeros(())
    if teacher is not None:
        if (not isinstance(teacher, NativePositiveExecutionTeacher)
                or observer_fingerprint != teacher.observer_fingerprint
                or selected_token != teacher.reference_token):
            raise ValueError('Positive teacher observer or reference identity differs.')
        positive = native_execution_kl(student_logits, teacher, site=site,
            round_executor_fingerprint=round_executor_fingerprint)
        return dict(positive=positive, retained_kl=zero, retained_margin=zero,
                    reference_kl=reference_kl, retained=False)
    margin, _ = selected_margin_penalty(student_logits, reference,
        site.legal_ids.index(selected_token))
    return dict(positive=zero, retained_kl=reference_kl, retained_margin=margin,
                reference_kl=reference_kl, retained=True)


def native_positive_free_distillation(policy, proposal, *, teachers=(),
        round_executor_fingerprint, observer_fingerprint):
    """Return per-site losses so the caller normalizes each supervision family."""
    by_prefix = {teacher.site.prefix: teacher for teacher in teachers}
    if len(by_prefix) != len(teachers):
        raise ValueError('Duplicate positive teachers at one prefix.')
    if not set(by_prefix).issubset({d.prefix for d in proposal.free_decisions}):
        raise ValueError('Positive teacher must address an actual observed native decision.')
    context, mask = policy.bundle.encode_event_requests([proposal.observation.request], device=policy.device)
    losses, details = [], []
    for decision in proposal.free_decisions:
        tokens = torch.tensor([decision.prefix], dtype=torch.long, device=policy.device)
        output = policy.bundle.ar(tokens, torch.ones_like(tokens, dtype=torch.bool), context, mask)
        student = output[0, -1, list(decision.legal_ids)]
        site = NativeDecisionSite.from_request(proposal.observation.sample_id,
            proposal.observation.request, decision.prefix, decision.legal_ids)
        loss = native_positive_site_objective(student, decision.teacher_logits, site=site,
            selected_token=decision.selected_token, teacher=by_prefix.get(decision.prefix),
            round_executor_fingerprint=round_executor_fingerprint, observer_fingerprint=observer_fingerprint)
        losses.append(loss)
        details.append(dict(prefix=list(decision.prefix), original_token=decision.selected_token,
            selected_token=decision.legal_ids[int(student.detach().argmax())], support_size=len(decision.legal_ids),
            retained=loss['retained'], **{k: float(v.detach()) for k, v in loss.items() if k != 'retained'}))
    return losses, details
