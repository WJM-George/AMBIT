"""Reward-sensitive native planning teachers with explicit measured support.

The candidate-conditional exponential tilt uses actual frozen execution. No
reward is invented for unmeasured choices, whose probability mass stays fixed.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .identity_preserving_kl import identity_preserving_kl
from .native_decision_retention import selected_margin_penalty
from .native_execution_teacher import NativeDecisionSite


@dataclass(frozen=True)
class RewardExecutionObservation:
    token: int
    seed: int
    reward: float
    protection: str


@dataclass(frozen=True)
class NativeRewardExecutionTeacher:
    site: NativeDecisionSite
    executor_fingerprint: str
    observer_fingerprint: str
    evidence_sha256: str
    paired_seeds: tuple[int, ...]
    reference_token: int
    target_logits: torch.Tensor
    measured_rewards: tuple[tuple[int, float], ...]
    temperature: float
    measured_probability_mass: float
    unsupported_probability_mass: float


def build_native_reward_execution_teacher(reference_logits, *, site, executor_fingerprint,
        observer_fingerprint, evidence_sha256, reference_token, paired_seeds,
        observations, temperature):
    """Maximize conditional mean reward minus tau*KL on protected support M.

    q(M)=p_ref(M), q(a|M) proportional to p_ref(a|M)*exp(mean_reward/tau).
    Outside M, q equals p_ref. Protected alternatives need a complete paired
    noise panel. This empirical optimum is not a global/model improvement
    guarantee. All initial priors, weights and measured targets stop gradients.
    """
    if (reference_logits.ndim!=1 or reference_logits.numel()!=len(site.legal_ids)
            or not reference_logits.is_floating_point() or not torch.isfinite(reference_logits).all()):
        raise ValueError('Finite logits must span the full native legal support.')
    identities=(executor_fingerprint,observer_fingerprint,evidence_sha256)
    if any(not isinstance(x,str) or len(x)!=64 or any(c not in '0123456789abcdef' for c in x) for x in identities):
        raise ValueError('Bind executor, observer and actual evidence identities.')
    if not math.isfinite(temperature) or temperature<=0:
        raise ValueError('Declare a finite positive reward temperature.')
    if reference_token not in site.legal_ids or site.legal_ids[int(reference_logits.argmax())]!=reference_token:
        raise ValueError('Reference must be the actual native greedy decision.')
    seeds=tuple(paired_seeds)
    if len(seeds)<2 or len(set(seeds))!=len(seeds) or any(type(x)is not int or x<0 for x in seeds):
        raise ValueError('Require at least two unique paired noises.')
    rows={}
    for value in observations:
        key=(value.token,value.seed)
        if (value.token not in site.legal_ids or value.seed not in seeds
                or not math.isfinite(value.reward) or value.protection not in {'passed','failed','uncertain'} or key in rows):
            raise ValueError('Invalid or duplicate observed execution.')
        rows[key]=value
    measured={token for token,_ in rows}
    if reference_token not in measured or len(measured)<2:
        raise ValueError('Need actual reference and alternative execution observations.')
    for token in measured:
        if {seed for action,seed in rows if action==token}!=set(seeds):
            raise ValueError('Every measured candidate needs the complete paired-noise panel.')
    if any(rows[reference_token,seed].protection!='passed' for seed in seeds):
        raise ValueError('Reference-relative identity protection must pass at every paired noise.')
    qualified=tuple(token for token in site.legal_ids if token in measured
        and all(rows[token,seed].protection=='passed' for seed in seeds))
    if len(qualified)<2:
        raise ValueError('No alternative has a complete protected execution panel.')
    means=tuple((token,math.fsum(rows[token,seed].reward for seed in seeds)/len(seeds)) for token in qualified)
    fixed=reference_logits.detach().double().clone()
    indices=[site.legal_ids.index(token) for token in qualified]
    reward=fixed.new_tensor([mean for _,mean in means])
    tilted=fixed[indices]+(reward-reward.max())/temperature
    target=fixed.clone()
    target[indices]=tilted+torch.logsumexp(fixed[indices],0)-torch.logsumexp(tilted,0)
    if not torch.isfinite(target).all():
        raise ValueError('Reward tilt exceeds finite numerical range.')
    mass=float(fixed.softmax(-1)[indices].sum())
    return NativeRewardExecutionTeacher(site,executor_fingerprint,observer_fingerprint,evidence_sha256,
        seeds,reference_token,target.detach(),means,float(temperature),mass,1-mass)


def native_reward_free_objective(policy, proposal, *, teachers=(),
        round_executor_fingerprint, observer_fingerprint):
    """Execution credit at measured sites; exact-fixed-point retention elsewhere.

    No extra CE sharpens an already-correct retained decision. A taught site
    replaces its original-choice KL/margin; the caller normalizes positive and
    retention families separately and retains complete native learning heads.
    """
    by_prefix={teacher.site.prefix:teacher for teacher in teachers}
    if len(by_prefix)!=len(teachers) or not set(by_prefix).issubset({x.prefix for x in proposal.free_decisions}):
        raise ValueError('Teachers must identify unique actual native decision sites.')
    context,mask=policy.bundle.encode_event_requests([proposal.observation.request],device=policy.device)
    losses,details=[],[]
    for decision in proposal.free_decisions:
        tokens=torch.tensor([decision.prefix],dtype=torch.long,device=policy.device)
        output=policy.bundle.ar(tokens,torch.ones_like(tokens,dtype=torch.bool),context,mask)
        student=output[0,-1,list(decision.legal_ids)]
        reference=decision.teacher_logits.detach().to(student.device)
        reference_kl=identity_preserving_kl(student,reference)
        zero=student.new_zeros(())
        teacher=by_prefix.get(decision.prefix)
        if teacher is None:
            margin,_=selected_margin_penalty(student,reference,decision.legal_ids.index(decision.selected_token))
            item=dict(positive=zero,retained_kl=reference_kl,retained_margin=margin,reference_kl=reference_kl,retained=True)
        else:
            site=NativeDecisionSite.from_request(proposal.observation.sample_id,proposal.observation.request,decision.prefix,decision.legal_ids)
            if (not isinstance(teacher,NativeRewardExecutionTeacher) or teacher.site!=site
                    or teacher.executor_fingerprint!=round_executor_fingerprint or teacher.observer_fingerprint!=observer_fingerprint
                    or teacher.reference_token!=decision.selected_token):
                raise ValueError('Execution teacher site, round, observer or original choice differs.')
            positive=identity_preserving_kl(student,teacher.target_logits.to(student.device))
            item=dict(positive=positive,retained_kl=zero,retained_margin=zero,reference_kl=reference_kl,retained=False)
        losses.append(item)
        details.append(dict(prefix=list(decision.prefix),original_token=decision.selected_token,
            selected_token=decision.legal_ids[int(student.detach().argmax())],retained=item['retained'],
            **{key:float(value.detach()) for key,value in item.items() if key!='retained'}))
    return losses,details
