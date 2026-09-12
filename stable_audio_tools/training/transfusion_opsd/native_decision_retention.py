"""Retain actual native greedy decisions when no new execution credit exists."""
from __future__ import annotations

import math

import torch


def selected_margin_penalty(student, teacher, selected_index, *, keep_fraction=.5, scale_floor=.05):
    """A zero-at-initialization barrier on the selected native logit margin.

    This is only a retention objective. Verified execution-advantage teachers
    must not simultaneously be constrained to retain the old selected token.
    Full legal support is supplied by the actual native decoder, not a new
    response-head action vocabulary or a single previously ranked competitor.
    """
    if (student.ndim!=1 or teacher.shape!=student.shape or student.numel()<2
            or not torch.isfinite(student).all() or not torch.isfinite(teacher).all()):
        raise ValueError('Native margin retention requires aligned finite legal logits.')
    if (not 0<keep_fraction<=1 or not math.isfinite(keep_fraction)
            or not scale_floor>0 or not math.isfinite(scale_floor)):
        raise ValueError('Declare a positive finite retention margin fraction and scale.')
    if type(selected_index) is not int or not 0<=selected_index<student.numel():
        raise ValueError('The selected native token is outside its legal support.')
    fixed=teacher.detach().to(student.device).double()
    if int(fixed.argmax())!=selected_index:
        raise ValueError('Retention target must be the actual original native argmax.')
    other=torch.arange(student.numel(),device=student.device)!=selected_index
    old_gap=fixed[selected_index]-fixed[other].max()
    gap=student.double()[selected_index]-student.double()[other].max()
    target_gap=keep_fraction*old_gap
    scale=old_gap.clamp_min(scale_floor)
    loss=((target_gap-gap).clamp_min(0)/scale).square().float()
    return loss,dict(original_margin=float(old_gap),student_margin=float(gap.detach()),
        target_margin=float(target_gap),selected_preserved=int(student.detach().argmax())==selected_index)


def native_free_retention(policy, proposal, *, keep_fraction=.5, scale_floor=.05):
    """KL and margin on every actual native free-decision prefix."""
    context,context_mask=policy.bundle.encode_event_requests([proposal.observation.request],device=policy.device)
    kls,margins,diagnostics=[],[],[]
    for decision in proposal.free_decisions:
        tokens=torch.tensor([decision.prefix],dtype=torch.long,device=policy.device)
        output=policy.bundle.ar(tokens,torch.ones_like(tokens,dtype=torch.bool),context,context_mask)
        student=output[0,-1,list(decision.legal_ids)]
        teacher=decision.teacher_logits.detach().to(policy.device)
        selected=decision.legal_ids.index(decision.selected_token)
        lp,lq=student.double().log_softmax(-1),teacher.double().log_softmax(-1)
        kls.append((lq.exp()*(lq-lp)).sum().float())
        penalty,details=selected_margin_penalty(student,teacher,selected,
            keep_fraction=keep_fraction,scale_floor=scale_floor)
        margins.append(penalty)
        diagnostics.append(dict(prefix=decision.prefix,selected_token=decision.selected_token,
            support_size=len(decision.legal_ids),**details))
    if not kls:raise ValueError('No actual native free decisions were observed.')
    return torch.stack(kls).mean(),torch.stack(margins).mean(),diagnostics
