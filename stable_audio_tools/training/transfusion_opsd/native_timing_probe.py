"""One-site interventions for a bounded native planning-control diagnostic.

No global decoder mutation and no new inference policy. Request compliance is
checked by the caller after decoding. These are controlled alternatives, not
on-policy samples or certified execution teachers.
"""
from types import FunctionType

import torch

from .native_stochastic_policy import _Recorder, sample_native_plan


@torch.no_grad()
def intervene_native_timing(policy, observation, *, slot, selected, seed=42):
    hits = []

    class ControlledRecorder(_Recorder):
        def categorical(self, logits, **site):
            choice = super().categorical(logits, **site)
            if (site.get('family') == 'qualitative' and site.get('kind') == 'timing'
                    and site.get('slot') == slot):
                if isinstance(selected, bool) or not 0 <= selected < logits.numel():
                    raise ValueError('Timing intervention is outside native feasible support.')
                probabilities = (logits.detach().double().cpu() / self.temperature).log_softmax(0)
                self.decisions[-1].update(selected=selected, old_log_prob=float(probabilities[selected]))
                hits.append(dict(site, previous=choice, selected=selected))
                return selected
            return choice

    original = sample_native_plan.__wrapped__
    local = FunctionType(original.__code__, dict(original.__globals__, _Recorder=ControlledRecorder),
                         name=original.__name__, argdefs=original.__defaults__)
    local.__kwdefaults__ = original.__kwdefaults__
    result = local(policy, observation, seed=seed, temperature=1., greedy_diagnostic=True)
    if len(hits) != 1 or result.failure:
        raise ValueError(f'One complete native intervention was required: {hits}; {result.failure}')
    return result, hits[0]
