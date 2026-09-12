"""Use a learned AR count expert while retaining the base decoder elsewhere.

The expert sees the same raw-request context and six tokens already generated
by the base model. No count, target or request annotation is supplied. Frozen
P10 weights are shared; only small AR weights are temporarily exchanged.
The wrapper is for serial evaluation, not concurrent calls on the same model.
"""
from dataclasses import dataclass, field
import torch


@dataclass
class CountExpertCache:
    base_cache: object
    context: torch.Tensor
    mask: torch.Tensor
    prefix: list = field(default_factory=list)
    count_done: bool = False


class CountExpertDecoder:
    def __init__(self, base, codec, expert_adapter, expert_lora):
        if base.training:
            raise ValueError('Count expert is inference-only')
        self.base = base
        self.count_tag = int(codec.token_to_id['<num_sources>'])
        self.modules = (base.ar_adapter, base.ar_lora)
        self.base_states = []
        self.expert_states = []
        for module, expert in zip(self.modules, (expert_adapter, expert_lora)):
            current = module.state_dict()
            if set(current) != set(expert):
                raise ValueError('Count expert state keys do not match base AR')
            converted = {}
            for name, tensor in current.items():
                value = expert[name]
                if tensor.shape != value.shape or not bool(torch.isfinite(value).all()):
                    raise ValueError('Invalid count expert weight: ' + name)
                converted[name] = value.to(device=tensor.device, dtype=tensor.dtype).clone()
            self.base_states.append({k: v.detach().clone() for k, v in current.items()})
            self.expert_states.append(converted)
        self.count_calls = 0

    def assert_restored(self):
        for module, saved in zip(self.modules, self.base_states):
            if any(not torch.equal(tensor, saved[name]) for name, tensor in module.state_dict().items()):
                raise RuntimeError('Base AR weights were not restored after count inference')

    def encode_requests(self, requests, *, device):
        return self.base.encode_requests(requests, device=device)

    def prepare_decode_cache(self, context, mask, *, max_plan_tokens):
        return CountExpertCache(self.base.prepare_decode_cache(
            context, mask, max_plan_tokens=max_plan_tokens), context, mask)

    @torch.no_grad()
    def decode_step(self, tokens, cache):
        # Advance base K/V even at the count position. Its continuation must
        # consume the selected count with the original base weights and cache.
        logits = self.base.decode_step(tokens, cache.base_cache)
        if cache.count_done:
            return logits
        cache.prefix.append(tokens.clone())
        if len(cache.prefix) < 6:
            return logits
        if not bool((tokens == self.count_tag).all()):
            raise ValueError('Count expert requires the existing six-token codec header')
        prefix = torch.stack(cache.prefix, dim=1)
        try:
            for module, state in zip(self.modules, self.expert_states):
                module.load_state_dict(state, strict=True)
            count_logits = self.base(prefix, torch.ones_like(prefix, dtype=torch.bool),
                                     cache.context, cache.mask)[:, -1]
        finally:
            for module, state in zip(self.modules, self.base_states):
                module.load_state_dict(state, strict=True)
        cache.count_done = True
        cache.prefix.clear()
        self.count_calls += 1
        return count_logits
