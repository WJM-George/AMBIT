"""Bounded memoization of immutable, single-row Qwen backbone outputs.

Only the frozen backbone is cached. Trainable prompt/role projections still
run on every use, with their usual gradients and original numerical kernels.
"""
from collections import OrderedDict
import copy

import torch


class FrozenForwardCache:
    def __init__(self, model, *, maximum_bytes=256 * 1024**2):
        if any(p.requires_grad for p in model.parameters()) or model.training:
            raise ValueError('Only an immutable evaluation backbone can be cached.')
        self.model, self.original = model, model.forward
        self.maximum_bytes = maximum_bytes
        self.entries = OrderedDict()
        self.bytes = self.hits = self.misses = 0
        self.enabled = True
        model.forward = self.forward

    def clear(self):
        self.entries.clear()
        self.bytes = 0

    def close(self):
        self.model.forward = self.original
        self.clear()

    def forward(self, *args, **kwargs):
        if (not self.enabled or args or self.model.training or torch.is_grad_enabled() or
                set(kwargs) != {'input_ids', 'attention_mask', 'use_cache', 'return_dict'} or
                kwargs['use_cache'] or not kwargs['return_dict']):
            return self.original(*args, **kwargs)
        ids, mask = kwargs['input_ids'], kwargs['attention_mask']
        if ids.ndim != 2 or ids.shape[0] != 1:
            return self.original(**kwargs)
        key = (str(ids.device), str(ids.dtype), tuple(ids.shape), str(mask.dtype),
               torch.is_autocast_enabled(), str(torch.get_autocast_dtype(ids.device.type)),
               ids.detach().cpu().contiguous().numpy().tobytes(),
               mask.detach().cpu().contiguous().numpy().tobytes())
        if key in self.entries:
            self.hits += 1
            result = self.entries.pop(key)
            self.entries[key] = result
            output = copy.copy(result)
            output.last_hidden_state = result.last_hidden_state.clone()
            return output
        self.misses += 1
        output = self.original(**kwargs)
        hidden = output.last_hidden_state
        size = hidden.numel() * hidden.element_size()
        if size <= self.maximum_bytes:
            while self.entries and self.bytes + size > self.maximum_bytes:
                _, retired = self.entries.popitem(last=False)
                self.bytes -= retired.last_hidden_state.numel() * retired.last_hidden_state.element_size()
            saved = copy.copy(output)
            saved.last_hidden_state = hidden.detach().clone()
            self.entries[key] = saved
            self.bytes += size
        return output

    def statistics(self):
        return dict(hits=self.hits, misses=self.misses, bytes=self.bytes, entries=len(self.entries),
                    maximum_bytes=self.maximum_bytes)
