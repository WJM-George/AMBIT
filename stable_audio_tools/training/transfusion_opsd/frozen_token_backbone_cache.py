"""Process-local memoization of a frozen text backbone, before learned projections.

This narrow adapter accepts exactly the QwenTextConditioner call contract.
It never caches AR states, projected conditioning or diffusion predictions.
"""
from collections import OrderedDict
from types import SimpleNamespace

import torch


class FrozenTokenBackboneCache:
    def __init__(self, backbone, *, maximum_bytes=128 << 20):
        if maximum_bytes <= 0:
            raise ValueError('A positive cache bound is required.')
        self.backbone = backbone
        self.original_forward = backbone.forward
        self.maximum_bytes = maximum_bytes
        self.values = OrderedDict()
        self.bytes = self.hits = self.misses = 0
        self.signature = self._signature()

    def _signature(self):
        if self.backbone.training or any(p.requires_grad for p in self.backbone.parameters()):
            raise ValueError('Only an eval-mode frozen text backbone may be cached.')
        return tuple((name, id(p), p._version, str(p.device), str(p.dtype))
                     for name, p in self.backbone.named_parameters())

    def __call__(self, *, input_ids, attention_mask, use_cache, return_dict):
        if use_cache is not False or return_dict is not True or torch.is_grad_enabled():
            raise ValueError('Cache only detached token-backbone calls with no decoding cache.')
        if self._signature() != self.signature:
            raise ValueError('Frozen backbone weights or runtime placement changed.')
        if (input_ids.ndim != 2 or attention_mask.shape != input_ids.shape
                or input_ids.device != attention_mask.device):
            raise ValueError('Aligned native token IDs and masks are required.')
        device_type = input_ids.device.type
        def key_tensor(tensor):
            return str(tensor.dtype), tuple(tensor.shape), tensor.contiguous().cpu().numpy().tobytes()
        key = (str(input_ids.device), torch.is_autocast_enabled(device_type),
               str(torch.get_autocast_dtype(device_type)), key_tensor(input_ids), key_tensor(attention_mask))
        if key in self.values:
            self.hits += 1
            self.values.move_to_end(key)
            return SimpleNamespace(last_hidden_state=self.values[key].clone())
        self.misses += 1
        output = self.original_forward(input_ids=input_ids, attention_mask=attention_mask,
                                       use_cache=False, return_dict=True)
        hidden = output.last_hidden_state
        if hidden.requires_grad or not torch.isfinite(hidden).all():
            raise ValueError('Frozen backbone must return finite detached hidden states.')
        size = hidden.numel() * hidden.element_size()
        if size <= self.maximum_bytes:
            while self.values and self.bytes + size > self.maximum_bytes:
                _, old = self.values.popitem(last=False)
                self.bytes -= old.numel() * old.element_size()
            self.values[key] = hidden.detach().clone()
            self.bytes += size
        return output

    def receipt(self):
        try:
            unchanged = self._signature() == self.signature
        except ValueError:
            unchanged = False
        return dict(contract='frozen_token_backbone_cache_v1', hits=self.hits, misses=self.misses,
                    bytes=self.bytes, maximum_bytes=self.maximum_bytes,
                    identity_unchanged=unchanged,
                    scope='Frozen raw token hidden states only; all learned projections and AR/DiT states recomputed.')
