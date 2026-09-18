"""Discard repeated deterministic evaluations inside one request collection.

The scope ends before any backward/update. Student training forwards bypass
the cache; reference native decoding, repeated prefixes and VAE decodes keep
their original kernels and inputs. The cache has a fixed memory ceiling.
"""
from contextlib import contextmanager
import copy

import torch


def tensor_bytes(value):
    if torch.is_tensor(value):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(tensor_bytes(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(map(tensor_bytes, value))
    return 0


@contextmanager
def request_evaluation_cache(learner, *, enabled=True, maximum_bytes=256 * 1024**2):
    originals, entries, retained_inputs = [], {}, []
    report = dict(bytes=0, maximum_bytes=maximum_bytes, hits={}, misses={})

    def install(owner, name, label, key_fn, *, clone_plan=False):
        original = getattr(owner, name)
        # Restore instance/class method ownership as well as behavior.
        existed, previous = name in owner.__dict__, owner.__dict__.get(name)
        originals.append((owner, name, existed, previous))

        def call(*args, **kwargs):
            if not enabled or torch.is_grad_enabled() or getattr(owner, 'training', False):
                return original(*args, **kwargs)
            key = key_fn(*args, **kwargs)
            if key is None:
                return original(*args, **kwargs)
            key = (label, key)
            if key in entries:
                report['hits'][label] = report['hits'].get(label, 0) + 1
                result = entries[key]
            else:
                report['misses'][label] = report['misses'].get(label, 0) + 1
                result = original(*args, **kwargs)
                size = tensor_bytes(result)
                if report['bytes'] + size <= maximum_bytes:
                    entries[key] = result
                    retained_inputs.append(args)  # Prevent pointer/id reuse within this scope.
                    report['bytes'] += size
            return (copy.deepcopy(result[0]), result[1].clone()) if clone_plan else result
        setattr(owner, name, call)

    def plan_key(obs, **kwargs):
        return None if kwargs else id(obs)

    def logits_key(obs, tokens, **kwargs):
        if kwargs:
            return None
        return (id(obs), str(tokens.device), str(tokens.dtype), tuple(tokens.shape),
                tuple(tokens.detach().flatten().tolist()))

    def decode_key(z, **kwargs):
        if set(kwargs) != {'model_num_samples'}:
            return None
        return (str(z.device), z.data_ptr(), z._version, tuple(z.shape), str(z.dtype),
                tuple(kwargs['model_num_samples']))

    try:
        install(learner.reference, 'native_plan', 'reference_native_plan', plan_key, clone_plan=True)
        for model, label in [(learner.adapter, 'student_logits'), (learner.reference, 'reference_logits')]:
            install(model, 'student_logits', label, logits_key)
        install(learner.pipeline, 'decode_foa_latents', 'FOA_decode', decode_key)
        yield report
    finally:
        for owner, name, existed, previous in reversed(originals):
            if existed:
                setattr(owner, name, previous)
            else:
                delattr(owner, name)
        entries.clear()
        retained_inputs.clear()
