"""Same greedy codec decisions with one GPU-to-CPU transfer per token step.

The caller owns autocast, evaluation mode, and any AR source-binding context.
This function does not change logits, legal token sets, or tie-breaking order.
It must pass real-model output parity before entering a new inference contract.
"""
import numpy as np
import torch


@torch.no_grad()
def generate_constrained_vectorized(model, requests, codec, *, device, max_plan_tokens=1024):
    context, context_mask = model.encode_requests(requests, device=device)
    cache = model.prepare_decode_cache(context, context_mask, max_plan_tokens=max_plan_tokens)
    prefixes = [[int(codec.bos_id)] for _ in requests]
    finished = [False] * len(requests)
    current = torch.full((len(requests),), int(codec.bos_id), device=context.device, dtype=torch.long)
    for _ in range(int(max_plan_tokens)-1):
        logits = model.decode_step(current, cache)
        legal = np.zeros(tuple(logits.shape), dtype=np.bool_)
        for index, prefix in enumerate(prefixes):
            allowed = [int(codec.eos_id)] if finished[index] else sorted(int(value) for value in codec.allowed_next_ids(prefix))
            if not allowed: raise RuntimeError('codec-v4 returned an empty legal next-token set')
            legal[index, allowed] = True
        # Vocab-order argmax preserves the original sorted-legal-ID tie break.
        mask = torch.from_numpy(legal).to(device=logits.device)
        current = logits.masked_fill(~mask, float('-inf')).argmax(dim=-1)
        for index, value in enumerate(current.tolist()):
            if finished[index]: continue
            prefixes[index].append(value)
            finished[index] = value == int(codec.eos_id)
        if all(finished): return prefixes
    incomplete = [index for index, value in enumerate(finished) if not value]
    raise RuntimeError(f'Generation AR did not emit EOS within {max_plan_tokens} tokens: rows={incomplete[:8]}')
