"""Batch forced tokens between learned AR decisions using full-prefix forwards.

This experimental backend is restricted to a learned inventory plus a
qualitative head that does not consume AR-prefix states. Every semantic field
still comes from those neural heads. AR predicts unconstrained room, duration
and gain tokens; grammar-forced tokens are appended without redundant AR calls.
An actual token-parity and throughput gate is required before promotion.
"""
from collections import deque
import importlib.util
from pathlib import Path

import torch


_spec = importlib.util.spec_from_file_location('_generation_ar_step_backend',
    Path(__file__).with_name('sceneplan_generation_ar_learned_copy.py'))
_step = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_step)
CopyDecodeLimitError = _step.CopyDecodeLimitError
predict_source_inventory = _step.predict_source_inventory


@torch.no_grad()
def generate_with_learned_copy(model, pointer, pointer_module, requests, codec, *, device,
        max_plan_tokens=512, observe_only=False, execution_head=None, execution_module=None,
        inventory_head=None, inventory_module=None):
    if (observe_only or inventory_head is None or execution_head is None or execution_head.use_ar_query):
        return _step.generate_with_learned_copy(model, pointer, pointer_module, requests, codec,
            device=device, max_plan_tokens=max_plan_tokens, observe_only=observe_only,
            execution_head=execution_head, execution_module=execution_module,
            inventory_head=inventory_head, inventory_module=inventory_module)
    context, mask = model.encode_requests(requests, device=device)
    alignment = pointer_module.encode_character_alignment(model.prompt_conditioner.tokenizer, requests, mask, device=device)
    inventory = predict_source_inventory(inventory_head, inventory_module, pointer_module,
        requests, context, mask, alignment)
    batch = len(requests)
    first = mask.long().argmax(-1, keepdim=True)
    starts = first.expand(-1, 4).clone()
    ends = starts.clone()
    fields = torch.zeros(batch, 4, device=context.device, dtype=torch.long)
    scopes = torch.stack((starts, ends), -1)
    local = bool(getattr(execution_head, 'source_local_attention', False))
    for i, row in enumerate(inventory):
        for j, source in enumerate(row['sources']):
            fields[i, j] = 1 if source['kind'] == 'speech' else 0
            starts[i, j] = alignment.token_indices[i, source['identity']['start']]
            ends[i, j] = alignment.token_indices[i, source['identity']['end']]
            if local:
                event = source['event']
                scopes[i, j, 0] = min(int(alignment.token_indices[i, event['start']]), int(starts[i, j]))
                scopes[i, j, 1] = max(int(alignment.token_indices[i, event['end']]), int(ends[i, j]))
    options = {'source_spans': scopes} if local else {}
    with torch.autocast(device_type=context.device.type, enabled=False):
        values = execution_head(context.new_zeros(batch, 4, 1024), fields, starts, ends, context, mask, **options)
    values = {name: value.cpu().tolist() for name, value in values.items()}
    prefixes = [[int(codec.bos_id)] for _ in requests]
    queues = [deque() for _ in requests]
    traces = [[] for _ in requests]
    prepared_text = [set() for _ in requests]
    controls = [None] * batch
    finished = [False] * batch
    text_begin = codec._tid('<text_begin>')
    field_tags = {codec._tid('<' + name + '>'): name for name in pointer_module.FIELD_TYPES}
    slot_tags = {codec._tid(f'<source_slot_{i}>'): i for i in range(4)}
    forward_calls = 0
    while True:
        decisions = {}
        for i, prefix in enumerate(prefixes):
            if finished[i]: continue
            while len(prefix) < max_plan_tokens:
                current = prefix[-1]
                if current == text_begin and len(prefix) not in prepared_text[i]:
                    prepared_text[i].add(len(prefix))
                    field = field_tags[prefix[-2]]
                    slot = next(slot_tags[t] for t in reversed(prefix) if t in slot_tags)
                    source = inventory[i]['sources'][slot]
                    span = source['transcript' if field == 'transcript' else 'identity']
                    pieces = codec._text(span['text'])[1:]
                    fits = len(prefix) + len(pieces) + 20 <= max_plan_tokens
                    if fits: queues[i].extend(pieces)
                    trace = {'field': field, 'query_position': len(prefix) - 1,
                        'generated_source_slot': slot, 'start': span['start'], 'end': span['end'],
                        'text': span['text'], 'copied': fits, 'observe_only': False,
                        'fallback_reason': None if fits else 'span_exceeds_remaining_token_budget',
                        'learned_inventory_count': inventory[i]['count'], 'learned_inventory_kind': source['kind'],
                        'inventory_kind_grammar_adjusted': source['kind'] != source['unconstrained_kind']}
                    if 'event' in source:
                        trace['learned_event_span'] = {key: source['event'][key] for key in ('start', 'end')}
                    if field != 'transcript':
                        controls[i] = execution_module.complete_from_logits({name: value[i][slot] for name, value in values.items()},
                            codec.frame_ids.index(prefix[2]), seed_key=f'42/{requests[i]}/{slot}', speech=field == 'speaker_description')
                        trace['qualitative_control'] = controls[i]
                    traces[i].append(trace)
                allowed = set(map(int, codec.allowed_next_ids(prefix)))
                if not allowed: raise RuntimeError('Codec returned an empty legal token set')
                if queues[i]:
                    token = int(queues[i].popleft())
                    if token not in allowed: raise RuntimeError('Learned literal span violates codec grammar')
                    allowed = {token}
                for token in (_step.qualitative_token(codec, current, prefix, controls[i]),
                              _step.source_inventory_token(codec, current, prefix, inventory[i])):
                    if token is not None:
                        if token not in allowed: raise RuntimeError('Learned planning decision violates codec grammar')
                        allowed = {int(token)}
                if len(allowed) > 1:
                    decisions[i] = sorted(allowed)
                    break
                token = next(iter(allowed))
                prefix.append(token)
                if token == int(codec.eos_id):
                    finished[i] = True
                    break
        if all(finished):
            for trace in traces:
                if trace: trace[0]['decision_block_forward_calls'] = forward_calls
            return prefixes, traces
        if any(len(prefix) >= max_plan_tokens and not done for prefix, done in zip(prefixes, finished)):
            raise CopyDecodeLimitError(max_plan_tokens, prefixes, traces, finished)
        if not decisions: raise RuntimeError('No progress and no pending AR decision')
        length = max(map(len, prefixes))
        tokens = torch.full((batch, length), codec.pad_id, device=context.device, dtype=torch.long)
        plan_mask = torch.zeros_like(tokens).bool()
        positions = []
        for i, prefix in enumerate(prefixes):
            tokens[i, :len(prefix)] = torch.tensor(prefix, device=context.device)
            plan_mask[i, :len(prefix)] = True
            positions.append(len(prefix) - 1)
        # Existing AR forward owns its FP32/LoRA scope. The prefix consists
        # entirely of decisions already made by this same inference run.
        logits = model(tokens, plan_mask, context, mask)
        forward_calls += 1
        for i, allowed in decisions.items():
            index = int(logits[i, positions[i], allowed].argmax())
            token = allowed[index]
            prefixes[i].append(token)
            finished[i] = token == int(codec.eos_id)
        del logits
