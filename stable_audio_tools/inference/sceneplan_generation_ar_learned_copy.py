"""Raw-only AR decoding with optional learned inventory and execution heads.

Learned count/kind, literal spans and qualitative controls are emitted through
the same AR cache and unchanged codec grammar. The raw request is never parsed
by a source/template rule and no target or request annotation is an input.
"""
from collections import deque

import numpy as np
import torch


class CopyDecodeLimitError(RuntimeError):
    """Keep finished batch members and diagnostics when one row exhausts its budget."""

    def __init__(self, maximum, prefixes, traces, finished):
        super().__init__(f'Generation AR did not emit EOS within {maximum} tokens: '
            f'rows={[i for i, value in enumerate(finished) if not value]}')
        self.prefixes = prefixes
        self.traces = traces
        self.finished = finished


def qualitative_token(codec, current, prefix, control):
    """Serialize learned controls; this receives no request or target annotation."""
    if control is None: return None
    for tag, key in [('<onset_frame>', 'onset_frame'), ('<offset_frame>', 'offset_frame')]:
        if current == codec._tid(tag): return codec.frame_ids[control[key]]
    trajectory = control['trajectory']
    if current == codec._tid('<trajectory_begin>'):
        return codec._tid('<motion_' + trajectory['type'] + '>')
    numeric = {codec._tid('<azimuth_bin>'): 'azimuth', codec._tid('<elevation_bin>'): 'elevation', codec._tid('<distance_bin>'): 'distance'}
    if current not in numeric: return None
    tags = {codec._tid('<' + point + '>'): point for point in ('position', 'start', 'end')}
    point = next(tags[token] for token in reversed(prefix) if token in tags)
    position = trajectory[point]; field = numeric[current]
    if field == 'azimuth': return codec.azimuth_ids[codec._azimuth_value(position['azimuth_deg']) + 180]
    if field == 'elevation': return codec.elevation_ids[codec._elevation_value(position['elevation_deg']) + 90]
    return codec.distance_ids[codec._distance_index(position['distance_m'])]


def predict_source_inventory(head, module, pointer_module, requests, context, mask, alignment):
    """Apply a neural inventory, including the codec's at-most-one-speech rule."""
    with torch.autocast(device_type=context.device.type, enabled=False):
        logits = head(context, mask, alignment)
        result = module.decode_inventory(logits, requests, alignment.endpoint_mask, pointer_module.best_ordered_span)
    for row, scores in zip(result, logits['kind'][..., :3].tolist()):
        count = row['count']
        # Maximize learned kind scores over the existing executable grammar.
        # This enumerates <=5 assignments, without consulting the request text.
        non_speech = [max(range(2), key=lambda kind: scores[i][kind]) for i in range(count)]
        options = [non_speech]
        for i in range(count):
            value = list(non_speech)
            value[i] = 2
            options.append(value)
        kinds = max(options, key=lambda value: sum(scores[i][kind] for i, kind in enumerate(value)))
        for source, kind in zip(row['sources'], kinds):
            source['unconstrained_kind'] = source['kind']
            source['kind'] = module.KINDS[kind]
    return result


def source_inventory_token(codec, current, prefix, inventory):
    """Serialize learned inventory decisions without any external count hint."""
    if current == codec._tid('<num_sources>'):
        return codec._tid(f'<num_sources_{inventory["count"]}>')
    if current == codec._tid('<kind>'):
        slots = {codec._tid(f'<source_slot_{n}>'): n for n in range(4)}
        slot = next(slots[token] for token in reversed(prefix) if token in slots)
        return codec._tid('<kind_' + inventory['sources'][slot]['kind'] + '>')
    return None


@torch.no_grad()
def generate_with_learned_copy(model, pointer, pointer_module, requests, codec, *, device,
                               max_plan_tokens=512, observe_only=False, execution_head=None, execution_module=None,
                               inventory_head=None, inventory_module=None):
    context, context_mask = model.encode_requests(requests, device=device)
    alignment = pointer_module.encode_character_alignment(model.prompt_conditioner.tokenizer, requests, context_mask, device=device)
    with torch.autocast(device_type=context.device.type, enabled=False):
        prepared = pointer.prepare_keys(context, context_mask, alignment)
        execution_context = execution_head.prepare_context(context, context_mask) if execution_head is not None else None
    if execution_head is not None and execution_module is None: raise ValueError('Missing learned execution module')
    if inventory_head is not None and inventory_module is None: raise ValueError('Missing learned inventory module')
    inventory = predict_source_inventory(inventory_head, inventory_module, pointer_module, requests,
        context, context_mask, alignment) if inventory_head is not None else None
    local_execution = bool(getattr(execution_head, 'source_local_attention', False))
    if local_execution and (inventory is None or any('event' not in s for row in inventory for s in row['sources'])):
        raise ValueError('Source-local execution requires learned inventory event spans')
    cache = model.prepare_decode_cache(context, context_mask, max_plan_tokens=max_plan_tokens)
    fields = {codec._tid('<' + name + '>'): (name, kind) for name, kind in pointer_module.FIELD_TYPES.items()}
    slots = {codec._tid(f'<source_slot_{n}>'): n for n in range(4)}
    text_begin = codec._tid('<text_begin>'); prefixes = [[int(codec.bos_id)] for _ in requests]
    queues = [deque() for _ in requests]; traces = [[] for _ in requests]; finished = [False] * len(requests)
    controls = [None] * len(requests)
    current = torch.full((len(requests),), int(codec.bos_id), device=context.device, dtype=torch.long)
    current_values = current.tolist(); captured = {}
    hook = model.ar_adapter.output_norm.register_forward_hook(lambda _module, _args, value: captured.update(hidden=value.detach()))
    try:
        for _ in range(max_plan_tokens - 1):
            logits = model.decode_step(current, cache); hidden = captured.pop('hidden')
            choose = [i for i, token in enumerate(current_values) if not finished[i] and token == text_begin]
            if choose:
                kinds = torch.zeros(len(requests), 1, device=context.device, dtype=torch.long)
                for i in choose:
                    assert len(prefixes[i]) >= 2 and prefixes[i][-2] in fields and not queues[i]
                    kinds[i, 0] = fields[prefixes[i][-2]][1]
                if inventory is not None and not observe_only:
                    starts, ends = [0] * len(requests), [0] * len(requests)
                    for i in choose:
                        slot = next(slots[t] for t in reversed(prefixes[i]) if t in slots)
                        field = 'transcript' if fields[prefixes[i][-2]][0] == 'transcript' else 'identity'
                        selected = inventory[i]['sources'][slot][field]
                        starts[i], ends[i] = selected['start'], selected['end']
                else:
                    with torch.autocast(device_type=context.device.type, enabled=False):
                        a, b = pointer.score_queries(hidden, kinds, prepared)
                        starts, ends = pointer_module.best_ordered_span(a, b, alignment.endpoint_mask)
                    starts, ends = starts[:, 0].tolist(), ends[:, 0].tolist()
                source_queries = [i for i in choose if fields[prefixes[i][-2]][0] != 'transcript']
                execution_logits = None
                if execution_head is not None and source_queries:
                    token_starts = context_mask.long().argmax(-1, keepdim=True); token_ends = token_starts.clone()
                    for i in source_queries:
                        token_starts[i, 0] = alignment.token_indices[i, starts[i]]
                        token_ends[i, 0] = alignment.token_indices[i, ends[i]]
                    execution_options = {}
                    if local_execution:
                        last_tokens = (context_mask.long() * torch.arange(context_mask.shape[1], device=context.device)).amax(-1)
                        source_spans = torch.stack((token_starts[:, 0], last_tokens), -1)[:, None]
                        for i in source_queries:
                            slot = next(slots[t] for t in reversed(prefixes[i]) if t in slots)
                            if observe_only and slot >= inventory[i]['count']:
                                continue
                            event = inventory[i]['sources'][slot]['event']
                            source_spans[i, 0, 0] = min(int(alignment.token_indices[i, event['start']]), int(token_starts[i, 0]))
                            source_spans[i, 0, 1] = max(int(alignment.token_indices[i, event['end']]), int(token_ends[i, 0]))
                        execution_options['source_spans'] = source_spans
                    with torch.autocast(device_type=context.device.type, enabled=False):
                        values = execution_head.score_sources(hidden, kinds, token_starts, token_ends, execution_context, **execution_options)
                    execution_logits = {name: value[:, 0].cpu().tolist() for name, value in values.items()}
                for i in choose:
                    start, end = starts[i], ends[i]
                    text = requests[i][start:end + 1]
                    pieces = codec._text(text)[1:]  # Current <text_begin> was already consumed.
                    # A bad predicted long span is not allowed to consume the
                    # whole token envelope. Fall back to the base AR text path;
                    # the unchanged EOS limit and actual evaluation expose it.
                    fits = len(prefixes[i]) + len(pieces) + 20 <= max_plan_tokens
                    if fits and not observe_only: queues[i].extend(pieces)
                    slot = next(slots[t] for t in reversed(prefixes[i]) if t in slots)
                    trace = {'field': fields[prefixes[i][-2]][0], 'query_position': len(prefixes[i]) - 1,
                        'generated_source_slot': slot,
                        'start': start, 'end': end, 'text': text, 'copied': fits and not observe_only,
                        'observe_only': observe_only, 'fallback_reason': None if fits else 'span_exceeds_remaining_token_budget'}
                    if inventory is not None and not observe_only:
                        trace['learned_inventory_count'] = inventory[i]['count']
                        trace['learned_inventory_kind'] = inventory[i]['sources'][slot]['kind']
                        trace['inventory_kind_grammar_adjusted'] = inventory[i]['sources'][slot]['kind'] != inventory[i]['sources'][slot]['unconstrained_kind']
                        if 'event' in inventory[i]['sources'][slot]:
                            event = inventory[i]['sources'][slot]['event']
                            trace['learned_event_span'] = {key: event[key] for key in ('start', 'end')}
                    if execution_logits is not None and i in source_queries:
                        control = execution_module.complete_from_logits({name: value[i] for name, value in execution_logits.items()},
                            codec.frame_ids.index(prefixes[i][2]), seed_key=f'42/{requests[i]}/{slot}', speech=trace['field'] == 'speaker_description')
                        trace['qualitative_control'] = control
                        if not observe_only: controls[i] = control
                    traces[i].append(trace)
            legal = np.zeros(tuple(logits.shape), dtype=np.bool_)
            for i, prefix in enumerate(prefixes):
                allowed = [int(codec.eos_id)] if finished[i] else sorted(map(int, codec.allowed_next_ids(prefix)))
                if not allowed: raise RuntimeError('Codec returned an empty legal token set')
                if queues[i] and not finished[i]:
                    token = int(queues[i].popleft())
                    if token not in allowed: raise RuntimeError('Learned copy queue violates the unchanged codec grammar')
                    allowed = [token]
                if execution_head is not None and not observe_only and not finished[i]:
                    token = qualitative_token(codec, current_values[i], prefix, controls[i])
                    if token is not None:
                        if token not in allowed: raise RuntimeError('Learned qualitative control violates the unchanged codec grammar')
                        allowed = [int(token)]
                if inventory is not None and not observe_only and not finished[i]:
                    token = source_inventory_token(codec, current_values[i], prefix, inventory[i])
                    if token is not None:
                        if token not in allowed: raise RuntimeError('Learned source inventory violates the unchanged codec grammar')
                        allowed = [int(token)]
                legal[i, allowed] = True
            current = logits.masked_fill(~torch.from_numpy(legal).to(logits.device), -torch.inf).argmax(-1)
            current_values = current.tolist()
            for i, token in enumerate(current_values):
                if finished[i]: continue
                prefixes[i].append(token); finished[i] = token == int(codec.eos_id)
            if all(finished): return prefixes, traces
        raise CopyDecodeLimitError(max_plan_tokens, prefixes, traces, finished)
    finally:
        hook.remove()
