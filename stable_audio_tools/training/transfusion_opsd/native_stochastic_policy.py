"""Sample and rescore the complete released EVENT planning decisions.

Sampling extends the released greedy decoder without modifying its source.
Inventory/count, grammar-constrained kinds, ordered copy/event spans, feasible
timing pairs, motion/direction/radial categories and free AR tokens all have
explicit categorical distributions. Seeded numerical completion and copied
literal text remain deterministic. No request annotations enter this policy.
Normal evaluation continues to use NativeEventPolicy.propose unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import itertools
from pathlib import Path
from types import SimpleNamespace

import torch

from ...models import sceneplan_generation_ar_copy_pointer as pointer_module
from ...models import sceneplan_generation_ar_source_inventory as inventory_module
from ...models import sceneplan_generation_ar_qualitative_head as qualitative_module
from .native_timing_execution import feasible_timing_pairs, timing_pair_logits


def ordered_span_log_partition(start, end, mask, *, temperature=1.):
    if (start.ndim != 1 or start.shape != end.shape or mask.shape != start.shape
            or mask.dtype != torch.bool or not mask.any() or temperature <= 0):
        raise ValueError('An ordered span requires matching endpoint logits and support.')
    a = (start.double() / temperature).masked_fill(~mask, -torch.inf)
    b = (end.double() / temperature).masked_fill(~mask, -torch.inf)
    return torch.logsumexp(torch.logcumsumexp(a, 0) + b, 0)


def ordered_span_log_prob(start, end, mask, selected, *, temperature=1.):
    low, high = selected
    if not 0 <= low <= high < start.numel() or not mask[low] or not mask[high]:
        raise ValueError('Selected literal span is outside the sampling support.')
    return ((start[low].double() + end[high].double()) / temperature
            - ordered_span_log_partition(start, end, mask, temperature=temperature))


def kind_assignments(count):
    if count not in range(1, 5):
        raise ValueError('The native codec supports one to four sources.')
    return tuple(values for values in itertools.product(range(3), repeat=count) if values.count(2) <= 1)


@dataclass
class SampledNativePlan:
    observation: object
    tokens: tuple[int, ...]
    trace: list[dict]
    plan: dict | None
    inventory: dict
    decisions: list[dict]
    temperature: float
    failure: str | None = None

    @property
    def old_log_prob(self):
        return sum(row['old_log_prob'] for row in self.decisions)


class _Recorder:
    def __init__(self, *, seed, temperature, greedy=False):
        if temperature <= 0:
            raise ValueError('Sampling temperature must be positive.')
        self.generator = torch.Generator(device='cpu').manual_seed(seed)
        self.temperature, self.greedy = float(temperature), bool(greedy)
        self.decisions = []

    def categorical(self, logits, **site):
        values = logits.detach().double().cpu() / self.temperature
        if values.ndim != 1 or not torch.isfinite(values).all():
            raise ValueError('A categorical decision requires finite supported logits.')
        log_probs = values.log_softmax(0)
        selected = int(values.argmax()) if self.greedy else int(torch.multinomial(log_probs.exp(), 1, generator=self.generator))
        self.decisions.append(dict(**site, selected=selected, old_log_prob=float(log_probs[selected])))
        return selected

    def span(self, start, end, mask, **site):
        a = (start.detach().double().cpu() / self.temperature).masked_fill(~mask.cpu(), -torch.inf)
        b = (end.detach().double().cpu() / self.temperature).masked_fill(~mask.cpu(), -torch.inf)
        if self.greedy:
            prefix, indices = a.cummax(0)
            high = int((prefix + b).argmax())
            low = int(indices[high])
        else:
            high = int(torch.multinomial((torch.logcumsumexp(a, 0) + b).softmax(0), 1, generator=self.generator))
            low = int(torch.multinomial(a[:high + 1].softmax(0), 1, generator=self.generator))
        log_prob = ordered_span_log_prob(start.detach(), end.detach(), mask, (low, high), temperature=self.temperature)
        self.decisions.append(dict(**site, selected=[low, high], old_log_prob=float(log_prob)))
        return low, high


def _qualitative_inputs(bundle, context, mask, alignment, inventory):
    # Always use four slots, exactly as the released decision-block decoder.
    first = mask.long().argmax(-1, keepdim=True)
    starts, ends = first.expand(-1, 4).clone(), first.expand(-1, 4).clone()
    fields = torch.zeros(1, 4, device=context.device, dtype=torch.long)
    scopes = torch.stack((starts, ends), -1)
    for slot, source in enumerate(inventory['sources']):
        fields[0, slot] = int(source['kind'] == 'speech')
        starts[0, slot] = alignment.token_indices[0, source['identity']['start']]
        ends[0, slot] = alignment.token_indices[0, source['identity']['end']]
        event = source['event']
        scopes[0, slot, 0] = min(int(alignment.token_indices[0, event['start']]), int(starts[0, slot]))
        scopes[0, slot, 1] = max(int(alignment.token_indices[0, event['end']]), int(ends[0, slot]))
    return bundle.qualitative_head(context.new_zeros(1, 4, 1024), fields, starts, ends,
        context, mask, source_spans=scopes)


@torch.no_grad()
def sample_native_plan(policy, observation, *, seed, temperature=1., greedy_diagnostic=False, max_tokens=512):
    """Sample once; malformed/over-budget plans return their trace as failures.

    greedy_diagnostic exists only for decoder-parity testing. Those samples
    must not be included in an on-policy GRPO batch.
    """
    bundle = policy.bundle
    if bundle.qualitative_head.use_ar_query or not bundle.qualitative_head.source_local_attention:
        raise ValueError('This adapter requires the released EVENT local-attention head.')
    recorder = _Recorder(seed=seed, temperature=temperature, greedy=greedy_diagnostic)
    inventory = {}

    def predict_inventory(head, module, pointer, requests, context, mask, alignment):
        if len(requests) != 1:
            raise ValueError('Collection and rescore both use batch size one.')
        with torch.autocast(device_type=context.device.type, enabled=False):
            values = head(context, mask, alignment)
        count = recorder.categorical(values['count'][0], family='inventory', kind='count') + 1
        assignments = kind_assignments(count)
        scores = torch.stack([sum(values['kind'][0, slot, k] for slot, k in enumerate(a)) for a in assignments])
        assignment = assignments[recorder.categorical(scores, family='inventory', kind='kinds', count=count)]
        sources = []
        for slot, kind in enumerate(assignment):
            source = dict(kind=module.KINDS[kind], unconstrained_kind=module.KINDS[kind])
            for field, name in enumerate(('identity', 'transcript', 'event')):
                if name == 'transcript' and kind != 2:
                    continue  # This branch never consumes a non-speech transcript.
                low, high = recorder.span(values['start'][0, slot, field], values['end'][0, slot, field],
                    alignment.endpoint_mask[0], family='inventory', kind='span', slot=slot, field=field)
                source[name] = dict(start=low, end=high, text=requests[0][low:high + 1])
            sources.append(source)
        inventory.update(count=count, sources=sources)
        return [inventory]

    source_slot = 0

    def complete(logits, frames, *, seed_key, speech):
        nonlocal source_slot
        slot, source_slot = source_slot, source_slot + 1
        scores = {name: torch.tensor(values, dtype=torch.float64) for name, values in logits.items()}
        def choose(values, kind):
            return recorder.categorical(values, family='qualitative', kind=kind, slot=slot)
        labels = {'motion': choose(scores['motion'], 'motion')}
        if labels['motion'] == 0:
            labels['start'] = labels['end'] = choose(scores['start'] + scores['end'], 'static_direction')
            labels['radial'] = 0
        else:
            for name in ('start', 'end', 'radial'):
                labels[name] = choose(scores[name], name)
        pair = recorder.categorical(timing_pair_logits(scores['onset'], scores['offset'], frames),
            family='qualitative', kind='timing', slot=slot, frames=frames)
        labels['onset'], labels['offset'] = feasible_timing_pairs(frames)[pair]
        forced = {name: [0. if i == labels[name] else -1000. for i in range(len(values))]
                  for name, values in logits.items()}
        return qualitative_module.complete_from_logits(forced, frames, seed_key=seed_key, speech=speech)

    # Give this call its own decoder globals: no monkeypatch of a shared import
    # or running OPSD/evaluation code is performed.
    path = Path(pointer_module.__file__).parents[1] / 'inference/sceneplan_generation_ar_decision_blocks.py'
    spec = importlib.util.spec_from_file_location('_native_stochastic_decoder', path)
    decoder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(decoder)
    decoder.predict_source_inventory = predict_inventory

    def sample_token(module, args, output):
        tokens, mask = args[:2]
        length = int(mask[0].sum())
        prefix = tuple(tokens[0, :length].tolist())
        support = sorted(bundle.codec.allowed_next_ids(prefix))
        if len(support) <= 1:
            raise ValueError('The decision-block decoder called AR without a free decision.')
        index = recorder.categorical(output[0, length - 1, support], family='token', kind='free',
            prefix=list(prefix), support=support)
        modified = output.clone()
        modified[0, length - 1, support] = -torch.inf
        modified[0, length - 1, support[index]] = 0.
        return modified

    handle = bundle.ar.register_forward_hook(sample_token)
    try:
        tokens, traces = decoder.generate_with_learned_copy(bundle.ar, bundle.copy_pointer, pointer_module,
            [observation.request], bundle.codec, device=bundle.device, max_plan_tokens=max_tokens,
            execution_head=bundle.qualitative_head, execution_module=SimpleNamespace(complete_from_logits=complete),
            inventory_head=bundle.source_inventory, inventory_module=inventory_module)
        plan = bundle.codec.decode(tokens[0], sample_id=observation.sample_id)
        return SampledNativePlan(observation, tuple(tokens[0]), traces[0], plan, inventory,
                                 recorder.decisions, temperature)
    except decoder.CopyDecodeLimitError as exc:
        return SampledNativePlan(observation, (), [], None, inventory, recorder.decisions, temperature,
                                 failure=f'{type(exc).__name__}: {exc}')
    finally:
        handle.remove()


def rescore_native_plan(policy, sample):
    """Differentiable probability of the recorded executed decision sequence."""
    bundle = policy.bundle
    context, mask = bundle.encode_event_requests([sample.observation.request], device=bundle.device)
    alignment = pointer_module.encode_character_alignment(bundle.prompt_conditioner.tokenizer,
        [sample.observation.request], mask, device=bundle.device)
    with torch.autocast(device_type=context.device.type, enabled=False):
        inventory = bundle.source_inventory(context, mask, alignment)
        qualitative = _qualitative_inputs(bundle, context, mask, alignment, sample.inventory)
    probabilities = []
    for row in sample.decisions:
        family, kind, selected = row['family'], row['kind'], row['selected']
        if family == 'inventory':
            if kind == 'count':
                logits = inventory['count'][0]
            elif kind == 'kinds':
                logits = torch.stack([sum(inventory['kind'][0, slot, k] for slot, k in enumerate(a))
                                      for a in kind_assignments(row['count'])])
            else:
                slot, field = row['slot'], row['field']
                probabilities.append(ordered_span_log_prob(inventory['start'][0, slot, field],
                    inventory['end'][0, slot, field], alignment.endpoint_mask[0], selected,
                    temperature=sample.temperature))
                continue
        elif family == 'qualitative':
            slot = row['slot']
            if kind == 'static_direction':
                logits = qualitative['start'][0, slot].double() + qualitative['end'][0, slot].double()
            elif kind == 'timing':
                logits = timing_pair_logits(qualitative['onset'][0, slot].double(), qualitative['offset'][0, slot].double(), row['frames'])
            else:
                logits = qualitative[kind][0, slot]
        elif family == 'token':
            tokens = torch.tensor([row['prefix']], dtype=torch.long, device=bundle.device)
            logits = bundle.ar(tokens, torch.ones_like(tokens, dtype=torch.bool), context, mask)[0, -1, row['support']]
        else:
            raise ValueError(f'Unknown native policy family {family!r}')
        probabilities.append((logits.double() / sample.temperature).log_softmax(0)[selected])
    return torch.stack(probabilities)
