"""Complete native EVENT planning, without the retired seven-action refiner.

Legacy full parents are imported by retaining every bundle tensor and omitting
only the explicitly inactive completion_head subtree. The original parent
file remains intact. New candidates use their own native policy contract.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .adapters import GenerationObservation


@dataclass(frozen=True)
class NativeFreeDecision:
    prefix: tuple[int, ...]
    legal_ids: tuple[int, ...]
    selected_token: int
    teacher_logits: torch.Tensor


@dataclass(frozen=True)
class NativeEventProposal:
    observation: GenerationObservation
    tokens: tuple[int, ...]
    trace: list[dict]
    plan: dict
    free_decisions: tuple[NativeFreeDecision, ...]


class NativeEventPolicy(nn.Module):
    contract = 'native_event_planning_and_execution_opsd_policy_v1'

    def __init__(self, bundle):
        super().__init__()
        self.bundle = bundle
        # The released inventory decoder does not invoke this older fallback.
        self.bundle.copy_pointer.requires_grad_(False)
        self.eval()

    @property
    def device(self):
        return self.bundle.device

    @property
    def runtime_contract(self):
        return dict(native_event_only=True, discrete_choice_gradients='distribution supervision',
            planning_input='raw request and preceding native plan tokens')

    def dependency_parameters(self):
        return {key: [('bundle.' + name, value) for name, value in items]
            for key, items in self.bundle.dependency_parameters().items()}

    @torch.no_grad()
    def propose(self, observations):
        captured = [[] for _ in observations]
        def hook(module, args, output):
            tokens, mask = args[:2]
            for index in range(tokens.shape[0]):
                length = int(mask[index].sum())
                prefix = tuple(tokens[index, :length].tolist())
                if prefix[-1] == self.bundle.codec.eos_id:
                    continue
                legal = tuple(sorted(self.bundle.codec.allowed_next_ids(prefix)))
                if len(legal) > 1:
                    captured[index].append((prefix, legal,
                        output[index, length - 1, list(legal)].detach().float().clone()))
        handle = self.bundle.ar.register_forward_hook(hook)
        try:
            tokens, traces = self.bundle.generate_released(observations)
        finally:
            handle.remove()
        proposals = []
        for observation, row, trace, decisions in zip(observations, tokens, traces, captured):
            free = []
            for prefix, legal, logits in decisions:
                if tuple(row[:len(prefix)]) != prefix:
                    raise ValueError('native decision is not on the actual sampled prefix')
                selected = int(row[len(prefix)])
                if selected != legal[int(logits.argmax())]:
                    raise ValueError('native logits do not explain the sampled greedy decision')
                free.append(NativeFreeDecision(prefix, legal, selected, logits))
            if not free:
                raise ValueError('native EVENT planner exposed no free decision')
            proposals.append(NativeEventProposal(observation, tuple(row), trace,
                self.bundle.codec.decode(row, sample_id=observation.sample_id), tuple(free)))
        return proposals

    def decision_forward(self, proposal):
        return self.bundle.planning_forward([proposal.observation], [proposal.tokens], [proposal.trace])


def import_legacy_parent(policy, payload):
    """Strict bundle import; unknown top-level subtrees are never discarded."""
    state = payload['model']
    retained = {name: tensor for name, tensor in state.items() if name.startswith('bundle.')}
    omitted = {name: tensor for name, tensor in state.items() if name.startswith('completion_head.')}
    if len(retained) + len(omitted) != len(state) or not retained:
        raise ValueError('legacy parent contains unknown or missing native model subtrees')
    policy.load_state_dict(retained, strict=True)
    return dict(retained_bundle_tensors=len(retained),
        omitted_inactive_completion_tensors=len(omitted),
        omitted_logical_tensor_elements=sum(tensor.numel() for tensor in omitted.values()),
        omitted_keys=sorted(omitted),
        original_parent_modified=False,
        optimizer='new optimizer must be declared; old completion momentum is not imported')


def load_native_event_policy(release_path, *, device, parent_path, parent_sha256):
    from .generation_event import load_event_generation
    from .provenance import sha256_file
    if not parent_sha256 or sha256_file(parent_path) != parent_sha256:
        raise ValueError('full native parent differs from the declared checkpoint')
    bundle, receipt = load_event_generation(release_path, device=device,
        qwen_runtime='torch_reference', dit_runtime='fp32')
    policy = NativeEventPolicy(bundle)
    payload = torch.load(parent_path, map_location='cpu', weights_only=True, mmap=True)
    if payload.get('qwen_runtime') != 'torch_reference' or payload.get('dit_runtime') != 'fp32':
        raise ValueError('native parent has a different numerical runtime')
    imported = import_legacy_parent(policy, payload)
    return policy, dict(release=receipt, parent=dict(path=str(parent_path), sha256=parent_sha256),
        imported=imported, policy_contract=policy.contract, runtime_contract=policy.runtime_contract)
