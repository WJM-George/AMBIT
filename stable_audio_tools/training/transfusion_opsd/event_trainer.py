"""Transactional fitting of fixed teachers on the full EVENT AR/DiT policy.

Collection and actual waveform validation remain explicit: a lower surrogate
loss is never sufficient to commit an EVENT update.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import tempfile

import torch

from .dual_route_batch import RouteLossTerm, dual_route_loss_closure
from .event_objectives import event_ar_loss
from .objectives import diffusion_loss
from .shared_step import (joint_adam_step, _cpu_clone, checkpoint_rng_state,
    decode_checkpoint_rng, _restore_rng)


@dataclass(frozen=True)
class EventFitConfig:
    ar_lr: float = 1e-5
    completion_lr: float = 1e-3
    shared_lr: float = 1e-7
    dit_lr: float = 1e-6
    min_loss_gain: float = 1e-8

    def __post_init__(self):
        if any(not math.isfinite(value) or value <= 0 for value in asdict(self).values()):
            raise ValueError('EVENT fitting rates and minimum gains must be finite and positive')


@dataclass(frozen=True)
class EventFitBatch:
    proposal: object
    teacher: object
    condition: object
    z: torch.Tensor
    time: torch.Tensor
    targets: object


def event_loss_closure(policy, batch):
    """Use identical autograd forward dispatch during fitting and certification.

    PyTorch eval attention may choose a different kernel under no_grad. Force
    the actual training path for both evaluations, detaching when the caller
    requests values only. This does not alter greedy inference or its baseline.
    """
    def closure():
        need_graph = torch.is_grad_enabled()
        with torch.enable_grad():
            output = policy.decision_forward(batch.proposal)
            ar = event_ar_loss(output, batch.teacher)
            velocity = policy.bundle.velocity_function(batch.condition, differentiable=True)
            dit = diffusion_loss(batch.z, batch.time, velocity(batch.z, batch.time), batch.targets)
        return (ar, dit, None) if need_graph else (ar.detach(), dit.detach(), None)
    return closure


class EventOPSDTrainer:
    checkpoint_contract = 'full_event_execution_coupled_opsd_candidate_v1'

    def __init__(self, policy, *, identity, config=EventFitConfig()):
        if policy.bundle.qwen_runtime != 'torch_reference':
            raise ValueError('EVENT pilot requires the independently matched Torch-reference runtime')
        if not identity or not all(key in identity for key in ('initialization', 'data', 'protocol')):
            raise ValueError('pin the complete initialization, data and protocol before fitting')
        self.policy, self.config = policy.eval(), config
        self.identity = json.loads(json.dumps(identity))
        self.partition = policy.dependency_parameters()
        ar_body = [(name, p) for name, p in self.partition['ar_private'] if not name.startswith('completion_head.')]
        completion = [(name, p) for name, p in self.partition['ar_private'] if name.startswith('completion_head.')]
        self.optimizer = torch.optim.AdamW([
            dict(params=[p for _, p in values], lr=lr, weight_decay=0., name=name)
            for name, values, lr in [('ar_body', ar_body, config.ar_lr),
                ('completion', completion, config.completion_lr),
                ('shared', self.partition['shared'], config.shared_lr),
                ('dit_private', self.partition['dit_private'], config.dit_lr)]], foreach=False)
        self.fit_attempts = 0
        self.commits = 0

    def fit(self, batch, *, validate_execution):
        if not callable(validate_execution):
            raise ValueError('actual post-update execution validation is mandatory')
        self.fit_attempts += 1
        result = joint_adam_step(self.policy, self.partition, self.optimizer,
            event_loss_closure(self.policy, batch), validate_execution=validate_execution,
            min_loss_gain=self.config.min_loss_gain)
        self.commits += int(result.committed)
        return result

    def save_candidate(self, path, *, collection_state):
        """Complete full-policy state; caller chooses a new, isolated artifact."""
        path = Path(path)
        if path.exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(contract=self.checkpoint_contract, policy_contract=self.policy.contract,
            policy_runtime_contract=getattr(self.policy, 'runtime_contract', None),
            identity=self.identity, config=asdict(self.config),
            qwen_runtime=self.policy.bundle.qwen_runtime, dit_runtime=self.policy.bundle.dit_runtime,
            qwen_fused_norm_runtime=getattr(self.policy.bundle, 'qwen_fused_norm_runtime', None),
            model=_cpu_clone(self.policy.state_dict()),
            optimizer=_cpu_clone(self.optimizer.state_dict()), rng=checkpoint_rng_state(),
            fit_attempts=self.fit_attempts, commits=self.commits,
            collection_state=json.loads(json.dumps(collection_state)),
            promotion_status='UNVALIDATED_CANDIDATE')
        fd, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
        os.close(fd)
        try:
            torch.save(payload, temporary)
            os.link(temporary, path)  # Atomic, and never overwrite an existing candidate.
        finally:
            os.unlink(temporary)

    def restore_candidate(self, path):
        payload = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        for key, expected in [('contract', self.checkpoint_contract), ('policy_contract', self.policy.contract),
                ('policy_runtime_contract', getattr(self.policy, 'runtime_contract', None)),
                ('identity', self.identity), ('config', asdict(self.config)),
                ('qwen_runtime', self.policy.bundle.qwen_runtime),
                ('qwen_fused_norm_runtime', getattr(self.policy.bundle, 'qwen_fused_norm_runtime', None))]:
            if payload.get(key) != expected:
                raise ValueError('EVENT candidate restore identity differs: ' + key)
        if payload.get('dit_runtime', 'native_bf16') != self.policy.bundle.dit_runtime:
            raise ValueError('EVENT candidate restore changed the declared DiT numerical runtime')
        # The policy must be freshly constructed, preserving scoped LoRA closures.
        self.policy.load_state_dict(payload['model'], strict=True)
        self.optimizer.load_state_dict(payload['optimizer'])
        self.fit_attempts, self.commits = int(payload['fit_attempts']), int(payload['commits'])
        _restore_rng(decode_checkpoint_rng(payload['rng']))
        return payload['collection_state']


@dataclass
class EventRouteSignals:
    ar: RouteLossTerm | None
    dit: RouteLossTerm | None
    record: dict
    behavior: object | None = None
    request_reward: object | None = None
    audio_reward: object | None = None
    proposal: object | None = None
    teacher: object | None = None
    condition: object | None = None
    z: object | None = None
    time: object | None = None
    targets: object | None = None

class EventBatchOPSDTrainer(EventOPSDTrainer):
    checkpoint_contract = 'full_event_batch_execution_coupled_opsd_candidate_v2'

    def fit_signals(self, signals, *, validate_execution):
        if not callable(validate_execution):
            raise ValueError('actual post-update execution validation is mandatory')
        closure = dual_route_loss_closure([value.ar for value in signals if value.ar is not None],
            [value.dit for value in signals if value.dit is not None], model_version=self.commits,
            ar_denominator=len(signals), dit_denominator=len(signals))
        self.fit_attempts += 1
        result = joint_adam_step(self.policy, self.partition, self.optimizer, closure,
            validate_execution=validate_execution, min_loss_gain=self.config.min_loss_gain)
        self.commits += int(result.committed)
        return result, closure.evidence


class EventRefinementTrainer(EventBatchOPSDTrainer):
    checkpoint_contract = 'full_event_greedy_refinement_coupled_opsd_candidate_v1'
