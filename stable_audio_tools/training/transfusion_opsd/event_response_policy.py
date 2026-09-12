"""Native EVENT interaction using current conditional DiT responses.

Current execution inputs are detached once for a fixed training transaction.
Actual inference recomputes them with the executor that produced the prefix,
including explicit frozen-DiT branch interventions.
"""
from dataclasses import dataclass

import torch

from .event_completion import ATTRIBUTES, completion_geometry
from .event_feedback_policy import EventFeedbackPolicy
from .event_refinement import refinement_candidates
from .event_refinement_policy import EventRefinementPolicy
from .execution_query import ExecutionQuery, execution_query_features
from .execution_response import CandidateResponseHead, execution_response_features
from .objectives import clean_prediction


@dataclass(frozen=True)
class ResponseExecutionQuery(ExecutionQuery):
    query_features: torch.Tensor
    response_features: torch.Tensor
    action_descriptors: torch.Tensor
    legal: torch.Tensor
    executor_identity: int

    def __post_init__(self):
        super().__post_init__()
        if (self.response_features.ndim != 3 or self.action_descriptors.ndim != 3
                or self.response_features.shape[:2] != self.action_descriptors.shape[:2]
                or self.legal.shape != self.response_features.shape[:2] or self.legal.dtype != torch.bool
                or self.query_features.ndim != 2 or self.query_features.shape[0] != self.state.shape[0]
                or self.response_features.shape[0] != self.state.shape[0]):
            raise ValueError('response inputs must align with the actual query and candidate support')
        if any(value.requires_grad or not torch.isfinite(value).all()
                for value in (self.query_features, self.response_features, self.action_descriptors)):
            raise ValueError('fixed response observations must be finite and detached')


class EventResponsePolicy(EventFeedbackPolicy):
    contract = 'event_current_response_interleaved_policy_v1_experimental'

    def __init__(self, bundle, *, latent_channels=64, temporal_bins=4, input_mode='current_response'):
        super().__init__(bundle, latent_channels=latent_channels, temporal_bins=temporal_bins)
        from .event_experiment import state_fingerprint
        self.initial_feedback_fingerprint = state_fingerprint(self)
        if input_mode not in ('request_only', 'query_moments', 'current_response'):
            raise ValueError('declare the candidate-response policy input mode')
        self.response_input_mode = input_mode
        self.attribute_order = sorted(ATTRIBUTES)
        self.response_head_shape = dict(context_dim=self.completion_head.hidden_dim,
            attribute_dim=sum(len(ATTRIBUTES[key]) for key in self.attribute_order),
            query_dim=latent_channels * temporal_bins * 4 + 1,
            response_dim=latent_channels * temporal_bins * 2 + 1, action_dim=8,
            width=128, initialization_seed=18325)
        # Retain complete original state for native planning and provenance.
        # Only the response scorer is used by the new feedback decision.
        self.completion_head.requires_grad_(False)
        self.completion_head.candidate_response = CandidateResponseHead(**self.response_head_shape).to(self.device)
        self.eval()

    @property
    def response_head(self):
        return self.completion_head.candidate_response

    @property
    def runtime_contract(self):
        return {'response_input_mode': self.response_input_mode,
            'latent_channels': self.latent_channels, 'temporal_bins': self.temporal_bins,
            'attribute_order': self.attribute_order, 'response_head_shape': self.response_head_shape,
            'original_action_prior': self.response_head.original_action_prior}

    @torch.no_grad()
    def prepare_feedback_query(self, proposal, query, *, executor=None):
        bundle = self.bundle if executor is None else executor
        if isinstance(query, ResponseExecutionQuery):
            if query.executor_identity != id(bundle):
                raise ValueError('cached response observations belong to a different executor')
            return query
        if len(proposal.plan['sources']) != 1 or query.state.shape[:2] != (1, self.latent_channels):
            raise ValueError('the first response policy fixture requires one native source/query')
        alternatives, legal = refinement_candidates(self.bundle.codec, proposal.plan, proposal.trace, 0)
        conditions = [bundle.render_condition(proposal.observation, plan) for plan in alternatives]
        cleans = []
        for action, condition in enumerate(conditions):
            if (not torch.equal(condition.mask, query.mask)
                    or condition.model_num_samples != conditions[0].model_num_samples):
                raise ValueError('response alternatives cannot change the already running trajectory geometry')
            if action and not legal[action]:
                cleans.append(cleans[0]); continue
            velocity = bundle.velocity_function(condition, differentiable=False)
            cleans.append(clean_prediction(query.state, query.time, velocity(query.state, query.time)).detach())
        current = ExecutionQuery(query.state, cleans[0], query.time, query.mask, query.model_version)
        return ResponseExecutionQuery(current.state, current.clean, current.time, current.mask, current.model_version,
            execution_query_features(current, bins=self.temporal_bins),
            execution_response_features(current, torch.stack(cleans, 1), bins=self.temporal_bins),
            torch.stack([completion_geometry(plan)[0] for plan in alternatives])[None].to(self.device),
            legal[None].to(self.device), id(bundle))

    def feedback_forward(self, proposal, query, *, use_feedback=True, executor=None):
        prepared = self.prepare_feedback_query(proposal, query, executor=executor)
        output = EventRefinementPolicy.decision_forward(self, proposal)
        if not torch.equal(output['completion_legal'][:, 0], prepared.legal):
            raise ValueError('fixed query response support differs from the actual native decision')
        attributes = torch.cat([output['qualitative'][key].float().softmax(-1)
            for key in self.attribute_order], -1)[:, 0]
        logits = self.response_head(output['source_queries'][:, 0], attributes, prepared.query_features,
            prepared.response_features, prepared.action_descriptors, prepared.legal,
            input_mode=self.response_input_mode if use_feedback else 'request_only')
        output['completion'] = logits[:, None]
        return output

    def frozen_copy(self):
        result, _ = load_response_policy(self.bundle.release_path, device=self.device,
            latent_channels=self.latent_channels, temporal_bins=self.temporal_bins, input_mode=self.response_input_mode)
        result.load_state_dict(self.state_dict(), strict=True)
        return result.eval().requires_grad_(False)


def load_response_policy(release_path, *, device, head_path=None, latent_channels=64, temporal_bins=4,
        input_mode='current_response'):
    from .generation_event import load_event_generation
    from .provenance import sha256_file
    bundle, receipt = load_event_generation(release_path, device=device,
        qwen_runtime='torch_reference', dit_runtime='fp32')
    policy = EventResponsePolicy(bundle, latent_channels=latent_channels, temporal_bins=temporal_bins, input_mode=input_mode)
    head_receipt = None
    if head_path is not None:
        payload = torch.load(head_path, weights_only=True, map_location='cpu')
        if (payload.get('head_kind') != 'candidate_response' or payload['full_rl_candidate']
                or payload['initial_model_fingerprint'] != policy.initial_feedback_fingerprint
                or payload['head_shape'] != policy.response_head_shape or payload['attribute_order'] != policy.attribute_order
                or payload['input_mode'] != input_mode):
            raise ValueError('response warm start must match the exact native foundation and head interface')
        policy.response_head.load_state_dict(payload['head_state'], strict=True)
        head_receipt = {'path': str(head_path), 'sha256': sha256_file(head_path),
            'objective': payload['objective'], 'steps': payload['steps'], 'scope': 'private-head warm start only',
            'initial_action_calibration': payload.get('initial_action_calibration'),
            'parent_head_sha256': payload.get('parent_head_sha256')}
    return policy, {**receipt, 'policy_contract': policy.contract, 'policy_runtime_contract': policy.runtime_contract,
        'response_head_shape': policy.response_head_shape,
        'attribute_order': policy.attribute_order, 'head_warm_start': head_receipt,
        'initial_feedback_fingerprint': policy.initial_feedback_fingerprint,
        'inference_schedule': 'full native AR proposal -> native DiT prefix -> current candidate predictions -> AR choice -> native suffix',
        'student_future_or_reward_input': False, 'fixed_query_responses_during_fitting': True,
        'cross_executor': 'prefix and candidate responses use the same explicitly selected executor',
        'initial_fixture': 'one source; full original native AR state retained'}
