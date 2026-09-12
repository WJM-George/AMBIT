"""Experimental EVENT planning decision after observing a real DiT prefix.

This changes the generation schedule to AR -> DiT prefix -> AR refinement ->
DiT suffix. Training-side execution scores choose teachers; the student only
receives its existing request/proposal states and the current latent query.
"""
import copy

import torch
from torch import nn

from .event_completion import completion_geometry
from .event_policy import EventBehavior
from .event_refinement import refinement_candidates
from .event_refinement_policy import EventRefinementPolicy
from .execution_query import execution_query_features


class EventFeedbackPolicy(EventRefinementPolicy):
    contract = 'event_dit_query_conditioned_refinement_policy_v1_experimental'

    def __init__(self, bundle, *, latent_channels=64, temporal_bins=4):
        super().__init__(bundle)
        from .event_experiment import state_fingerprint
        self.initial_refinement_fingerprint = state_fingerprint(self)
        self.latent_channels, self.temporal_bins = latent_channels, temporal_bins
        size = latent_channels * temporal_bins * 4 + 1
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(18323)
            self.completion_head.query_adapter = nn.Sequential(nn.LayerNorm(size), nn.Linear(size, 128),
                nn.SiLU(), nn.Linear(128, self.completion_head.hidden_dim)).to(self.device)
        # The existing tanh/output bound preserves initial greedy action zero
        # for every possible query, including this new adapter's output.
        self.eval()

    def prepare_feedback_query(self, proposal, query, *, executor=None):
        """Detach execution observations once before fixed-target fitting."""
        return query

    def feedback_forward(self, proposal, query, *, use_feedback=True, executor=None):
        if query.state.shape[:2] != (1, self.latent_channels):
            raise ValueError('EVENT feedback requires one actual model query')
        output = super().decision_forward(proposal)
        features = execution_query_features(query, bins=self.temporal_bins).to(self.device)
        if not use_feedback:
            features = torch.zeros_like(features)
        message = self.completion_head.query_adapter(features)[:, None]
        geometry = completion_geometry(proposal.plan)[None].to(self.device)
        output['completion'] = self.completion_head(output['source_queries'] + message,
            output['qualitative'], geometry)
        return output

    @torch.no_grad()
    def feedback_act(self, proposal, query, *, use_feedback=True, executor=None):
        output = self.feedback_forward(proposal, query, use_feedback=use_feedback, executor=executor)
        logits = output['completion'][0].double().masked_fill(~output['completion_legal'][0], -torch.inf)
        actions = tuple(logits.argmax(-1).tolist())
        plan = copy.deepcopy(proposal.plan)
        for slot, action in enumerate(actions):
            alternatives, legal = refinement_candidates(self.bundle.codec, proposal.plan, proposal.trace, slot)
            if not legal[action]:
                raise ValueError('feedback policy selected an inadmissible action')
            plan['sources'][slot] = copy.deepcopy(alternatives[action]['sources'][slot])
        return EventBehavior(proposal, actions, plan, None)

    @torch.no_grad()
    def generate_feedback_audio(self, observation, *, seed, query_index, model_version,
            use_feedback=True, executor=None, reference_frames=None, return_trace=False):
        """Run the actual interleaved schedule without computing a future first.

        An explicit executor supports A-new/D-old branch interventions. It is
        never substituted silently in the ordinary student generation path.
        """
        from ...inference.sampling import sample_discrete_euler
        from .execution_query import ExecutionQuery
        from .objectives import clean_prediction
        if type(query_index) is not int or not 0 < query_index < 100:
            raise ValueError('feedback must occur inside the native 100-step schedule')
        bundle = self.bundle if executor is None else executor
        proposal = self.propose([observation])[0]
        condition = bundle.render_condition(observation, proposal.plan)
        from .event_counterfactual import paired_gaussian_noise
        noise = paired_gaussian_noise(seed, (1, self.latent_channels, condition.mask.shape[-1]),
            reference_frames=condition.mask.shape[-1] if reference_frames is None else reference_frames).to(bundle.device)
        times = torch.tensor(bundle.schedule(100, noise.shape[-1]), device=bundle.device, dtype=torch.float32)
        original_velocity = bundle.velocity_function(condition, differentiable=False)
        states = []
        callback = (lambda values: states.append(values['x'].detach().clone())) if return_trace else None
        state = sample_discrete_euler(original_velocity, noise, times[:query_index + 1], disable_tqdm=True, callback=callback)
        time = times[query_index:query_index + 1]
        query = ExecutionQuery(state.detach(), clean_prediction(state, time,
            original_velocity(state, time)).detach(), time, condition.mask, model_version)
        query = self.prepare_feedback_query(proposal, query, executor=bundle)
        behavior = self.feedback_act(proposal, query, use_feedback=use_feedback, executor=bundle)
        selected_condition = bundle.render_condition(observation, behavior.plan)
        if (selected_condition.model_num_samples != condition.model_num_samples
                or not torch.equal(selected_condition.mask, condition.mask)):
            raise ValueError('a refinement action cannot change the already running trajectory geometry')
        selected_velocity = (original_velocity if behavior.plan == proposal.plan
            else bundle.velocity_function(selected_condition, differentiable=False))
        final = sample_discrete_euler(selected_velocity, state, times[query_index:], disable_tqdm=True, callback=callback)
        audio = bundle.decode_for_reward(final, selected_condition.model_num_samples)
        if return_trace:
            from .objectives import EulerTrace
            states.append(final.detach().clone())
            context = {'trace': EulerTrace(tuple(states), tuple(times.tolist()), condition.mask),
                'original_velocity': original_velocity, 'selected_velocity': selected_velocity,
                'original_condition': condition, 'selected_condition': selected_condition}
            return behavior, query, audio, context
        return behavior, query, audio

    def frozen_copy(self):
        result, _ = load_feedback_policy(self.bundle.release_path, device=self.device,
            latent_channels=self.latent_channels, temporal_bins=self.temporal_bins)
        result.load_state_dict(self.state_dict(), strict=True)
        return result.eval().requires_grad_(False)


def load_feedback_policy(release_path, *, device, latent_channels=64, temporal_bins=4):
    from .generation_event import load_event_generation
    bundle, receipt = load_event_generation(release_path, device=device, qwen_runtime='torch_reference', dit_runtime='fp32')
    policy = EventFeedbackPolicy(bundle, latent_channels=latent_channels, temporal_bins=temporal_bins)
    return policy, {**receipt, 'policy_contract': policy.contract,
        'query_features': 'masked temporal means/std of actual query state and current clean prediction, plus time',
        'latent_channels': latent_channels, 'temporal_bins': temporal_bins,
        'inference_schedule': 'native AR proposal -> native DiT prefix -> query-conditioned AR action -> native DiT suffix',
        'student_observes_execution_rewards': False, 'student_observes_future_suffix': False,
        'initial_greedy': 'existing output bound preserves action 0 for every query'}
