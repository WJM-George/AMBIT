"""Explicit task roles for a common AR/DiT repair teacher.

The spatial profile changes the objective, not the measured direction metric.
It keeps every content/ASR protection and does not relabel semantic-only runs.
"""
from __future__ import annotations

import math
import copy

from .event_latent_target import protected_latent_objective
from .event_task_quality import semantic_similarity, select_query_teacher
from .joint_repair import select_joint_repair_teacher
from .objectives import RewardScore


class EventCommonRepairTask:
    def __init__(self, *, kind, tolerances, minimum_gain, utility_tolerance,
                 content_tolerance=None):
        if kind not in ('semantic_v1', 'requested_sector_v1'):
            raise ValueError('declare a supported common repair task')
        values = [utility_tolerance, *tolerances.values()]
        if (not math.isfinite(minimum_gain) or minimum_gain <= 0
                or any(not math.isfinite(x) or x < 0 for x in values)):
            raise ValueError('finite protection and a positive task gain required')
        if kind == 'requested_sector_v1' and (
                'requested_sector_failure' not in tolerances
                or content_tolerance is None or not math.isfinite(content_tolerance)
                or content_tolerance < 0):
            raise ValueError('spatial repair requires explicit sector and content protection')
        if any(key.startswith('clap_content_cost/') for key in tolerances):
            raise ValueError('source content coverage comes from the measured request')
        self.kind = kind
        self.tolerances = dict(tolerances)
        self.minimum_gain = minimum_gain
        self.utility_tolerance = utility_tolerance
        self.content_tolerance = content_tolerance

    @classmethod
    def from_protocol(cls, protocol):
        spec = protocol.get('repair_task')
        if spec is None:
            return cls(kind='semantic_v1', tolerances=protocol['tolerances'],
                minimum_gain=protocol['minimum_semantic_gain'],
                utility_tolerance=protocol['semantic_tolerance'])
        return cls(tolerances=protocol['tolerances'], **spec)

    @property
    def spatial(self):
        return self.kind == 'requested_sector_v1'

    def quality(self, raw):
        noncontent = {k: v for k, v in raw.costs.items() if not k.startswith('clap_content_cost/')}
        if noncontent.keys() != self.tolerances.keys():
            raise ValueError('common repair task cannot drop protection coverage')
        semantic = semantic_similarity(raw)  # Also require measured source content.
        if self.spatial:
            failure = raw.costs['requested_sector_failure']
            if not 0 <= failure <= 1:
                raise ValueError('requested sector failure must be a fraction')
            return RewardScore(1. - failure, dict(raw.costs))
        return RewardScore(semantic, noncontent)

    def score_tolerances(self, score):
        extra = set(score.costs) - set(self.tolerances)
        if (not set(self.tolerances) <= set(score.costs)
                or any(not key.startswith('clap_content_cost/') for key in extra)
                or self.spatial != bool(extra)):
            raise ValueError('task score must retain its declared protection coverage')
        return {**self.tolerances, **{key: self.content_tolerance for key in extra}}

    def limits(self, original_raw):
        score = self.quality(original_raw)
        return {key: score.costs[key] + tol for key, tol in self.score_tolerances(score).items()}

    def certificate(self, raw, *, limits):
        score = self.quality(raw)
        if score.costs.keys() != limits.keys():
            raise ValueError('certification cannot replace source content or fixed limits')
        return RewardScore(score.utility,
            {key: max(0., value - limits[key]) for key, value in score.costs.items()})

    def latent_objective(self, latent, *, decode, scorer, requirements,
                         directional_reward, diagnostics):
        content = lambda audio: scorer.differentiable_content_cost(audio, requirements)
        if self.spatial:
            utility = lambda audio: -directional_reward.soft_sector_cost(audio)
            protected = content
        else:
            utility = lambda audio: -content(audio)
            protected = directional_reward.soft_sector_cost
        return protected_latent_objective(latent, decode=decode, utility=utility,
            protected_cost=protected, diagnostics=diagnostics)

    def select(self, variants, *, legal, repair_penalty):
        result = select_joint_repair_teacher(variants, legal=legal,
            tolerances=self.score_tolerances(variants[0][0]['score']),
            minimum_gain=self.minimum_gain, semantic_tolerance=self.utility_tolerance,
            repair_penalty=repair_penalty)
        return self.label_selection(result)

    def label_selection(self, result):
        if self.spatial:
            result = copy.deepcopy(result)
            # The generic selector retains its historical semantic_gain label.
            # A spatial report must state what that quantity now measures.
            for candidate in result['improving_plans']:
                candidate['utility_gain'] = candidate.pop('semantic_gain')
            result['utility_metric'] = 'one_minus_requested_sector_failure'
        return result

    def label_distillation(self, evidence):
        if not self.spatial:
            return evidence
        evidence = copy.deepcopy(evidence)
        evidence['selected_joint_teacher'] = self.label_selection(evidence['selected_joint_teacher'])
        for candidate in evidence['candidates']:
            if candidate is not None:
                candidate['utility_gain'] = candidate.pop('semantic_gain')
        evidence['utility_metric'] = 'one_minus_requested_sector_failure'
        return evidence

    def select_raw(self, raw_scores, variants, *, legal):
        if self.spatial:
            return self.select([row[:1] for row in variants], legal=legal, repair_penalty=0.)
        return select_query_teacher(raw_scores, legal=legal, tolerances=self.tolerances,
            minimum_semantic_gain=self.minimum_gain)
