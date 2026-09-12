"""Separate capability acceptance from successful minibatch distillation.

Inputs are paired, higher-is-better metrics from a predeclared evaluation.
Bootstrap units are independent scene/asset-family clusters, not K*M renders.
These finite-data checks are not a universal no-regression theorem.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Sequence
import math

import numpy as np


ARMS = frozenset({"new_ar_old_dit", "old_ar_new_dit", "new_ar_new_dit", "canonical_plan_new_dit"})


@dataclass(frozen=True)
class PromotionRule:
    arm: str
    group: str
    metric: str
    strict_gain: bool
    absolute_floor: float
    min_clusters: int = 32


@dataclass(frozen=True)
class PairedMeasurement:
    arm: str
    group: str
    metric: str
    cluster_id: str
    baseline: float
    candidate: float


def assess_promotion(measurements: Sequence[PairedMeasurement], rules: Sequence[PromotionRule], *,
                     protocol_id: str, candidate_id: str, alpha: float = .05,
                     candidate_looks: int = 1, resamples: int = 20000, seed: int = 42):
    """Zero regression margins, absolute floors, and complete declared coverage.

    candidate_looks must be fixed before repeated promotion attempts. Bonferroni
    allocates the error budget across looks and rules; bootstrap inference still
    requires enough representative independent clusters and calibrated metrics.
    """
    if not protocol_id or not candidate_id or not 0 < alpha < 1 or candidate_looks < 1:
        raise ValueError("promotion requires a pinned protocol/candidate and valid error budget")
    keys = [(r.arm, r.group, r.metric) for r in rules]
    if not rules or len(set(keys)) != len(keys) or {r.arm for r in rules if r.strict_gain} != ARMS:
        raise ValueError("all four comparisons require independent primary improvement rules")
    if {r.arm for r in rules if not r.strict_gain} != ARMS:
        raise ValueError("all four comparisons also require explicit capability protection rules")
    if any(r.arm not in ARMS or r.min_clusters < 2 or not math.isfinite(r.absolute_floor) for r in rules):
        raise ValueError("invalid promotion rule")
    tail = alpha / (candidate_looks * len(rules))
    if resamples * tail < 10:
        raise ValueError("too few bootstrap resamples for the declared multiplicity")
    grouped = {key: {} for key in keys}
    for m in measurements:
        if not m.cluster_id or not all(math.isfinite(v) for v in (m.baseline, m.candidate)):
            raise ValueError("invalid measurement; failures need explicit fixed-denominator scores")
        key = (m.arm, m.group, m.metric)
        if key not in grouped:
            raise ValueError("measurement was not declared in the promotion protocol")
        grouped[key].setdefault(m.cluster_id, []).append((m.baseline, m.candidate))
    rng = np.random.default_rng(seed)
    results = []
    for rule, key in zip(rules, keys):
        clusters = grouped[key]
        result = {**asdict(rule), "clusters": len(clusters), "passed": False}
        if len(clusters) < rule.min_clusters:
            result["reason"] = "insufficient_independent_coverage"
        else:
            pairs = np.asarray([np.mean(clusters[name], axis=0) for name in sorted(clusters)])
            delta = pairs[:, 1] - pairs[:, 0]
            # Batch the bootstrap so large validation sets do not allocate a
            # resamples x scenes matrix all at once.
            means = []
            for begin in range(0, resamples, 256):
                indices = rng.integers(0, len(delta), size=(min(256, resamples - begin), len(delta)))
                means.append(delta[indices].mean(1))
            lower = float(np.quantile(np.concatenate(means), tail))
            mean_candidate = float(pairs[:, 1].mean())
            passed = (lower > 0 if rule.strict_gain else lower >= 0) and mean_candidate >= rule.absolute_floor
            result.update(lower_bound=lower, mean_gain=float(delta.mean()), mean_candidate=mean_candidate,
                          passed=bool(passed), reason="passed" if passed else "gain_protection_or_floor_failed")
        results.append(result)
    return {"protocol_id": protocol_id, "candidate_id": candidate_id,
            "accepted": all(row["passed"] for row in results), "rules": results,
            "alpha": alpha, "candidate_looks": candidate_looks, "seed": seed,
            "regression_margin": 0., "bootstrap_resamples": resamples}


def evaluate_cross_swaps(incumbent, candidate, *, evaluate, evaluate_canonical):
    """Evaluate full route objects, including their conditioner projections.

    evaluate(actor, renderer) returns paired-scene raw metrics. Canonical-plan
    evaluation bypasses AR. No shared parameter tensor is overwritten to fake
    a swap; independently owned complete bundles are required.
    """
    if incumbent is candidate or incumbent.mode != candidate.mode:
        raise ValueError("cross-swaps require two distinct bundles of the same mode")
    if {id(p) for p in incumbent.parameters()} & {id(p) for p in candidate.parameters()}:
        raise ValueError("candidate and incumbent still alias mutable route parameters")
    return {
        "old_ar_old_dit": evaluate(incumbent, incumbent),
        "new_ar_old_dit": evaluate(candidate, incumbent),
        "old_ar_new_dit": evaluate(incumbent, candidate),
        "new_ar_new_dit": evaluate(candidate, candidate),
        "canonical_plan_old_dit": evaluate_canonical(incumbent),
        "canonical_plan_new_dit": evaluate_canonical(candidate),
    }
