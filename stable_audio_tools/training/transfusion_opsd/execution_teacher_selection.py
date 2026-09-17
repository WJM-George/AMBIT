"""Select useful self-generated terminals within each *same-plan* group.

The input contains request-side execution observations only. Qualification is
partial evidence, not a claim that an entire audio target is correct. Failed
executions remain in the group baseline, but never become regression targets.
"""
import math


def same_plan_improvement_weights(metrics, plan_weights, *, temperature=.05,
                                  minimum_advantage=0., semantic_tolerance=.01,
                                  preservation_reward_weight=.5):
    if temperature <= 0 or minimum_advantage < 0 or semantic_tolerance < 0:
        raise ValueError('Invalid execution-teacher selection parameters.')
    masses = [float(x) for x in plan_weights]
    if not all(math.isfinite(x) and x >= 0 for x in masses):
        raise ValueError('Plan weights must be finite nonnegative stopped values.')
    coefficients = [0.] * len(metrics)
    groups = []
    for plan_index, mass in enumerate(masses):
        indices = [i for i, m in enumerate(metrics) if m['plan_index'] == plan_index]
        if not indices:
            continue
        values, eligible, failures = [], [], {}
        for i in indices:
            m = metrics[i]
            keep = m.get('unchanged_windows', {})
            penalty = min(1., float(keep['covariance_error'])) if keep.get('available') else 0.
            value = (float(m['reward']) - preservation_reward_weight * penalty
                     - float(not m.get('qualified_terminal', False)))
            if not math.isfinite(value):
                raise ValueError('Nonfinite observed execution reward.')
            values.append(value)
            reasons = []
            if not m.get('qualified_terminal', False):
                reasons.append('failed_existing_qualification')
            # A comparison against the fixed reference must have actually
            # been measured; a missing observation is never a semantic pass.
            score, reference = m.get('anchored_semantic'), m.get('fixed_reference_semantic')
            if score is None or reference is None or not all(math.isfinite(float(v)) for v in (score, reference)):
                reasons.append('missing_fixed_semantic_observation')
            elif score < reference - semantic_tolerance:
                reasons.append('fixed_semantic_floor')
            if reasons:
                failures[i] = reasons
            else:
                eligible.append(i)
        baseline = sum(values) / len(values)
        advantages = {i: v - baseline for i, v in zip(indices, values)}
        # Two repeated records of one noise are not two execution samples.
        independent = len({metrics[i]['seed'] for i in indices}) >= 2
        selected = [i for i in eligible if independent and advantages[i] > minimum_advantage]
        # Keep the previous qualified terminal mass ceiling. Selecting one
        # execution must not silently double the DiT objective for that plan.
        qualified_fraction = sum(bool(metrics[i].get('qualified_terminal')) for i in indices) / len(indices)
        budget = mass * qualified_fraction
        if selected:
            peak = max(advantages[i] / temperature for i in selected)
            raw = [math.exp(advantages[i] / temperature - peak) for i in selected]
            normalizer = sum(raw)
            for i, value in zip(selected, raw):
                coefficients[i] = budget * value / normalizer
        groups.append(dict(plan_index=plan_index, indices=indices, baseline=baseline,
                           advantages=advantages, selected=selected, rejected=failures,
                           independent_noises=independent, mass_ceiling=budget,
                           actual_mass=sum(coefficients[i] for i in indices)))
    if any(not 0 <= m['plan_index'] < len(masses) for m in metrics):
        raise ValueError('An execution has no matching plan weight.')
    return coefficients, dict(groups=groups, selected=sum(c > 0 for c in coefficients),
                             proposed=len(metrics), information='Observed same-plan execution reward and fixed semantic floor; no target audio',
                             objective='Positive within-plan improvement terminal RF; not intermediate-state repair')
