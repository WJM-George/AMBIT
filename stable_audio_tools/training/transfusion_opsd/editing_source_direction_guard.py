"""Preserve observed source direction without requiring an erroneous old plan.

This is a finite, read-only update constraint. It does not give a new request
paired target labels, certify output quality, or make discrete choices smooth.
"""
from __future__ import annotations

import copy
import math

import torch

from .coarse_native_plan_retention import coarse_native_plan_retention
from .editing_native_update import name_editing_optimizer_groups
from .guarded_joint_update import guarded_joint_adam_update
from .native_spatial_credit import SpatialCreditWindow, native_spatial_credit


def angular_distance(a, b):
    return abs((a - b + 180.) % 360. - 180.)


@torch.no_grad()
def source_direction_evidence(waveform, *, sample_rate, source_kind, source_count,
                              window_seconds, quantiles, minimum_energy,
                              minimum_coherence, maximum_span_deg):
    """Derive fixed windows from source W energy, never generated output/GT.

    Single-source input identity is an explicit dataset input constraint.
    Multiple/uncertain sources and variable or unobservable direction cannot
    authorize exceptions. The evidence is detached and fixed for the block.
    """
    if source_kind not in ('speech', 'music', 'sound') or type(source_count) is not int:
        raise ValueError('Declare the existing input source kind and count.')
    if (not isinstance(sample_rate, int) or sample_rate <= 0 or not math.isfinite(window_seconds)
            or window_seconds <= 0 or not math.isfinite(maximum_span_deg)
            or not 0 <= maximum_span_deg <= 180):
        raise ValueError('Invalid source observation configuration.')
    if (waveform.ndim != 3 or waveform.shape[:2] != (1, 4)
            or waveform.dtype not in (torch.float32, torch.float64)
            or not torch.isfinite(waveform).all()):
        raise ValueError('Require finite FP32/64 source FOA in WYZX order.')
    quantiles = tuple(quantiles)
    if len(quantiles) < 3 or sorted(set(quantiles)) != list(quantiles) or not 0 < quantiles[0] < quantiles[-1] < 1:
        raise ValueError('Declare at least three increasing interior energy quantiles.')
    result = dict(available=False, source_kind=source_kind, source_count=source_count, windows=[],
                  window_rule='Whole source-input W energy quantiles; fixed before updating or sampling candidates.')
    if source_count != 1:
        return dict(result, reason='Input has multiple sources; no direction binding is authorized.')
    width = round(sample_rate * window_seconds)
    if width < 1 or width > waveform.shape[-1]:
        return dict(result, reason='Input is shorter than the fixed observation width.')
    cdf = waveform[0, 0].double().square().cumsum(-1)
    if cdf[-1] <= 0:
        return dict(result, reason='Silent source input.')
    windows = []
    for fraction in quantiles:
        center = int(torch.searchsorted(cdf, cdf[-1]*fraction))
        left = max(0, min(center-width//2, waveform.shape[-1]-width))
        windows.append(SpatialCreditWindow(left, left+width, 0., 32.5, 'source_input'))
    observed = native_spatial_credit(waveform, windows, minimum_energy=minimum_energy,
                                    minimum_coherence=minimum_coherence)
    for fraction, window, row in zip(quantiles, windows, observed['windows']):
        result['windows'].append(dict(quantile=fraction, start_sample=window.start_sample,
            end_sample=window.end_sample, observable=bool(row['observable']),
            energy=float(row['energy']), coherence=float(row['coherence']),
            azimuth_deg=math.degrees(math.atan2(float(row['unit'][1]),float(row['unit'][0])))
            if bool(row['observable']) else None))
    if not all(w['observable'] for w in result['windows']):
        return dict(result, reason='One or more declared source windows are unobservable.')
    angles = [w['azimuth_deg'] for w in result['windows']]
    span = max(angular_distance(a,b) for a in angles for b in angles)
    result['observed_span_deg'] = span
    if span > maximum_span_deg:
        return dict(result, reason='Source direction varies across windows; no static exception.')
    return dict(result, available=True, reason='Single source with consistent observable horizontal direction.')


def source_supported_plan_retention(before, after, *, tolerance, preserve_room,
                                     source_evidence=None, source_tolerance_deg=32.5):
    """Only waive a static existing-source azimuth change supported by input.

    Each observed window must become at least as accurate, and the candidate
    must lie in every registered source cone. Other failures remain failures.
    Core meaning/actual generated audio still require separate evaluation.
    """
    if not math.isfinite(source_tolerance_deg) or not 0 < source_tolerance_deg <= 180:
        raise ValueError('Declare a finite coarse source direction tolerance.')
    result = copy.deepcopy(coarse_native_plan_retention(before, after,
        tolerance=tolerance, preserve_room=preserve_room))
    result.update(source_supported_exceptions=[], old_plan_change_budget_passed=result['passed'])
    if not result['available'] or not source_evidence or not source_evidence['available']:
        return result
    if source_evidence['source_count'] != 1:
        raise ValueError('Evidence cannot bind a multi-source input.')
    kind = source_evidence['source_kind']
    sources = [[s for s in p['sources'] if s['kind']==kind] for p in (before,after)]
    if any(len(s)!=1 for s in sources):
        return result
    old, new = (s[0]['trajectory'] for s in sources)
    if old['type'] != 'static' or new['type'] != 'static':
        return result
    field = 'sources.'+kind+'.trajectory.position.azimuth_deg'
    if not any(f['field']==field for f in result['failures']):
        return result
    windows = source_evidence['windows']
    if len(windows)<3 or not all(w['observable'] and math.isfinite(w['azimuth_deg']) for w in windows):
        raise ValueError('A claimed source exception needs all fixed observations.')
    comparisons = [dict(start_sample=w['start_sample'],end_sample=w['end_sample'],
        before_error_deg=angular_distance(old['position']['azimuth_deg'],w['azimuth_deg']),
        after_error_deg=angular_distance(new['position']['azimuth_deg'],w['azimuth_deg'])) for w in windows]
    if all(w['after_error_deg'] <= source_tolerance_deg+1e-6
           and w['after_error_deg'] <= w['before_error_deg']+1e-6 for w in comparisons):
        result['failures'] = [f for f in result['failures'] if f['field'] != field]
        result['source_supported_exceptions'].append(dict(field=field,source_tolerance_deg=source_tolerance_deg,
            comparisons=comparisons,reason='Candidate is no worse in every source window and inside every coarse source cone.'))
    result['passed'] = not result['failures']
    return result


def guarded_source_supported_editing_update(adapter, optimizer, *, anchors, tolerance,
        trial_scales, preserve_room, source_tolerance_deg, stop_on_first_failure, scheduler=None):
    if adapter.training or type(stop_on_first_failure) is not bool:
        raise ValueError('Use eval mode and an explicit finite validation policy.')
    anchors = list(anchors)
    if not anchors or len({a['identifier'] for a in anchors}) != len(anchors):
        raise ValueError('Declare distinct native anchors.')
    name_editing_optimizer_groups(optimizer.param_groups)
    for item in anchors:
        if not coarse_native_plan_retention(item['plan'],item['plan'],tolerance=tolerance,
                                             preserve_room=preserve_room)['passed']:
            raise ValueError('The native anchor lacks supported source binding.')

    def validate():
        rows = []
        for item in anchors:
            plan,_ = adapter.native_plan(item['observation'])
            result = source_supported_plan_retention(item['plan'],plan,tolerance=tolerance,
                preserve_room=preserve_room,source_evidence=item.get('source_direction_evidence'),
                source_tolerance_deg=source_tolerance_deg)
            rows.append(dict(identifier=item['identifier'],plan=plan,result=result))
            if stop_on_first_failure and not result['passed']:
                break
        return dict(passed=len(rows)==len(anchors) and all(r['result']['passed'] for r in rows),
            rows=rows,expected_anchor_count=len(anchors),checked_anchor_count=len(rows),
            scope='Actual greedy plans and frozen source-input direction evidence; no generated-audio quality claim.')

    return guarded_joint_adam_update(adapter,optimizer,trial_scales=trial_scales,
                                     validate_native=validate,scheduler=scheduler)
