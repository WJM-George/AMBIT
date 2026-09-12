"""Coarse rendered-audio retention; callers supply calibrated spatial errors."""
from __future__ import annotations

import math

import numpy as np

from ...data.audio_activity import measure_audio_activity


def foa_activity(wave, sample_rate=44100):
    """W-channel extent, retaining level and silence evidence without rescaling."""
    value = np.asarray(wave)
    if value.ndim != 2 or value.shape[1] != 4 or not np.isfinite(value).all():
        raise ValueError('Expected finite [samples, W Y Z X] FOA audio.')
    return measure_audio_activity(value[:, :1], sample_rate)


def coarse_audio_retention(before, after, target, *, angle_tolerance=5., time_tolerance=.25):
    """Compare normal audio with itself and a paired rendered reference.

    Spatial inputs use one fixed observer/profile and the paired trajectory.
    The reference's own error is a calibration floor. Timing is audible extent
    relative to the paired audio, with coarse slack, not unique plan seconds.
    These observations do not certify source identity, transcript or room.
    """
    if any(not math.isfinite(x) or x < 0 for x in (angle_tolerance, time_tolerance)):
        raise ValueError('Declare finite nonnegative coarse tolerances.')
    for row in (before, after, target):
        for key in ('angle_deg', 'unobservable_fraction'):
            if not math.isfinite(row[key]):
                raise ValueError('Nonfinite spatial observations cannot pass retention.')
    angle_limit = max(before['angle_deg'], target['angle_deg'])+angle_tolerance
    failures = []
    if after['angle_deg'] > angle_limit:
        failures.append('audio_direction')
    if after['unobservable_fraction'] > before['unobservable_fraction']+1e-9:
        failures.append('direction_observability')
    errors = {}
    if target['activity']['all_silent']:
        failures.append('paired_timing_reference_unobservable')
    elif after['activity']['all_silent']:
        failures.append('generated_audio_silent')
    else:
        for key in ('activity_onset_sec', 'activity_offset_sec'):
            ref = target['activity'][key]
            a, b = before['activity'][key], after['activity'][key]
            if ref is None or b is None or not math.isfinite(ref) or not math.isfinite(b):
                raise ValueError('Audible extent is missing or nonfinite.')
            old_error = abs(a-ref) if a is not None else target['activity']['audio_duration_sec']
            new_error = abs(b-ref)
            errors[key] = dict(before_error_sec=old_error, after_error_sec=new_error,
                               allowed_error_sec=old_error+time_tolerance)
            if new_error > old_error+time_tolerance:
                failures.append(key)
    return dict(passed=not failures, failures=failures, timing=errors,
                direction=dict(before_deg=before['angle_deg'], after_deg=after['angle_deg'],
                               paired_reference_deg=target['angle_deg'], allowed_deg=angle_limit),
                scope='Coarse single-target audio retention only; separate native/content/source checks remain required.')
