"""Bounded horizontal correction of the same fixed-window FOA moment scored.

Unlike a per-frame direction objective, this proposal targets the observable
window-average direction. It is still only a physical teacher proposal;
tapering, VAE transport and real continuation must each be validated.
"""
import math

import torch

from .native_spatial_credit import SpatialCreditWindow, native_spatial_credit
from .source_window_horizontal_repair import native_azimuth_at


@torch.no_grad()
def repair_source_moments(waveform, source, windows, *, max_rotation_deg=30.,
                          tolerance_deg=30., interior_margin_deg=2.5, taper_seconds=.02):
    if (not 0 < max_rotation_deg <= 90 or not 0 < interior_margin_deg < tolerance_deg < 90
            or not math.isfinite(taper_seconds) or not 0 <= taper_seconds <= .1):
        raise ValueError('Require bounded rotation, a valid coarse cone and a short taper.')
    if not windows or any(w.source_id != source['source_id'] for w in windows):
        raise ValueError('Every window must identify the same native source.')
    desired = [SpatialCreditWindow(w.start_sample, w.end_sample,
        native_azimuth_at(source, (w.start_sample+w.end_sample)/2/44100),
        tolerance_deg, w.source_id) for w in windows]
    measured = native_spatial_credit(waveform, desired)
    support = torch.zeros(waveform.shape[-1], device=waveform.device, dtype=torch.bool)
    angle = torch.zeros(waveform.shape[-1], device=waveform.device, dtype=torch.float64)
    receipts = []
    for window, observed in zip(desired, measured['windows']):
        left, right = window.start_sample, window.end_sample
        if support[left:right].any():
            raise ValueError('Repair windows must not overlap.')
        support[left:right] = True
        unit = observed['unit']
        actual = math.degrees(math.atan2(float(unit[1]), float(unit[0])))
        difference = (window.azimuth_deg-actual+180.) % 360.-180.
        observable = bool(observed['observable'])
        shift = 0.
        if observable and abs(difference) > tolerance_deg:
            # A bounded partial correction is allowed even when one proposal
            # cannot reach the cone. Its actual benefit remains to be tested.
            shift = math.copysign(min(max_rotation_deg,
                abs(difference)-(tolerance_deg-interior_margin_deg)), difference)
        envelope = torch.ones(right-left, device=waveform.device, dtype=torch.float64)
        ramp = min(round(taper_seconds*44100), (right-left)//2)
        if ramp:
            fade = (1-torch.cos(torch.linspace(0, math.pi, ramp, device=waveform.device, dtype=torch.float64)))/2
            envelope[:ramp] = fade
            envelope[-ramp:] = fade.flip(0)
        angle[left:right] = math.radians(shift)*envelope
        receipts.append(dict(source_id=window.source_id, start_sample=left, end_sample=right,
            requested_native_azimuth_deg=window.azimuth_deg, observed_azimuth_deg=actual,
            signed_error_deg=difference, observable=observable, proposed_rotation_deg=shift,
            energy=float(observed['energy']), coherence=float(observed['coherence'])))
    result = waveform.clone()
    x, y = waveform[:,3].double(), waveform[:,1].double()
    result[:,3] = (x*angle.cos()-y*angle.sin()).to(waveform.dtype)
    result[:,1] = (x*angle.sin()+y*angle.cos()).to(waveform.dtype)
    result[..., ~support] = waveform[..., ~support]
    before = x.square()+y.square()
    after = result[:,3].double().square()+result[:,1].double().square()
    energy_error = float((before-after).norm()/before.norm().clamp_min(1e-30))
    evidence = dict(windows=receipts, w_exact=torch.equal(result[:,0],waveform[:,0]),
        z_exact=torch.equal(result[:,2],waveform[:,2]), outside_fixed_windows_exact=torch.equal(result[...,~support],waveform[...,~support]),
        xy_instantaneous_energy_relative_error=energy_error, no_op_exact=torch.equal(result,waveform),
        max_applied_rotation_deg=float(angle.abs().max())*180/math.pi,
        config=dict(max_rotation_deg=max_rotation_deg,tolerance_deg=tolerance_deg,
            interior_margin_deg=interior_margin_deg,taper_seconds=taper_seconds),
        direction_authority='Window-average active intensity compared with the active native plan at the same time.',
        scope='A tapered physical moment correction. No source separation, independent content certification, VAE transport or student improvement is implied.')
    assert evidence['w_exact'] and evidence['z_exact'] and evidence['outside_fixed_windows_exact']
    assert energy_error<1e-6 and evidence['max_applied_rotation_deg']<=max_rotation_deg+1e-8
    return result,evidence
