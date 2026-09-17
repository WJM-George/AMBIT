"""Bounded physical rotation proposals for one stationary FOA source.

Only WYZX waveform channels are physical axes. This is a stopped training-side
proposal; VAE transport, final continuation, content and room need validation.
"""
from __future__ import annotations

import math
import torch


@torch.no_grad()
def bounded_static_rotation(waveform, *, azimuth_deg, elevation_deg,
                            maximum_rotation_deg=20., tolerance_deg=15.,
                            minimum_coherence=.1, source_count=1):
    if (waveform.ndim != 3 or waveform.shape[:2] != (1, 4)
            or not waveform.is_floating_point() or waveform.shape[-1] == 0
            or not torch.isfinite(waveform).all() or source_count != 1):
        raise ValueError('Require a finite batch-one WYZX waveform for one stationary source.')
    if (not all(math.isfinite(x) for x in [azimuth_deg, elevation_deg,
            maximum_rotation_deg, tolerance_deg, minimum_coherence])
            or not -90 <= elevation_deg <= 90 or not 0 < maximum_rotation_deg <= 30
            or not 0 <= tolerance_deg < 90 or not 0 < minimum_coherence <= 1):
        raise ValueError('Invalid explicit static direction or bounded rotation settings.')
    wave = waveform[0].double()
    axes = wave[[3, 1, 2]]
    intensity = (wave[0, None]*axes).sum(-1)
    norm = intensity.norm()
    coherence = norm/(wave[0].square().sum()*axes.square().sum()).sqrt().clamp_min(1e-30)
    observable = bool(norm > 1e-12 and coherence >= minimum_coherence)
    a, e = math.radians(azimuth_deg), math.radians(elevation_deg)
    desired = torch.tensor([math.cos(e)*math.cos(a), math.cos(e)*math.sin(a), math.sin(e)],
                           device=waveform.device, dtype=torch.float64)
    evidence = dict(observable=observable, coherence=float(coherence),
        requested_direction=dict(azimuth_deg=azimuth_deg, elevation_deg=elevation_deg),
        maximum_rotation_deg=maximum_rotation_deg, tolerance_deg=tolerance_deg,
        source_count=source_count, physical_channel_order='W,Y,Z,X',
        scope='One global physical rotation of a stationary source. W and directional energy conservation do not certify full semantic, radial, room or student quality.')
    if not observable:
        evidence.update(no_op_exact=True, applied_rotation_deg=0., w_exact=True,
                        reason='No reliable aggregate active-intensity direction.')
        return waveform.detach().clone(), evidence
    current = intensity/norm
    error = math.degrees(math.acos(float((current@desired).clamp(-1, 1))))
    angle = min(maximum_rotation_deg, max(0., error-tolerance_deg))
    evidence.update(aggregate_error_before_deg=error, applied_rotation_deg=angle)
    if angle <= 1e-10:
        evidence.update(no_op_exact=True, w_exact=True, aggregate_error_after_deg=error,
                        directional_energy_relative_error=0., reason='Already inside the requested tolerance.')
        return waveform.detach().clone(), evidence
    axis = torch.linalg.cross(current, desired)
    if float(axis.norm()) < 1e-12:
        # Opposite vectors admit many shortest rotations; use a fixed orthogonal axis.
        basis = torch.eye(3, device=waveform.device, dtype=torch.float64)[int(current.abs().argmin())]
        axis = torch.linalg.cross(current, basis)
    axis = axis/axis.norm()
    kx, ky, kz = axis.unbind()
    zero = torch.zeros_like(kx)
    skew = torch.stack([zero, -kz, ky, kz, zero, -kx, -ky, kx, zero]).reshape(3, 3)
    radians = math.radians(angle)
    rotation = torch.eye(3, device=waveform.device, dtype=torch.float64) + math.sin(radians)*skew + (1-math.cos(radians))*(skew@skew)
    repaired = waveform.detach().clone()
    repaired[0, [3, 1, 2]] = (rotation@axes).to(waveform.dtype)
    old_energy = axes.square().sum(0)
    new_energy = repaired[0, [3, 1, 2]].double().square().sum(0)
    energy_error = float((new_energy-old_energy).norm()/old_energy.norm().clamp_min(1e-30))
    new_intensity = (repaired[0, 0, None].double()*repaired[0, [3, 1, 2]].double()).sum(-1)
    after = math.degrees(math.acos(float(((new_intensity/new_intensity.norm())@desired).clamp(-1, 1))))
    evidence.update(w_exact=torch.equal(repaired[:, 0], waveform[:, 0]),
        no_op_exact=torch.equal(repaired, waveform), rotation_matrix=rotation.cpu().tolist(),
        aggregate_error_after_deg=after, directional_energy_relative_error=energy_error,
        rotation_orthogonality_error=float((rotation.T@rotation-torch.eye(3, device=rotation.device)).abs().max()),
        rotation_determinant=float(torch.linalg.det(rotation)))
    if not evidence['w_exact'] or energy_error > 1e-5 or abs(after-(error-angle)) > .02:
        raise RuntimeError('Physical rotation failed its waveform invariants.')
    return repaired, evidence
