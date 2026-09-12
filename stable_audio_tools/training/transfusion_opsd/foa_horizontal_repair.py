"""Bounded horizontal FOA repair for a training-side common target.

This operates on WYZX waveforms, never on VAE channels as physical axes.
Horizontal scores are components, not complete spatial/content certificates.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch


@dataclass(frozen=True)
class HorizontalRepairConfig:
    max_rotation_deg: float = 30.
    interior_margin_deg: float = 2.5
    magnitude_penalty: float = .02
    smoothness_penalty: float = 4.
    iterations: int = 128

    def __post_init__(self):
        if (not 0 < self.max_rotation_deg <= 90 or not 0 < self.interior_margin_deg < 10
                or self.magnitude_penalty <= 0 or self.smoothness_penalty < 0
                or type(self.iterations) is not int or self.iterations < 1
                or not all(math.isfinite(x) for x in [self.max_rotation_deg,
                    self.interior_margin_deg,self.magnitude_penalty,self.smoothness_penalty])):
            raise ValueError('Declare a finite bounded horizontal repair.')


@torch.no_grad()
def horizontal_components(reference, waveform):
    """Retain the reference windows and reliability definition; free elevation.

    The caller must report every other requested constraint separately. This
    helper intentionally does not return an overall request-admissibility flag.
    """
    unit, coherence, level = reference._directions(waveform)
    horizontal = unit[:,:2]
    norm = horizontal.norm(dim=-1, keepdim=True)
    unit_xy = horizontal/norm.clamp_min(1e-6)
    observable = (coherence >= reference.min_coherence) & (level >= .5) & (norm[:,0] > 1e-6)
    windows = reference.windows.to(waveform.device)
    cosine = reference.directions.to(waveform.device)[:,:2] @ unit_xy.T
    fraction = lambda mask: float(((windows & mask).sum(-1).float()/windows.sum(-1)).mean())
    base_costs = reference._measure(waveform)[1]
    return dict(
        requested_horizontal_failure=fraction((cosine < reference.cos_tolerance) | ~observable),
        direction_unobservable_fraction=fraction(~observable),
        observably_wrong_horizontal=fraction((cosine < reference.cos_tolerance) & observable),
        source_presence_failure=float(base_costs['source_presence_failure']),
        forbidden_activity_fraction=float(base_costs['forbidden_activity_fraction']),
        clipping=float((waveform.abs()>1).float().mean()),
    )


@torch.no_grad()
def bounded_horizontal_repair(reference, waveform, *, config=HorizontalRepairConfig()):
    """Find a small smooth angle correction, then apply an actual SO(2) field.

    The frame-angle problem is a convex quadratic with box constraints. Frames
    already inside their requested sector only constrain the allowed movement;
    they have no extra reward for reaching an exact center. Initially wrong,
    observable, reachable frames supply the repair target. Unobservable frames
    and frames outside requested windows are held fixed in the frame solution.
    The interpolated sample rotation must still be scored after application.
    """
    if waveform.shape[0:2] != (1,4) or not torch.isfinite(waveform).all():
        raise ValueError('Expected finite batch-one WYZX waveform.')
    windows = reference.windows.detach().cpu().numpy().astype(bool)
    identities = reference.evidence['requested_windows']
    if len({x['source_id'] for x in identities}) != 1:
        raise ValueError('A global field repair cannot certify independent multi-source correction.')
    unit, coherence, level = reference._directions(waveform)
    horizontal_norm = unit[:,:2].norm(dim=-1)
    observable = ((coherence >= reference.min_coherence) & (level >= .5) & (horizontal_norm > 1e-6)).cpu().numpy()
    actual = np.degrees(np.arctan2(unit[:,1].cpu().numpy(),unit[:,0].cpu().numpy())).astype(np.float64)
    desired = reference.directions.detach().cpu().numpy()
    desired = np.degrees(np.arctan2(desired[:,1],desired[:,0])).astype(np.float64)
    covered = windows.any(axis=0)
    target = np.zeros(len(actual),dtype=np.float64)
    assigned = np.zeros(len(actual),dtype=bool)
    wrap = lambda value: (value+180.) % 360.-180.
    for region, angle in zip(windows,desired):
        if np.any(region & assigned & (np.abs(wrap(target-angle))>1e-4)):
            raise ValueError('Overlapping requested windows disagree about their compass direction.')
        target[region]=angle
        assigned |= region
    error = wrap(actual-target)
    tolerance = math.degrees(math.acos(reference.cos_tolerance))
    if config.interior_margin_deg >= tolerance:
        raise ValueError('Repair margin must stay inside the declared request sector.')
    inside = covered & observable & (np.abs(error)<=tolerance)
    reachable = covered & observable & (np.abs(error)>tolerance) & (
        np.abs(error)<=tolerance+config.max_rotation_deg)
    lower = np.full(len(actual),-config.max_rotation_deg,dtype=np.float64)
    upper = -lower
    lower[inside] = np.maximum(lower[inside],-tolerance-error[inside])
    upper[inside] = np.minimum(upper[inside],tolerance-error[inside])
    fixed = ~covered | ~observable
    lower[fixed]=0.; upper[fixed]=0.
    desired_shift = -np.sign(error)*(np.abs(error)-(tolerance-config.interior_margin_deg))
    weights = reachable.astype(np.float64)
    alpha = np.zeros_like(error)
    momentum, acceleration = alpha.copy(),1.
    step = 1./(2.*(1.+config.magnitude_penalty+4.*config.smoothness_penalty))
    def objective(value):
        return float(np.sum(weights*(value-desired_shift)**2)
            + config.magnitude_penalty*np.sum(value**2)
            + config.smoothness_penalty*np.sum(np.diff(value)**2))
    initial_objective = objective(alpha)
    if reachable.any():
        for _ in range(config.iterations):
            gradient = 2.*weights*(momentum-desired_shift)+2.*config.magnitude_penalty*momentum
            difference = np.diff(momentum)
            gradient[:-1] -= 2.*config.smoothness_penalty*difference
            gradient[1:] += 2.*config.smoothness_penalty*difference
            updated = np.clip(momentum-step*gradient,lower,upper)
            next_acceleration = (1.+math.sqrt(1.+4.*acceleration**2))/2.
            momentum = updated+((acceleration-1.)/next_acceleration)*(updated-alpha)
            alpha,acceleration = updated,next_acceleration
    frames = np.arange(len(alpha),dtype=np.float64)*1024.+511.5
    sample_angles = np.interp(np.arange(waveform.shape[-1]),frames,alpha,left=alpha[0],right=alpha[-1])
    angle = torch.as_tensor(np.radians(sample_angles),device=waveform.device,dtype=waveform.dtype)
    cosine,sine = angle.cos(),angle.sin()
    repaired = waveform.clone()
    x,y = waveform[:,3],waveform[:,1]
    repaired[:,3] = x*cosine-y*sine
    repaired[:,1] = x*sine+y*cosine
    original_energy = waveform[:,1].double().square()+waveform[:,3].double().square()
    repaired_energy = repaired[:,1].double().square()+repaired[:,3].double().square()
    relative_energy_error = float((repaired_energy-original_energy).norm()/original_energy.norm().clamp_min(1e-30))
    evidence = dict(config=config.__dict__, requested_frames=int(covered.sum()),
        originally_observable_correct_frames=int(inside.sum()), reachable_wrong_frames=int(reachable.sum()),
        objective_before=initial_objective, objective_after=objective(alpha),
        max_applied_rotation_deg=float(np.max(np.abs(sample_angles))),
        normalized_mean_squared_rotation=float(np.mean((alpha[covered]/config.max_rotation_deg)**2)),
        frame_rotation_deg=alpha.tolist(),
        frame_solution_respects_originally_correct_sectors=bool(np.all(np.abs(error[inside]+alpha[inside])<=tolerance+1e-8)),
        w_exact=torch.equal(repaired[:,0],waveform[:,0]), z_exact=torch.equal(repaired[:,2],waveform[:,2]),
        xy_instantaneous_energy_relative_error=relative_energy_error,
        no_op_exact=torch.equal(repaired,waveform),
        extra_spatial_requirements_certified=False,
        scope='Training-side physical target proposal. W/Z and instantaneous XY energy conservation do not prove full radial, room, binaural or student-generated quality.')
    assert evidence['w_exact'] and evidence['z_exact'] and relative_energy_error<1e-6
    assert evidence['max_applied_rotation_deg']<=config.max_rotation_deg+1e-8
    assert evidence['frame_solution_respects_originally_correct_sectors']
    return repaired,evidence
