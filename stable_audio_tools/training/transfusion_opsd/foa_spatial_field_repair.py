"""Smooth bounded SO(3) waveform fields for a single stationary target.

Local wrong directions can cancel in an aggregate intensity vector. Keep the
same angular tolerance while addressing only observable out-of-cone frames.
"""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F


@torch.no_grad()
def bounded_static_field_rotation(waveform, *, azimuth_deg, elevation_deg,
        maximum_rotation_deg=20., tolerance_deg=15., minimum_coherence=.1,
        source_count=1, hop=1024, iterations=128, smoothness_penalty=4., magnitude_penalty=.02):
    if (waveform.ndim != 3 or waveform.shape[:2] != (1,4) or waveform.shape[-1] == 0
            or waveform.dtype not in (torch.float32,torch.float64)
            or not torch.isfinite(waveform).all() or source_count != 1):
        raise ValueError('Require finite FP32/FP64 WYZX audio for one stationary source.')
    if (not all(math.isfinite(x) for x in [azimuth_deg,elevation_deg,maximum_rotation_deg,
            tolerance_deg,minimum_coherence,smoothness_penalty,magnitude_penalty])
            or not -90<=elevation_deg<=90 or not 0<maximum_rotation_deg<=30
            or not 0<=tolerance_deg<90 or not 0<minimum_coherence<=1
            or smoothness_penalty<0 or magnitude_penalty<=0 or type(hop) is not int or hop<=0
            or type(iterations) is not int or iterations<1):
        raise ValueError('Invalid bounded waveform-field configuration.')
    n=waveform.shape[-1];frames=math.ceil(n/hop)
    padded=F.pad(waveform[0].double(),(0,frames*hop-n)).reshape(4,frames,hop)
    w=padded[0];xyz=padded[[3,1,2]]
    energy=w.square().mean(-1)
    intensity=(w[None]*xyz).mean(-1).T
    norms=intensity.norm(dim=-1)
    coherence=norms/(energy*xyz.square().sum(0).mean(-1)).sqrt().clamp_min(1e-30)
    level=10*energy.clamp_min(1e-30).log10()
    active=level>=max(-55.,float(level.max())-40.)
    reliable=active & (coherence>=minimum_coherence) & (norms>1e-12)
    current=intensity/norms[:,None].clamp_min(1e-30)
    az,el=math.radians(azimuth_deg),math.radians(elevation_deg)
    desired=torch.tensor([math.cos(el)*math.cos(az),math.cos(el)*math.sin(az),math.sin(el)],device=waveform.device,dtype=torch.float64)
    errors=torch.rad2deg(torch.acos((current@desired).clamp(-1,1)))
    wrong=reliable & (errors>tolerance_deg)
    axis=torch.linalg.cross(current,desired.expand_as(current),dim=-1)
    opposite=wrong & (axis.norm(dim=-1)<1e-12)
    if opposite.any():
        basis=torch.eye(3,device=waveform.device,dtype=torch.float64)[current[opposite].abs().argmin(-1)]
        axis[opposite]=torch.linalg.cross(current[opposite],basis,dim=-1)
    axis=axis/axis.norm(dim=-1,keepdim=True).clamp_min(1e-30)
    requested=axis*(errors-tolerance_deg).clamp(0,maximum_rotation_deg)[:,None]
    requested[~wrong]=0
    value=torch.zeros_like(requested);momentum=value.clone();accel=1.
    step=1/(2*(1+magnitude_penalty+4*smoothness_penalty))
    for _ in range(iterations):
        grad=2*wrong[:,None]*(momentum-requested)+2*magnitude_penalty*momentum
        difference=momentum[1:]-momentum[:-1]
        grad[:-1]-=2*smoothness_penalty*difference
        grad[1:]+=2*smoothness_penalty*difference
        updated=momentum-step*grad
        updated*= (maximum_rotation_deg/updated.norm(dim=-1,keepdim=True).clamp_min(1e-30)).clamp(max=1)
        # Already-good and unreliable frame centers remain exact no-ops.
        updated[~wrong]=0
        next_accel=(1+math.sqrt(1+4*accel**2))/2
        momentum=updated+((accel-1)/next_accel)*(updated-value)
        value,accel=updated,next_accel
    field=F.interpolate(value.T[None],size=frames*hop,mode='linear',align_corners=False)[0,:,:n]
    angles=field.norm(dim=0)
    unit=field/angles[None].clamp_min(1e-30)
    theta=torch.deg2rad(angles)
    axes=waveform[0,[3,1,2]].double()
    rotated=axes*theta.cos()[None]+torch.linalg.cross(unit,axes,dim=0)*theta.sin()[None]+unit*(unit*axes).sum(0)[None]*(1-theta.cos())[None]
    repaired=waveform.detach().clone();repaired[0,[3,1,2]]=rotated.to(waveform.dtype)
    energy_before=axes.square().sum(0)
    energy_after=repaired[0,[3,1,2]].double().square().sum(0)
    error=float((energy_after-energy_before).norm()/energy_before.norm().clamp_min(1e-30))
    evidence=dict(local_frames=frames,reliable_frames=int(reliable.sum()),wrong_frames=int(wrong.sum()),
        inside_frames=int((reliable & ~wrong).sum()),unreliable_frame_centers_fixed=True,
        already_good_frame_centers_fixed=bool((value[reliable & ~wrong]==0).all()),
        max_applied_rotation_deg=float(angles.max()),frame_rotation_vectors_deg=value.cpu().tolist(),
        local_error_before_deg=errors[reliable].cpu().tolist(),w_exact=torch.equal(repaired[:,0],waveform[:,0]),
        directional_energy_relative_error=error,no_op_exact=torch.equal(repaired,waveform),
        config=dict(azimuth_deg=azimuth_deg,elevation_deg=elevation_deg,maximum_rotation_deg=maximum_rotation_deg,
            tolerance_deg=tolerance_deg,minimum_coherence=minimum_coherence,hop=hop,iterations=iterations,
            smoothness_penalty=smoothness_penalty,magnitude_penalty=magnitude_penalty),
        scope='Smooth local rotation field on physical waveform XYZ only. Frame-center constraints do not certify interpolated audio or VAE transport; actual continuation is evaluated separately. No independent multi-source, distance or room claim.')
    assert evidence['w_exact'] and error<1e-6 and float(angles.max())<=maximum_rotation_deg+1e-8
    return repaired,evidence
