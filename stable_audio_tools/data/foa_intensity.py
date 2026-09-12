"""FOA -> per-frame spatial trajectory (active-intensity DoA + diffuseness).

This analytic spatial sequence is retained for P10 FOA evaluation and spatial
calibration.  It is not a trainable P11 input and does not define a planner
architecture.

Ambisonic convention
--------------------
Input is First-Order Ambisonics in **ACN / SN3D (AmbiX)** channel order
``[W, Y, Z, X]`` (ACN indices 0,1,2,3), which is what the 4ch VAE line uses.
The velocity-proportional components map to Cartesian axes as::

    X (ACN 3) -> +x (front)
    Y (ACN 1) -> +y (left)
    Z (ACN 2) -> +z (up)

Active sound intensity is ``I = <p * u>`` with pressure ``p = W`` and particle
velocity ``u ∝ [X, Y, Z]``. Under SN3D encoding a source at unit direction
``d`` maps to ``[X,Y,Z] = d * s`` and ``W = s/√2``, so ``<W·[X,Y,Z]> ∝ +d`` —
i.e. the intensity vector already points **toward the source**. We therefore
return the DoA (source direction) as ``+I / ||I||`` by default
(``source_direction=True``); set it False to return the energy-flow direction.

Energy-normalized diffuseness (DirAC-style coherence):
    ``psi = 1 - ||<W [X,Y,Z]>|| / sqrt(<W^2><X^2+Y^2+Z^2>)``
    in ``[0, 1]``
where ``psi≈0`` => one clear direction and ``psi≈1`` => fully diffuse.

Output
------
``[T, 4]`` per frame ``[dx, dy, dz, diffuseness]`` where ``[dx,dy,dz]`` is a unit
(or zero, when fully diffuse) DoA vector and ``T = n_samples // hop`` frames,
aligned to the VAE latent frame rate when ``hop == downsampling_ratio``.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

__all__ = [
    "foa_to_intensity_trajectory",
    "standardize_trajectory",
    "TRAJ_COMPONENTS",
]

TRAJ_COMPONENTS = ("dx", "dy", "dz", "diffuseness")


def _as_tensor(foa) -> Tensor:
    if isinstance(foa, np.ndarray):
        foa = torch.from_numpy(foa)
    return foa.float()


def foa_to_intensity_trajectory(
    foa,
    hop: int = 1024,
    eps: float = 1e-8,
    source_direction: bool = True,
) -> Tensor:
    """FOA ``[4, N]`` (ACN ``[W,Y,Z,X]``) -> trajectory ``[T, 4]``.

    Each frame aggregates a non-overlapping block of ``hop`` samples into
    ``[dx, dy, dz, diffuseness]``. Set ``hop = downsampling_ratio`` (e.g. 1024)
    so ``T`` matches the VAE latent length.
    """
    x = _as_tensor(foa)
    if x.dim() != 2 or x.shape[0] < 4:
        raise ValueError(f"expected FOA of shape [4, N] (ACN W,Y,Z,X), got {tuple(x.shape)}")

    W, Y, Z, X = x[0], x[1], x[2], x[3]
    n = x.shape[-1]
    T = n // hop
    if T == 0:
        raise ValueError(f"signal length {n} shorter than one hop ({hop})")
    trim = T * hop

    def blocks(v: Tensor) -> Tensor:
        return v[:trim].reshape(T, hop)

    Wb, Yb, Zb, Xb = blocks(W), blocks(Y), blocks(Z), blocks(X)

    # active intensity vector per block, Cartesian (x=front, y=left, z=up)
    Ix = (Wb * Xb).mean(dim=-1)
    Iy = (Wb * Yb).mean(dim=-1)
    Iz = (Wb * Zb).mean(dim=-1)
    I = torch.stack([Ix, Iy, Iz], dim=-1)  # [T, 3]
    mag = I.norm(dim=-1)  # [T]

    # Normalize intensity by the geometric mean of omni and directional energy.
    # This is invariant to the W-channel normalization convention and equals one
    # for an ideal plane wave. The previous ``mag / (W2 + .5 * XYZ2)`` formula
    # assigned a pure source a spurious diffuseness of roughly 0.3.
    omni_energy = (Wb ** 2).mean(dim=-1)
    dir_energy = (Xb ** 2 + Yb ** 2 + Zb ** 2).mean(dim=-1)
    coherence = mag / (omni_energy * dir_energy).clamp_min(0.0).sqrt().add(eps)
    coherence = coherence.clamp(0.0, 1.0)
    diffuseness = 1.0 - coherence  # [T]

    doa = I / (mag.unsqueeze(-1) + eps)  # [T, 3] unit; +I points toward source (SN3D)
    if not source_direction:
        doa = -doa  # energy-flow direction (receiver-ward)

    # Damp unstable directions in diffuse/silent frames. For a coherent plane
    # wave this leaves a unit vector; for diffuse audio it approaches zero.
    doa = doa * coherence.unsqueeze(-1)

    return torch.cat([doa, diffuseness.unsqueeze(-1)], dim=-1)  # [T, 4]


def standardize_trajectory(traj: Tensor, mean=None, std=None):
    """Optionally standardize per component to ~unit std (config: per_component_unit_std).

    Returns ``(traj_std, mean, std)``. If ``mean``/``std`` are None they are
    computed from ``traj`` (per component over time). Pass dataset-level stats
    for consistent train/infer scaling.
    """
    if mean is None:
        mean = traj.mean(dim=0, keepdim=True)
    if std is None:
        std = traj.std(dim=0, keepdim=True).clamp_min(1e-4)
    return (traj - mean) / std, mean, std
