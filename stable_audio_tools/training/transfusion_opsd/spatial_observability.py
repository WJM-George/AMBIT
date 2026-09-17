"""Keep missing-sound penalties, without interpreting inaudible FOA direction."""
from __future__ import annotations

import torch

from .native_spatial_credit import native_spatial_credit


def observable_spatial_credit(waveform, windows, **thresholds):
    """Same fixed windows and denominator as the original geometry proxy.

    Direction is undefined below the registered energy/coherence floors.
    Those windows keep their presence/coherence penalties and remain in the
    denominator. The binary observation boundary is a piecewise training
    proxy, with no derivative through its threshold. Callers must first
    calibrate the fixed windows using an independent, frozen target.
    """
    result = native_spatial_credit(waveform, windows, **thresholds)
    rows = []
    for row in result['windows']:
        direction = row['direction'] * row['observable'].to(row['direction'].dtype)
        rows.append(dict(row, raw_direction=row['direction'], direction=direction,
                         loss=direction + row['energy_loss'] + row['coherence_loss']))
    return dict(loss=torch.stack([row['loss'] for row in rows]).mean(), windows=rows,
                scope='Fixed windows, observable direction plus missing-energy/coherence penalties; not content, identity, exact timing or full-path certification.')
