"""Check that a cached repair still supervises the actual initial prediction.

Model hashes identify stored tensors; they do not certify identical runtime
conditioning. This is a numerical target-coordinate check, not an audio
quality tolerance or a requirement for bitwise waveform equality.
"""
import math

import torch

from .objectives import valid_mask


class RepairCorrespondenceError(ValueError):
    def __init__(self, diagnostics):
        self.diagnostics = diagnostics
        super().__init__('cached repair anchor differs from the actual student prediction; '
            'recollect and certify the repair with the current executor: '+str(diagnostics))


def certify_repair_correspondence(prediction, anchor, positive, mask, *,
        max_relative_drift=.01, absolute_rms_tolerance=1e-7):
    """Allow tiny arithmetic differences, reject fitting a different displacement.

The discrepancy is measured against the certified repair itself. A substantial
anchor shift must not be silently counted as learning that repair, even if an
absolute MSE happens to decrease. This does not re-certify cached future scores
or establish that an old trajectory was visited by the current policy.
    """
    if (not math.isfinite(max_relative_drift) or not 0 <= max_relative_drift < 1
            or not math.isfinite(absolute_rms_tolerance) or absolute_rms_tolerance < 0):
        raise ValueError('declare finite nonnegative numerical correspondence tolerances')
    if prediction.shape != anchor.shape or positive.shape != anchor.shape:
        raise ValueError('repair correspondence needs identical query geometry')
    keep = valid_mask(anchor, mask)
    values = [x.detach().double()[keep] for x in (prediction, anchor, positive)]
    if not all(torch.isfinite(x).all() for x in values):
        raise ValueError('active prediction and repair values must be finite')
    current, base, target = values
    repair, drift = target-base, current-base
    repair_rms = float(repair.square().mean().sqrt())
    if repair_rms <= absolute_rms_tolerance:
        raise ValueError('a positive repair must exceed its numerical resolution floor')
    drift_rms = float(drift.square().mean().sqrt())
    allowed = absolute_rms_tolerance + max_relative_drift*repair_rms
    ratio = drift_rms/repair_rms
    diagnostics = dict(certified=drift_rms <= allowed, repair_rms=repair_rms,
        anchor_drift_rms=drift_rms, drift_to_repair_ratio=ratio, allowed_drift_rms=allowed,
        max_relative_drift=max_relative_drift, absolute_rms_tolerance=absolute_rms_tolerance,
        nominal_squared_error=float(repair.square().mean()),
        actual_squared_error=float((current-target).square().mean()),
        relative_squared_error_deviation_bound=2*ratio+ratio*ratio)
    if not diagnostics['certified']:
        raise RepairCorrespondenceError(diagnostics)
    return diagnostics


def repair_fit_gain_decomposition(before, after, anchor, positive, mask):
    """Separate repair alignment from removal of a pre-existing anchor shift.

At a fixed query, let r=positive-anchor, b=before-anchor and u=after-before.
The exact squared-error reduction is 2<r,u>-2<b,u>-||u||^2. These are loss
components, not independent audio-quality improvements or causal mechanisms.
    """
    if any(x.shape != anchor.shape for x in (before, after, positive)) or anchor.shape[0] != 1:
        raise ValueError('loss decomposition requires one aligned query')
    keep = valid_mask(anchor, mask)
    first, last, base, target = [x.detach().double()[keep] for x in (before, after, anchor, positive)]
    if not all(torch.isfinite(x).all() for x in (first, last, base, target)):
        raise ValueError('finite active predictions and targets required')
    repair, drift, update = target-base, first-base, last-first
    normalizer = float(repair.abs().mean().clamp_min(1e-5))
    repair_gain = float(2*(repair*update).mean())/normalizer
    drift_gain = float(-2*(drift*update).mean())/normalizer
    curvature = float(update.square().mean())/normalizer
    initial = float((first-target).square().mean())/normalizer
    final = float((last-target).square().mean())/normalizer
    return dict(initial_loss=initial, final_loss=final, observed_loss_gain=initial-final,
        repair_alignment_gain=repair_gain, anchor_drift_removal_gain=drift_gain,
        update_curvature_cost=curvature, component_sum=repair_gain+drift_gain-curvature,
        identity_error=(initial-final)-(repair_gain+drift_gain-curvature))
