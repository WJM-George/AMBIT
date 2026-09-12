from __future__ import annotations

import numpy as np

from scripts.t2a.data.sceneplan_v2_renderer import _rir_direction_diagnostic


def test_direction_diagnostic_uses_aligned_direct_arrival_not_stronger_reflection() -> None:
    source_xyz = (0.0, 1.0, 0.0)
    microphone_xyz = (0.0, 0.0, 0.0)
    rirs = [np.zeros(300, dtype=np.float32) for _ in range(4)]

    # ACN channel ratios are Y/Z/X. The direct response is correctly aligned
    # at sample 40, while an unrelated later reflection is slightly stronger.
    rirs[0][40] = 0.08
    rirs[1][40] = 0.08
    rirs[2][40] = 0.0
    rirs[3][40] = 0.0
    rirs[0][247] = 0.09
    rirs[1][247] = -0.09
    rirs[2][247] = 0.09
    rirs[3][247] = 0.09

    diagnostic = _rir_direction_diagnostic(rirs, source_xyz, microphone_xyz)

    assert diagnostic["residual_direct_peak_sample"] == 40
    assert diagnostic["max_abs_direction_ratio_error"] == 0.0
