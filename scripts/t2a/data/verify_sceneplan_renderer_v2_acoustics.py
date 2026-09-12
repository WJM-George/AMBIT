#!/usr/bin/env python3
"""Deterministic acoustic-contract checks for ScenePlan renderer v2."""

from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import pyroomacoustics as pra
import soundfile as sf


REPO_ROOT = Path(__file__).resolve().parents[3]
SYNTHESIS_ROOT = REPO_ROOT / "dataset/synthesis"
for value in (REPO_ROOT, SYNTHESIS_ROOT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from render_spatial_edit_families import (  # noqa: E402
    _align_rirs_to_direct_arrival,
    _load_activity_signal,
)
from synthesize_foa_pyroom import (  # noqa: E402
    N3D_TO_SN3D_FIRST_ORDER_GAIN,
    pyroom_n3d_to_sn3d_foa,
    room_rirs,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def expect_runtime_error(callable_value, message: str) -> None:
    try:
        callable_value()
    except RuntimeError:
        return
    raise AssertionError(message)


def verify_normalization() -> dict[str, object]:
    require(
        math.isclose(N3D_TO_SN3D_FIRST_ORDER_GAIN, 1.0 / math.sqrt(3.0), abs_tol=1e-15),
        "N3D-to-SN3D first-order gain changed",
    )
    native = np.asarray([[1.0], [math.sqrt(3.0)], [0.0], [0.0]], dtype=np.float32)
    converted = pyroom_n3d_to_sn3d_foa(native)
    require(np.allclose(converted[:, 0], [1.0, 1.0, 0.0, 0.0], atol=1e-7), "N3D conversion failed")

    microphone = (4.0, 4.0, 2.0)
    directions = {
        "front": ((6.0, 4.0, 2.0), np.asarray([0.0, 0.0, 1.0])),
        "left": ((4.0, 6.0, 2.0), np.asarray([1.0, 0.0, 0.0])),
        "up": ((4.0, 4.0, 3.0), np.asarray([0.0, 1.0, 0.0])),
    }
    errors = {}
    residual_delays = {}
    for name, (source, target_yzx) in directions.items():
        rirs = room_rirs(
            (8.0, 8.0, 4.0),
            0.3,
            0,
            microphone,
            source,
            44_100,
        )
        w_peak = int(np.argmax(np.abs(rirs[0])))
        w = float(rirs[0][w_peak])
        measured = np.asarray(
            [rirs[1][w_peak] / w, rirs[2][w_peak] / w, rirs[3][w_peak] / w],
            dtype=np.float64,
        )
        errors[name] = float(np.max(np.abs(measured - target_yzx)))
        require(errors[name] < 1e-5, f"{name} WYZX/SN3D direction mismatch: {measured}")
        aligned = _align_rirs_to_direct_arrival(
            rirs,
            source_xyz=source,
            microphone_xyz=microphone,
            sample_rate=44_100,
        )
        residual = int(np.argmax(np.abs(aligned[0])))
        residual_delays[name] = residual
        require(39 <= residual <= 41, f"unexpected Pyroom residual delay: {residual}")
    return {
        "n3d_to_sn3d_first_order_gain": N3D_TO_SN3D_FIRST_ORDER_GAIN,
        "direction_max_abs_errors": errors,
        "residual_algorithmic_delay_samples": residual_delays,
        "pyroom_fractional_delay_filter_samples": int(pra.constants.get("frac_delay_length")),
    }


def source_record(path: Path, channels: int, frames: int, activity_frames: int) -> dict:
    return {
        "dry_audio": {
            "path": str(path),
            "native_sample_rate": 44_100,
            "crop_start_native_sample": 0,
            "input_audio_domain": "dry_mono",
            "spatialization_passes_before_scene": 0,
        },
        "playback": {
            "offset_in_crop_native_sample": 0,
            "num_native_samples": frames,
            "loudness": {"gain_linear": 1.0},
        },
        "activity": {
            "onset_sec": 0.0,
            "offset_sec": activity_frames / 44_100,
        },
        "_channels": channels,
    }


def verify_dry_source_guards() -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="renderer_v2_contract_") as directory:
        root = Path(directory)
        mono_path = root / "mono.wav"
        stereo_path = root / "stereo.wav"
        sf.write(mono_path, np.ones(1000, dtype=np.float32) * 0.01, 44_100)
        sf.write(stereo_path, np.ones((1000, 2), dtype=np.float32) * 0.01, 44_100)
        valid = _load_activity_signal(
            source_record(mono_path, 1, 1000, 1000),
            sample_rate=44_100,
            num_samples=1000,
        )
        require(valid.shape == (1000,), "valid dry mono shape changed")
        expect_runtime_error(
            lambda: _load_activity_signal(
                source_record(stereo_path, 2, 1000, 1000),
                sample_rate=44_100,
                num_samples=1000,
            ),
            "multichannel source was not rejected",
        )
        expect_runtime_error(
            lambda: _load_activity_signal(
                source_record(mono_path, 1, 1000, 900),
                sample_rate=44_100,
                num_samples=1000,
            ),
            "source longer than activity window was silently truncated",
        )
    return {
        "valid_dry_mono": True,
        "multichannel_rejected": True,
        "activity_truncation_rejected": True,
    }


def main() -> int:
    import json

    report = {
        "schema": "stable_audio_tools.sceneplan_renderer_v2_acoustic_verification",
        "ok": True,
        "normalization_and_timing": verify_normalization(),
        "dry_source_guards": verify_dry_source_guards(),
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
