"""Frozen timing-grid contract for the offline speech alignment teacher."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


NOMINAL_ENDPOINT_OVERSHOOT_GRID_BINS = 2


def load_aligner_timing_contract(model_root: str | Path) -> dict[str, Any]:
    """Read the aligner's timestamp grid and derive its terminal clip bound.

    Qwen3-ForcedAligner emits timestamp tokens on a discrete grid.  Its final
    lexical suffix can therefore land just beyond the authoritative waveform
    boundary (including a zero-width final word whose start and end share the
    first grid point after the waveform).  We retain the raw timestamps and
    clamp that bounded terminal suffix only when creating frame targets.  Two
    grid bins are the nominal distribution gate; rare larger clips remain
    visible instead of corrupting an otherwise valid 500k-row registry.
    """

    root = Path(model_root).expanduser().resolve(strict=True)
    config_path = root / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    segment_ms = float(config.get("timestamp_segment_time", 0.0))
    if not 0.0 < segment_ms <= 1000.0:
        raise RuntimeError(
            f"invalid timestamp_segment_time in {config_path}: {segment_ms!r}"
        )
    grid_sec = segment_ms / 1000.0
    return {
        "timestamp_grid_sec": grid_sec,
        "nominal_endpoint_overshoot_grid_bins": (
            NOMINAL_ENDPOINT_OVERSHOOT_GRID_BINS
        ),
        "nominal_endpoint_grid_overshoot_sec": grid_sec
        * NOMINAL_ENDPOINT_OVERSHOOT_GRID_BINS,
    }


def check_alignment_items(
    items: list[dict[str, Any]],
    duration_sec: float,
    *,
    nominal_endpoint_grid_overshoot_sec: float,
) -> dict[str, Any]:
    """Validate raw word timing without hiding bounded terminal quantization.

    ``raw_in_bounds`` records the literal waveform-boundary result. ``in_bounds``
    is the training eligibility decision: a monotonic terminal suffix may lie
    on a nearby aligner grid point, but no timestamp may exceed the frozen
    nominal grid bound.  This distinction is needed for cases such as a final
    zero-width word at 1.76 s for a 1.75-s waveform.
    """

    tolerance = 0.0011
    monotonic = bool(items)
    in_bounds = bool(items)
    raw_in_bounds = bool(items)
    previous_end = 0.0
    max_endpoint_overshoot = 0.0
    for item in items:
        start = float(item["start_sec"])
        end = float(item["end_sec"])
        if start + tolerance < previous_end or end + tolerance < start:
            monotonic = False
        if start < -tolerance:
            in_bounds = False
            raw_in_bounds = False
        elif start > duration_sec + tolerance:
            raw_in_bounds = False
            if start - duration_sec > nominal_endpoint_grid_overshoot_sec + tolerance:
                in_bounds = False
        if end > duration_sec + tolerance:
            raw_in_bounds = False
        max_endpoint_overshoot = max(
            max_endpoint_overshoot, max(0.0, end - duration_sec)
        )
        previous_end = max(previous_end, end)
    return {
        "nonempty": bool(items),
        "monotonic": monotonic,
        "in_bounds": in_bounds,
        "raw_in_bounds": raw_in_bounds,
        "first_item_start_sec": float(items[0]["start_sec"]) if items else None,
        "last_item_end_sec": float(items[-1]["end_sec"]) if items else None,
        "endpoint_grid_overshoot_sec": float(max_endpoint_overshoot),
        "endpoint_grid_overshoot_within_nominal_bound": bool(
            max_endpoint_overshoot
            <= nominal_endpoint_grid_overshoot_sec + tolerance
        ),
        "nominal_endpoint_grid_overshoot_sec": float(
            nominal_endpoint_grid_overshoot_sec
        ),
    }
