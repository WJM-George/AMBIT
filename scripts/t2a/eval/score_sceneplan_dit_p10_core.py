#!/usr/bin/env python3
"""Score QC, activity, and FOA DoA on a frozen P10 panel."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F

from scripts.t2a.eval.sceneplan_44_eval_common import atomic_wav, virtual_stereo
from scripts.t2a.eval.sceneplan_dit_p10_panel_common import (
    DEFAULT_EVAL_ROOT,
    DOMAINS,
    atomic_json,
    checkpoint_steps,
    load_foa,
    load_output_rows,
    sha256_file,
    summarize,
)
from stable_audio_tools.data.foa_intensity import foa_to_intensity_trajectory
from stable_audio_tools.data.model_sceneplan import compile_model_44_controls


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def _frame_audio(audio: torch.Tensor, frames: int, hop: int) -> torch.Tensor:
    target = int(frames) * int(hop)
    if int(audio.shape[-1]) > target:
        audio = audio[:, :target]
    elif int(audio.shape[-1]) < target:
        audio = F.pad(audio, (0, target - int(audio.shape[-1])))
    return audio.reshape(4, frames, hop)


def _doa_metrics(
    audio: torch.Tensor,
    scene_plan: dict[str, Any],
    *,
    model_num_samples: int,
    latent_frames: int,
    hop: int = 1024,
    min_coherence: float = 0.1,
) -> dict[str, Any]:
    controls = compile_model_44_controls(
        scene_plan,
        model_num_samples=model_num_samples,
        latent_frames_valid=latent_frames,
    )
    features = torch.from_numpy(controls["source_trajectory_features"]).float()
    active_by_source = torch.from_numpy(
        controls["source_event_frame_ids"]
    ).gt(0)
    active_count = active_by_source.sum(dim=0)
    scene_active = active_count.gt(0)
    # A single ground-truth direction is only well-defined when exactly one
    # source is active.  Overlapping frames are evaluated separately against
    # the rendered reference mixture by `_paired_doa_metrics`.
    active = active_count.eq(1)
    source_directions = torch.stack(
        [
            features[..., 3] * features[..., 1],
            features[..., 3] * features[..., 0],
            features[..., 2],
        ],
        dim=-1,
    )
    expected = (
        source_directions * active_by_source[..., None].to(source_directions.dtype)
    ).sum(dim=0)
    expected_norm = expected.norm(dim=-1)
    expected = expected / expected_norm[:, None].clamp_min(1.0e-8)

    framed = _frame_audio(audio, latent_frames, hop)
    padded = framed.reshape(4, latent_frames * hop)
    trajectory = foa_to_intensity_trajectory(padded, hop=hop)
    generated = trajectory[:, :3]
    generated_norm = generated.norm(dim=-1)
    generated_unit = generated / generated_norm[:, None].clamp_min(1.0e-8)
    coherence = (1.0 - trajectory[:, 3]).clamp(0.0, 1.0)
    energy = framed.square().mean(dim=(0, 2))
    energy_floor = max(1.0e-12, float(energy.max()) * 1.0e-4)
    valid = (
        active
        & expected_norm.gt(1.0e-6)
        & generated_norm.gt(1.0e-6)
        & coherence.ge(min_coherence)
        & energy.ge(energy_floor)
    )
    result: dict[str, Any] = {
        "frames": latent_frames,
        "active_frames": int(active.sum()),
        "scene_active_frames": int(scene_active.sum()),
        "overlap_frames": int(active_count.gt(1).sum()),
        "plan_doa_policy": "exactly_one_active_source_frames",
        "valid_direction_frames": int(valid.sum()),
        "valid_direction_fraction": (
            float(valid.sum() / active.sum()) if bool(active.any()) else None
        ),
        "min_coherence": min_coherence,
        "spherical_error_mean_deg": None,
        "spherical_error_median_deg": None,
        "spherical_error_p95_deg": None,
        "start_spherical_error_mean_deg": None,
        "end_spherical_error_mean_deg": None,
        "trajectory_extent_error_deg": None,
        "azimuth_circular_mae_deg": None,
        "elevation_mae_deg": None,
        "direction_cosine_mean": None,
        "generated_coherence_mean": (
            float(coherence[active].mean()) if bool(active.any()) else None
        ),
    }
    if bool(valid.any()):
        cosine = (generated_unit[valid] * expected[valid]).sum(dim=-1).clamp(-1.0, 1.0)
        spherical = torch.rad2deg(torch.acos(cosine))
        generated_az = torch.atan2(generated_unit[valid, 1], generated_unit[valid, 0])
        expected_az = torch.atan2(expected[valid, 1], expected[valid, 0])
        az_delta = torch.atan2(
            torch.sin(generated_az - expected_az),
            torch.cos(generated_az - expected_az),
        ).abs()
        generated_el = torch.asin(generated_unit[valid, 2].clamp(-1.0, 1.0))
        expected_el = torch.asin(expected[valid, 2].clamp(-1.0, 1.0))
        weights = energy[valid].clamp_min(1.0e-12)
        weighted = lambda value: float((value * weights).sum() / weights.sum())
        result.update(
            {
                "spherical_error_mean_deg": weighted(spherical),
                "spherical_error_median_deg": float(spherical.median()),
                "spherical_error_p95_deg": float(torch.quantile(spherical, 0.95)),
                "azimuth_circular_mae_deg": math.degrees(weighted(az_delta)),
                "elevation_mae_deg": math.degrees(weighted((generated_el - expected_el).abs())),
                "direction_cosine_mean": weighted(cosine),
            }
        )
        # Start/end and extent are source-trajectory metrics, so they remain
        # defined only for a single-source scene.  Combining endpoints from
        # different sources would produce a plausible-looking but invalid
        # number.
        if len(scene_plan["sources"]) == 1:
            valid_indices = torch.nonzero(valid).flatten()
            endpoint_frames = max(1, int(math.ceil(int(valid_indices.numel()) * 0.2)))
            start_indices = valid_indices[:endpoint_frames]
            end_indices = valid_indices[-endpoint_frames:]

            def endpoint_unit(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
                endpoint_weights = energy[indices].clamp_min(1.0e-12)
                vector = (values[indices] * endpoint_weights[:, None]).sum(dim=0)
                return vector / vector.norm().clamp_min(1.0e-8)

            generated_start = endpoint_unit(generated_unit, start_indices)
            generated_end = endpoint_unit(generated_unit, end_indices)
            expected_start = endpoint_unit(expected, start_indices)
            expected_end = endpoint_unit(expected, end_indices)
            generated_extent = torch.rad2deg(
                torch.acos((generated_start * generated_end).sum().clamp(-1.0, 1.0))
            )
            expected_extent = torch.rad2deg(
                torch.acos((expected_start * expected_end).sum().clamp(-1.0, 1.0))
            )
            result.update(
                {
                    "start_spherical_error_mean_deg": float(
                        torch.rad2deg(
                            torch.acos(
                                (generated_unit[start_indices] * expected[start_indices])
                                .sum(dim=-1)
                                .clamp(-1.0, 1.0)
                            )
                        ).mean()
                    ),
                    "end_spherical_error_mean_deg": float(
                        torch.rad2deg(
                            torch.acos(
                                (generated_unit[end_indices] * expected[end_indices])
                                .sum(dim=-1)
                                .clamp(-1.0, 1.0)
                            )
                        ).mean()
                    ),
                    "trajectory_extent_error_deg": float(
                        (generated_extent - expected_extent).abs()
                    ),
                }
            )
    return result


def _activity_metrics(
    audio: torch.Tensor,
    scene_plan: dict[str, Any],
    *,
    model_num_samples: int,
    latent_frames: int,
    hop: int = 1024,
) -> dict[str, Any]:
    controls = compile_model_44_controls(
        scene_plan,
        model_num_samples=model_num_samples,
        latent_frames_valid=latent_frames,
    )
    expected = torch.from_numpy(controls["source_event_frame_ids"]).gt(0).any(dim=0)
    framed = _frame_audio(audio, latent_frames, hop)
    frame_rms = framed.square().mean(dim=(0, 2)).sqrt()
    active_median = float(frame_rms[expected].median()) if bool(expected.any()) else 0.0
    threshold = max(10.0 ** (-60.0 / 20.0), active_median * 0.1)
    detected = frame_rms >= threshold
    intersection = int((detected & expected).sum())
    union = int((detected | expected).sum())
    active_rms = float(framed[:, expected].square().mean().sqrt()) if bool(expected.any()) else None
    inactive_rms = (
        float(framed[:, ~expected].square().mean().sqrt()) if bool((~expected).any()) else None
    )
    active_to_inactive_db = None
    if active_rms is not None and inactive_rms is not None:
        active_to_inactive_db = 20.0 * math.log10(
            max(active_rms, 1.0e-12) / max(inactive_rms, 1.0e-12)
        )
    expected_indices = torch.nonzero(expected).flatten()
    detected_indices = torch.nonzero(detected).flatten()
    onset_error = None
    offset_error = None
    if int(expected_indices.numel()) and int(detected_indices.numel()):
        seconds_per_frame = hop / 44_100.0
        onset_error = float(
            abs(int(detected_indices[0]) - int(expected_indices[0])) * seconds_per_frame
        )
        offset_error = float(
            abs(int(detected_indices[-1]) - int(expected_indices[-1])) * seconds_per_frame
        )
    return {
        "expected_active_frames": int(expected.sum()),
        "detected_active_frames": int(detected.sum()),
        "energy_threshold": threshold,
        "temporal_iou": intersection / union if union else None,
        "onset_abs_error_sec": onset_error,
        "offset_abs_error_sec": offset_error,
        "active_rms": active_rms,
        "inactive_rms": inactive_rms,
        "active_to_inactive_db": active_to_inactive_db,
    }


def _paired_doa_metrics(
    generated_audio: torch.Tensor,
    reference_audio: torch.Tensor,
    scene_plan: dict[str, Any],
    *,
    model_num_samples: int,
    latent_frames: int,
    hop: int = 1024,
    min_coherence: float = 0.1,
) -> dict[str, Any]:
    """Compare generated and rendered-reference FOA intensity trajectories."""

    controls = compile_model_44_controls(
        scene_plan,
        model_num_samples=model_num_samples,
        latent_frames_valid=latent_frames,
    )
    active = torch.from_numpy(controls["source_event_frame_ids"]).gt(0).any(dim=0)
    trajectories = []
    energies = []
    coherences = []
    for audio in (generated_audio, reference_audio):
        framed = _frame_audio(audio, latent_frames, hop)
        trajectory = foa_to_intensity_trajectory(
            framed.reshape(4, latent_frames * hop), hop=hop
        )
        vector = trajectory[:, :3]
        norm = vector.norm(dim=-1)
        trajectories.append(vector / norm[:, None].clamp_min(1.0e-8))
        energies.append(framed.square().mean(dim=(0, 2)))
        coherences.append((1.0 - trajectory[:, 3]).clamp(0.0, 1.0))
    valid = active.clone()
    for energy, coherence, unit in zip(energies, coherences, trajectories):
        floor = max(1.0e-12, float(energy.max()) * 1.0e-4)
        valid &= energy.ge(floor) & coherence.ge(min_coherence) & unit.norm(dim=-1).gt(1.0e-6)
    result: dict[str, Any] = {
        "active_frames": int(active.sum()),
        "valid_direction_frames": int(valid.sum()),
        "valid_direction_fraction": float(valid.sum() / active.sum()) if bool(active.any()) else None,
        "spherical_error_mean_deg": None,
        "spherical_error_median_deg": None,
        "spherical_error_p95_deg": None,
    }
    if bool(valid.any()):
        cosine = (trajectories[0][valid] * trajectories[1][valid]).sum(dim=-1).clamp(-1.0, 1.0)
        spherical = torch.rad2deg(torch.acos(cosine))
        weights = energies[1][valid].clamp_min(1.0e-12)
        result.update(
            {
                "spherical_error_mean_deg": float((spherical * weights).sum() / weights.sum()),
                "spherical_error_median_deg": float(spherical.median()),
                "spherical_error_p95_deg": float(torch.quantile(spherical, 0.95)),
            }
        )
    return result


def _aggregate(rows: list[dict[str, Any]], steps: tuple[int, ...]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for step in steps:
        result[str(step)] = {}
        for domain in DOMAINS:
            chosen = [
                row for row in rows if row["checkpoint_step"] == step and row["domain"] == domain
            ]
            result[str(step)][domain] = {
                "rows": len(chosen),
                "raw_peak": summarize(row["qc"]["peak"] for row in chosen),
                "raw_rms": summarize(row["qc"]["rms"] for row in chosen),
                "raw_fraction_abs_ge_1": summarize(
                    row["qc"]["fraction_abs_ge_1"] for row in chosen
                ),
                "plan_spherical_error_mean_deg": summarize(
                    row["generated_doa"]["spherical_error_mean_deg"] for row in chosen
                ),
                "reference_plan_spherical_error_mean_deg": summarize(
                    row["reference_doa"]["spherical_error_mean_deg"] for row in chosen
                ),
                "generated_reference_spherical_error_mean_deg": summarize(
                    row["generated_reference_doa"]["spherical_error_mean_deg"]
                    for row in chosen
                ),
                "plan_spherical_error_p95_deg": summarize(
                    row["generated_doa"]["spherical_error_p95_deg"] for row in chosen
                ),
                "plan_start_spherical_error_mean_deg": summarize(
                    row["generated_doa"]["start_spherical_error_mean_deg"]
                    for row in chosen
                ),
                "plan_end_spherical_error_mean_deg": summarize(
                    row["generated_doa"]["end_spherical_error_mean_deg"]
                    for row in chosen
                ),
                "plan_trajectory_extent_error_deg": summarize(
                    row["generated_doa"]["trajectory_extent_error_deg"]
                    for row in chosen
                ),
                "reference_plan_trajectory_extent_error_deg": summarize(
                    row["reference_doa"]["trajectory_extent_error_deg"]
                    for row in chosen
                ),
                "plan_azimuth_circular_mae_deg": summarize(
                    row["generated_doa"]["azimuth_circular_mae_deg"] for row in chosen
                ),
                "plan_elevation_mae_deg": summarize(
                    row["generated_doa"]["elevation_mae_deg"] for row in chosen
                ),
                "valid_direction_fraction": summarize(
                    row["generated_doa"]["valid_direction_fraction"] for row in chosen
                ),
                "activity_temporal_iou": summarize(
                    row["generated_activity"]["temporal_iou"] for row in chosen
                ),
                "reference_activity_temporal_iou": summarize(
                    row["reference_activity"]["temporal_iou"] for row in chosen
                ),
                "activity_onset_abs_error_sec": summarize(
                    row["generated_activity"]["onset_abs_error_sec"] for row in chosen
                ),
                "activity_offset_abs_error_sec": summarize(
                    row["generated_activity"]["offset_abs_error_sec"] for row in chosen
                ),
                "activity_active_to_inactive_db": summarize(
                    row["generated_activity"]["active_to_inactive_db"] for row in chosen
                ),
            }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", type=Path, default=DEFAULT_EVAL_ROOT)
    args = parser.parse_args()
    root = args.eval_root.expanduser().resolve(strict=True)
    steps = checkpoint_steps(root)
    output_rows = load_output_rows(root)
    reference_cache: dict[str, dict[str, Any]] = {}
    scored: list[dict[str, Any]] = []
    for index, row in enumerate(output_rows, start=1):
        expected_samples = int(row["model_num_samples"])
        generated_path = Path(row["generated_foa_path"])
        reference_path = Path(row["reference_foa_path"])
        if sha256_file(generated_path) != row["generated_foa_sha256"]:
            raise RuntimeError(f"generated hash mismatch: {generated_path}")
        if sha256_file(reference_path) != row["reference_foa_sha256"]:
            raise RuntimeError(f"reference hash mismatch: {reference_path}")
        generated, generated_rate = load_foa(generated_path, expected_samples=expected_samples)
        reference, reference_rate = load_foa(reference_path, expected_samples=expected_samples)
        if generated_rate != 44_100 or reference_rate != 44_100:
            raise RuntimeError("P10 benchmark sample rate changed")
        latent_frames = int(row["latent_frames"])
        generated_doa = _doa_metrics(
            generated,
            row["scene_plan"],
            model_num_samples=expected_samples,
            latent_frames=latent_frames,
        )
        reference_doa = _doa_metrics(
            reference,
            row["scene_plan"],
            model_num_samples=expected_samples,
            latent_frames=latent_frames,
        )
        generated_activity = _activity_metrics(
            generated,
            row["scene_plan"],
            model_num_samples=expected_samples,
            latent_frames=latent_frames,
        )
        reference_activity = _activity_metrics(
            reference,
            row["scene_plan"],
            model_num_samples=expected_samples,
            latent_frames=latent_frames,
        )
        generated_reference_doa = _paired_doa_metrics(
            generated,
            reference,
            row["scene_plan"],
            model_num_samples=expected_samples,
            latent_frames=latent_frames,
        )

        panel_id = str(row["panel_id"])
        if panel_id not in reference_cache:
            preview, preview_info = virtual_stereo(reference)
            preview_path = root / "references" / row["domain"] / panel_id / "reference_stereo.wav"
            atomic_wav(preview_path, preview, 44_100, subtype="PCM_16")
            reference_cache[panel_id] = {
                "panel_id": panel_id,
                "domain": row["domain"],
                "sample_id": row["sample_id"],
                "reference_foa_path": str(reference_path.resolve()),
                "reference_foa_sha256": row["reference_foa_sha256"],
                "reference_stereo_path": str(preview_path.resolve()),
                "reference_stereo_sha256": sha256_file(preview_path),
                "reference_doa": reference_doa,
                "reference_activity": reference_activity,
                **preview_info,
            }
        scored.append(
            {
                "checkpoint_step": int(row["checkpoint_step"]),
                "panel_id": panel_id,
                "domain": row["domain"],
                "sample_id": row["sample_id"],
                "generated_foa_path": str(generated_path.resolve()),
                "generated_stereo_path": row["generated_stereo_path"],
                "reference_foa_path": str(reference_path.resolve()),
                "reference_stereo_path": reference_cache[panel_id]["reference_stereo_path"],
                "qc": row["qc"],
                "generated_doa": generated_doa,
                "reference_doa": reference_doa,
                "generated_reference_doa": generated_reference_doa,
                "generated_activity": generated_activity,
                "reference_activity": reference_activity,
            }
        )
        print(
            json.dumps(
                {
                    "event": "core_scored",
                    "index": index,
                    "count": len(output_rows),
                    "step": row["checkpoint_step"],
                    "panel_id": panel_id,
                    "doa_deg": generated_doa["spherical_error_mean_deg"],
                }
            ),
            flush=True,
        )

    metrics_root = root / "metrics"
    _atomic_jsonl(metrics_root / "core_per_output.jsonl", scored)
    _atomic_jsonl(metrics_root / "reference_panel.jsonl", list(reference_cache.values()))
    summary = {
        "schema": "stable_audio_tools.sceneplan_dit_p10_core_metrics",
        "schema_version": 1,
        "status": "PASS",
        "evaluation_rows": len(reference_cache),
        "checkpoint_outputs": len(scored),
        "raw_foa_files": len(scored),
        "reference_previews": len(reference_cache),
        "all_generated_finite": all(row["qc"]["finite"] for row in scored),
        "all_generated_four_channel": all(row["qc"]["channels"] == 4 for row in scored),
        "aggregates": _aggregate(scored, steps),
    }
    atomic_json(metrics_root / "CORE_SUMMARY.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
