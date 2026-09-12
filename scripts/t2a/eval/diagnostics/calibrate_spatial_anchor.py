#!/usr/bin/env python3
"""Calibrate the deterministic ScenePlan anchor against retained FOA.

This is a data/representation diagnostic, not a model score.  It compares the
Plan-derived ``[mx, my, mz, geometric_dispersion]`` anchor with intensity fields
measured from the original rendered FOA and, when available, its frozen-VAE
reconstruction.  Single-source and overlapping frames are reported separately
so source-level geometry is not conflated with mixture-energy approximation.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torchaudio

from stable_audio_tools.data.foa_intensity import foa_to_intensity_trajectory
from stable_audio_tools.data.spatial_conversation_metadata import SpatialFamilyMetadata
from stable_audio_tools.data.spatial_family_dataset import SpatialFamilyDataset
from stable_audio_tools.data.spatial_story import (
    compile_source_tracks,
    source_tracks_to_mixture_trajectory,
)


def _provider(codec_root: Path) -> SpatialFamilyMetadata:
    return SpatialFamilyMetadata(
        {
            "codec_path": str(codec_root),
            "family_record_key": "spatial_family",
            "turns_key": "turns",
            "target_plan_path": "after.scene_plan",
            "previous_plan_path": "before.scene_plan",
            "plan_output_key": "spatial_plan_tokens",
            "previous_plan_output_key": "previous_plan_tokens",
            "previous_foa_key": "previous_foa",
            "context_presence_key": "previous_foa_present",
            "tracks_output_key": "source_tracks",
            "tracks_layout": "packed_source_features",
            "max_sources": 4,
            "max_distance_m": 20.0,
            "max_tokens": 1024,
        }
    )


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _parse_ranks(value: str) -> list[int]:
    ranks = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not ranks or any(rank < 0 for rank in ranks) or len(set(ranks)) != len(ranks):
        raise argparse.ArgumentTypeError("family ranks must be unique non-negative integers")
    return ranks


def _load_index(latent_root: Path, requested: Iterable[int]) -> dict[int, dict[str, Any]]:
    wanted = set(requested)
    records: dict[int, dict[str, Any]] = {}
    with (latent_root / "index.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            rank = int(record["family_rank"])
            if rank in wanted:
                records[rank] = record
    missing = sorted(wanted - records.keys())
    if missing:
        raise RuntimeError(f"family ranks absent from latent index: {missing}")
    return records


def _direction_rows(
    anchor: torch.Tensor,
    field: torch.Tensor,
    tracks: torch.Tensor,
    segment: str,
    *,
    min_coherence: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    active_sources = tracks[:, 0].gt(0.5)
    if segment == "single_source":
        mask = active_sources.sum(dim=0).eq(1)
    elif segment == "overlap":
        mask = active_sources.sum(dim=0).ge(2)
    elif segment == "active":
        mask = active_sources.any(dim=0)
    else:
        raise ValueError(segment)
    anchor_coherence = (1.0 - anchor[:, 3]).clamp(0.0, 1.0)
    field_coherence = (1.0 - field[:, 3]).clamp(0.0, 1.0)
    mask &= anchor_coherence.ge(min_coherence)
    mask &= field_coherence.ge(min_coherence)
    mask &= anchor[:, :3].norm(dim=-1).gt(1.0e-6)
    mask &= field[:, :3].norm(dim=-1).gt(1.0e-6)
    return anchor[mask], field[mask], mask


def _segment_metrics(
    anchor: torch.Tensor,
    field: torch.Tensor,
    tracks: torch.Tensor,
    segment: str,
    *,
    min_coherence: float,
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor]:
    selected_anchor, selected_field, mask = _direction_rows(
        anchor, field, tracks, segment, min_coherence=min_coherence
    )
    if selected_anchor.numel() == 0:
        return {"valid_frames": 0}, selected_anchor[:, :3], selected_field[:, :3]
    anchor_unit = torch.nn.functional.normalize(selected_anchor[:, :3], dim=-1)
    field_unit = torch.nn.functional.normalize(selected_field[:, :3], dim=-1)
    cosine = (anchor_unit * field_unit).sum(dim=-1).clamp(-1.0, 1.0)
    angle = torch.rad2deg(torch.acos(cosine))
    anchor_diffuseness = selected_anchor[:, 3]
    field_diffuseness = selected_field[:, 3]
    return (
        {
            "valid_frames": int(mask.sum()),
            "direction_cosine_mean": float(cosine.mean()),
            "angular_error_mean_deg": float(angle.mean()),
            "angular_error_median_deg": float(angle.median()),
            "angular_error_p90_deg": float(torch.quantile(angle, 0.9)),
            "anchor_diffuseness_mean": float(anchor_diffuseness.mean()),
            "field_diffuseness_mean": float(field_diffuseness.mean()),
            "diffuseness_mae": float(
                (anchor_diffuseness - field_diffuseness).abs().mean()
            ),
        },
        anchor_unit,
        field_unit,
    )


def _signed_permutation_audit(
    anchors: list[torch.Tensor], fields: list[torch.Tensor]
) -> dict[str, Any]:
    if not anchors:
        return {"valid_frames": 0}
    anchor = torch.cat(anchors, dim=0)
    field = torch.cat(fields, dim=0)
    candidates = []
    for permutation in itertools.permutations(range(3)):
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            transformed = anchor[:, permutation] * anchor.new_tensor(signs)
            cosine = (transformed * field).sum(dim=-1).clamp(-1.0, 1.0)
            mean_angle = float(torch.rad2deg(torch.acos(cosine)).mean())
            matrix = torch.zeros(3, 3)
            for row, (column, sign) in enumerate(zip(permutation, signs)):
                matrix[row, column] = sign
            candidates.append((mean_angle, permutation, signs, float(torch.det(matrix))))
    candidates.sort(key=lambda item: item[0])
    best = candidates[0]
    identity = next(
        item
        for item in candidates
        if item[1] == (0, 1, 2) and item[2] == (1.0, 1.0, 1.0)
    )
    return {
        "valid_frames": int(anchor.shape[0]),
        "identity_angular_error_mean_deg": identity[0],
        "best_angular_error_mean_deg": best[0],
        "best_permutation": list(best[1]),
        "best_signs": list(best[2]),
        "best_determinant": best[3],
        "identity_is_best": best[1] == (0, 1, 2)
        and best[2] == (1.0, 1.0, 1.0),
    }


def _aggregate_segments(
    records: list[dict[str, Any]], source: str
) -> dict[str, Any]:
    aggregate: dict[str, Any] = {}
    metric_names = (
        "direction_cosine_mean",
        "angular_error_mean_deg",
        "anchor_diffuseness_mean",
        "field_diffuseness_mean",
        "diffuseness_mae",
    )
    for segment in ("single_source", "overlap", "active"):
        rows = []
        for record in records:
            source_metrics = record.get(source)
            if source_metrics is None:
                continue
            metrics = source_metrics[segment]
            count = int(metrics.get("valid_frames", 0))
            if count:
                rows.append((count, metrics))
        total = sum(count for count, _metrics in rows)
        result: dict[str, Any] = {"valid_frames": total}
        for name in metric_names:
            values = [
                (count, metrics.get(name)) for count, metrics in rows
                if metrics.get(name) is not None
            ]
            result[name] = (
                sum(count * float(value) for count, value in values)
                / sum(count for count, _value in values)
                if values
                else None
            )
        aggregate[segment] = result
    return aggregate


def _audio_metrics(
    plan: dict[str, Any], audio: torch.Tensor, *, min_coherence: float
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor]:
    field = foa_to_intensity_trajectory(audio, hop=1024)
    tracks = compile_source_tracks(
        plan, num_frames=int(field.shape[0]), max_sources=4
    )["tracks"]
    anchor = source_tracks_to_mixture_trajectory(tracks)
    segments: dict[str, Any] = {}
    single_anchor = torch.empty(0, 3)
    single_field = torch.empty(0, 3)
    for segment in ("single_source", "overlap", "active"):
        metrics, anchor_rows, field_rows = _segment_metrics(
            anchor,
            field,
            tracks,
            segment,
            min_coherence=min_coherence,
        )
        segments[segment] = metrics
        if segment == "single_source":
            single_anchor, single_field = anchor_rows, field_rows
    return segments, single_anchor, single_field


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Spatial anchor compiler calibration",
        "",
        f"Families: `{', '.join(map(str, report['family_ranks']))}`.",
        "",
    ]
    for source in ("raw", "vae_reconstruction"):
        aggregate = report["aggregate"].get(source)
        if aggregate is None:
            continue
        audit = aggregate["single_source_axis_audit"]
        lines.extend(
            [
                f"## {source}",
                "",
                f"Single-source identity error: `{audit.get('identity_angular_error_mean_deg')}` degrees.",
                f"Best signed permutation: `{audit.get('best_permutation')}` with signs "
                f"`{audit.get('best_signs')}` and error "
                f"`{audit.get('best_angular_error_mean_deg')}` degrees.",
                "",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--latent-root",
        type=Path,
        default=Path("/mnt/sdb/audio_dataset/spatial_cot_v1/latents/validation"),
    )
    parser.add_argument(
        "--codec-root",
        type=Path,
        default=Path("/mnt/sdb/audio_dataset/spatial_cot_v1/codec"),
    )
    parser.add_argument(
        "--render-root",
        type=Path,
        default=Path("/mnt/sdb/audio_dataset/spatial_cot_v1/eval_rendered/validation"),
    )
    parser.add_argument("--family-ranks", type=_parse_ranks, default=[274, 8179])
    parser.add_argument(
        "--reconstruction-root",
        type=Path,
        help="optional sequential-sweep root containing step0 target WAVs",
    )
    parser.add_argument("--min-coherence", type=float, default=0.1)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    for path in (
        args.latent_root / "READY",
        args.codec_root / "READY",
        args.render_root / "QUALITY.json",
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not 0.0 <= args.min_coherence <= 1.0:
        raise ValueError("min-coherence must be in [0,1]")

    index = _load_index(args.latent_root, args.family_ranks)
    dataset = SpatialFamilyDataset(
        [
            {
                "path": str(args.latent_root),
                "custom_metadata_fn": _provider(args.codec_root),
            }
        ],
        require_ready=True,
        max_open_shards=1,
    )
    records = []
    aggregate_rows: dict[str, tuple[list[torch.Tensor], list[torch.Tensor]]] = {
        "raw": ([], []),
        "vae_reconstruction": ([], []),
    }
    for rank in args.family_ranks:
        _latents, family = dataset[rank]
        family_id = str(family["family_id"])
        work_shard = int(index[rank]["work_shard"])
        family_render_root = (
            args.render_root
            / f"work-{work_shard:05d}"
            / "render"
            / "families"
            / family_id
        )
        for turn_index, turn in enumerate(family["family_turn_metadata"]):
            raw_path = (
                family_render_root
                / "audio"
                / f"turn_{turn_index:03d}_WYZX_4ch.flac"
            )
            raw, sample_rate = torchaudio.load(raw_path)
            if sample_rate != 44_100 or raw.shape[0] != 4:
                raise RuntimeError(
                    f"unexpected retained FOA format at {raw_path}: "
                    f"shape={tuple(raw.shape)} sr={sample_rate}"
                )
            raw_metrics, raw_anchor, raw_field = _audio_metrics(
                turn["scene_plan"], raw, min_coherence=args.min_coherence
            )
            aggregate_rows["raw"][0].append(raw_anchor)
            aggregate_rows["raw"][1].append(raw_field)
            reconstruction_metrics = None
            reconstruction_path = None
            if args.reconstruction_root is not None:
                candidate = (
                    args.reconstruction_root
                    / f"family{rank}_sweep"
                    / "step0"
                    / f"family_{rank}"
                    / f"turn_{turn_index:02d}.target.wav"
                )
                if candidate.is_file():
                    reconstruction_path = candidate
                    reconstruction, reconstruction_rate = torchaudio.load(candidate)
                    if reconstruction_rate != 44_100 or reconstruction.shape[0] != 4:
                        raise RuntimeError(f"unexpected reconstruction format: {candidate}")
                    reconstruction_metrics, recon_anchor, recon_field = _audio_metrics(
                        turn["scene_plan"],
                        reconstruction,
                        min_coherence=args.min_coherence,
                    )
                    aggregate_rows["vae_reconstruction"][0].append(recon_anchor)
                    aggregate_rows["vae_reconstruction"][1].append(recon_field)
            records.append(
                {
                    "family_id": family_id,
                    "family_rank": rank,
                    "turn": turn_index,
                    "raw_path": str(raw_path),
                    "raw": raw_metrics,
                    "reconstruction_path": (
                        str(reconstruction_path) if reconstruction_path else None
                    ),
                    "vae_reconstruction": reconstruction_metrics,
                }
            )

    aggregate = {}
    for source, (anchors, fields) in aggregate_rows.items():
        nonempty = [
            (anchor, field)
            for anchor, field in zip(anchors, fields)
            if anchor.numel() > 0
        ]
        if not nonempty:
            aggregate[source] = None
            continue
        aggregate[source] = {
            "single_source_axis_audit": _signed_permutation_audit(
                [item[0] for item in nonempty], [item[1] for item in nonempty]
            ),
            "segments": _aggregate_segments(records, source),
        }
    report = {
        "schema": "stable_audio_tools.spatial_anchor_calibration",
        "schema_version": 1,
        "family_ranks": args.family_ranks,
        "latent_root": str(args.latent_root.resolve()),
        "render_root": str(args.render_root.resolve()),
        "reconstruction_root": (
            str(args.reconstruction_root.resolve())
            if args.reconstruction_root is not None
            else None
        ),
        "min_coherence": args.min_coherence,
        "aggregate": aggregate,
        "cases": records,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(args.output_dir / "RESULT.json", report)
    _write_markdown(args.output_dir / "SUMMARY.md", report)
    print(json.dumps(report["aggregate"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
