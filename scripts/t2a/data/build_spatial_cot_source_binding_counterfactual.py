#!/usr/bin/env python3
"""Build fixed-10-second matched source-binding counterfactuals.

Each output item contains two balanced pairs:

    original target, control-swapped target,
    original target, control-swapped target.

Within a pair, semantic content, dry sources, room, and the Plan-derived global
field anchor are fixed.  Complete activity/motion/gain bundles are exchanged
between two persistent source IDs.  The resulting target FOA therefore differs
only in which source owns which execution track.  All four rows remain
independent creation examples; this curriculum does not invent an edit chain.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Mapping

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.synthesis.render_spatial_edit_families import (  # noqa: E402
    _flac_pcm24_memory_roundtrip,
)
from scripts.t2a.data.build_spatial_cot_source_curriculum import (  # noqa: E402
    EXPECTED_VAE_SHA256,
    _creation_record,
    _encode_audio,
    _load_recipe_families,
    _pcm24_track,
    _sha256,
    _signal_stats,
    _write_store,
)
from scripts.t2a.data.preencode_spatial_cot_family_shard import (  # noqa: E402
    _load_vae,
)
from stable_audio_tools.data.spatial_counterfactual import (  # noqa: E402
    swap_recipe_source_controls,
    swap_scene_plan_source_controls,
)
from stable_audio_tools.data.spatial_edit_recipe import (  # noqa: E402
    describe_recipe,
    recipe_fingerprint,
    source_render_signature,
)
from stable_audio_tools.data.spatial_family_dataset import (  # noqa: E402
    SpatialFamilyDataset,
)
from stable_audio_tools.data.spatial_story import (  # noqa: E402
    compile_source_tracks,
    source_tracks_to_mixture_trajectory,
)


SCHEMA = "stable_audio_tools.spatial_cot_source_binding_counterfactual"
SCHEMA_VERSION = 2
WINDOW_SAMPLES = 442_368
LATENT_FRAMES = 432
ANCHOR_NUMERICAL_TOLERANCE = 1.0e-6
PAIR_PEAK_CEILING = 0.95


def _sources_by_id(plan: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    sources = ((plan.get("scene") or {}).get("sources") or [])
    result = {str(source.get("source_id") or ""): source for source in sources}
    if "" in result or len(result) != len(sources):
        raise RuntimeError("ScenePlan source IDs must be present and unique")
    return result


def _candidate_score(
    plan: Mapping[str, Any], source_a: str, source_b: str
) -> float:
    """Prefer semantically distinct sources with dissimilar execution tracks."""

    compiled = compile_source_tracks(plan, num_frames=LATENT_FRAMES)
    tracks = compiled["tracks"].float()
    source_ids = list(compiled["source_ids"])
    left = source_ids.index(source_a)
    right = source_ids.index(source_b)
    track_scale = tracks.square().mean().sqrt().clamp_min(1.0e-6)
    track_delta = float((tracks[left] - tracks[right]).square().mean().sqrt() / track_scale)
    sources = _sources_by_id(plan)
    left_event = sources[source_a].get("event") or {}
    right_event = sources[source_b].get("event") or {}
    category_bonus = float(left_event.get("category") != right_event.get("category"))
    label_bonus = float(left_event.get("label") != right_event.get("label"))
    return track_delta + 0.5 * category_bonus + 0.25 * label_bonus


def _rank_candidates(
    turns: list[Mapping[str, Any]], recipes: list[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for turn_index, (turn, recipe) in enumerate(zip(turns, recipes)):
        plan = turn["after"]["scene_plan"]
        plan_ids = set(_sources_by_id(plan))
        recipe_ids = [str(source["source_id"]) for source in recipe["sources"]]
        if plan_ids != set(recipe_ids):
            raise RuntimeError(f"turn {turn_index} Plan/recipe source IDs differ")
        for left_index, source_a in enumerate(recipe_ids):
            for source_b in recipe_ids[left_index + 1 :]:
                candidates.append(
                    {
                        "turn_index": turn_index,
                        "source_a": source_a,
                        "source_b": source_b,
                        "score": _candidate_score(plan, source_a, source_b),
                    }
                )
    return sorted(
        candidates,
        key=lambda item: (
            -float(item["score"]),
            int(item["turn_index"]),
            str(item["source_a"]),
            str(item["source_b"]),
        ),
    )


def _render_mix(
    recipe: Mapping[str, Any],
    *,
    master_gain: float,
    track_cache: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, float]]:
    components: list[np.ndarray] = []
    for source in recipe["sources"]:
        track_id = source_render_signature(recipe, source)
        track = track_cache.get(track_id)
        if track is None:
            track = _pcm24_track(recipe, source)
            track_cache[track_id] = track
        gain = 10.0 ** (float(source.get("gain_db", 0.0)) / 20.0)
        components.append((track * gain * master_gain).astype(np.float32, copy=False))
    raw = np.sum(np.stack(components, axis=0), axis=0, dtype=np.float32)
    if tuple(raw.shape) != (4, WINDOW_SAMPLES) or not np.isfinite(raw).all():
        raise RuntimeError(f"invalid rendered FOA shape/values: {raw.shape}")
    clipped_fraction = float(np.mean(np.abs(raw) > 1.0))
    clipped = np.clip(raw, -1.0, 1.0).astype(np.float32, copy=False)
    rendered = _flac_pcm24_memory_roundtrip(
        clipped, int(recipe["audio"]["sample_rate"])
    )
    stats = _signal_stats(rendered)
    stats.update(
        {
            "raw_peak": float(np.max(np.abs(raw))),
            "clipped_fraction": clipped_fraction,
        }
    )
    return rendered, stats


def _same_anchor_audit(
    base_plan: Mapping[str, Any], swapped_plan: Mapping[str, Any]
) -> dict[str, float]:
    base_tracks = compile_source_tracks(base_plan, num_frames=LATENT_FRAMES)["tracks"]
    swapped_tracks = compile_source_tracks(
        swapped_plan, num_frames=LATENT_FRAMES
    )["tracks"]
    if not torch.equal(
        base_tracks.sort(dim=0).values, swapped_tracks.sort(dim=0).values
    ):
        raise RuntimeError("source-control swap changed the track multiset")
    report: dict[str, float] = {}
    for fourth in ("received_level", "geometric_dispersion"):
        base_anchor = source_tracks_to_mixture_trajectory(
            base_tracks, fourth_component=fourth
        )
        swapped_anchor = source_tracks_to_mixture_trajectory(
            swapped_tracks, fourth_component=fourth
        )
        maximum = float((base_anchor - swapped_anchor).abs().max())
        report[f"{fourth}_max_abs"] = maximum
        if maximum > ANCHOR_NUMERICAL_TOLERANCE:
            raise RuntimeError(
                f"{fourth} anchor changed by {maximum}, exceeding "
                f"{ANCHOR_NUMERICAL_TOLERANCE}"
            )
    return report


def _swapped_state(
    *,
    base_turn: Mapping[str, Any],
    recipe: Mapping[str, Any],
    source_a: str,
    source_b: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    base_plan = base_turn["after"]["scene_plan"]
    swapped_plan = swap_scene_plan_source_controls(base_plan, source_a, source_b)
    swapped_recipe = swap_recipe_source_controls(recipe, source_a, source_b)
    description = describe_recipe(swapped_recipe)
    swapped_plan["caption"] = description
    swapped_recipe.update(
        {
            "scene_plan": copy.deepcopy(swapped_plan),
            "scene_description": description,
            "instruction": description,
            "planner_prompt": description,
        }
    )
    swapped_recipe.pop("recipe_fingerprint", None)
    swapped_recipe["recipe_fingerprint"] = recipe_fingerprint(swapped_recipe)

    base_sources = _sources_by_id(base_plan)
    swapped_sources = _sources_by_id(swapped_plan)
    for source_id in base_sources:
        for semantic_key in ("event", "content"):
            if base_sources[source_id].get(semantic_key) != swapped_sources[
                source_id
            ].get(semantic_key):
                raise RuntimeError(
                    f"counterfactual changed {source_id}.{semantic_key}"
                )
    _same_anchor_audit(base_plan, swapped_plan)
    return swapped_plan, swapped_recipe


def _curriculum_record(
    base_turn: Mapping[str, Any],
    *,
    family_id: str,
    turn_index: int,
    plan: Mapping[str, Any],
    curriculum: Mapping[str, Any],
    signal_stats: Mapping[str, float],
) -> dict[str, Any]:
    plan_copy = copy.deepcopy(dict(plan))
    plan_copy["sample_id"] = f"{family_id}_turn_{turn_index:03d}"
    record = _creation_record(
        base_turn,
        family_id=family_id,
        turn_index=turn_index,
        plan=plan_copy,
        semantic_caption=str(base_turn["semantic_caption"]),
        semantic_caption_metadata=base_turn["semantic_caption_metadata"],
        curriculum=curriculum,
        signal_stats=signal_stats,
    )
    record["after"]["foa_sha256"] = None
    record["after"]["source_track_refs"] = []
    return record


def build(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output_root.expanduser().resolve()
    if (output / "READY").is_file():
        report = json.loads((output / "AUDIT.json").read_text(encoding="utf-8"))
        if report.get("status") != "PASS":
            raise RuntimeError(f"cached counterfactual audit is not PASS: {output}")
        print(json.dumps({"status": "CACHED", **report}, indent=2))
        return report
    if output.exists():
        raise RuntimeError(f"refusing to overwrite incomplete curriculum: {output}")

    family_store = args.family_store.expanduser().resolve()
    caption_overlay = args.caption_overlay.expanduser().resolve()
    vae_checkpoint = args.vae_checkpoint.expanduser().resolve()
    vae_config = args.vae_config.expanduser().resolve()
    if _sha256(vae_checkpoint) != EXPECTED_VAE_SHA256:
        raise RuntimeError(f"frozen VAE checksum mismatch: {vae_checkpoint}")

    dataset = SpatialFamilyDataset(
        [{"path": family_store, "caption_overlay_path": caption_overlay}],
        require_ready=True,
    )
    if not 0 < args.families <= len(dataset):
        raise ValueError("--families is outside the source store")
    source_families: list[dict[str, Any]] = []
    source_latents: list[torch.Tensor] = []
    for rank in range(args.families):
        latent, info = dataset[rank]
        family = copy.deepcopy(info["spatial_family"])
        if int(family["family_rank"]) != rank or tuple(latent.shape) != (
            4,
            64,
            LATENT_FRAMES,
        ):
            raise RuntimeError(f"source family rank/shape mismatch at {rank}")
        source_families.append(family)
        source_latents.append(latent.to(torch.float16).contiguous())
    recipes = _load_recipe_families(source_families)

    swap_audio: list[np.ndarray] = []
    prepared: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    anchor_max = 0.0
    pair_relative_rms: list[float] = []
    swap_raw_peaks: list[float] = []
    original_swap_raw_peaks: list[float] = []
    swap_clip_fractions: list[float] = []
    pair_master_scales: list[float] = []
    base_peak_errors: list[float] = []
    base_rms_errors: list[float] = []

    for source_rank, (family, mixture) in enumerate(
        zip(source_families, source_latents)
    ):
        original_id = str(family["family_id"])
        recipe_family = recipes[original_id]
        recipe_turns = recipe_family["recipes"]
        candidates = _rank_candidates(family["turns"], recipe_turns)
        if not candidates:
            excluded.append(
                {
                    "source_family_rank": source_rank,
                    "source_family_id": original_id,
                    "reason": "no_turn_with_two_sources",
                }
            )
            continue
        chosen = candidates[:2]
        while len(chosen) < 2:
            chosen.append(copy.deepcopy(chosen[0]))

        track_cache: dict[str, np.ndarray] = {}
        pair_rows: list[dict[str, Any]] = []
        for pair_index, candidate in enumerate(chosen):
            source_turn_index = int(candidate["turn_index"])
            source_a = str(candidate["source_a"])
            source_b = str(candidate["source_b"])
            base_turn = family["turns"][source_turn_index]
            recipe = recipe_turns[source_turn_index]
            master_gain = float(
                (family.get("render_provenance") or {}).get(
                    "family_master_gain_linear"
                )
            )
            if not math.isfinite(master_gain) or master_gain <= 0.0:
                raise RuntimeError(f"{original_id} has invalid family master gain")

            base_audio, canonical_base_stats = _render_mix(
                recipe, master_gain=master_gain, track_cache=track_cache
            )
            target_stats = base_turn["after"]["signal_stats"]
            base_peak_errors.append(
                abs(canonical_base_stats["peak"] - float(target_stats["peak"]))
            )
            base_rms_errors.append(
                abs(canonical_base_stats["rms"] - float(target_stats["rms"]))
            )
            if base_peak_errors[-1] > 2.0e-5 or base_rms_errors[-1] > 2.0e-5:
                raise RuntimeError(
                    f"base renderer does not reproduce canonical stats: {original_id}"
                )

            swapped_plan, swapped_recipe = _swapped_state(
                base_turn=base_turn,
                recipe=recipe,
                source_a=source_a,
                source_b=source_b,
            )
            anchor_report = _same_anchor_audit(
                base_turn["after"]["scene_plan"], swapped_plan
            )
            anchor_max = max(anchor_max, *anchor_report.values())
            rendered, stats = _render_mix(
                swapped_recipe,
                master_gain=master_gain,
                track_cache=track_cache,
            )
            original_swap_raw_peaks.append(stats["raw_peak"])
            pair_scale = min(
                1.0,
                PAIR_PEAK_CEILING
                / max(
                    PAIR_PEAK_CEILING,
                    canonical_base_stats["raw_peak"],
                    stats["raw_peak"],
                ),
            )
            base_audio_index: int | None = None
            if pair_scale < 1.0:
                effective_master_gain = master_gain * pair_scale
                base_audio, base_stats = _render_mix(
                    recipe,
                    master_gain=effective_master_gain,
                    track_cache=track_cache,
                )
                rendered, stats = _render_mix(
                    swapped_recipe,
                    master_gain=effective_master_gain,
                    track_cache=track_cache,
                )
                base_audio_index = len(swap_audio)
                swap_audio.append(base_audio)
            else:
                base_stats = canonical_base_stats
            relative_rms = float(
                np.sqrt(np.mean(np.square(rendered - base_audio, dtype=np.float64)))
                / max(1.0e-12, base_stats["rms"])
            )
            if relative_rms < args.min_pair_relative_rms:
                raise RuntimeError(
                    f"counterfactual target is too similar for {original_id}: "
                    f"relative_rms={relative_rms:.6f}"
                )
            audio_index = len(swap_audio)
            swap_audio.append(rendered)
            pair_rows.append(
                {
                    "pair_index": pair_index,
                    "source_turn_index": source_turn_index,
                    "source_a": source_a,
                    "source_b": source_b,
                    "candidate_score": float(candidate["score"]),
                    "base_turn": base_turn,
                    "base_plan": base_turn["after"]["scene_plan"],
                    "base_latent": mixture[source_turn_index],
                    "base_audio_index": base_audio_index,
                    "swapped_plan": swapped_plan,
                    "swapped_audio_index": audio_index,
                    "base_stats": base_stats,
                    "swapped_stats": stats,
                    "relative_rms": relative_rms,
                    "pair_master_scale": pair_scale,
                }
            )
            pair_relative_rms.append(relative_rms)
            swap_raw_peaks.append(stats["raw_peak"])
            swap_clip_fractions.append(stats["clipped_fraction"])
            pair_master_scales.append(pair_scale)
        prepared.append(
            {
                "source_family_rank": source_rank,
                "source_family_id": original_id,
                "pairs": pair_rows,
            }
        )

    if not prepared:
        raise RuntimeError("no source-binding counterfactuals were built")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    model, _ = _load_vae(vae_config, vae_checkpoint, device)
    encoded_swaps = _encode_audio(
        model,
        swap_audio,
        device=device,
        batch_size=args.encode_batch_size,
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    records: list[dict[str, Any]] = []
    tensors: dict[str, torch.Tensor] = {}
    for item in prepared:
        family_id = f"{item['source_family_id']}__source_binding_cf"
        turns: list[dict[str, Any]] = []
        latents: list[torch.Tensor] = []
        for pair_index, pair in enumerate(item["pairs"]):
            pair_id = (
                f"turn{pair['source_turn_index']}:"
                f"{pair['source_a']}<->{pair['source_b']}"
            )
            common = {
                "source_family_id": item["source_family_id"],
                "source_family_rank": item["source_family_rank"],
                "source_turn_index": pair["source_turn_index"],
                "pair_index": pair_index,
                "pair_id": pair_id,
                "source_a": pair["source_a"],
                "source_b": pair["source_b"],
                "same_global_anchor": True,
                "pair_relative_rms": pair["relative_rms"],
                "pair_master_scale": pair["pair_master_scale"],
            }
            base_index = 2 * pair_index
            turns.append(
                _curriculum_record(
                    pair["base_turn"],
                    family_id=family_id,
                    turn_index=base_index,
                    plan=pair["base_plan"],
                    curriculum={
                        "kind": "source_binding_base_creation",
                        "role": "base",
                        **common,
                    },
                    signal_stats=pair["base_stats"],
                )
            )
            turns.append(
                _curriculum_record(
                    pair["base_turn"],
                    family_id=family_id,
                    turn_index=base_index + 1,
                    plan=pair["swapped_plan"],
                    curriculum={
                        "kind": "source_binding_counterfactual_creation",
                        "role": "control_swapped",
                        **common,
                    },
                    signal_stats=pair["swapped_stats"],
                )
            )
            base_latent = pair["base_latent"]
            if pair["base_audio_index"] is not None:
                base_latent = encoded_swaps[int(pair["base_audio_index"])]
            latents.extend(
                [base_latent, encoded_swaps[int(pair["swapped_audio_index"])]]
            )
        tensor = torch.stack(latents).to(torch.float16).contiguous()
        if tuple(tensor.shape) != (4, 64, LATENT_FRAMES):
            raise RuntimeError(f"invalid paired latent tensor: {tensor.shape}")
        records.append(
            {
                "schema": SCHEMA,
                "schema_version": SCHEMA_VERSION,
                "family_id": family_id,
                "family_rank": len(records),
                "split": "train",
                "curriculum_kind": "source_binding_counterfactual_pair",
                "source_family_id": item["source_family_id"],
                "turns": turns,
            }
        )
        tensors[family_id] = tensor

    audit = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": "BUILDING",
        "contract": (
            "Fixed 442368-sample/432-frame paired creation targets. Semantic "
            "content, dry sources, room, and global anchor are identical within "
            "each pair; complete activity/motion/gain ownership is swapped. A "
            "common per-pair master scale caps both targets at 0.95 peak without "
            "changing relative source gains."
        ),
        "source_family_store": str(family_store),
        "source_caption_overlay": str(caption_overlay),
        "requested_source_families": args.families,
        "included_source_families": len(prepared),
        "excluded_source_families": excluded,
        "matched_pairs": len(pair_relative_rms),
        "vae_encoded_states": len(swap_audio),
        "states": 4 * len(records),
        "window_samples": WINDOW_SAMPLES,
        "latent_frames": LATENT_FRAMES,
        "global_anchor_max_abs": anchor_max,
        "global_anchor_numerical_tolerance": ANCHOR_NUMERICAL_TOLERANCE,
        "pair_relative_rms_min": min(pair_relative_rms),
        "pair_relative_rms_max": max(pair_relative_rms),
        "pair_relative_rms_mean": sum(pair_relative_rms) / len(pair_relative_rms),
        "swap_raw_peak_max": max(swap_raw_peaks),
        "original_swap_raw_peak_max": max(original_swap_raw_peaks),
        "swap_clipped_fraction_max": max(swap_clip_fractions),
        "pair_peak_ceiling": PAIR_PEAK_CEILING,
        "pair_master_scale_min": min(pair_master_scales),
        "normalized_pairs": sum(scale < 1.0 for scale in pair_master_scales),
        "base_peak_abs_error_max": max(base_peak_errors),
        "base_rms_abs_error_max": max(base_rms_errors),
        "vae_config": str(vae_config),
        "vae_checkpoint": str(vae_checkpoint),
        "vae_checkpoint_sha256": EXPECTED_VAE_SHA256,
        "vae_rng_seed": args.seed,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.", suffix=".building", dir=output.parent
        )
    )
    try:
        _write_store(
            temporary,
            tensors=tensors,
            records=records,
            audit=audit,
            schema=SCHEMA,
            schema_version=SCHEMA_VERSION,
        )
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    report = json.loads((output / "AUDIT.json").read_text(encoding="utf-8"))
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--family-store",
        type=Path,
        default=Path(os.environ.get("AMBIT_DATA_ROOT", "data") + "/spatial_cot_v1/latents/train"),
    )
    parser.add_argument(
        "--caption-overlay",
        type=Path,
        default=Path(
            os.environ.get("AMBIT_DATA_ROOT", "data") + "/spatial_cot_v1/captions/"
            "spatial_source_regions_v3/train"
        ),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--families", type=int, default=48)
    parser.add_argument(
        "--vae-config",
        type=Path,
        default=REPO_ROOT
        / "stable_audio_tools/configs/model_configs/autoencoders/"
        "stable_audio_4ch_vae_ds1024_z64_wdmix_scm.json",
    )
    parser.add_argument(
        "--vae-checkpoint",
        type=Path,
        default=Path(
            os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/compareVAE_ckpt/unwrapped_wdmix_1350000.ckpt"
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--encode-batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--min-pair-relative-rms", type=float, default=0.25)
    args = parser.parse_args()
    if args.families <= 0 or args.encode_batch_size <= 0:
        parser.error("--families and --encode-batch-size must be positive")
    if not 0.0 < args.min_pair_relative_rms < 10.0:
        parser.error("--min-pair-relative-rms must lie in (0, 10)")
    build(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
