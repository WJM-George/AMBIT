#!/usr/bin/env python3
"""Audit source semantics before and after the frozen FOA VAE.

The renderer can only learn source identity that survives its target codec.  This
diagnostic reconstructs each turn-0 source component from the authoritative edit
recipe, decodes the corresponding source-curriculum latent, and compares the two
active-window signals in CLAP space.  It deliberately scores exact 10-second
repetitions of the planned activity window so classifier padding or random crops
cannot turn source duration into a semantic score.

The output is diagnostic evidence, not a training objective.  Caption scores are
reported alongside caption-independent post-to-pre audio assignment, and every
family fails closed when its pre-VAE sources are not themselves distinguishable.
"""
from __future__ import annotations
import os

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torchaudio
from safetensors import safe_open

from dataset.synthesis.render_spatial_edit_families import (
    _flac_pcm24_memory_roundtrip,
)
from scripts.t2a.data.build_spatial_cot_source_curriculum import (
    _load_recipe_families,
    _pcm24_track,
)
from scripts.t2a.data.preencode_spatial_cot_family_shard import _load_vae
from stable_audio_tools.data.spatial_edit_recipe import source_render_signature
from stable_audio_tools.data.spatial_family_dataset import SpatialFamilyDataset
from stable_audio_tools.training.metrics.fad_metrics import load_clap_model


DEFAULT_FAMILY_STORE = Path(os.environ.get("AMBIT_DATA_ROOT", "data") + "/spatial_cot_v1/latents/train")
DEFAULT_CAPTION_OVERLAY = Path(
    os.environ.get("AMBIT_DATA_ROOT", "data") + "/spatial_cot_v1/captions/spatial_source_regions_v3/train"
)
DEFAULT_SOURCE_CURRICULUM = Path(
    os.environ.get("AMBIT_DATA_ROOT", "data") + "/spatial_cot_v1/curriculum/"
    "source_identifiability_fixed48_v2"
)
DEFAULT_VAE_CONFIG = REPO_ROOT / (
    "stable_audio_tools/configs/model_configs/autoencoders/"
    "stable_audio_4ch_vae_ds1024_z64_wdmix_scm.json"
)
DEFAULT_VAE_CHECKPOINT = Path(
    os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/compareVAE_ckpt/unwrapped_wdmix_1350000.ckpt"
)


def _cosine(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left = torch.nn.functional.normalize(left.float(), dim=-1)
    right = torch.nn.functional.normalize(right.float(), dim=-1)
    return left @ right.transpose(0, 1)


def _source_by_id(plan: Mapping[str, Any], source_id: str) -> dict[str, Any]:
    sources = ((plan.get("scene") or {}).get("sources") or [])
    matches = [source for source in sources if str(source.get("source_id")) == source_id]
    if len(matches) != 1:
        raise RuntimeError(f"expected one {source_id}, found {len(matches)}")
    return copy.deepcopy(matches[0])


def _active_clap_waveform(
    audio: torch.Tensor,
    *,
    sample_rate: int,
    onset_sec: float,
    offset_sec: float,
    device: torch.device,
) -> torch.Tensor:
    """Return a deterministic, peak-normalized 10-second mono CLAP input."""

    audio = torch.as_tensor(audio, dtype=torch.float32).reshape(-1)
    start = max(0, round(float(onset_sec) * sample_rate))
    stop = min(audio.numel(), round(float(offset_sec) * sample_rate))
    if stop <= start:
        raise ValueError(f"empty source activity window: {onset_sec}..{offset_sec}")
    active = audio[start:stop].to(device)
    peak = active.abs().amax()
    if not bool(torch.isfinite(active).all()) or float(peak) <= 1.0e-8:
        raise RuntimeError("source activity window is silent or non-finite")
    active = active / peak * (10.0 ** (-1.0 / 20.0))
    target_samples = 10 * int(sample_rate)
    active = active.repeat(math.ceil(target_samples / active.numel()))[:target_samples]
    if sample_rate != 48_000:
        active = torchaudio.functional.resample(active[None], sample_rate, 48_000)[0]
    if active.numel() != 480_000:
        raise RuntimeError(f"CLAP input length changed: {active.numel()}")
    return active.clamp(-1.0, 1.0)


def _text_embeddings(clap_model, labels: Sequence[str]) -> torch.Tensor:
    values = list(labels)
    if not values:
        raise ValueError("at least one source label is required")
    requested = values if len(values) > 1 else values * 2
    embeddings = clap_model.get_text_embedding(requested, use_tensor=True).float()
    return embeddings[: len(values)]


def _load_curriculum_records(root: Path) -> dict[int, list[dict[str, Any]]]:
    metadata = root / "metadata" / "families-00000.jsonl"
    by_rank: dict[int, list[dict[str, Any]]] = {}
    with metadata.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("curriculum_kind") != "isolated_source_creation":
                continue
            turns = []
            for turn in record["turns"]:
                curriculum = turn.get("curriculum") or {}
                if int(curriculum.get("source_turn_index", -1)) == 0:
                    turns.append(turn)
            if not turns:
                continue
            rank = int(turns[0]["curriculum"]["source_family_rank"])
            by_rank.setdefault(rank, []).append(
                {"family_id": str(record["family_id"]), "turns": turns}
            )
    return by_rank


def _recipe_source(
    recipe: Mapping[str, Any],
    *,
    source_id: str,
    track_id: str,
    gain_db: float,
) -> Mapping[str, Any]:
    matches = []
    for source in recipe["sources"]:
        if str(source.get("source_id")) != source_id:
            continue
        if source_render_signature(recipe, source) != track_id:
            continue
        if not math.isclose(float(source.get("gain_db", 0.0)), gain_db, abs_tol=1.0e-6):
            continue
        matches.append(source)
    if len(matches) != 1:
        raise RuntimeError(
            f"recipe source resolution changed for {source_id}/{track_id}: {len(matches)}"
        )
    return matches[0]


def _assignment_rows(
    post_to_pre: torch.Tensor,
    pre_to_pre: torch.Tensor,
    *,
    source_ids: Sequence[str],
    min_pre_gap: float,
) -> list[dict[str, Any]]:
    rows = []
    for index, source_id in enumerate(source_ids):
        other = [column for column in range(len(source_ids)) if column != index]
        pre_gap = float(pre_to_pre[index, index]) - max(
            (float(pre_to_pre[index, column]) for column in other), default=-1.0
        )
        post_gap = float(post_to_pre[index, index]) - max(
            (float(post_to_pre[index, column]) for column in other), default=-1.0
        )
        predicted = int(torch.argmax(post_to_pre[index]))
        rows.append(
            {
                "source_id": source_id,
                "pre_discriminability_gap": pre_gap,
                "pre_valid": pre_gap >= min_pre_gap,
                "post_own_similarity": float(post_to_pre[index, index]),
                "post_assignment_margin": post_gap,
                "post_assignment": source_ids[predicted],
                "post_correct": predicted == index,
            }
        )
    return rows


def audit(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    ranks = sorted(set(int(rank) for rank in args.family_ranks))
    if not ranks or min(ranks) < 0:
        raise ValueError("--family-ranks must contain non-negative ranks")

    family_store = args.family_store.expanduser().resolve()
    caption_overlay = args.caption_overlay.expanduser().resolve()
    curriculum_root = args.source_curriculum.expanduser().resolve()
    source_dataset = SpatialFamilyDataset(
        [{"path": family_store, "caption_overlay_path": caption_overlay}],
        require_ready=True,
    )
    if max(ranks) >= len(source_dataset):
        raise ValueError("family rank is outside the canonical source store")
    families = []
    for rank in ranks:
        _, info = source_dataset[rank]
        family = copy.deepcopy(info["spatial_family"])
        if int(family["family_rank"]) != rank:
            raise RuntimeError(f"family rank changed at {rank}")
        families.append(family)
    recipes = _load_recipe_families(families)
    curriculum = _load_curriculum_records(curriculum_root)

    vae, vae_config = _load_vae(
        args.vae_config.expanduser().resolve(),
        args.vae_checkpoint.expanduser().resolve(),
        device,
    )
    clap_model = load_clap_model(args.clap_model, device=str(device))
    tensor_path = curriculum_root / "shards" / "families-00000.safetensors"

    family_inputs: list[dict[str, Any]] = []
    with safe_open(str(tensor_path), framework="pt", device="cpu") as tensors:
        for family in families:
            rank = int(family["family_rank"])
            recipe_family = recipes[str(family["family_id"])]
            recipe = recipe_family["recipes"][0]
            master_gain = float(
                (family.get("render_provenance") or {}).get(
                    "family_master_gain_linear", 1.0
                )
            )
            if not math.isfinite(master_gain) or master_gain <= 0.0:
                raise RuntimeError(f"family {rank} has invalid master gain")
            rows = []
            seen_source_ids: dict[str, str] = {}
            for group in curriculum.get(rank, []):
                latents = tensors.get_tensor(group["family_id"])
                for turn in group["turns"]:
                    curriculum_info = turn["curriculum"]
                    source_id = str(curriculum_info["source_id"])
                    execution_key = str(curriculum_info["execution_key"])
                    previous_execution = seen_source_ids.get(source_id)
                    if previous_execution is not None:
                        # The groups-of-four writer pads its final group with
                        # existing execution rows.  Exact duplicates are not
                        # additional semantic sources; a different execution
                        # under the same turn-0 slot would be an ambiguity.
                        if previous_execution != execution_key:
                            raise RuntimeError(
                                f"ambiguous turn-0 source {rank}/{source_id}: "
                                f"{previous_execution} != {execution_key}"
                            )
                        continue
                    seen_source_ids[source_id] = execution_key
                    latent_index = int(turn["after"]["latent_state_index"])
                    source_plan = _source_by_id(turn["after"]["scene_plan"], source_id)
                    activity = source_plan.get("activity") or {}
                    onset = float(activity["onset_sec"])
                    offset = float(activity["offset_sec"])
                    track_id = str(curriculum_info["track_id"])
                    gain_db = float(
                        execution_key.rsplit("gain=", 1)[1]
                    )
                    source_recipe = _recipe_source(
                        recipe,
                        source_id=source_id,
                        track_id=track_id,
                        gain_db=gain_db,
                    )
                    track = _pcm24_track(recipe, source_recipe)
                    component = (
                        track * (10.0 ** (gain_db / 20.0)) * master_gain
                    ).astype("float32", copy=False)
                    component = _flac_pcm24_memory_roundtrip(
                        component.clip(-1.0, 1.0), int(recipe["audio"]["sample_rate"])
                    )
                    latent = latents[latent_index].float().to(device)
                    with torch.inference_mode():
                        decoded = vae.decode(latent[None])[0].float().cpu()
                    label = str((source_plan.get("event") or {}).get("label") or source_id)
                    pre = _active_clap_waveform(
                        torch.from_numpy(component[0]) * math.sqrt(2.0),
                        sample_rate=int(recipe["audio"]["sample_rate"]),
                        onset_sec=onset,
                        offset_sec=offset,
                        device=device,
                    )
                    post = _active_clap_waveform(
                        decoded[0] * math.sqrt(2.0),
                        sample_rate=int(vae_config["sample_rate"]),
                        onset_sec=onset,
                        offset_sec=offset,
                        device=device,
                    )
                    rows.append(
                        {
                            "source_id": source_id,
                            "label": label,
                            "track_id": track_id,
                            "gain_db": gain_db,
                            "onset_sec": onset,
                            "offset_sec": offset,
                            "pre": pre,
                            "post": post,
                        }
                    )
            expected_ids = {
                str(source["source_id"])
                for source in recipe["scene_plan"]["scene"]["sources"]
            }
            if set(seen_source_ids) != expected_ids:
                raise RuntimeError(
                    f"turn-0 source set changed for family {rank}: "
                    f"{sorted(seen_source_ids)} != {sorted(expected_ids)}"
                )
            rows.sort(key=lambda row: row["source_id"])
            family_inputs.append(
                {
                    "family_rank": rank,
                    "family_id": str(family["family_id"]),
                    "rows": rows,
                }
            )

    results = []
    for family in family_inputs:
        rows = family["rows"]
        labels = [row["label"] for row in rows]
        with torch.inference_mode():
            pre_embeddings = torch.cat(
                [
                    clap_model.get_audio_embedding_from_data(
                        x=row["pre"][None], use_tensor=True
                    ).float()
                    for row in rows
                ],
                dim=0,
            )
            post_embeddings = torch.cat(
                [
                    clap_model.get_audio_embedding_from_data(
                        x=row["post"][None], use_tensor=True
                    ).float()
                    for row in rows
                ],
                dim=0,
            )
            text_embeddings = _text_embeddings(clap_model, labels)
        pre_to_text = _cosine(pre_embeddings, text_embeddings).cpu()
        post_to_text = _cosine(post_embeddings, text_embeddings).cpu()
        post_to_pre = _cosine(post_embeddings, pre_embeddings).cpu()
        pre_to_pre = _cosine(pre_embeddings, pre_embeddings).cpu()
        source_ids = [row["source_id"] for row in rows]
        assignments = _assignment_rows(
            post_to_pre,
            pre_to_pre,
            source_ids=source_ids,
            min_pre_gap=args.min_pre_audio_gap,
        )
        source_results = []
        caption_valid = []
        for index, row in enumerate(rows):
            other = [column for column in range(len(rows)) if column != index]
            pre_caption_gap = float(pre_to_text[index, index]) - max(
                (float(pre_to_text[index, column]) for column in other), default=-1.0
            )
            post_caption_gap = float(post_to_text[index, index]) - max(
                (float(post_to_text[index, column]) for column in other), default=-1.0
            )
            valid = pre_caption_gap >= args.min_pre_caption_gap
            if valid:
                caption_valid.append(int(torch.argmax(post_to_text[index])) == index)
            source_results.append(
                {
                    **{key: row[key] for key in (
                        "source_id",
                        "label",
                        "track_id",
                        "gain_db",
                        "onset_sec",
                        "offset_sec",
                    )},
                    "pre_caption_similarity": float(pre_to_text[index, index]),
                    "post_caption_similarity": float(post_to_text[index, index]),
                    "caption_similarity_delta": float(
                        post_to_text[index, index] - pre_to_text[index, index]
                    ),
                    "pre_caption_discriminability_gap": pre_caption_gap,
                    "pre_caption_valid": valid,
                    "post_caption_assignment_margin": post_caption_gap,
                    "post_caption_assignment": source_ids[
                        int(torch.argmax(post_to_text[index]))
                    ],
                    "post_to_pre_own_similarity": float(post_to_pre[index, index]),
                }
            )
        valid_audio = [row for row in assignments if row["pre_valid"]]
        family_status = "PASS" if (
            len(valid_audio) == len(rows)
            and all(row["post_correct"] for row in valid_audio)
            and len(caption_valid) == len(rows)
            and all(caption_valid)
        ) else "BLOCK"
        results.append(
            {
                "family_rank": family["family_rank"],
                "family_id": family["family_id"],
                "status": family_status,
                "source_count": len(rows),
                "audio_assignment": assignments,
                "sources": source_results,
            }
        )

    return {
        "schema": "stable_audio_tools.vae_source_semantic_audit",
        "schema_version": 1,
        "family_ranks": ranks,
        "family_count": len(results),
        "pass_count": sum(row["status"] == "PASS" for row in results),
        "block_count": sum(row["status"] != "PASS" for row in results),
        "min_pre_audio_gap": float(args.min_pre_audio_gap),
        "min_pre_caption_gap": float(args.min_pre_caption_gap),
        "clap_model": args.clap_model,
        "vae_config": str(args.vae_config.expanduser().resolve()),
        "vae_checkpoint": str(args.vae_checkpoint.expanduser().resolve()),
        "activity_window_policy": "planned activity cropped then repeated to exactly 10 seconds",
        "families": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family-ranks", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--family-store", type=Path, default=DEFAULT_FAMILY_STORE)
    parser.add_argument("--caption-overlay", type=Path, default=DEFAULT_CAPTION_OVERLAY)
    parser.add_argument("--source-curriculum", type=Path, default=DEFAULT_SOURCE_CURRICULUM)
    parser.add_argument("--vae-config", type=Path, default=DEFAULT_VAE_CONFIG)
    parser.add_argument("--vae-checkpoint", type=Path, default=DEFAULT_VAE_CHECKPOINT)
    parser.add_argument("--clap-model", default="630k-audioset-fusion-best.pt")
    parser.add_argument("--min-pre-audio-gap", type=float, default=0.05)
    parser.add_argument("--min-pre-caption-gap", type=float, default=0.01)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = audit(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "family_count": report["family_count"],
                "pass_count": report["pass_count"],
                "block_count": report["block_count"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
