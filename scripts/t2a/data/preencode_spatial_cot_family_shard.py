#!/usr/bin/env python3
"""QC and VAE-preencode one rendered Spatial-CoT work shard.

One output tensor is stored per conversation family with shape ``[4,64,432]``.
The work shard is atomic and resumable; train renders may only be deleted after
the safetensors and metadata index have been reopened and fully verified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np
import soundfile as sf
import torch
from safetensors import safe_open
from safetensors.torch import save_file

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.t2a.data.finalize_spatial_edit_conversations import _turn_row  # noqa: E402
from stable_audio_tools.configuration import load_config  # noqa: E402
from stable_audio_tools.data.spatial_edit_recipe import (  # noqa: E402
    STATE_RENDER_INPUT,
    source_render_signature,
    validate_edit_family,
)
from stable_audio_tools.data.spatial_story import diff_scene_plans  # noqa: E402
from stable_audio_tools.data.t2a_artifacts import atomic_write_json, atomic_write_jsonl  # noqa: E402
from stable_audio_tools.models.factory import create_model_from_config  # noqa: E402
from stable_audio_tools.models.utils import copy_state_dict, load_ckpt_state_dict  # noqa: E402


SCHEMA = "stable_audio_tools.spatial_family_latent_shard"
VERSION = 1
RENDER_MARKER = ".spatial_cot_render_root.json"
RENDER_READY_MARKER = "READY"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _family_manifests(root: Path) -> Iterator[Path]:
    yield from sorted(root.glob("families/*/family.json"))


def _validate_diff_and_tracks(family: Mapping[str, Any]) -> None:
    recipes = family["recipes"]
    for index, recipe in enumerate(recipes):
        outputs = recipe.get("outputs") or {}
        if (
            outputs.get("state_input") != STATE_RENDER_INPUT
            or outputs.get("uses_previous_foa") is not False
        ):
            raise RuntimeError(
                f"turn was not independently rendered from dry sources: "
                f"{family['family_id']} turn={index}"
            )
        if index == 0:
            expected = diff_scene_plans(None, recipe["scene_plan"])
        else:
            expected = diff_scene_plans(
                recipes[index - 1]["scene_plan"], recipe["scene_plan"]
            )
        if expected != family["turns"][index]["diff"]:
            raise RuntimeError(
                f"diff scope mismatch: {family['family_id']} turn={index}"
            )
        if index == 0:
            continue
        before = recipes[index - 1]
        before_ids = {source["source_id"]: source for source in before["sources"]}
        after_ids = {source["source_id"]: source for source in recipe["sources"]}
        before_refs = {
            row["source_id"]: row["track_id"]
            for row in before["outputs"]["source_track_refs"]
        }
        after_refs = {
            row["source_id"]: row["track_id"]
            for row in recipe["outputs"]["source_track_refs"]
        }
        changed = {
            str(row["source_id"])
            for row in (family["turns"][index]["diff"].get("changed") or [])
        }
        for source_id in set(before_ids) & set(after_ids):
            if source_id not in changed and before_refs[source_id] != after_refs[source_id]:
                raise RuntimeError(
                    f"unchanged track was rerendered: {family['family_id']} "
                    f"turn={index} source={source_id}"
                )
        if recipe["edit"]["type"] == "change_gain":
            source_id = str(recipe["edit"]["target_source_id"])
            if before_refs[source_id] != after_refs[source_id]:
                raise RuntimeError(
                    f"gain-only edit changed pre-mix track: {family['family_id']}"
                )
        for source in recipe["sources"]:
            source_id = str(source["source_id"])
            if after_refs[source_id] != source_render_signature(recipe, source):
                raise RuntimeError(
                    f"source-track signature mismatch: {family['family_id']} {source_id}"
                )


def _load_family_audio(
    family: Mapping[str, Any],
    *,
    sample_rate: int,
    num_samples: int,
    minimum_rms: float,
    minimum_active_100ms_fraction: float,
    active_frame_rms_threshold: float,
    max_peak: float,
    verify_checksums: bool,
) -> torch.Tensor:
    states = []
    master_gains = set()
    for recipe in family["recipes"]:
        path = Path(recipe["outputs"]["foa_path"])
        audio, rate = sf.read(str(path), always_2d=True, dtype="float32")
        if int(rate) != sample_rate or audio.shape != (num_samples, 4):
            raise RuntimeError(
                f"invalid FOA {path}: rate={rate} shape={audio.shape}; "
                f"expected {sample_rate} {(num_samples, 4)}"
            )
        if not np.isfinite(audio).all():
            raise RuntimeError(f"non-finite FOA: {path}")
        peak = float(np.max(np.abs(audio)))
        rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
        block = max(1, sample_rate // 10)
        active_fraction = float(
            np.mean(
                [
                    np.sqrt(
                        np.mean(
                            np.square(audio[start : start + block], dtype=np.float64)
                        )
                    )
                    >= active_frame_rms_threshold
                    for start in range(0, len(audio), block)
                ]
            )
        )
        if (
            peak > max_peak + 1e-6
            or rms < minimum_rms
            or active_fraction < minimum_active_100ms_fraction
        ):
            raise RuntimeError(
                f"FOA QC failed: {path} peak={peak} rms={rms} "
                f"active_100ms_fraction={active_fraction}"
            )
        expected_hash = recipe["outputs"].get("foa_sha256")
        if verify_checksums and expected_hash and _sha256(path) != expected_hash:
            raise RuntimeError(f"FOA checksum mismatch: {path}")
        master_gains.add(round(float(recipe["mix"]["family_master_gain_linear"]), 12))
        states.append(torch.from_numpy(audio.T.copy()))
    if len(master_gains) != 1:
        raise RuntimeError(f"family does not share one master gain: {family['family_id']}")
    return torch.stack(states, dim=0)


def _atomic_safetensors(
    path: Path, tensors: Mapping[str, torch.Tensor], metadata: Mapping[str, str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        save_file(dict(tensors), str(temporary), metadata=dict(metadata))
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_metadata(path: Path, records: list[Mapping[str, Any]]) -> list[tuple[int, int]]:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    offsets = []
    try:
        with os.fdopen(fd, "wb") as handle:
            for record in records:
                payload = json.dumps(
                    record, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                ).encode("utf-8")
                offset = handle.tell()
                handle.write(payload + b"\n")
                offsets.append((offset, len(payload)))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)
    return offsets


def _load_vae(config_path: Path, checkpoint: Path, device: torch.device):
    config = load_config(config_path)
    model = create_model_from_config(config)
    copy_state_dict(model, load_ckpt_state_dict(str(checkpoint)))
    return model.eval().requires_grad_(False).to(device), config


def encode_work_shard(
    *,
    spec: Mapping[str, Any],
    render_root: Path,
    recipe_root: Path,
    output_root: Path,
    split: str,
    work_shard: int,
    model,
    vae_config: Mapping[str, Any],
    vae_config_path: Path,
    vae_checkpoint: Path,
    device: torch.device,
    family_batch_size: int,
    cleanup_rendered: bool,
    verify_foa_checksums: bool,
) -> dict[str, Any]:
    """Encode one shard with an already-loaded VAE.

    Keeping model ownership outside this function lets an eight-GPU worker
    pool process thousands of work shards without reloading 595 MiB of weights
    for every 256 families.
    """

    if split not in {"train", "validation", "test"}:
        raise ValueError(f"unknown split: {split}")
    if work_shard < 0 or family_batch_size <= 0:
        raise ValueError("work-shard must be non-negative and family-batch-size positive")
    audio_spec = spec["audio"]
    qc = spec["quality_control"]
    render_root = render_root.expanduser().resolve()
    recipe_root = recipe_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    vae_config_path = vae_config_path.expanduser().resolve()
    vae_checkpoint = vae_checkpoint.expanduser().resolve()
    done = output_root / "work_done" / f"work-{work_shard:05d}.json"
    if done.is_file():
        print(f"[spatial-cot-preencode] cached DONE {done}")
        return json.loads(done.read_text(encoding="utf-8"))
    if not (render_root / RENDER_MARKER).is_file():
        raise SystemExit(f"unrecognized render root: {render_root}")
    if not (render_root / RENDER_READY_MARKER).is_file():
        raise SystemExit(f"render shard is not READY: {render_root}")
    if not (recipe_root / "READY").is_file():
        raise SystemExit(f"recipe shard is not READY: {recipe_root}")
    recipe_details = json.loads(
        (recipe_root / "details.json").read_text(encoding="utf-8")
    )
    recipe_jsonl_sha256 = str(recipe_details.get("recipe_sha256") or "")
    if not recipe_jsonl_sha256:
        raise RuntimeError(f"recipe shard has no persistent checksum: {recipe_root}")
    if cleanup_rendered and bool(spec["splits"][split]["retain_rendered_foa"]):
        raise ValueError(f"refusing cleanup for retained split {split}")
    manifests = list(_family_manifests(render_root))
    if not manifests:
        raise SystemExit(f"no rendered family manifests under {render_root}")

    expected_families = json.loads(
        (recipe_root / "READY").read_text(encoding="utf-8")
    )["families"]
    render_ready = json.loads(
        (render_root / RENDER_READY_MARKER).read_text(encoding="utf-8")
    )
    if int(render_ready.get("families", -1)) != int(expected_families):
        raise SystemExit(
            f"render READY family count {render_ready.get('families')} "
            f"!= recipes {expected_families}"
        )
    expected_profile = str(
        spec["storage"][
            "train_render_profile" if split == "train" else "eval_render_profile"
        ]
    )
    actual_profile = str(render_ready.get("storage_profile") or "retained_flac_pcm24")
    if actual_profile != expected_profile:
        raise SystemExit(
            f"render storage profile {actual_profile} != expected {expected_profile}"
        )
    if len(manifests) != int(expected_families):
        raise SystemExit(
            f"rendered family count {len(manifests)} != recipes {expected_families}"
        )
    torch.manual_seed(int(spec["seed"]) + int(work_shard))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(spec["seed"]) + int(work_shard))

    tensors: dict[str, torch.Tensor] = {}
    metadata_records: list[dict[str, Any]] = []
    pending_audio: list[torch.Tensor] = []
    pending_families: list[dict[str, Any]] = []

    def encode_pending() -> None:
        if not pending_families:
            return
        batch = torch.cat(pending_audio, dim=0).to(device, non_blocking=True)
        with torch.inference_mode():
            latents = model.encode(batch)
        if latents.ndim != 3 or tuple(latents.shape[1:]) != (
            int(audio_spec["latent_channels"]),
            int(audio_spec["latent_frames"]),
        ):
            raise RuntimeError(f"unexpected VAE latent shape: {tuple(latents.shape)}")
        latents = latents.float().cpu()
        cursor = 0
        for family in pending_families:
            count = len(family["turns"])
            family_latent = latents[cursor : cursor + count].to(torch.float16).contiguous()
            cursor += count
            family_id = str(family["family_id"])
            if family_id in tensors:
                raise RuntimeError(f"duplicate family id: {family_id}")
            tensors[family_id] = family_latent
            turns = []
            for index in range(count):
                row = _turn_row(family, index)
                recipe = family["recipes"][index]
                previous = family["recipes"][index - 1] if index else None
                row["schema_version"] = "1.1"
                row["audio_path"] = (
                    row.get("audio_path") if not cleanup_rendered else None
                )
                row["after"]["latent_state_index"] = index
                row["after"]["foa_sha256"] = recipe["outputs"].get(
                    "foa_sha256"
                )
                row["after"]["rendered_foa_retained"] = not cleanup_rendered
                row["before"]["latent_state_index"] = index - 1 if index else None
                row["before"]["foa_sha256"] = (
                    previous["outputs"].get("foa_sha256") if previous else None
                )
                row["before"]["rendered_foa_retained"] = bool(
                    previous and not cleanup_rendered
                )
                if cleanup_rendered:
                    row["after"]["audio_path"] = None
                    row["before"]["audio_path"] = None
                    for reference in row["after"].get("source_track_refs") or []:
                        reference["path"] = None
                        reference["retained"] = False
                else:
                    for reference in row["after"].get("source_track_refs") or []:
                        reference["retained"] = True
                turns.append(row)
            recipe_shard = (
                recipe_root
                / "shards"
                / f"recipes-{split}-{work_shard:05d}.jsonl"
            )
            render_summary = family.get("render_summary") or {}
            metadata_records.append(
                {
                    "schema": "stable_audio_tools.spatial_family_training_record",
                    "schema_version": 2,
                    "family_id": family_id,
                    "split": family.get("split") or split,
                    "family_rank": int(family["family_rank"]),
                    "work_shard": int(family["work_shard"]),
                    "source_lineage": family.get("source_lineage"),
                    "recipe_store_ref": str(recipe_root),
                    "recipe_jsonl_ref": str(recipe_shard),
                    "recipe_jsonl_sha256": recipe_jsonl_sha256,
                    "recipe_family_id": family_id,
                    "render_provenance": {
                        "recipe_schema": family.get("schema"),
                        "recipe_schema_version": family.get("schema_version"),
                        "renderer": family["recipes"][0].get("renderer"),
                        "source_loudness": family["recipes"][0]
                        .get("render_contract", {})
                        .get("source_loudness"),
                        "family_peak_before_master": render_summary.get(
                            "family_peak_before_master"
                        ),
                        "family_master_gain_linear": render_summary.get(
                            "family_master_gain_linear"
                        ),
                        "peak_target": render_summary.get("peak_target"),
                        "unique_source_tracks": render_summary.get(
                            "unique_source_tracks"
                        ),
                        "state_input": render_summary.get("state_input"),
                        "uses_previous_foa": render_summary.get(
                            "uses_previous_foa"
                        ),
                        "independent_state_mix": render_summary.get(
                            "independent_state_mix"
                        ),
                        "storage_profile": render_summary.get(
                            "storage_profile"
                        ),
                        "foa_container": render_summary.get("foa_container"),
                        "foa_subtype": render_summary.get("foa_subtype"),
                        "minimum_rms": render_summary.get("minimum_rms"),
                        "minimum_active_100ms_fraction": render_summary.get(
                            "minimum_active_100ms_fraction"
                        ),
                        "active_frame_rms_threshold": render_summary.get(
                            "active_frame_rms_threshold"
                        ),
                        "state_rms_min": render_summary.get("state_rms_min"),
                        "state_active_100ms_fraction_min": render_summary.get(
                            "state_active_100ms_fraction_min"
                        ),
                        "rendered_foa_retained": not cleanup_rendered,
                        "source_tracks_retained": not cleanup_rendered,
                    },
                    "turns": turns,
                }
            )
        pending_audio.clear()
        pending_families.clear()

    for manifest in manifests:
        family = json.loads(manifest.read_text(encoding="utf-8"))
        validate_edit_family(family, require_outputs=True)
        if family.get("split") != split or int(family.get("work_shard", -1)) != work_shard:
            raise RuntimeError(f"render family is in the wrong shard: {family['family_id']}")
        _validate_diff_and_tracks(family)
        audio = _load_family_audio(
            family,
            sample_rate=int(audio_spec["sample_rate"]),
            num_samples=int(audio_spec["num_samples"]),
            minimum_rms=float(qc["minimum_rms"]),
            minimum_active_100ms_fraction=float(
                qc["minimum_active_100ms_fraction"]
            ),
            active_frame_rms_threshold=float(qc["active_frame_rms_threshold"]),
            max_peak=float(qc["max_abs_peak"]),
            verify_checksums=bool(verify_foa_checksums),
        )
        pending_audio.append(audio)
        pending_families.append(family)
        if len(pending_families) >= family_batch_size:
            encode_pending()
    encode_pending()
    metadata_records.sort(key=lambda row: int(row["family_rank"]))

    stem = f"families-{work_shard:05d}"
    tensor_path = output_root / "shards" / f"{stem}.safetensors"
    metadata_path = output_root / "metadata" / f"{stem}.jsonl"
    index_path = output_root / "shard_indexes" / f"{stem}.jsonl"
    _atomic_safetensors(
        tensor_path,
        tensors,
        {
            "schema": SCHEMA,
            "schema_version": str(VERSION),
            "split": split,
            "work_shard": str(work_shard),
            "vae_checkpoint": str(vae_checkpoint),
        },
    )
    offsets = _atomic_metadata(metadata_path, metadata_records)
    index_records = []
    for record, (offset, length) in zip(metadata_records, offsets):
        tensor = tensors[record["family_id"]]
        index_records.append(
            {
                "family_rank": int(record["family_rank"]),
                "family_id": record["family_id"],
                "split": split,
                "work_shard": work_shard,
                "tensor_shard": tensor_path.relative_to(output_root).as_posix(),
                "tensor_key": record["family_id"],
                "metadata_shard": metadata_path.relative_to(output_root).as_posix(),
                "metadata_offset": offset,
                "metadata_length": length,
                "num_turns": int(tensor.shape[0]),
                "channels": int(tensor.shape[1]),
                "frames": int(tensor.shape[2]),
                "dtype": "float16",
            }
        )
    atomic_write_jsonl(index_path, index_records)

    # Reopen every entry before publishing DONE.  A partial/corrupt shard is
    # therefore never eligible for finalization or render cleanup.
    with safe_open(str(tensor_path), framework="pt", device="cpu") as handle:
        if set(handle.keys()) != set(tensors):
            raise RuntimeError("written safetensors key set mismatch")
        for row in index_records:
            value = handle.get_tensor(row["tensor_key"])
            if tuple(value.shape) != (
                row["num_turns"], row["channels"], row["frames"]
            ) or not torch.isfinite(value).all():
                raise RuntimeError(f"written latent QC failed: {row['family_id']}")
    done.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        done,
        {
            "schema": SCHEMA,
            "schema_version": VERSION,
            "split": split,
            "work_shard": work_shard,
            "families": len(index_records),
            "states": sum(row["num_turns"] for row in index_records),
            "tensor_shard": str(tensor_path),
            "tensor_sha256": _sha256(tensor_path),
            "metadata_sha256": _sha256(metadata_path),
            "index_sha256": _sha256(index_path),
            "vae_config": str(vae_config_path),
            "vae_checkpoint": str(vae_checkpoint),
            "vae_model_config": dict(vae_config),
            "vae_rng_seed": int(spec["seed"]) + int(work_shard),
        },
    )
    if cleanup_rendered:
        shutil.rmtree(render_root)
        try:
            render_root.parent.rmdir()
        except OSError:
            # Another transient child or a concurrent cleanup may still own
            # the parent. The next resume/status pass can prune it safely.
            pass
    result = {
        "status": "DONE",
        "split": split,
        "work_shard": work_shard,
        "families": len(index_records),
        "cleaned_rendered": bool(cleanup_rendered),
    }
    print(json.dumps(result, indent=2))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-spec", type=Path, required=True)
    parser.add_argument("--render-root", type=Path, required=True)
    parser.add_argument("--recipe-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), required=True)
    parser.add_argument("--work-shard", type=int, required=True)
    parser.add_argument("--vae-config", type=Path, required=True)
    parser.add_argument("--vae-checkpoint", type=Path, required=True)
    parser.add_argument("--family-batch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cleanup-rendered", action="store_true")
    parser.add_argument(
        "--verify-foa-checksums",
        action="store_true",
        help="Extra encoded-file reread; useful for smoke/audit, disabled at scale.",
    )
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    spec = json.loads(args.build_spec.resolve().read_text(encoding="utf-8"))
    model, vae_config = _load_vae(
        args.vae_config.expanduser().resolve(),
        args.vae_checkpoint.expanduser().resolve(),
        device,
    )
    encode_work_shard(
        spec=spec,
        render_root=args.render_root,
        recipe_root=args.recipe_root,
        output_root=args.output_root,
        split=args.split,
        work_shard=args.work_shard,
        model=model,
        vae_config=vae_config,
        vae_config_path=args.vae_config,
        vae_checkpoint=args.vae_checkpoint,
        device=device,
        family_batch_size=args.family_batch_size,
        cleanup_rendered=bool(args.cleanup_rendered),
        verify_foa_checksums=bool(args.verify_foa_checksums),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
