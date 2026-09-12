#!/usr/bin/env python3
"""Build a small source-identifiability curriculum from canonical FOA families.

The immutable 1M-family store supervises complete mixtures.  In strongly
overlapping scenes that objective can be minimized while omitting a quieter
source.  This builder adds no new model route: it deterministically re-renders
each persistent source execution state in isolation, encodes it with the exact
frozen FOA VAE, and pairs those examples with the original mixture latents.

Every loader item still contains four target states.  ``independent_turns`` in
the dataset metadata config makes all four states creation/no-previous examples
instead of pretending that the curriculum rows form an edit conversation.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import tempfile
from typing import Any, Iterable, Mapping
import uuid

import numpy as np
from safetensors import safe_open
from safetensors.torch import save_file
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.synthesis.render_spatial_edit_families import (  # noqa: E402
    _flac_pcm24_memory_roundtrip,
    _render_track,
)
from scripts.t2a.data.preencode_spatial_cot_family_shard import (  # noqa: E402
    _load_vae,
)
from stable_audio_tools.data.spatial_caption_templates import (  # noqa: E402
    render_semantic_caption,
    validate_semantic_caption_metadata,
)
from stable_audio_tools.data.spatial_edit_recipe import (  # noqa: E402
    source_render_signature,
)
from stable_audio_tools.data.spatial_family_dataset import (  # noqa: E402
    SpatialFamilyDataset,
)
from stable_audio_tools.data.t2a_artifacts import atomic_write_json  # noqa: E402


SCHEMA = "stable_audio_tools.spatial_cot_source_curriculum"
SCHEMA_VERSION = 1
EXPECTED_VAE_SHA256 = (
    "0229e48729bb6cf138c277d37c598d659000cf78e0a16f498171f2f1f83e8a87"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_safetensors(
    path: Path,
    tensors: Mapping[str, torch.Tensor],
    *,
    metadata: Mapping[str, str],
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


def _atomic_jsonl(
    path: Path, records: Iterable[Mapping[str, Any]]
) -> list[tuple[int, int]]:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    offsets: list[tuple[int, int]] = []
    try:
        with os.fdopen(fd, "wb") as handle:
            for record in records:
                payload = json.dumps(
                    record,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
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


def _load_recipe_families(
    family_records: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    by_path: dict[Path, set[str]] = {}
    expected_sha: dict[Path, str] = {}
    for family in family_records:
        path = Path(str(family["recipe_jsonl_ref"])).expanduser().resolve()
        by_path.setdefault(path, set()).add(str(family["family_id"]))
        expected = str(family.get("recipe_jsonl_sha256") or "")
        if not expected:
            raise RuntimeError(f"{family['family_id']} has no recipe JSONL checksum")
        previous = expected_sha.setdefault(path, expected)
        if previous != expected:
            raise RuntimeError(f"conflicting recipe checksums for {path}")

    resolved: dict[str, dict[str, Any]] = {}
    for path, wanted in by_path.items():
        if not path.is_file() or _sha256(path) != expected_sha[path]:
            raise RuntimeError(f"recipe JSONL checksum mismatch: {path}")
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                family = json.loads(line)
                family_id = str(family.get("family_id") or "")
                if family_id in wanted:
                    resolved[family_id] = family
                    if wanted <= resolved.keys():
                        break
    missing = sorted(
        str(family["family_id"])
        for family in family_records
        if str(family["family_id"]) not in resolved
    )
    if missing:
        raise RuntimeError(f"recipe families are missing: {missing[:8]}")
    return resolved


def _pcm24_track(recipe: Mapping[str, Any], source: Mapping[str, Any]) -> np.ndarray:
    track = _render_track(dict(recipe), dict(source))
    peak = float(np.max(np.abs(track)))
    storage_scale = min(1.0, 0.95 / peak) if peak > 0.0 else 1.0
    quantized = _flac_pcm24_memory_roundtrip(
        track * storage_scale,
        int(recipe["audio"]["sample_rate"]),
    )
    return (quantized / storage_scale).astype(np.float32, copy=False)


def _signal_stats(audio: np.ndarray) -> dict[str, float]:
    return {
        "peak": float(np.max(np.abs(audio))),
        "rms": float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))),
    }


def _source_by_id(plan: Mapping[str, Any], source_id: str) -> dict[str, Any]:
    sources = (((plan.get("scene") or {}).get("sources")) or [])
    matches = [source for source in sources if str(source.get("source_id")) == source_id]
    if len(matches) != 1:
        raise RuntimeError(
            f"ScenePlan must contain one {source_id}, found {len(matches)}"
        )
    return copy.deepcopy(matches[0])


def _isolated_plan(
    plan: Mapping[str, Any], source_id: str, *, sample_id: str
) -> dict[str, Any]:
    isolated = copy.deepcopy(dict(plan))
    source = _source_by_id(isolated, source_id)
    isolated.setdefault("scene", {})["sources"] = [source]
    isolated.setdefault("mix", {})["num_sources"] = 1
    # Keep the existing, well-covered mixture type token. Source isolation is
    # a supervision curriculum, not a new user-facing task or plan grammar.
    isolated["mix"]["type"] = "mixture"
    isolated["sample_id"] = sample_id
    label = str((source.get("event") or {}).get("label") or source_id)
    isolated["caption"] = f"An isolated {label} source in the specified FOA field."
    return isolated


def _creation_record(
    base: Mapping[str, Any],
    *,
    family_id: str,
    turn_index: int,
    plan: Mapping[str, Any],
    semantic_caption: str,
    semantic_caption_metadata: Mapping[str, Any],
    curriculum: Mapping[str, Any],
    signal_stats: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    record = copy.deepcopy(dict(base))
    turn_id = f"turn_{turn_index:03d}"
    source_ids = [
        str(source["source_id"])
        for source in (((plan.get("scene") or {}).get("sources")) or [])
    ]
    prompt = str(plan.get("caption") or semantic_caption)
    record.update(
        {
            "sample_id": f"{family_id}_{turn_id}",
            "conversation_id": family_id,
            "turn_id": turn_id,
            "turn_index": turn_index,
            "parent_turn_id": None,
            "task": "text_to_spatial_audio",
            "instruction": prompt,
            "planner_prompt": prompt,
            "semantic_caption": semantic_caption,
            "semantic_caption_metadata": copy.deepcopy(
                dict(semantic_caption_metadata)
            ),
            "edit": {
                "type": "create",
                "target_source_id": None,
                "changed_fields": [],
            },
            "diff": {"added": source_ids, "removed": [], "changed": []},
            "curriculum": copy.deepcopy(dict(curriculum)),
        }
    )
    after = copy.deepcopy(dict(record.get("after") or {}))
    after.update(
        {
            "scene_plan": copy.deepcopy(dict(plan)),
            "latent_state_index": turn_index,
            "audio_path": None,
            "rendered_foa_retained": False,
        }
    )
    if curriculum.get("kind") == "isolated_source_creation":
        after["foa_sha256"] = None
        after["source_track_refs"] = [
            reference
            for reference in (after.get("source_track_refs") or [])
            if str(reference.get("source_id")) in source_ids
        ]
    if signal_stats is not None:
        after["signal_stats"] = copy.deepcopy(dict(signal_stats))
    record["after"] = after
    record["before"] = {
        "scene_plan": None,
        "audio_path": None,
        "foa_sha256": None,
        "latent_state_index": None,
        "rendered_foa_retained": False,
    }
    record["audio_path"] = None
    record["reference_audio"] = None
    record["supervision"] = {
        "paired_edit": False,
        "planner": False,
        "renderer": True,
        "understanding": False,
        "unchanged_source_consistency": False,
    }
    validate_semantic_caption_metadata(
        semantic_caption,
        semantic_caption_metadata,
        expected_source_ids=source_ids,
    )
    return record


def _groups_of_four(
    rows: list[dict[str, Any]],
) -> tuple[list[list[dict[str, Any]]], int]:
    """Pack source states into fixed-size items, padding with quiet rows."""

    if not rows:
        raise RuntimeError("a curriculum family has no source rows")
    ordered = sorted(
        rows,
        key=lambda row: (
            int(row["turn_index"]),
            int(str(row["source_id"]).split("_")[-1]),
            str(row["execution_key"]),
        ),
    )
    groups = [ordered[start : start + 4] for start in range(0, len(ordered), 4)]
    quiet = sorted(ordered, key=lambda row: (float(row["rms"]), row["execution_key"]))
    padding = 0
    while len(groups[-1]) < 4:
        groups[-1].append(quiet[padding % len(quiet)])
        padding += 1
    return groups, padding


def _encode_audio(
    model,
    audio: list[np.ndarray],
    *,
    device: torch.device,
    batch_size: int,
) -> list[torch.Tensor]:
    latents: list[torch.Tensor] = []
    for start in range(0, len(audio), batch_size):
        batch = torch.from_numpy(
            np.stack(audio[start : start + batch_size], axis=0)
        ).to(device=device, dtype=torch.float32, non_blocking=True)
        with torch.inference_mode():
            encoded = model.encode(batch)
        if encoded.ndim != 3 or tuple(encoded.shape[1:]) != (64, 432):
            raise RuntimeError(f"unexpected curriculum VAE latent shape: {encoded.shape}")
        if not torch.isfinite(encoded).all():
            raise RuntimeError("curriculum VAE encoding produced non-finite values")
        latents.extend(encoded.to(dtype=torch.float16, device="cpu").unbind(0))
    return latents


def _write_store(
    root: Path,
    *,
    tensors: Mapping[str, torch.Tensor],
    records: list[dict[str, Any]],
    audit: dict[str, Any],
    schema: str = SCHEMA,
    schema_version: int = SCHEMA_VERSION,
) -> None:
    tensor_path = root / "shards" / "families-00000.safetensors"
    metadata_path = root / "metadata" / "families-00000.jsonl"
    _atomic_safetensors(
        tensor_path,
        tensors,
        metadata={
            "schema": schema,
            "schema_version": str(schema_version),
            "vae_checkpoint_sha256": EXPECTED_VAE_SHA256,
        },
    )
    offsets = _atomic_jsonl(metadata_path, records)
    index_path = root / "index.sqlite"
    connection = sqlite3.connect(index_path)
    try:
        connection.execute(
            "CREATE TABLE families ("
            "family_rank INTEGER PRIMARY KEY, family_id TEXT NOT NULL UNIQUE, "
            "tensor_shard TEXT NOT NULL, tensor_key TEXT NOT NULL, "
            "metadata_shard TEXT NOT NULL, metadata_offset INTEGER NOT NULL, "
            "metadata_length INTEGER NOT NULL, num_turns INTEGER NOT NULL, "
            "channels INTEGER NOT NULL, frames INTEGER NOT NULL, dtype TEXT NOT NULL)"
        )
        for rank, (record, (offset, length)) in enumerate(zip(records, offsets)):
            tensor = tensors[str(record["family_id"])]
            connection.execute(
                "INSERT INTO families VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    rank,
                    record["family_id"],
                    tensor_path.relative_to(root).as_posix(),
                    record["family_id"],
                    metadata_path.relative_to(root).as_posix(),
                    offset,
                    length,
                    int(tensor.shape[0]),
                    int(tensor.shape[1]),
                    int(tensor.shape[2]),
                    "float16",
                ),
            )
        connection.commit()
    finally:
        connection.close()

    with safe_open(str(tensor_path), framework="pt", device="cpu") as handle:
        if set(handle.keys()) != set(tensors):
            raise RuntimeError("written curriculum tensor keys changed")
        for family_id in tensors:
            value = handle.get_tensor(family_id)
            if tuple(value.shape) != (4, 64, 432) or not torch.isfinite(value).all():
                raise RuntimeError(f"written curriculum latent failed QC: {family_id}")
    with sqlite3.connect(index_path) as connection:
        count = int(connection.execute("SELECT COUNT(*) FROM families").fetchone()[0])
    if count != len(records):
        raise RuntimeError("curriculum SQLite count mismatch")

    audit.update(
        {
            "status": "PASS",
            "families": len(records),
            "states": 4 * len(records),
            "tensor_sha256": _sha256(tensor_path),
            "metadata_sha256": _sha256(metadata_path),
            "index_sha256": _sha256(index_path),
        }
    )
    atomic_write_json(root / "AUDIT.json", audit)
    atomic_write_json(
        root / "READY",
        {
            "schema": schema,
            "schema_version": schema_version,
            "families": len(records),
            "states": 4 * len(records),
            # The temporary build directory is atomically renamed after this
            # manifest is written, so keep the reference relocation-safe.
            "audit": "AUDIT.json",
            "vae_checkpoint_sha256": EXPECTED_VAE_SHA256,
        },
    )


def build(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output_root.expanduser().resolve()
    ready = output / "READY"
    if ready.is_file():
        report = json.loads((output / "AUDIT.json").read_text(encoding="utf-8"))
        if report.get("status") != "PASS":
            raise RuntimeError(f"cached curriculum audit is not PASS: {output}")
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

    source_dataset = SpatialFamilyDataset(
        [
            {
                "path": family_store,
                "caption_overlay_path": caption_overlay,
            }
        ],
        require_ready=True,
    )
    if not 0 < args.families <= len(source_dataset):
        raise ValueError("--families is outside the source store")
    source_families: list[dict[str, Any]] = []
    mixture_latents: list[torch.Tensor] = []
    for rank in range(args.families):
        latent, info = source_dataset[rank]
        family = copy.deepcopy(info["spatial_family"])
        if int(family["family_rank"]) != rank or tuple(latent.shape) != (4, 64, 432):
            raise RuntimeError(f"source family rank/shape mismatch at {rank}")
        source_families.append(family)
        mixture_latents.append(latent.to(torch.float16).contiguous())
    recipes = _load_recipe_families(source_families)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    model, _ = _load_vae(vae_config, vae_checkpoint, device)

    source_rows_by_family: list[list[dict[str, Any]]] = []
    source_audio: list[np.ndarray] = []
    slot_counts = {f"source_{slot}": 0 for slot in range(4)}
    category_counts: dict[str, int] = {}
    peak_errors: list[float] = []
    rms_errors: list[float] = []
    isolated_peaks: list[float] = []
    isolated_rms: list[float] = []
    unique_tracks: set[str] = set()
    unique_execution_states: set[str] = set()

    for family, recipe_family in zip(source_families, map(recipes.get, [str(f["family_id"]) for f in source_families])):
        if recipe_family is None:
            raise RuntimeError(f"recipe resolution failed for {family['family_id']}")
        family_id = str(family["family_id"])
        turns = family["turns"]
        recipe_turns = recipe_family["recipes"]
        if len(turns) != 4 or len(recipe_turns) != 4:
            raise RuntimeError(f"{family_id} must contain four states")
        master_gain = float((family.get("render_provenance") or {}).get("family_master_gain_linear"))
        if not math.isfinite(master_gain) or master_gain <= 0.0:
            raise RuntimeError(f"{family_id} has invalid family master gain")

        track_cache: dict[str, np.ndarray] = {}
        execution_rows: dict[str, dict[str, Any]] = {}
        for turn_index, (turn, recipe) in enumerate(zip(turns, recipe_turns)):
            plan = turn["after"]["scene_plan"]
            components: list[np.ndarray] = []
            for source in recipe["sources"]:
                source_id = str(source["source_id"])
                track_id = source_render_signature(recipe, source)
                track = track_cache.get(track_id)
                if track is None:
                    track = _pcm24_track(recipe, source)
                    track_cache[track_id] = track
                gain_db = float(source.get("gain_db", 0.0))
                component = (
                    track
                    * (10.0 ** (gain_db / 20.0))
                    * master_gain
                ).astype(np.float32, copy=False)
                components.append(component)
                execution_key = f"{track_id}:gain={gain_db:.6f}"
                if execution_key in execution_rows:
                    continue
                isolated = np.clip(component, -1.0, 1.0).astype(np.float32, copy=False)
                isolated = _flac_pcm24_memory_roundtrip(
                    isolated, int(recipe["audio"]["sample_rate"])
                )
                stats = _signal_stats(isolated)
                if stats["rms"] <= 0.0 or not np.isfinite(isolated).all():
                    raise RuntimeError(f"invalid isolated source: {execution_key}")
                plan_source = _source_by_id(plan, source_id)
                category = str((plan_source.get("event") or {}).get("category") or "unknown")
                row = {
                    "family_id": family_id,
                    "turn_index": turn_index,
                    "source_id": source_id,
                    "track_id": track_id,
                    "execution_key": execution_key,
                    "gain_db": gain_db,
                    "rms": stats["rms"],
                    "peak": stats["peak"],
                    "category": category,
                    "base_turn": turn,
                    "plan": plan,
                    "audio_index": len(source_audio),
                }
                execution_rows[execution_key] = row
                source_audio.append(isolated)
                unique_tracks.add(track_id)
                unique_execution_states.add(execution_key)
                slot_counts[source_id] = slot_counts.get(source_id, 0) + 1
                category_counts[category] = category_counts.get(category, 0) + 1
                isolated_peaks.append(stats["peak"])
                isolated_rms.append(stats["rms"])

            reconstructed = np.clip(
                np.sum(np.stack(components, axis=0), axis=0), -1.0, 1.0
            )
            reconstructed_stats = _signal_stats(reconstructed)
            target_stats = turn["after"]["signal_stats"]
            peak_errors.append(
                abs(reconstructed_stats["peak"] - float(target_stats["peak"]))
            )
            rms_errors.append(
                abs(reconstructed_stats["rms"] - float(target_stats["rms"]))
            )
        if max(peak_errors[-4:]) > 2.0e-5 or max(rms_errors[-4:]) > 2.0e-5:
            raise RuntimeError(
                f"source components do not reconstruct canonical mixture stats: {family_id}"
            )
        source_rows_by_family.append(list(execution_rows.values()))

    encoded_sources = _encode_audio(
        model,
        source_audio,
        device=device,
        batch_size=args.encode_batch_size,
    )
    if len(encoded_sources) != len(source_audio):
        raise RuntimeError("isolated VAE encoding count mismatch")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    output_records: list[dict[str, Any]] = []
    output_tensors: dict[str, torch.Tensor] = {}
    padding_states = 0
    source_items = 0
    for source_rank, (family, mixture, rows) in enumerate(
        zip(source_families, mixture_latents, source_rows_by_family)
    ):
        original_id = str(family["family_id"])
        mixture_id = f"{original_id}__mixture_curriculum"
        mixture_turns = []
        for turn_index, base in enumerate(family["turns"]):
            plan = copy.deepcopy(base["after"]["scene_plan"])
            mixture_turns.append(
                _creation_record(
                    base,
                    family_id=mixture_id,
                    turn_index=turn_index,
                    plan=plan,
                    semantic_caption=str(base["semantic_caption"]),
                    semantic_caption_metadata=base["semantic_caption_metadata"],
                    curriculum={
                        "kind": "mixture_creation",
                        "source_family_id": original_id,
                        "source_family_rank": source_rank,
                        "source_turn_index": turn_index,
                    },
                    signal_stats=base["after"].get("signal_stats"),
                )
            )
        output_records.append(
            {
                "schema": SCHEMA,
                "schema_version": SCHEMA_VERSION,
                "family_id": mixture_id,
                "family_rank": len(output_records),
                "split": "train",
                "curriculum_kind": "mixture_creation",
                "source_family_id": original_id,
                "turns": mixture_turns,
            }
        )
        output_tensors[mixture_id] = mixture.clone()

        groups, padding = _groups_of_four(rows)
        padding_states += padding
        for group_index, group in enumerate(groups):
            curriculum_id = f"{original_id}__source_curriculum_{group_index:02d}"
            turns: list[dict[str, Any]] = []
            latents: list[torch.Tensor] = []
            for turn_index, row in enumerate(group):
                sample_id = f"{curriculum_id}_turn_{turn_index:03d}"
                plan = _isolated_plan(
                    row["plan"], row["source_id"], sample_id=sample_id
                )
                plan_source = _source_by_id(plan, row["source_id"])
                caption = render_semantic_caption(
                    [plan_source],
                    template_key=(curriculum_id, turn_index, row["execution_key"]),
                )
                turns.append(
                    _creation_record(
                        row["base_turn"],
                        family_id=curriculum_id,
                        turn_index=turn_index,
                        plan=plan,
                        semantic_caption=caption.text,
                        semantic_caption_metadata=caption.metadata(),
                        curriculum={
                            "kind": "isolated_source_creation",
                            "source_family_id": original_id,
                            "source_family_rank": source_rank,
                            "source_turn_index": row["turn_index"],
                            "source_id": row["source_id"],
                            "track_id": row["track_id"],
                            "execution_key": row["execution_key"],
                        },
                        signal_stats={"peak": row["peak"], "rms": row["rms"]},
                    )
                )
                latents.append(encoded_sources[int(row["audio_index"])])
            tensor = torch.stack(latents).to(torch.float16).contiguous()
            output_records.append(
                {
                    "schema": SCHEMA,
                    "schema_version": SCHEMA_VERSION,
                    "family_id": curriculum_id,
                    "family_rank": len(output_records),
                    "split": "train",
                    "curriculum_kind": "isolated_source_creation",
                    "source_family_id": original_id,
                    "turns": turns,
                }
            )
            output_tensors[curriculum_id] = tensor
            source_items += 1

    audit = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": "BUILDING",
        "source_family_store": str(family_store),
        "source_caption_overlay": str(caption_overlay),
        "source_families": args.families,
        "mixture_items": args.families,
        "source_items": source_items,
        "unique_source_tracks": len(unique_tracks),
        "unique_source_execution_states": len(unique_execution_states),
        "isolated_encoded_states": len(source_audio),
        "isolated_padding_states": padding_states,
        "source_slot_counts": slot_counts,
        "source_category_counts": dict(sorted(category_counts.items())),
        "isolated_peak_min": min(isolated_peaks),
        "isolated_peak_max": max(isolated_peaks),
        "isolated_rms_min": min(isolated_rms),
        "isolated_rms_max": max(isolated_rms),
        "mixture_reconstruction_peak_abs_error_max": max(peak_errors),
        "mixture_reconstruction_rms_abs_error_max": max(rms_errors),
        "vae_config": str(vae_config),
        "vae_checkpoint": str(vae_checkpoint),
        "vae_checkpoint_sha256": EXPECTED_VAE_SHA256,
        "vae_rng_seed": args.seed,
        "contract": (
            "One canonical renderer; independent source-isolated creation "
            "targets are interleaved with exact original mixture latents."
        ),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", suffix=".building", dir=output.parent)
    )
    try:
        _write_store(
            temporary,
            tensors=output_tensors,
            records=output_records,
            audit=audit,
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
    parser.add_argument("--encode-batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260812)
    args = parser.parse_args()
    if args.families <= 0 or args.encode_batch_size <= 0:
        parser.error("--families and --encode-batch-size must be positive")
    build(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
