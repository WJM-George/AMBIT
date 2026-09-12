#!/usr/bin/env python3
"""Render and VAE-encode one revision-5 model ScenePlan JSONL shard.

P7.5 deliberately split each sample into three immutable views: a compact
model ScenePlan, a renderer-only recipe, and deterministic conditioning.  P8
joins those views in memory; asset locators and renderer lineage never leak
back into the model ScenePlan stored for P10/P11.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
import sys

for value in (SCRIPT_DIR, REPO_ROOT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from materialize_sceneplan_v2_shard import (  # noqa: E402
    atomic_safetensors,
    load_vae,
    remove_interrupted_render_temporaries,
    render_one as render_execution_record,
    sha256_file,
    tensor_sha256,
)
from sceneplan_v2_common import (  # noqa: E402
    DATASET_ROOT,
    MODEL_SAMPLE_RATE,
    atomic_write_json,
    deterministic_digest,
    require_dataset_not_frozen,
)
from stable_audio_tools.data.model_sceneplan import (  # noqa: E402
    compile_model_renderer_caption,
    validate_model_sceneplan,
)


MATERIALIZED_SCHEMA = pa.schema(
    [
        ("sample_id", pa.string()),
        ("split", pa.string()),
        ("family", pa.string()),
        ("source_count", pa.int8()),
        ("model_num_samples", pa.int32()),
        ("latent_frames_valid", pa.int16()),
        ("vae_encode_seed", pa.uint64()),
        ("planned_bundle_sha256", pa.string()),
        ("model_sceneplan_sha256", pa.string()),
        ("render_recipe_sha256", pa.string()),
        ("renderer_caption_sha256", pa.string()),
        ("render_result_json", pa.string()),
        ("foa_path", pa.string()),
        ("foa_sha256", pa.string()),
        ("latent_ref", pa.string()),
        ("latent_tensor_sha256", pa.string()),
        ("latent_shard_sha256", pa.string()),
        ("work_shard", pa.int32()),
        ("row_in_shard", pa.int16()),
    ]
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def public_render_result(
    result: dict[str, Any],
    *,
    cleanup_foa: bool,
    logical_render_root: Path | None,
    split: str,
    shard: int,
    sample_id: str,
) -> dict[str, Any]:
    value = dict(result)
    if cleanup_foa and logical_render_root is not None:
        value["foa_path"] = str(
            logical_render_root
            / split
            / f"work-{shard:05d}"
            / sample_id
            / Path(result["foa_path"]).name
        )
    return value


def shard_number(path: Path) -> int:
    return int(path.stem.rsplit("-", 1)[-1])


def companion_paths(model_path: Path) -> tuple[Path, Path]:
    name = model_path.name
    if not name.startswith("model-sceneplans-") or model_path.suffix != ".jsonl":
        raise ValueError(f"not a model ScenePlan shard: {model_path}")
    suffix = name[len("model-sceneplans-") :]
    return (
        model_path.with_name(f"render-recipes-{suffix}"),
        model_path.with_name(f"conditioning-{suffix}"),
    )


def _canonical_line(raw: str, *, path: Path, row_index: int) -> tuple[dict[str, Any], str]:
    text = raw.rstrip("\n")
    if not text or "\r" in text:
        raise RuntimeError(f"{path}:{row_index + 1}: invalid JSONL line")
    value = json.loads(text)
    if canonical_json(value) != text:
        raise RuntimeError(f"{path}:{row_index + 1}: JSON is not canonical")
    return value, text


def _trajectory_keyframes(source: dict[str, Any]) -> list[dict[str, Any]]:
    trajectory = source["trajectory"]
    motion_type = str(trajectory["type"])
    onset = float(source["activity"]["onset_sec"])
    offset = float(source["activity"]["offset_sec"])
    if motion_type == "static":
        return [{"time_sec": onset, "position": trajectory["position"]}]
    if motion_type == "linear":
        return [
            {"time_sec": onset, "position": trajectory["start"]},
            {"time_sec": offset, "position": trajectory["end"]},
        ]
    if motion_type == "keyframed":
        return [dict(item) for item in trajectory["keyframes"]]
    raise RuntimeError(f"unsupported trajectory type: {motion_type!r}")


def expand_render_execution(
    model_sceneplan: dict[str, Any], render_recipe: dict[str, Any]
) -> dict[str, Any]:
    """Join compact model state with renderer-only execution state in memory."""

    validate_model_sceneplan(model_sceneplan)
    sample_id = str(model_sceneplan["sample_id"])
    if render_recipe.get("sample_id") != sample_id:
        raise RuntimeError(f"{sample_id}: render recipe sample id mismatch")
    audio = render_recipe["audio_execution"]
    model_num_samples = int(audio["model_num_samples"])
    latent_frames = int(audio["latent_frames_valid"])
    if latent_frames != math.ceil(model_num_samples / 1024):
        raise RuntimeError(f"{sample_id}: latent frame geometry mismatch")
    duration = float(model_sceneplan["duration_sec"])
    if not math.isclose(
        duration, model_num_samples / MODEL_SAMPLE_RATE, rel_tol=0.0, abs_tol=1.1e-6
    ):
        raise RuntimeError(f"{sample_id}: compact duration/sample geometry mismatch")

    recipe_sources = {
        str(source["source_id"]): source for source in render_recipe["sources"]
    }
    if len(recipe_sources) != len(model_sceneplan["sources"]):
        raise RuntimeError(f"{sample_id}: recipe/model source count mismatch")
    sources: list[dict[str, Any]] = []
    for source in model_sceneplan["sources"]:
        source_id = str(source["source_id"])
        recipe_source = recipe_sources.pop(source_id, None)
        if recipe_source is None or recipe_source["kind"] != source["kind"]:
            raise RuntimeError(f"{sample_id}: recipe/model source identity mismatch")
        window = recipe_source["exact_source_sample_window"]
        onset_sample = int(window["model_onset_sample"])
        offset_sample = int(window["model_offset_sample"])
        dry_start = int(window["dry_start_sample"])
        dry_end = int(window["dry_end_sample"])
        if dry_start != 0 or offset_sample - onset_sample != dry_end:
            raise RuntimeError(f"{sample_id}: source window is not a complete dry asset")
        activity = source["activity"]
        if (
            abs(float(activity["onset_sec"]) - onset_sample / MODEL_SAMPLE_RATE) > 1.1e-6
            or abs(float(activity["offset_sec"]) - offset_sample / MODEL_SAMPLE_RATE) > 1.1e-6
        ):
            raise RuntimeError(f"{sample_id}: source time/sample window mismatch")
        expanded: dict[str, Any] = {
            "source_id": source_id,
            "slot": int(source_id[7:]),
            "present": True,
            "kind": str(source["kind"]),
            "description": str(
                source.get("speaker_description") or source.get("description")
            ),
            "gain_db": float(source["gain_db"]),
            "activity": [
                {
                    "onset_sec": float(activity["onset_sec"]),
                    "offset_sec": float(activity["offset_sec"]),
                    "model_onset_sample": onset_sample,
                    "model_offset_sample": offset_sample,
                    "dry_start_sample": dry_start,
                    "dry_end_sample": dry_end,
                }
            ],
            "motion": {
                "type": str(source["trajectory"]["type"]),
                "keyframes": _trajectory_keyframes(source),
            },
            "asset_ref": recipe_source["asset_ref"],
        }
        if source["kind"] == "speech":
            expanded["speech"] = {
                "speaker_description": str(source["speaker_description"]),
                "speaker_id": str(recipe_source["speaker_id"]),
                "transcript": str(source["transcript"]),
                "transcript_normalization": "punctuation_case_whitespace_only_v1",
            }
        sources.append(expanded)
    if recipe_sources:
        raise RuntimeError(f"{sample_id}: unmatched render-recipe sources")

    resolved_room = render_recipe["resolved_room"]
    speech_count = sum(source["kind"] == "speech" for source in sources)
    background_count = sum(source["kind"] != "speech" for source in sources)
    default_mixing_mode = (
        "overlap_calibrated"
        if speech_count and background_count
        else "not_applicable"
    )
    mixing_mode = str(
        (render_recipe.get("mixing") or {}).get(
            "speech_background_mode", default_mixing_mode
        )
    )
    if mixing_mode not in {
        "not_applicable",
        "overlap_calibrated",
        "sequential_nonoverlap",
    }:
        raise RuntimeError(f"{sample_id}: unsupported renderer mixing mode")
    if bool(speech_count and background_count) == (mixing_mode == "not_applicable"):
        raise RuntimeError(f"{sample_id}: renderer mixing mode/source kinds disagree")
    return {
        "audio": {
            "channel_layout": "WYZX_ACN_SN3D",
            "channels": 4,
            "duration_sec": duration,
            "latent_frames_valid": latent_frames,
            "model_num_samples": model_num_samples,
            "model_sample_rate_hz": MODEL_SAMPLE_RATE,
            "render_num_samples": model_num_samples,
            "render_sample_rate_hz": MODEL_SAMPLE_RATE,
            "render_tail_samples": int(audio["render_tail_samples"]),
            "vae_hop_samples": 1024,
            "vae_padded_num_samples": int(audio["vae_padded_num_samples"]),
        },
        "room": {
            "class": str(model_sceneplan["room"]["type"]),
            "room_id": str(resolved_room["room_id"]),
            "dimensions_m": list(map(float, resolved_room["dimensions_m"])),
            "rt60_sec": float(resolved_room["rt60_sec"]),
            "max_order": int(resolved_room["max_order"]),
            "microphone_xyz_m": list(map(float, resolved_room["microphone_xyz_m"])),
            "material_model": "pyroom_inverse_sabine_v1",
        },
        "mixing": {"speech_background_mode": mixing_mode},
        "sources": sources,
    }


def load_shard_rows(model_path: Path, *, max_rows: int | None = None) -> list[dict[str, Any]]:
    model_path = model_path.expanduser().resolve(strict=True)
    recipe_path, conditioning_path = companion_paths(model_path)
    recipe_path = recipe_path.resolve(strict=True)
    conditioning_path = conditioning_path.resolve(strict=True)
    split = model_path.parent.name
    work_shard = shard_number(model_path)
    rows: list[dict[str, Any]] = []
    with (
        model_path.open("r", encoding="utf-8") as model_handle,
        recipe_path.open("r", encoding="utf-8") as recipe_handle,
        conditioning_path.open("r", encoding="utf-8") as conditioning_handle,
    ):
        iterator = itertools.zip_longest(
            model_handle, recipe_handle, conditioning_handle, fillvalue=None
        )
        for row_index, triple in enumerate(iterator):
            if max_rows is not None and row_index >= int(max_rows):
                break
            if any(value is None for value in triple):
                raise RuntimeError(f"{model_path}: companion JSONL row counts differ")
            model, model_text = _canonical_line(
                triple[0], path=model_path, row_index=row_index
            )
            recipe, recipe_text = _canonical_line(
                triple[1], path=recipe_path, row_index=row_index
            )
            conditioning, conditioning_text = _canonical_line(
                triple[2], path=conditioning_path, row_index=row_index
            )
            validate_model_sceneplan(model)
            sample_id = str(model["sample_id"])
            if recipe.get("sample_id") != sample_id or conditioning.get("sample_id") != sample_id:
                raise RuntimeError(f"{sample_id}: P7.5 three-view sample id mismatch")
            model_sha = sha256_text(model_text)
            recipe_sha = sha256_text(recipe_text)
            caption = conditioning.get("renderer_caption")
            if caption != compile_model_renderer_caption(model):
                raise RuntimeError(f"{sample_id}: deterministic conditioning drift")
            caption_text = canonical_json(caption)
            # P7.5 freezes the complete canonical caption object, including
            # all exact character regions, rather than only its display text.
            caption_sha = sha256_text(caption_text)
            if recipe.get("model_sceneplan_sha256") != model_sha:
                raise RuntimeError(f"{sample_id}: render recipe/model SHA mismatch")
            execution = expand_render_execution(model, recipe)
            planned_bundle_sha = sha256_text(
                canonical_json(
                    {
                        "model_sceneplan_sha256": model_sha,
                        "render_recipe_sha256": recipe_sha,
                        "renderer_caption_sha256": caption_sha,
                    }
                )
            )
            family = "speech" if any(
                source["kind"] == "speech" for source in model["sources"]
            ) else "no_speech"
            audio = recipe["audio_execution"]
            target = {
                "materialization_state": "planned",
                "retention": (
                    "transient_train_foa" if split == "train" else "retained_eval_foa"
                ),
                "latent_channels": 64,
                "latent_dtype": "float16",
                "foa_path": None,
                "foa_sha256": None,
                "latent_ref": None,
                "latent_sha256": None,
                "vae_encode_seed": None,
            }
            execution_record = {
                "schema": "stable_audio_tools.sceneplan_materialization_input",
                "schema_version": 3,
                "sample_id": sample_id,
                "split": split,
                "scene_plan": execution,
                "target": target,
            }
            rows.append(
                {
                    "sample_id": sample_id,
                    "split": split,
                    "family": family,
                    "source_count": len(model["sources"]),
                    "model_num_samples": int(audio["model_num_samples"]),
                    "latent_frames_valid": int(audio["latent_frames_valid"]),
                    "planned_bundle_sha256": planned_bundle_sha,
                    "model_sceneplan_sha256": model_sha,
                    "render_recipe_sha256": recipe_sha,
                    "renderer_caption_sha256": caption_sha,
                    "model_sceneplan_json": model_text,
                    "render_recipe_json": recipe_text,
                    "renderer_caption_json": caption_text,
                    "record_json": canonical_json(execution_record),
                    # The proven revision-4 renderer uses this generic lineage
                    # field.  In revision 5 it is the immutable three-view
                    # bundle hash, not a hash of a model-facing record.
                    "record_sha256": planned_bundle_sha,
                    "work_shard": work_shard,
                    "row_in_shard": row_index,
                }
            )
    if not rows:
        raise RuntimeError(f"empty model ScenePlan shard: {model_path}")
    return rows


def render_one(row: dict[str, Any], output_root: str, retain_stems: bool) -> dict[str, Any]:
    result = render_execution_record(row, output_root, retain_stems)
    if result.get("status") != "ok":
        return result
    execution = result.pop("materialized_scene_plan", None)
    if execution is not None:
        result["materialized_execution_sha256"] = sha256_text(canonical_json(execution))
    contract_revision = int(
        json.loads(row["render_recipe_json"]).get(
            "dataset_contract_revision", 5
        )
    )
    result.update(
        {
            "schema_version": 3,
            "dataset_contract_revision": contract_revision,
            "planned_bundle_sha256": row["planned_bundle_sha256"],
            "model_sceneplan_sha256": row["model_sceneplan_sha256"],
            "render_recipe_sha256": row["render_recipe_sha256"],
            "renderer_caption_sha256": row["renderer_caption_sha256"],
            "model_sceneplan_immutable_during_materialization": True,
        }
    )
    result_path = Path(result["foa_path"]).parent / "render_result.json"
    atomic_write_json(result_path, result)
    return result


def encode_results(
    rows: list[dict[str, Any]],
    results: list[dict[str, Any]],
    *,
    output_root: Path,
    device: torch.device,
    batch_size: int,
    cleanup_foa: bool,
    retain_stems: bool,
    model: Any | None = None,
    logical_render_root: Path | None = None,
) -> tuple[list[dict[str, Any]], Path]:
    shard = int(rows[0]["work_shard"])
    split = str(rows[0]["split"])
    seed = int(deterministic_digest(20260814, "vae-encode", split, shard)[:16], 16)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    owns_model = model is None
    if model is None:
        model = load_vae(device)
    row_by_id = {str(row["sample_id"]): row for row in rows}
    tensors: dict[str, torch.Tensor] = {}
    pending: list[dict[str, Any]] = []

    def flush() -> None:
        if not pending:
            return
        audios: list[torch.Tensor] = []
        valid: list[tuple[str, int, int]] = []
        max_padded = 0
        for result in pending:
            row = row_by_id[str(result["sample_id"])]
            audio_spec = json.loads(row["render_recipe_json"])["audio_execution"]
            audio, rate = sf.read(result["foa_path"], dtype="float32", always_2d=True)
            if rate != MODEL_SAMPLE_RATE or audio.shape != (
                int(audio_spec["model_num_samples"]),
                4,
            ):
                raise RuntimeError("FOA geometry changed before VAE encoding")
            padded = int(audio_spec["vae_padded_num_samples"])
            max_padded = max(max_padded, padded)
            audios.append(torch.from_numpy(audio.T.copy()))
            valid.append(
                (str(result["sample_id"]), int(audio_spec["latent_frames_valid"]), padded)
            )
        batch = torch.zeros((len(audios), 4, max_padded), dtype=torch.float32)
        for index, (audio, (_, _, padded)) in enumerate(zip(audios, valid)):
            batch[index, :, : audio.shape[-1]] = audio
            if padded < max_padded:
                batch[index, :, padded:] = 0
        with torch.inference_mode():
            latent_batch = model.encode(batch.to(device, non_blocking=True))
        if latent_batch.ndim != 3 or latent_batch.shape[1] != 64:
            raise RuntimeError(f"unexpected VAE latent shape: {tuple(latent_batch.shape)}")
        for index, (sample_id, frames, _) in enumerate(valid):
            latent = latent_batch[index, :, :frames].to(torch.float16).cpu().contiguous()
            if latent.shape != (64, frames) or not torch.isfinite(latent).all():
                raise RuntimeError(f"invalid variable-length latent: {sample_id}")
            tensors[sample_id] = latent
        pending.clear()

    for result in results:
        pending.append(result)
        if len(pending) >= int(batch_size):
            flush()
    flush()
    if owns_model:
        del model
    if owns_model and device.type == "cuda":
        torch.cuda.empty_cache()

    latent_path = (
        output_root / "latents" / split / f"latents-{split}-{shard:05d}.safetensors"
    )
    atomic_safetensors(latent_path, tensors)
    latent_shard_sha = sha256_file(latent_path)
    result_by_id = {str(result["sample_id"]): result for result in results}
    materialized: list[dict[str, Any]] = []
    for row in rows:
        sample_id = str(row["sample_id"])
        result = result_by_id[sample_id]
        public_result = public_render_result(
            result,
            cleanup_foa=cleanup_foa,
            logical_render_root=logical_render_root,
            split=split,
            shard=shard,
            sample_id=sample_id,
        )
        materialized.append(
            {
                "sample_id": sample_id,
                "split": row["split"],
                "family": row["family"],
                "source_count": int(row["source_count"]),
                "model_num_samples": int(row["model_num_samples"]),
                "latent_frames_valid": int(row["latent_frames_valid"]),
                "vae_encode_seed": seed,
                "planned_bundle_sha256": row["planned_bundle_sha256"],
                "model_sceneplan_sha256": row["model_sceneplan_sha256"],
                "render_recipe_sha256": row["render_recipe_sha256"],
                "renderer_caption_sha256": row["renderer_caption_sha256"],
                "render_result_json": canonical_json(public_result),
                "foa_path": None if cleanup_foa else result["foa_path"],
                "foa_sha256": result["foa_sha256"],
                "latent_ref": f"{latent_path}#{sample_id}",
                "latent_tensor_sha256": tensor_sha256(tensors[sample_id]),
                "latent_shard_sha256": latent_shard_sha,
                "work_shard": int(row["work_shard"]),
                "row_in_shard": int(row["row_in_shard"]),
            }
        )

    manifest = (
        output_root
        / "manifests"
        / split
        / f"materialized-{split}-{shard:05d}.parquet"
    )
    manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest.with_name(manifest.name + f".tmp.{os.getpid()}")
    pq.write_table(
        pa.Table.from_pylist(materialized, schema=MATERIALIZED_SCHEMA),
        temporary,
        compression="zstd",
    )
    if pq.read_metadata(temporary).num_rows != len(materialized):
        raise RuntimeError("materialized manifest row count changed after reopen")
    os.replace(temporary, manifest)

    if cleanup_foa:
        for result in results:
            sample_root = Path(result["foa_path"]).parent
            Path(result["foa_path"]).unlink(missing_ok=True)
            for reference in result.get("stem_refs") or []:
                Path(reference["path"]).unlink(missing_ok=True)
            (sample_root / "render_result.json").unlink(missing_ok=True)
            remove_interrupted_render_temporaries(sample_root)
            sample_root.rmdir()
        Path(results[0]["foa_path"]).parent.parent.rmdir()
    return materialized, manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-sceneplan-shard", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--retain-stems", action="store_true")
    parser.add_argument("--cleanup-foa", action="store_true")
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--supplement-mode", action="store_true")
    parser.add_argument(
        "--revision-mode",
        action="store_true",
        help="materialize an append-only dataset revision below DATASET_ROOT/revisions",
    )
    args = parser.parse_args()
    model_path = args.model_sceneplan_shard.expanduser().resolve(strict=True)
    output_root = args.output_root.expanduser().resolve(strict=False)
    try:
        output_root.relative_to(os.environ.get("AMBIT_DATA_ROOT", "data"))
    except ValueError as error:
        raise ValueError(f"materialized outputs must be on SDB: {output_root}") from error
    if args.supplement_mode and args.revision_mode:
        raise ValueError("--supplement-mode and --revision-mode are mutually exclusive")
    if args.supplement_mode:
        supplement_root = (DATASET_ROOT / "supplements").resolve(strict=True)
        for path in (model_path, output_root):
            try:
                path.relative_to(supplement_root)
            except ValueError as error:
                raise ValueError(
                    f"supplement materialization must stay below {supplement_root}: {path}"
                ) from error
        if not (DATASET_ROOT / "FROZEN_P9.json").is_file():
            raise RuntimeError("supplement mode requires an already-frozen base P9")
    elif args.revision_mode:
        revision_root = (DATASET_ROOT / "revisions").resolve(strict=True)
        for path in (model_path, output_root):
            try:
                path.relative_to(revision_root)
            except ValueError as error:
                raise ValueError(
                    f"revision materialization must stay below {revision_root}: {path}"
                ) from error
        if not (DATASET_ROOT / "FROZEN_P9.json").is_file():
            raise RuntimeError("revision mode requires an already-frozen base P9")
    else:
        require_dataset_not_frozen()
    rows = load_shard_rows(model_path, max_rows=args.max_rows)
    if args.revision_mode and any(
        int(json.loads(row["render_recipe_json"]).get("dataset_contract_revision", -1))
        != 6
        for row in rows
    ):
        raise RuntimeError("revision mode requires contract-revision-6 render recipes")
    import concurrent.futures
    import multiprocessing as mp

    context = mp.get_context("spawn")
    results: list[dict[str, Any]] = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.jobs, mp_context=context
    ) as pool:
        futures = [
            pool.submit(render_one, row, str(output_root / "renders"), args.retain_stems)
            for row in rows
        ]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda result: str(result["sample_id"]))
    failures = [result for result in results if result.get("status") != "ok"]
    if failures:
        quarantine = (
            output_root
            / "quarantine"
            / rows[0]["split"]
            / f"work-{rows[0]['work_shard']:05d}.json"
        )
        atomic_write_json(quarantine, {"error_count": len(failures), "errors": failures})
        raise RuntimeError(f"render shard has {len(failures)} quarantines")
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    materialized, manifest = encode_results(
        rows,
        results,
        output_root=output_root,
        device=device,
        batch_size=args.batch_size,
        cleanup_foa=args.cleanup_foa,
        retain_stems=args.retain_stems,
    )
    # A resumed shard may have a fail-closed quarantine receipt from an
    # earlier attempt.  Remove it only after every row has rendered and the
    # replacement latent/manifest have been atomically written.
    (
        output_root
        / "quarantine"
        / rows[0]["split"]
        / f"work-{rows[0]['work_shard']:05d}.json"
    ).unlink(missing_ok=True)
    revisions = {
        int(json.loads(row["render_recipe_json"]).get("dataset_contract_revision", 5))
        for row in rows
    }
    if len(revisions) != 1:
        raise RuntimeError("one materialization shard cannot mix contract revisions")
    print(
        json.dumps(
            {
                "ok": True,
                "dataset_contract_revision": revisions.pop(),
                "rows": len(materialized),
                "manifest": str(manifest),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
