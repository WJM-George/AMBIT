#!/usr/bin/env python3
"""Fail-closed quality gate for retained Spatial-CoT validation/test data.

The regular metadata audit proves the planner/loader/latent contract.  This
gate additionally treats retained evaluation audio as a reproducible test
fixture: every FOA and pre-mix track is PCM24 FLAC, every persisted checksum is
re-read, selected families are independently remixed from their source tracks,
and signal statistics are measured from decoded audio.  ``QUALITY.json`` is
published only after every check succeeds.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import copy
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.t2a.audit.audit_spatial_cot_data import audit_view  # noqa: E402
from scripts.t2a.data.preencode_spatial_cot_family_shard import (  # noqa: E402
    _validate_diff_and_tracks,
)
from stable_audio_tools.data.spatial_edit_recipe import (  # noqa: E402
    RECIPE_VERSION,
    validate_edit_family,
)
from stable_audio_tools.data.t2a_artifacts import atomic_write_json  # noqa: E402


QUALITY_SCHEMA = "stable_audio_tools.spatial_cot_eval_quality"
QUALITY_VERSION = 2
RETAINED_PROFILE = "retained_flac_pcm24"


def _json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(f"invalid JSONL {path}:{line_number}") from error


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _inside(path: Path, root: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise RuntimeError(f"{label} escapes retained split root: {resolved}") from error
    return resolved


def _audio_info(
    path: Path,
    *,
    sample_rate: int,
    channels: int,
    num_samples: int,
) -> None:
    info = sf.info(str(path))
    if (
        info.format != "FLAC"
        or info.subtype != "PCM_24"
        or int(info.samplerate) != sample_rate
        or int(info.channels) != channels
        or int(info.frames) != num_samples
    ):
        raise RuntimeError(
            f"invalid retained PCM24 audio {path}: format={info.format} "
            f"subtype={info.subtype} rate={info.samplerate} "
            f"channels={info.channels} frames={info.frames}"
        )


def _sample_ranks(total: int, requested: int) -> set[int]:
    count = min(total, max(1, requested))
    if count == 1:
        return {0}
    return {
        round(index * (total - 1) / (count - 1))
        for index in range(count)
    }


def _recipe_without_render_outputs(family: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(family))
    result.pop("render_status", None)
    result.pop("render_summary", None)
    for recipe in result.get("recipes") or []:
        recipe.pop("outputs", None)
        (recipe.get("mix") or {})["family_master_gain_linear"] = None
    for turn in result.get("turns") or []:
        (turn.get("before") or {})["audio_ref"] = None
        (turn.get("after") or {})["audio_ref"] = None
    return result


def _read_track(metadata: Mapping[str, Any]) -> np.ndarray:
    audio, rate = sf.read(
        str(metadata["path"]), always_2d=True, dtype="float32"
    )
    if int(rate) != int(metadata["sample_rate"]) or audio.shape[1] != 4:
        raise RuntimeError(f"source-track layout changed: {metadata['path']}")
    return (
        audio.T.astype(np.float32, copy=False)
        * float(metadata["restore_gain"])
    ).astype(np.float32, copy=False)


def _signal_stats(audio: np.ndarray, sample_rate: int) -> dict[str, float]:
    if not np.isfinite(audio).all():
        raise RuntimeError("decoded FOA contains NaN or Inf")
    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
    block = max(1, sample_rate // 10)
    frame_rms = []
    for start in range(0, audio.shape[0], block):
        frame = audio[start : start + block]
        frame_rms.append(
            float(np.sqrt(np.mean(np.square(frame, dtype=np.float64))))
        )
    return {
        "peak": peak,
        "rms": rms,
        "active_100ms_fraction": float(
            np.mean(np.asarray(frame_rms, dtype=np.float64) >= 1.0e-4)
        ),
        "near_clip_fraction": float(np.mean(np.abs(audio) >= 0.999)),
    }


def _remix_error(
    family: Mapping[str, Any],
    track_metadata: Mapping[str, Mapping[str, Any]],
) -> float:
    cache: dict[str, np.ndarray] = {}
    worst = 0.0
    for recipe in family["recipes"]:
        refs = recipe["outputs"]["source_track_refs"]
        target_path = Path(recipe["outputs"]["foa_path"])
        target, rate = sf.read(str(target_path), always_2d=True, dtype="float32")
        expected = np.zeros((4, target.shape[0]), dtype=np.float32)
        for reference in refs:
            track_id = str(reference["track_id"])
            track = cache.get(track_id)
            if track is None:
                track = _read_track(track_metadata[track_id])
                cache[track_id] = track
            gain = 10.0 ** (float(reference["gain_db"]) / 20.0)
            expected += track[:, : expected.shape[1]] * gain
        expected = np.clip(
            expected * float(recipe["mix"]["family_master_gain_linear"]),
            -1.0,
            1.0,
        )
        if int(rate) != int(recipe["audio"]["sample_rate"]):
            raise RuntimeError(f"FOA sample-rate changed: {target_path}")
        error = float(np.max(np.abs(expected.T - target)))
        # The target is one final PCM24 quantization of the independently
        # reconstructed float32 mix.  Two PCM24 least-significant steps allow
        # for libsndfile boundary behavior while still catching any changed
        # source, gain, trajectory, room, ordering, or family master gain.
        if error > 2.0 / (2**23):
            raise RuntimeError(
                f"independent source-track remix mismatch: {target_path} "
                f"max_abs_error={error:.9g}"
            )
        worst = max(worst, error)
    return worst


def _check_hash(item: tuple[Path, str]) -> tuple[Path, str]:
    path, expected = item
    actual = _sha256(path)
    if actual != expected:
        raise RuntimeError(f"retained audio checksum mismatch: {path}")
    return path, actual


def audit_retained_audio(
    *,
    spec: Mapping[str, Any],
    split: str,
    expected_families: int,
    recipe_split_root: Path,
    render_split_root: Path,
    latent_root: Path,
    codec_root: Path,
    audio_sample_families: int,
    remix_sample_families: int,
    hash_workers: int,
) -> dict[str, Any]:
    audio_spec = spec["audio"]
    quality = spec["quality_control"]
    minimum_rms = float(quality["minimum_rms"])
    minimum_active_fraction = float(
        quality["minimum_active_100ms_fraction"]
    )
    active_frame_threshold = float(quality["active_frame_rms_threshold"])
    per_shard = int(spec["sharding"]["families_per_latent_shard"])
    work_shards = math.ceil(expected_families / per_shard)
    # First prove the complete recipe/latent/loader contract.  This includes
    # every float16 tensor, every shard checksum, codec round-trips, persistent
    # edit diffs, previous-latent alignment, and retained path semantics.
    structural = audit_view(
        recipe_split_root,
        latent_root,
        codec_root,
        spec,
        split=split,
        expected_families=expected_families,
        runtime_samples=min(256, expected_families),
    )

    audio_ranks = _sample_ranks(expected_families, audio_sample_families)
    remix_ranks = _sample_ranks(expected_families, remix_sample_families)
    expected_hashes: dict[Path, str] = {}
    signal_rows: list[dict[str, float]] = []
    remix_errors: list[float] = []
    edit_counts = Counter()
    families_seen = 0
    states_seen = 0
    tracks_seen: set[Path] = set()

    for work_shard in range(work_shards):
        expected_local = min(
            per_shard, expected_families - work_shard * per_shard
        )
        recipe_root = recipe_split_root / f"work-{work_shard:05d}"
        render_root = render_split_root / f"work-{work_shard:05d}" / "render"
        recipe_ready = _json(recipe_root / "READY")
        render_ready = _json(render_root / "READY")
        marker = _json(render_root / ".spatial_cot_render_root.json")
        if (
            int(recipe_ready.get("families", -1)) != expected_local
            or int(render_ready.get("families", -1)) != expected_local
            or render_ready.get("storage_profile") != RETAINED_PROFILE
            or marker.get("storage_profile") != RETAINED_PROFILE
        ):
            raise RuntimeError(f"retained shard READY mismatch: {work_shard}")

        recipe_path = (
            recipe_root / "shards" / f"recipes-{split}-{work_shard:05d}.jsonl"
        )
        raw_by_id = {str(row["family_id"]): row for row in _jsonl(recipe_path)}
        metadata_path = (
            latent_root / "metadata" / f"families-{work_shard:05d}.jsonl"
        )
        metadata_by_id = {
            str(row["family_id"]): row for row in _jsonl(metadata_path)
        }
        manifest_paths = sorted(render_root.glob("families/*/family.json"))
        if (
            len(raw_by_id) != expected_local
            or len(metadata_by_id) != expected_local
            or len(manifest_paths) != expected_local
        ):
            raise RuntimeError(f"retained shard family count mismatch: {work_shard}")

        for manifest_path in manifest_paths:
            family = _json(manifest_path)
            validate_edit_family(family, require_outputs=True)
            _validate_diff_and_tracks(family)
            family_id = str(family["family_id"])
            raw = raw_by_id.get(family_id)
            metadata_record = metadata_by_id.get(family_id)
            if raw is None or metadata_record is None:
                raise RuntimeError(
                    f"render manifest lacks recipe/latent metadata: {family_id}"
                )
            if str(family.get("schema_version")) != RECIPE_VERSION:
                raise RuntimeError(f"stale retained recipe: {family_id}")
            if _recipe_without_render_outputs(family) != _recipe_without_render_outputs(raw):
                raise RuntimeError(f"renderer changed recipe semantics: {family_id}")
            rank = int(family["family_rank"])
            if rank != families_seen:
                raise RuntimeError(
                    f"retained families are not rank contiguous: {rank} != {families_seen}"
                )
            summary = family.get("render_summary") or {}
            if (
                summary.get("storage_profile") != RETAINED_PROFILE
                or summary.get("foa_container") != "flac"
                or summary.get("foa_subtype") != "PCM_24"
                or summary.get("source_tracks_retained") is not True
                or summary.get("independent_state_mix") is not True
                or summary.get("uses_previous_foa") is not False
                or not math.isclose(
                    float(summary.get("minimum_rms", -1.0)), minimum_rms
                )
                or not math.isclose(
                    float(summary.get("minimum_active_100ms_fraction", -1.0)),
                    minimum_active_fraction,
                )
                or not math.isclose(
                    float(summary.get("active_frame_rms_threshold", -1.0)),
                    active_frame_threshold,
                )
            ):
                raise RuntimeError(f"invalid retained render summary: {family_id}")

            track_metadata: dict[str, dict[str, Any]] = {}
            metadata_turns = metadata_record.get("turns") or []
            if len(metadata_turns) != len(family["recipes"]):
                raise RuntimeError(f"retained metadata turn mismatch: {family_id}")
            for turn_index, recipe in enumerate(family["recipes"]):
                edit_counts[str(recipe["edit"]["type"])] += 1
                output = recipe["outputs"]
                persisted_stats = output.get("signal_stats") or {}
                try:
                    persisted_peak = float(persisted_stats["peak"])
                    persisted_rms = float(persisted_stats["rms"])
                    persisted_active = float(
                        persisted_stats["active_100ms_fraction"]
                    )
                    persisted_threshold = float(
                        persisted_stats["active_frame_rms_threshold"]
                    )
                except (KeyError, TypeError, ValueError) as error:
                    raise RuntimeError(
                        f"missing persisted signal QC: {family_id}:{turn_index}"
                    ) from error
                if (
                    not all(
                        math.isfinite(value)
                        for value in (
                            persisted_peak,
                            persisted_rms,
                            persisted_active,
                            persisted_threshold,
                        )
                    )
                    or persisted_rms < minimum_rms
                    or persisted_active < minimum_active_fraction
                    or persisted_peak > float(quality["max_abs_peak"]) + 1.0e-6
                    or not math.isclose(
                        persisted_threshold, active_frame_threshold
                    )
                ):
                    raise RuntimeError(
                        f"persisted FOA signal QC failed: "
                        f"{family_id}:{turn_index}: {persisted_stats}"
                    )
                foa_path = _inside(Path(output["foa_path"]), render_root, "FOA")
                metadata_turn = metadata_turns[turn_index]
                if (
                    metadata_turn.get("audio_path") != str(foa_path)
                    or metadata_turn["after"].get("audio_path") != str(foa_path)
                    or metadata_turn["after"].get("foa_sha256")
                    != output.get("foa_sha256")
                    or metadata_turn["after"].get("source_track_refs")
                    != output.get("source_track_refs")
                ):
                    raise RuntimeError(
                        f"render/latent metadata provenance mismatch: "
                        f"{family_id}:{turn_index}"
                    )
                expected_before = (
                    family["recipes"][turn_index - 1]["outputs"]["foa_path"]
                    if turn_index
                    else None
                )
                if metadata_turn["before"].get("audio_path") != expected_before:
                    raise RuntimeError(
                        f"retained before-FOA mismatch: {family_id}:{turn_index}"
                    )
                _audio_info(
                    foa_path,
                    sample_rate=int(audio_spec["sample_rate"]),
                    channels=4,
                    num_samples=int(audio_spec["num_samples"]),
                )
                foa_hash = str(output.get("foa_sha256") or "")
                if len(foa_hash) != 64:
                    raise RuntimeError(f"missing FOA checksum: {foa_path}")
                prior = expected_hashes.setdefault(foa_path, foa_hash)
                if prior != foa_hash:
                    raise RuntimeError(f"conflicting FOA checksum: {foa_path}")
                for reference in output.get("source_track_refs") or []:
                    if reference.get("retained") is not True:
                        raise RuntimeError(f"source track was not retained: {family_id}")
                    track_id = str(reference["track_id"])
                    track_path = _inside(
                        Path(reference["path"]), render_root, "source track"
                    )
                    _audio_info(
                        track_path,
                        sample_rate=int(audio_spec["sample_rate"]),
                        channels=4,
                        num_samples=int(audio_spec["num_samples"]),
                    )
                    metadata_path = track_path.with_suffix(".json")
                    metadata = _json(metadata_path)
                    if (
                        metadata.get("track_id") != track_id
                        or Path(metadata.get("path", "")).resolve() != track_path
                        or metadata.get("channel_layout") != audio_spec["channel_layout"]
                        or int(metadata.get("sample_rate", -1))
                        != int(audio_spec["sample_rate"])
                        or int(metadata.get("channels", -1)) != 4
                        or int(metadata.get("num_samples", -1))
                        != int(audio_spec["num_samples"])
                        or not float(metadata.get("restore_gain") or 0.0) > 0
                    ):
                        raise RuntimeError(f"invalid source-track metadata: {metadata_path}")
                    track_hash = str(metadata.get("sha256") or "")
                    if len(track_hash) != 64:
                        raise RuntimeError(f"missing track checksum: {track_path}")
                    prior = expected_hashes.setdefault(track_path, track_hash)
                    if prior != track_hash:
                        raise RuntimeError(f"conflicting track checksum: {track_path}")
                    prior_metadata = track_metadata.setdefault(track_id, metadata)
                    if prior_metadata != metadata:
                        raise RuntimeError(f"conflicting track metadata: {track_id}")
                    tracks_seen.add(track_path)
                if rank in audio_ranks:
                    audio, rate = sf.read(
                        str(foa_path), always_2d=True, dtype="float32"
                    )
                    if int(rate) != int(audio_spec["sample_rate"]):
                        raise RuntimeError(f"sampled FOA rate changed: {foa_path}")
                    row = _signal_stats(audio, int(rate))
                    if (
                        row["rms"] < minimum_rms
                        or row["peak"]
                        > float(quality["max_abs_peak"]) + 1.0e-6
                        or row["active_100ms_fraction"] < minimum_active_fraction
                    ):
                        raise RuntimeError(f"sampled FOA signal QC failed: {foa_path}: {row}")
                    signal_rows.append(row)
                states_seen += 1
            if rank in remix_ranks:
                remix_errors.append(_remix_error(family, track_metadata))
            families_seen += 1

    if families_seen != expected_families or states_seen != expected_families * 4:
        raise RuntimeError(
            f"retained counts mismatch: families={families_seen} states={states_seen}"
        )
    with concurrent.futures.ThreadPoolExecutor(max_workers=hash_workers) as executor:
        for _ in executor.map(_check_hash, sorted(expected_hashes.items()), chunksize=8):
            pass

    if not signal_rows or not remix_errors:
        raise RuntimeError("quality audit selected no audio/remix samples")
    return {
        "structural": structural,
        "families": families_seen,
        "states": states_seen,
        "work_shards": work_shards,
        "edit_counts": dict(edit_counts),
        "retained_foa_files_hashed": states_seen,
        "unique_source_track_files_hashed": len(tracks_seen),
        "all_retained_audio_files_hashed": len(expected_hashes),
        "all_state_signal_stats_verified": states_seen,
        "sampled_audio_families": len(audio_ranks),
        "sampled_audio_states": len(signal_rows),
        "sampled_remix_families": len(remix_errors),
        "rms_min": min(row["rms"] for row in signal_rows),
        "rms_mean": sum(row["rms"] for row in signal_rows) / len(signal_rows),
        "peak_max": max(row["peak"] for row in signal_rows),
        "active_100ms_fraction_min": min(
            row["active_100ms_fraction"] for row in signal_rows
        ),
        "near_clip_fraction_max": max(
            row["near_clip_fraction"] for row in signal_rows
        ),
        "independent_remix_max_abs_error": max(remix_errors),
        "storage_profile": RETAINED_PROFILE,
        "foa_format": "FLAC/PCM_24/WYZX_ACN_SN3D",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-spec", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--expected-families", type=int, default=None)
    parser.add_argument("--audio-sample-families", type=int, default=512)
    parser.add_argument("--remix-sample-families", type=int, default=128)
    parser.add_argument("--hash-workers", type=int, default=8)
    parser.add_argument("--recipe-split-root", type=Path, default=None)
    parser.add_argument("--render-split-root", type=Path, default=None)
    parser.add_argument("--latent-root", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()
    if min(
        args.audio_sample_families,
        args.remix_sample_families,
        args.hash_workers,
    ) <= 0:
        raise SystemExit("sample and worker counts must be positive")
    build_spec_path = args.build_spec.expanduser().resolve()
    spec = _json(build_spec_path)
    build_spec_sha256 = _sha256(build_spec_path)
    expected = int(
        args.expected_families
        if args.expected_families is not None
        else spec["splits"][args.split]["families"]
    )
    explicit_roots = (
        args.recipe_split_root,
        args.render_split_root,
        args.latent_root,
    )
    if any(root is not None for root in explicit_roots) and not all(
        root is not None for root in explicit_roots
    ):
        raise SystemExit(
            "--recipe-split-root, --render-split-root, and --latent-root "
            "must be provided together"
        )
    full_expected = int(spec["splits"][args.split]["families"])
    if expected != full_expected and not all(root is not None for root in explicit_roots):
        raise SystemExit(
            "a retained eval subset requires explicit isolated recipe/render/latent roots"
        )
    if not 0 < expected <= full_expected:
        raise SystemExit(f"expected-families must lie in [1, {full_expected}]")
    if (
        spec["splits"][args.split].get("retain_rendered_foa") is not True
        or spec["splits"][args.split].get("retain_source_tracks") is not True
        or spec["storage"].get("eval_render_profile") != RETAINED_PROFILE
    ):
        raise SystemExit(f"split {args.split} is not configured as retained PCM24 eval")

    recipe_split_root = (
        args.recipe_split_root.expanduser().resolve()
        if args.recipe_split_root is not None
        else Path(spec["storage"]["catalog_root"]).expanduser().resolve()
        / "recipes"
        / args.split
    )
    render_split_root = (
        args.render_split_root.expanduser().resolve()
        if args.render_split_root is not None
        else Path(spec["storage"]["retained_eval_render_root"]).expanduser().resolve()
        / args.split
    )
    latent_root = (
        args.latent_root.expanduser().resolve()
        if args.latent_root is not None
        else Path(spec["storage"]["latent_root"]).expanduser().resolve()
        / args.split
    )
    codec_root = Path(spec["storage"]["codec_root"]).expanduser().resolve()
    output = (
        args.output_json.expanduser().resolve()
        if args.output_json is not None
        else render_split_root / "QUALITY.json"
    )
    if output.is_file():
        existing = _json(output)
        if (
            existing.get("status") == "PASS"
            and existing.get("split") == args.split
            and int(existing.get("families", -1)) == expected
            and existing.get("recipe_version") == RECIPE_VERSION
            and existing.get("build_spec_sha256") == build_spec_sha256
        ):
            print(json.dumps({"status": "ALREADY_PASS", "quality": str(output)}, indent=2))
            return 0
        raise SystemExit(f"refusing to overwrite non-matching quality marker: {output}")

    result = audit_retained_audio(
        spec=spec,
        split=args.split,
        expected_families=expected,
        recipe_split_root=recipe_split_root,
        render_split_root=render_split_root,
        latent_root=latent_root,
        codec_root=codec_root,
        audio_sample_families=args.audio_sample_families,
        remix_sample_families=args.remix_sample_families,
        hash_workers=args.hash_workers,
    )
    report = {
        "schema": QUALITY_SCHEMA,
        "schema_version": QUALITY_VERSION,
        "status": "PASS",
        "split": args.split,
        "recipe_version": RECIPE_VERSION,
        "build_spec_sha256": build_spec_sha256,
        **result,
    }
    atomic_write_json(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
