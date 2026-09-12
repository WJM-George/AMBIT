#!/usr/bin/env python3
"""Fail-closed audit for Spatial-CoT metadata, recipes, and latent views.

READY markers are necessary but not sufficient.  This command independently
checks structural ScenePlan invariants, codec round-trips/FSM constraints,
source-disjoint catalog policy, persistent executable recipes, and finalized
family safetensors.  It never renders production data or mutates an artifact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sqlite3
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterator, Mapping

import torch
from safetensors import safe_open

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stable_audio_tools.data.spatial_conversation_metadata import (  # noqa: E402
    SpatialFamilyMetadata,
)
from stable_audio_tools.data.spatial_edit_recipe import (  # noqa: E402
    RECIPE_VERSION,
    STATE_RENDER_INPUT,
    is_silent_source,
    is_speech_source,
    stable_digest,
    validate_edit_family,
)
from stable_audio_tools.data.spatial_family_dataset import (  # noqa: E402
    SpatialFamilyDataset,
)
from stable_audio_tools.data.spatial_plan_codec import SpatialPlanCodec  # noqa: E402
from stable_audio_tools.data.spatial_story import (  # noqa: E402
    compile_source_tracks,
    diff_scene_plans,
)
from stable_audio_tools.data.t2a_artifacts import atomic_write_json  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(paths: Iterator[Path]) -> Iterator[dict[str, Any]]:
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as error:
                    raise RuntimeError(f"invalid JSONL {path}:{line_number}") from error


def _sqlite(path: Path) -> sqlite3.Connection:
    if not path.is_file():
        raise FileNotFoundError(path)
    connection = sqlite3.connect(
        f"file:{path.as_posix()}?mode=ro&immutable=1", uri=True
    )
    if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
        connection.close()
        raise RuntimeError(f"SQLite integrity check failed: {path}")
    return connection


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{label} must be numeric, found {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError(f"{label} is non-finite")
    return number


def _validate_plan(
    plan: Mapping[str, Any],
    *,
    expected_version: str,
    allow_sparse_slots: bool = False,
) -> None:
    if plan.get("schema") != "stable_audio_tools.spatial_scene_plan":
        raise RuntimeError(f"invalid ScenePlan schema: {plan.get('schema')}")
    if str(plan.get("schema_version")) != expected_version:
        raise RuntimeError(
            f"invalid ScenePlan version: {plan.get('schema_version')} != {expected_version}"
        )
    if not isinstance(plan.get("sample_id"), str) or not plan["sample_id"]:
        raise RuntimeError("ScenePlan has no sample_id")
    audio = plan.get("audio") or {}
    duration = _finite_number(audio.get("duration_sec"), "audio.duration_sec")
    if duration <= 0 or audio.get("spatial_format") != "foa":
        raise RuntimeError(f"invalid ScenePlan audio contract: {audio}")
    if audio.get("channel_layout") != "WYZX_ACN_SN3D":
        raise RuntimeError(f"invalid FOA layout: {audio.get('channel_layout')}")
    sources = ((plan.get("scene") or {}).get("sources") or [])
    if not 1 <= len(sources) <= 4:
        raise RuntimeError(f"ScenePlan requires 1..4 sources, found {len(sources)}")
    ids = [str(source.get("source_id") or "") for source in sources]
    if allow_sparse_slots:
        valid = {f"source_{index}" for index in range(4)}
        if len(ids) != len(set(ids)) or any(source_id not in valid for source_id in ids):
            raise RuntimeError(f"invalid persistent source slots: {ids}")
    elif ids != [f"source_{index}" for index in range(len(sources))]:
        raise RuntimeError(f"non-canonical source slots: {ids}")
    for source in sources:
        event = source.get("event") or {}
        if not event.get("label") or not event.get("category"):
            raise RuntimeError(f"source {source['source_id']} has no event semantics")
        activity = source.get("activity") or {}
        onset, offset = activity.get("onset_sec"), activity.get("offset_sec")
        if (onset is None) != (offset is None):
            raise RuntimeError("activity onset/offset must both be known or unknown")
        if onset is not None:
            onset = _finite_number(onset, "activity.onset_sec")
            offset = _finite_number(offset, "activity.offset_sec")
            if not 0 <= onset < offset <= duration + 1e-6:
                raise RuntimeError(f"activity outside clip: {onset}, {offset}, {duration}")
        motion = source.get("motion") or {}
        keyframes = motion.get("keyframes") or []
        if not keyframes:
            raise RuntimeError("motion has no keyframes")
        times = []
        for keyframe in keyframes:
            t_norm = _finite_number(keyframe.get("t_norm"), "motion.t_norm")
            if not 0 <= t_norm <= 1:
                raise RuntimeError(f"motion keyframe outside [0,1]: {t_norm}")
            times.append(t_norm)
            position = keyframe.get("position") or {}
            for key, lower, upper in (
                ("azimuth_deg", -180.0, 180.0),
                ("elevation_deg", -90.0, 90.0),
            ):
                if position.get(key) is not None:
                    number = _finite_number(position[key], f"position.{key}")
                    if not lower <= number <= upper:
                        raise RuntimeError(f"position.{key} outside range: {number}")
            if position.get("distance_m") is not None:
                if _finite_number(position["distance_m"], "position.distance_m") <= 0:
                    raise RuntimeError("distance must be positive")
        if times != sorted(times):
            raise RuntimeError(f"motion keyframes are not sorted: {times}")


def audit_scene_and_codec(
    scene_root: Path,
    codec_root: Path,
    *,
    expected_samples: int,
    codec_samples: int,
) -> dict[str, Any]:
    ready = _json(scene_root / "READY")
    if int(ready.get("samples", -1)) != expected_samples:
        raise RuntimeError("ScenePlan READY count mismatch")
    connection = _sqlite(scene_root / "index.sqlite")
    try:
        row = connection.execute(
            "SELECT COUNT(*), COUNT(DISTINCT sample_id), COUNT(DISTINCT audio_path) "
            "FROM samples"
        ).fetchone()
    finally:
        connection.close()
    if tuple(map(int, row)) != (expected_samples, expected_samples, expected_samples):
        raise RuntimeError(f"ScenePlan SQLite uniqueness mismatch: {row}")
    if (scene_root / "index.jsonl").is_file():
        if _sha256(scene_root / "index.jsonl") != ready.get("index_sha256"):
            raise RuntimeError("ScenePlan portable-index checksum mismatch")

    rng = random.Random(20260808)
    reservoir: list[dict[str, Any]] = []
    count = 0
    datasets = Counter()
    annotated = 0
    speech = 0
    dynamic = 0
    for plan in _jsonl(iter(sorted((scene_root / "shards").glob("*.jsonl")))):
        _validate_plan(plan, expected_version=str(ready["schema_version"]))
        datasets[str(plan.get("dataset_id") or "unknown")] += 1
        sources = ((plan.get("scene") or {}).get("sources") or [])
        annotated += int(
            all((source.get("activity") or {}).get("onset_sec") is not None for source in sources)
        )
        speech += int(any(is_speech_source(source) for source in sources))
        dynamic += int(any(len((source.get("motion") or {}).get("keyframes") or []) > 1 for source in sources))
        count += 1
        if count % 100_000 == 0:
            print(
                f"[spatial-cot-audit] ScenePlans checked: {count:,}",
                file=sys.stderr,
                flush=True,
            )
        if len(reservoir) < codec_samples:
            reservoir.append(dict(plan))
        else:
            replacement = rng.randrange(count)
            if replacement < codec_samples:
                reservoir[replacement] = dict(plan)
    if count != expected_samples:
        raise RuntimeError(f"ScenePlan JSONL count {count} != {expected_samples}")

    codec = SpatialPlanCodec(codec_root)
    if codec.codec_name != "spatial_plan_codec_v2" or not codec.has_gain:
        raise RuntimeError("production codec must be gain-aware spatial_plan_codec_v2")
    token_lengths = []
    fsm_checked = 0
    for index, plan in enumerate(reservoir):
        encoded = codec.encode(plan, max_tokens=1024)
        decoded = codec.decode(encoded["input_ids"])
        repeated = codec.encode(decoded, max_tokens=1024)
        if not torch.equal(encoded["input_ids"], repeated["input_ids"]):
            raise RuntimeError(f"codec id round-trip failed: {plan['sample_id']}")
        compiled = compile_source_tracks(decoded, num_frames=432, max_sources=4)
        tracks = compiled["tracks"]
        if tuple(tracks.shape) != (4, 8, 432) or not torch.isfinite(tracks).all():
            raise RuntimeError(f"compiled source-track QC failed: {plan['sample_id']}")
        ids = encoded["input_ids"].tolist()
        token_lengths.append(len(ids))
        if index < min(64, len(reservoir)):
            for position, token in enumerate(ids):
                allowed = codec.allowed_next_ids(
                    ids[:position], min_sources=1, max_sources=4
                )
                if token not in allowed:
                    raise RuntimeError(
                        f"codec FSM rejected ground truth at {plan['sample_id']}:{position}"
                    )
            if codec.allowed_next_ids(ids, min_sources=1, max_sources=4):
                raise RuntimeError("codec FSM did not accept a complete plan")
            fsm_checked += 1
    if not token_lengths:
        raise RuntimeError("codec audit received no ScenePlans")
    return {
        "samples": count,
        "datasets": dict(datasets),
        "fully_annotated_activity_samples": annotated,
        "speech_samples": speech,
        "dynamic_samples": dynamic,
        "codec_name": codec.codec_name,
        "codec_fingerprint": codec.fingerprint,
        "codec_samples": len(reservoir),
        "codec_fsm_samples": fsm_checked,
        "token_length_min": min(token_lengths),
        "token_length_max": max(token_lengths),
        "token_length_mean": sum(token_lengths) / len(token_lengths),
    }


def audit_catalog(catalog_root: Path, spec: Mapping[str, Any]) -> dict[str, Any]:
    qc = spec.get("quality_control") or {}
    if (
        qc.get("acoustic_speech_screening") != "disabled"
        or qc.get("hidden_unlabelled_background_speech_policy")
        != "accepted_as_rare_metadata_noise"
    ):
        raise RuntimeError("speech-screening risk policy is not explicit")
    ready = _json(catalog_root / "READY")
    if int(ready.get("schema_version", -1)) != 5:
        raise RuntimeError("production catalog must use schema version 5")
    if ready.get("build_spec_fingerprint") != stable_digest(spec, size=32):
        raise RuntimeError("production catalog was built from a different build spec")
    expected_counts = {
        split: int(config["families"]) for split, config in spec["splits"].items()
    }
    if ready.get("counts") != expected_counts:
        raise RuntimeError(f"catalog counts mismatch: {ready.get('counts')}")

    source_db = _sqlite(catalog_root / "sources/index.sqlite")
    try:
        source_rows = source_db.execute(
            "SELECT template_id, asset_id, split, kind FROM sources"
        ).fetchall()
        leaked = source_db.execute(
            "SELECT COUNT(*) FROM (SELECT asset_id FROM sources GROUP BY asset_id "
            "HAVING COUNT(DISTINCT split) > 1)"
        ).fetchone()[0]
    finally:
        source_db.close()
    if leaked:
        raise RuntimeError(f"{leaked} dry assets cross catalog splits")
    source_info = {
        str(template): (str(asset), str(split), str(kind))
        for template, asset, split, kind in source_rows
    }
    if len(source_info) != int(ready["source_templates"]):
        raise RuntimeError("source catalog count mismatch")

    # Do not trust the compact SQLite kind alone: independently derive speech
    # from every persisted source template.  A transcript always wins over an
    # imprecise event label, preventing speech from entering a donor pool.
    source_json_count = 0
    source_semantics: dict[str, str] = {}
    for row in _jsonl(iter(sorted((catalog_root / "sources/shards").glob("*.jsonl")))):
        template_id = str(row["template_id"])
        if template_id not in source_info:
            raise RuntimeError(f"source JSONL missing from SQLite: {template_id}")
        event = row["source"].get("event") or {}
        if is_silent_source(row["source"]):
            raise RuntimeError(
                f"catalog contains an explicitly silent source: {template_id}"
            )
        derived_kind = "speech" if is_speech_source(row["source"]) else "sound_music"
        if derived_kind == "speech":
            derived_semantic = "speech"
        else:
            category = " ".join(
                str(event.get("category") or "sound").strip().lower().split()
            )
            label = " ".join(
                str(event.get("label") or category).strip().lower().split()
            )
            derived_semantic = f"{category}\x1f{label}"
        asset, split, indexed_kind = source_info[template_id]
        if (
            str(row["asset_id"]) != asset
            or str(row["split"]) != split
            or str(row["kind"]) != indexed_kind
            or derived_kind != indexed_kind
            or str(row.get("semantic_key") or "") != derived_semantic
        ):
            raise RuntimeError(
                f"source kind/index mismatch: {template_id} "
                f"json={row.get('kind')} indexed={indexed_kind} derived={derived_kind}"
            )
        source_json_count += 1
        if source_json_count % 100_000 == 0:
            print(
                f"[spatial-cot-audit] source templates checked: "
                f"{source_json_count:,}",
                file=sys.stderr,
                flush=True,
            )
        source_semantics[template_id] = derived_semantic
    if source_json_count != len(source_info):
        raise RuntimeError(
            f"source JSONL count {source_json_count} != SQLite count {len(source_info)}"
        )

    family_db = _sqlite(catalog_root / "families/index.sqlite")
    try:
        for split, expected in expected_counts.items():
            row = family_db.execute(
                "SELECT COUNT(*), MIN(family_rank), MAX(family_rank), "
                "COUNT(DISTINCT family_id) FROM families WHERE split=?",
                (split,),
            ).fetchone()
            if tuple(map(int, row)) != (expected, 0, expected - 1, expected):
                raise RuntimeError(f"family SQLite mismatch for {split}: {row}")
    finally:
        family_db.close()

    next_rank = {split: 0 for split in expected_counts}
    scanned = Counter()
    speech_donor_families = 0
    for row in _jsonl(iter(sorted((catalog_root / "families/shards").glob("*.jsonl")))):
        split = str(row["split"])
        rank = int(row["family_rank"])
        if rank != next_rank[split]:
            raise RuntimeError(f"non-contiguous catalog rank {split}:{rank}")
        next_rank[split] += 1
        if int(row["work_shard"]) != rank // int(
            spec["sharding"]["families_per_latent_shard"]
        ):
            raise RuntimeError(f"wrong work shard: {row['family_id']}")
        refs = list(row["base_source_refs"]) + list(row["donor_source_refs"])
        try:
            info = [source_info[str(reference)] for reference in refs]
        except KeyError as error:
            raise RuntimeError(f"family references unknown source: {row['family_id']}") from error
        if any(source_split != split for _, source_split, _ in info):
            raise RuntimeError(f"family crosses source split: {row['family_id']}")
        assets = [asset for asset, _, _ in info]
        if len(assets) != len(set(assets)):
            raise RuntimeError(f"family repeats one dry asset: {row['family_id']}")
        base_info = info[: len(row["base_source_refs"])]
        donor_info = info[len(row["base_source_refs"]) :]
        if sum(kind == "speech" for _, _, kind in base_info) > 1:
            raise RuntimeError(f"speech+speech base family: {row['family_id']}")
        base_speech = sum(kind == "speech" for _, _, kind in base_info)
        donor_speech = sum(kind == "speech" for _, _, kind in donor_info)
        if donor_speech > 1 or (base_speech and donor_speech):
            raise RuntimeError(
                f"speech donor can create speech+speech: {row['family_id']}"
            )
        declared_donor_kinds = list(row.get("donor_source_kinds") or [])
        actual_donor_kinds = [kind for _, _, kind in donor_info]
        if declared_donor_kinds != actual_donor_kinds:
            raise RuntimeError(f"donor kind contract mismatch: {row['family_id']}")
        if row.get("speech_donor_policy") != "at_most_one_if_base_has_no_speech":
            raise RuntimeError(f"missing speech donor policy: {row['family_id']}")
        speech_donor_families += int(donor_speech == 1)
        semantics = [source_semantics[str(reference)] for reference in refs]
        if len(semantics) != len(set(semantics)):
            raise RuntimeError(
                f"family contains planner-visible duplicate semantics: {row['family_id']}"
            )
        if row.get("unique_semantic_sources_per_family") is not True:
            raise RuntimeError(f"family lacks semantic uniqueness contract: {row['family_id']}")
        active = len(base_info)
        for operation in row["edit_types"]:
            if operation == "add_source":
                active += 1
            elif operation == "remove_source":
                active -= 1
            if not 1 <= active <= int(spec["family_distribution"]["max_sources"]):
                raise RuntimeError(f"invalid edit source count: {row['family_id']}")
        scanned[split] += 1
        total_scanned = sum(scanned.values())
        if total_scanned % 100_000 == 0:
            print(
                f"[spatial-cot-audit] family specs checked: {total_scanned:,}",
                file=sys.stderr,
                flush=True,
            )
    if dict(scanned) != expected_counts:
        raise RuntimeError(f"catalog JSONL counts mismatch: {dict(scanned)}")

    for name, expected in (
        ("smoke", int(spec["sharding"]["smoke_families"])),
        ("pilot", int(spec["sharding"]["pilot_families"])),
    ):
        rows = list(_jsonl(iter([catalog_root / "views" / f"{name}.jsonl"])))
        if len(rows) != expected or [int(row["family_rank"]) for row in rows] != list(
            range(expected)
        ):
            raise RuntimeError(f"catalog {name} view is not the nested first-{expected}")
    return {
        "source_templates": len(source_info),
        "source_templates_semantically_checked": source_json_count,
        "build_spec_fingerprint": ready["build_spec_fingerprint"],
        "families": dict(scanned),
        "source_split_leaks": 0,
        "speech_policy": (
            "at_most_one_per_state; one speech donor allowed only for a "
            "no-speech base"
        ),
        "speech_detection_policy": (
            "metadata_only; hidden_unlabelled_background_speech_risk_accepted"
        ),
        "speech_donor_families": speech_donor_families,
        "semantic_pairing_policy": "all_base_and_donor_semantic_keys_are_distinct",
        "smoke_nested": int(spec["sharding"]["smoke_families"]),
        "pilot_nested": int(spec["sharding"]["pilot_families"]),
    }


def _metadata_rows(path: Path) -> Iterator[dict[str, Any]]:
    yield from _jsonl(iter([path]))


def _planner_semantic_key(source: Mapping[str, Any]) -> str:
    if is_speech_source(source):
        return "speech"
    event = source.get("event") or {}
    category = " ".join(
        str(event.get("category") or "sound").strip().lower().split()
    )
    label = " ".join(
        str(event.get("label") or category).strip().lower().split()
    )
    return f"{category}\x1f{label}"


def audit_view(
    recipe_split_root: Path,
    latent_root: Path,
    codec_root: Path,
    spec: Mapping[str, Any],
    *,
    split: str,
    expected_families: int,
    runtime_samples: int,
) -> dict[str, Any]:
    """Audit one finalized family store and its persistent recipe shards.

    ``recipe_split_root`` is the directory containing ``work-00000`` and
    ``latent_root`` is the finalized split store.  Keeping these roots
    explicit makes the same audit usable for nested smoke/pilot views and for
    the production train/validation/test layout without fabricating a common
    parent directory.
    """

    per_shard = int(spec["sharding"]["families_per_latent_shard"])
    expected_work_shards = (expected_families + per_shard - 1) // per_shard
    retain_rendered = bool(spec["splits"][split]["retain_rendered_foa"])
    expected_profile = str(
        spec["storage"][
            "eval_render_profile" if retain_rendered else "train_render_profile"
        ]
    )
    signal_qc = spec["quality_control"]
    recipe_ids: set[str] = set()
    recipe_count = 0
    edit_counts = Counter()
    transcript_quality_assets = Counter()
    speech_turns = 0
    family_dry_asset_occurrences = 0
    globally_seen_dry_assets: set[str] = set()
    playback_rms_min = math.inf
    speech_introduction_edits = 0
    spatial_move_edits = 0
    independently_mixed_states = 0
    source_substitutions = 0
    for work_shard in range(expected_work_shards):
        recipe_root = recipe_split_root / f"work-{work_shard:05d}"
        recipe_ready = _json(recipe_root / "READY")
        shard = recipe_root / "shards" / f"recipes-{split}-{work_shard:05d}.jsonl"
        local_count = 0
        for family in _jsonl(iter([shard])):
            validate_edit_family(family, require_outputs=False)
            if str(family.get("schema_version")) != RECIPE_VERSION:
                raise RuntimeError(f"stale recipe schema: {family['family_id']}")
            family_id = str(family["family_id"])
            if family_id in recipe_ids:
                raise RuntimeError(f"duplicate recipe family: {family_id}")
            recipe_ids.add(family_id)
            if int(family["family_rank"]) >= expected_families:
                raise RuntimeError(f"recipe outside nested view: {family_id}")
            path_semantics: dict[str, str] = {}
            path_transcript_quality: dict[str, str] = {}
            for index, recipe in enumerate(family["recipes"]):
                render_contract = recipe.get("render_contract") or {}
                if (
                    render_contract.get("state_input") != STATE_RENDER_INPUT
                    or render_contract.get("uses_previous_foa") is not False
                    or render_contract.get("independent_state_mix") is not True
                ):
                    raise RuntimeError(
                        f"state is not dry-source rendered: {family_id}:{index}"
                    )
                independently_mixed_states += 1
                plan = recipe["scene_plan"]
                _validate_plan(
                    plan,
                    expected_version="1.2-paired",
                    allow_sparse_slots=True,
                )
                if index:
                    expected_diff = diff_scene_plans(
                        family["recipes"][index - 1]["scene_plan"], plan
                    )
                else:
                    expected_diff = diff_scene_plans(None, plan)
                if expected_diff != family["turns"][index]["diff"]:
                    raise RuntimeError(f"recipe diff mismatch: {family_id}:{index}")
                if any(
                    "render." in field
                    for changed in expected_diff.get("changed") or []
                    for field in changed.get("fields") or []
                ):
                    raise RuntimeError(f"renderer artifact leaked into ScenePlan diff: {family_id}")
                speech_count = sum(is_speech_source(source) for source in recipe["sources"])
                if speech_count > 1:
                    raise RuntimeError(f"speech+speech recipe turn: {family_id}:{index}")
                speech_turns += int(speech_count == 1)
                if index:
                    previous_recipe = family["recipes"][index - 1]
                    previous_speech = sum(
                        is_speech_source(source)
                        for source in previous_recipe["sources"]
                    )
                    edit_type = str(recipe["edit"]["type"])
                    if (
                        previous_speech == 0
                        and speech_count == 1
                        and edit_type in {"add_source", "replace_source"}
                    ):
                        speech_introduction_edits += 1
                    if edit_type == "move_source":
                        target_id = str(recipe["edit"]["target_source_id"])
                        before_by_id = {
                            str(source["source_id"]): source
                            for source in previous_recipe["sources"]
                        }
                        after_by_id = {
                            str(source["source_id"]): source
                            for source in recipe["sources"]
                        }
                        if (
                            target_id not in before_by_id
                            or target_id not in after_by_id
                            or before_by_id[target_id]["dry_audio"]
                            != after_by_id[target_id]["dry_audio"]
                            or before_by_id[target_id]["motion"]
                            == after_by_id[target_id]["motion"]
                        ):
                            raise RuntimeError(
                                f"invalid same-source spatial move: {family_id}:{index}"
                            )
                        spatial_move_edits += 1
                for source in recipe["sources"]:
                    dry = source["dry_audio"]
                    dry_path = str(Path(dry["path"]).resolve())
                    if not Path(dry_path).is_file():
                        raise FileNotFoundError(dry_path)
                    semantic = _planner_semantic_key(source)
                    previous_semantic = path_semantics.setdefault(dry_path, semantic)
                    if previous_semantic != semantic:
                        raise RuntimeError(
                            f"one dry asset changes semantics inside family: {family_id}"
                        )
                    quality = str(
                        (source.get("content") or {}).get("transcript_quality")
                        or "missing"
                    )
                    previous_quality = path_transcript_quality.setdefault(
                        dry_path, quality
                    )
                    if previous_quality != quality:
                        raise RuntimeError(
                            f"one dry asset changes transcript quality: {family_id}"
                        )
                    playback_rms_min = min(
                        playback_rms_min, float(source["playback"]["mono_rms"])
                    )
                edit_counts[str(recipe["edit"]["type"])] += 1
            lineage = family.get("source_lineage") or {}
            substitutions = list(lineage.get("source_substitutions") or [])
            if substitutions:
                if lineage.get("substitution_policy") != (
                    "deterministic_same_split_same_kind_after_dry_playback_qc"
                ):
                    raise RuntimeError(
                        f"invalid source substitution policy: {family_id}"
                    )
                requested = {
                    "base": list(lineage.get("requested_base_source_refs") or []),
                    "donor": list(lineage.get("requested_donor_source_refs") or []),
                }
                effective = {
                    "base": list(lineage.get("base_source_refs") or []),
                    "donor": list(lineage.get("donor_source_refs") or []),
                }
                if any(
                    len(requested[branch]) != len(effective[branch])
                    for branch in ("base", "donor")
                ):
                    raise RuntimeError(
                        f"source substitution changes family arity: {family_id}"
                    )
                seen_slots = set()
                for substitution in substitutions:
                    slot = str(substitution.get("slot") or "")
                    match = re.fullmatch(r"(base|donor)_(\d+)", slot)
                    if match is None or slot in seen_slots:
                        raise RuntimeError(
                            f"invalid source substitution slot: {family_id}:{slot}"
                        )
                    seen_slots.add(slot)
                    branch, index = match.group(1), int(match.group(2))
                    if index >= len(effective[branch]):
                        raise RuntimeError(
                            f"source substitution slot out of range: {family_id}:{slot}"
                        )
                    original = str(substitution.get("original_template_ref") or "")
                    replacement = str(
                        substitution.get("replacement_template_ref") or ""
                    )
                    if (
                        original != requested[branch][index]
                        or replacement != effective[branch][index]
                        or original == replacement
                        or substitution.get("reason")
                        != "dry_playback_below_minimum_input_rms"
                        or not math.isclose(
                            float(substitution.get("minimum_input_rms", -1.0)),
                            float(
                                signal_qc["source_loudness"]["minimum_input_rms"]
                            ),
                        )
                    ):
                        raise RuntimeError(
                            f"invalid source substitution lineage: {family_id}:{slot}"
                        )
                    rejected = list(substitution.get("rejected_candidates") or [])
                    if not rejected or any(
                        float(item.get("mono_rms", math.inf))
                        >= float(signal_qc["source_loudness"]["minimum_input_rms"])
                        for item in rejected
                    ):
                        raise RuntimeError(
                            f"invalid rejected-source evidence: {family_id}:{slot}"
                        )
                source_substitutions += len(substitutions)
            expected_dry_assets = len(lineage.get("base_source_refs") or []) + len(
                lineage.get("donor_source_refs") or []
            )
            if len(path_semantics) != expected_dry_assets:
                raise RuntimeError(
                    f"donor/base consumption mismatch: {family_id} "
                    f"actual_dry={len(path_semantics)} expected={expected_dry_assets}"
                )
            if len(set(path_semantics.values())) != len(path_semantics):
                raise RuntimeError(
                    f"planner-visible duplicate semantics in recipe family: {family_id}"
                )
            family_dry_asset_occurrences += len(path_semantics)
            globally_seen_dry_assets.update(path_semantics)
            transcript_quality_assets.update(path_transcript_quality.values())
            local_count += 1
            recipe_count += 1
        if local_count != int(recipe_ready["families"]):
            raise RuntimeError(f"recipe shard count mismatch: {work_shard}")
        if (work_shard + 1) % 100 == 0 or work_shard + 1 == expected_work_shards:
            print(
                f"[spatial-cot-audit] recipe shards checked: "
                f"{work_shard + 1:,}/{expected_work_shards:,}; "
                f"families={recipe_count:,}",
                file=sys.stderr,
                flush=True,
            )
    if recipe_count != expected_families:
        raise RuntimeError(f"recipe family count {recipe_count} != {expected_families}")

    ready = _json(latent_root / "READY")
    if int(ready.get("families", -1)) != expected_families:
        raise RuntimeError("latent READY family count mismatch")
    if int(ready.get("states", -1)) != expected_families * int(
        spec["splits"][split]["states_per_family"]
    ):
        raise RuntimeError("latent READY state count mismatch")
    if _sha256(latent_root / "index.jsonl") != ready.get("index_sha256"):
        raise RuntimeError("latent portable-index checksum mismatch")
    if _sha256(latent_root / "index.sqlite") != ready.get("sqlite_sha256"):
        raise RuntimeError("latent SQLite checksum mismatch")
    connection = _sqlite(latent_root / "index.sqlite")
    try:
        count, minimum, maximum = connection.execute(
            "SELECT COUNT(*), MIN(family_rank), MAX(family_rank) FROM families"
        ).fetchone()
    finally:
        connection.close()
    if tuple(map(int, (count, minimum, maximum))) != (
        expected_families,
        0,
        expected_families - 1,
    ):
        raise RuntimeError("latent SQLite ranks are not contiguous")

    latent_families = 0
    metadata_families = 0
    recipe_hash_cache: dict[Path, str] = {}
    for work_shard in range(expected_work_shards):
        done = _json(latent_root / "work_done" / f"work-{work_shard:05d}.json")
        tensor_path = Path(done["tensor_shard"])
        index_path = latent_root / "shard_indexes" / f"families-{work_shard:05d}.jsonl"
        metadata_path = latent_root / "metadata" / f"families-{work_shard:05d}.jsonl"
        if _sha256(tensor_path) != done["tensor_sha256"]:
            raise RuntimeError(f"tensor shard checksum mismatch: {tensor_path}")
        if _sha256(index_path) != done["index_sha256"]:
            raise RuntimeError(f"index shard checksum mismatch: {index_path}")
        if _sha256(metadata_path) != done["metadata_sha256"]:
            raise RuntimeError(f"metadata shard checksum mismatch: {metadata_path}")
        with safe_open(str(tensor_path), framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
            if len(keys) != int(done["families"]):
                raise RuntimeError(f"safetensors key count mismatch: {tensor_path}")
            for key in keys:
                tensor = handle.get_tensor(key)
                if tuple(tensor.shape) != (4, 64, 432) or tensor.dtype != torch.float16:
                    raise RuntimeError(f"invalid latent tensor {key}: {tensor.shape}")
                if not torch.isfinite(tensor).all():
                    raise RuntimeError(f"non-finite latent tensor: {key}")
                latent_families += 1
        for record in _metadata_rows(metadata_path):
            if int(record.get("schema_version", -1)) != 2:
                raise RuntimeError(f"stale family metadata: {record.get('family_id')}")
            family_id = str(record["family_id"])
            if family_id not in recipe_ids:
                raise RuntimeError(f"latent metadata lacks persistent recipe: {family_id}")
            recipe_jsonl = Path(record["recipe_jsonl_ref"])
            if not recipe_jsonl.is_file() or record.get("recipe_family_id") != family_id:
                raise RuntimeError(f"invalid recipe reference: {family_id}")
            actual_recipe_hash = recipe_hash_cache.get(recipe_jsonl)
            if actual_recipe_hash is None:
                actual_recipe_hash = _sha256(recipe_jsonl)
                recipe_hash_cache[recipe_jsonl] = actual_recipe_hash
            if actual_recipe_hash != record.get("recipe_jsonl_sha256"):
                raise RuntimeError(f"persistent recipe checksum mismatch: {family_id}")
            provenance = record.get("render_provenance") or {}
            if not float(provenance.get("family_master_gain_linear") or 0.0) > 0:
                raise RuntimeError(f"missing family master gain: {family_id}")
            if (
                provenance.get("state_input") != STATE_RENDER_INPUT
                or provenance.get("uses_previous_foa") is not False
                or provenance.get("independent_state_mix") is not True
            ):
                raise RuntimeError(
                    f"metadata lacks independent dry-source provenance: {family_id}"
                )
            source_loudness = provenance.get("source_loudness") or {}
            if source_loudness != signal_qc.get("source_loudness"):
                raise RuntimeError(
                    f"metadata source loudness contract mismatch: {family_id}"
                )
            if (
                not math.isclose(
                    float(provenance.get("minimum_rms", -1.0)),
                    float(signal_qc["minimum_rms"]),
                )
                or not math.isclose(
                    float(
                        provenance.get("minimum_active_100ms_fraction", -1.0)
                    ),
                    float(signal_qc["minimum_active_100ms_fraction"]),
                )
                or not math.isclose(
                    float(provenance.get("active_frame_rms_threshold", -1.0)),
                    float(signal_qc["active_frame_rms_threshold"]),
                )
                or float(provenance.get("state_rms_min", -1.0))
                < float(signal_qc["minimum_rms"])
                or float(
                    provenance.get("state_active_100ms_fraction_min", -1.0)
                )
                < float(signal_qc["minimum_active_100ms_fraction"])
            ):
                raise RuntimeError(
                    f"metadata audibility provenance mismatch: {family_id}"
                )
            if str(provenance.get("storage_profile") or "") != expected_profile:
                raise RuntimeError(
                    f"metadata storage profile mismatch: {family_id}: "
                    f"{provenance.get('storage_profile')} != {expected_profile}"
                )
            if provenance.get("rendered_foa_retained") is not retain_rendered:
                raise RuntimeError(f"FOA retention mismatch: {family_id}")
            if provenance.get("source_tracks_retained") is not retain_rendered:
                raise RuntimeError(f"source-track retention mismatch: {family_id}")
            turns = record.get("turns") or []
            if len(turns) != 4:
                raise RuntimeError(f"family metadata turn count mismatch: {family_id}")
            for index, turn in enumerate(turns):
                turn_render = turn["after"].get("render_contract") or {}
                if (
                    turn_render.get("state_input") != STATE_RENDER_INPUT
                    or turn_render.get("uses_previous_foa") is not False
                ):
                    raise RuntimeError(
                        f"turn lacks independent render contract: {family_id}:{index}"
                    )
                signal_stats = turn["after"].get("signal_stats") or {}
                if (
                    float(signal_stats.get("rms", -1.0))
                    < float(signal_qc["minimum_rms"])
                    or float(signal_stats.get("active_100ms_fraction", -1.0))
                    < float(signal_qc["minimum_active_100ms_fraction"])
                    or float(signal_stats.get("peak", math.inf))
                    > float(signal_qc["max_abs_peak"]) + 1.0e-6
                    or not math.isclose(
                        float(
                            signal_stats.get("active_frame_rms_threshold", -1.0)
                        ),
                        float(signal_qc["active_frame_rms_threshold"]),
                    )
                ):
                    raise RuntimeError(
                        f"turn audibility stats failed: {family_id}:{index}"
                    )
                after_path = turn["after"].get("audio_path")
                before_path = turn["before"].get("audio_path")
                if retain_rendered:
                    if (
                        not after_path
                        or turn.get("audio_path") != after_path
                        or not Path(after_path).is_file()
                    ):
                        raise RuntimeError(
                            f"retained target FOA is missing or inconsistent: "
                            f"{family_id}:{index}"
                        )
                    if index:
                        if not before_path or not Path(before_path).is_file():
                            raise RuntimeError(
                                f"retained context FOA is missing: {family_id}:{index}"
                            )
                    elif before_path is not None:
                        raise RuntimeError(
                            f"initial retained turn has context FOA: {family_id}"
                        )
                elif (
                    turn.get("audio_path") is not None
                    or after_path is not None
                    or before_path is not None
                ):
                    raise RuntimeError(
                        f"cleaned train metadata retained stale audio path: "
                        f"{family_id}:{index}"
                    )
                if int(turn["after"]["latent_state_index"]) != index:
                    raise RuntimeError(f"target latent index mismatch: {family_id}")
                expected_previous = index - 1 if index else None
                if turn["before"].get("latent_state_index") != expected_previous:
                    raise RuntimeError(f"previous latent index mismatch: {family_id}")
                if not turn["after"].get("foa_sha256"):
                    raise RuntimeError(f"missing recoverability FOA hash: {family_id}")
                for reference in turn["after"].get("source_track_refs") or []:
                    path = reference.get("path")
                    if retain_rendered:
                        if (
                            not path
                            or reference.get("retained") is not True
                            or not Path(path).is_file()
                        ):
                            raise RuntimeError(
                                f"retained source track is missing: {family_id}:{index}"
                            )
                    elif path is not None or reference.get("retained") is not False:
                        raise RuntimeError(f"stale source-track path: {family_id}")
            metadata_families += 1
        if (work_shard + 1) % 100 == 0 or work_shard + 1 == expected_work_shards:
            print(
                f"[spatial-cot-audit] latent shards checked: "
                f"{work_shard + 1:,}/{expected_work_shards:,}; "
                f"tensors={latent_families:,}; metadata={metadata_families:,}",
                file=sys.stderr,
                flush=True,
            )
    if latent_families != expected_families or metadata_families != expected_families:
        raise RuntimeError(
            f"latent/metadata counts mismatch: {latent_families}/{metadata_families}"
        )

    codec = SpatialPlanCodec(codec_root)
    provider = SpatialFamilyMetadata(
        {
            "codec_path": str(codec_root),
            "max_tokens": 1024,
            "max_sources": 4,
            "max_distance_m": 20.0,
        }
    )
    dataset = SpatialFamilyDataset(
        [
            {
                "path": str(latent_root),
                "custom_metadata_fn": provider,
            }
        ],
        require_ready=True,
        max_open_shards=2,
    )
    runtime_count = min(max(1, runtime_samples), len(dataset))
    ranks = sorted(
        set(
            round(index * (len(dataset) - 1) / max(1, runtime_count - 1))
            for index in range(runtime_count)
        )
    )
    runtime_turns = 0
    for rank in ranks:
        family_latent, info = dataset[rank]
        if tuple(family_latent.shape) != (4, 64, 432):
            raise RuntimeError(f"runtime family shape mismatch: rank={rank}")
        turns = info.get("family_turn_metadata") or []
        if len(turns) != 4:
            raise RuntimeError(f"runtime metadata turn mismatch: rank={rank}")
        for index, turn in enumerate(turns):
            tokens = turn["spatial_plan_tokens"]["input_ids"]
            if len(tokens) > 1024:
                raise RuntimeError(f"runtime plan exceeds max tokens: rank={rank}")
            decoded = codec.decode(tokens)
            repeated = codec.encode(decoded, max_tokens=1024)["input_ids"]
            if not torch.equal(tokens, repeated):
                raise RuntimeError(f"runtime plan round-trip failed: rank={rank}")
            if tuple(turn["source_tracks"].shape) != (32, 432):
                raise RuntimeError(f"runtime source tracks mismatch: rank={rank}")
            if not torch.isfinite(turn["source_tracks"]).all():
                raise RuntimeError(f"runtime source tracks are non-finite: rank={rank}")
            if index == 0:
                if turn["previous_foa_present"] or torch.count_nonzero(turn["previous_foa"]):
                    raise RuntimeError("initial turn previous-FOA contract failed")
            else:
                if not turn["previous_foa_present"] or not torch.equal(
                    turn["previous_foa"], family_latent[index - 1].float()
                ):
                    raise RuntimeError("previous-FOA alignment failed")
            runtime_turns += 1
    dataset.roots[0].close()
    return {
        "families": expected_families,
        "states": expected_families * 4,
        "work_shards": expected_work_shards,
        "recipe_schema_version": RECIPE_VERSION,
        "edit_counts": dict(edit_counts),
        "speech_turns": speech_turns,
        "speech_introduction_edits": speech_introduction_edits,
        "same_source_spatial_move_edits": spatial_move_edits,
        "source_substitutions": source_substitutions,
        "independently_mixed_states_verified": independently_mixed_states,
        "family_dry_asset_occurrences_verified": family_dry_asset_occurrences,
        "globally_unique_dry_assets_verified": len(globally_seen_dry_assets),
        "transcript_quality_asset_occurrences": dict(transcript_quality_assets),
        "minimum_dry_playback_rms": playback_rms_min,
        "latent_families_verified": latent_families,
        "metadata_families_verified": metadata_families,
        "runtime_families_verified": len(ranks),
        "runtime_turns_verified": runtime_turns,
        "rendered_foa_retained": retain_rendered,
        "storage_profile": expected_profile,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-spec", type=Path, required=True)
    parser.add_argument("--scene-plan-root", type=Path, required=True)
    parser.add_argument("--codec-root", type=Path, required=True)
    parser.add_argument("--catalog-root", type=Path, required=True)
    parser.add_argument("--scene-samples", type=int, default=1_018_957)
    parser.add_argument("--codec-samples", type=int, default=4096)
    parser.add_argument("--view-root", type=Path, default=None)
    parser.add_argument(
        "--recipe-split-root",
        type=Path,
        default=None,
        help="Production recipe split root containing work-* directories.",
    )
    parser.add_argument(
        "--latent-root",
        type=Path,
        default=None,
        help="Production finalized latent split root.",
    )
    parser.add_argument("--split", choices=("train", "validation", "test"), default="train")
    parser.add_argument("--expected-families", type=int, default=None)
    parser.add_argument("--runtime-samples", type=int, default=64)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument(
        "--static-audit-json",
        type=Path,
        default=None,
        help=(
            "Reuse a prior PASS of the immutable ScenePlan/codec/catalog layer; "
            "build-spec and codec fingerprints are revalidated."
        ),
    )
    args = parser.parse_args()
    if args.codec_samples <= 0 or args.scene_samples <= 0:
        raise SystemExit("sample counts must be positive")
    spec = _json(args.build_spec.expanduser().resolve())
    report: dict[str, Any] = {
        "schema": "stable_audio_tools.spatial_cot_audit",
        "schema_version": 1,
    }
    if args.static_audit_json is None:
        report["scene_and_codec"] = audit_scene_and_codec(
            args.scene_plan_root.expanduser().resolve(),
            args.codec_root.expanduser().resolve(),
            expected_samples=args.scene_samples,
            codec_samples=args.codec_samples,
        )
        report["catalog"] = audit_catalog(
            args.catalog_root.expanduser().resolve(), spec
        )
    else:
        static_path = args.static_audit_json.expanduser().resolve()
        static = _json(static_path)
        static_scene = static.get("scene_and_codec") or {}
        static_catalog = static.get("catalog") or {}
        current_codec = SpatialPlanCodec(args.codec_root.expanduser().resolve())
        if (
            static.get("schema") != "stable_audio_tools.spatial_cot_audit"
            or static.get("status") != "PASS"
            or int(static_scene.get("samples", -1)) != args.scene_samples
            or static_scene.get("codec_fingerprint") != current_codec.fingerprint
            or static_catalog.get("build_spec_fingerprint")
            != stable_digest(spec, size=32)
        ):
            raise SystemExit(f"static audit is stale or incompatible: {static_path}")
        catalog_ready = _json(
            args.catalog_root.expanduser().resolve() / "READY"
        )
        if catalog_ready.get("build_spec_fingerprint") != static_catalog.get(
            "build_spec_fingerprint"
        ):
            raise SystemExit("static audit catalog no longer matches catalog READY")
        report["scene_and_codec"] = static_scene
        report["catalog"] = static_catalog
        report["static_audit_reused"] = str(static_path)
    explicit_roots = args.recipe_split_root is not None or args.latent_root is not None
    if args.view_root is not None and explicit_roots:
        raise SystemExit("use either --view-root or explicit production roots, not both")
    if explicit_roots and (args.recipe_split_root is None or args.latent_root is None):
        raise SystemExit("--recipe-split-root and --latent-root must be provided together")
    if args.view_root is not None or explicit_roots:
        if args.expected_families is None or args.expected_families <= 0:
            raise SystemExit("split artifact audit requires positive --expected-families")
        if args.view_root is not None:
            view_root = args.view_root.expanduser().resolve()
            recipe_split_root = view_root / "recipes" / args.split
            latent_root = view_root / "latents" / args.split
        else:
            recipe_split_root = args.recipe_split_root.expanduser().resolve()
            latent_root = args.latent_root.expanduser().resolve()
        report["view"] = audit_view(
            recipe_split_root,
            latent_root,
            args.codec_root.expanduser().resolve(),
            spec,
            split=args.split,
            expected_families=args.expected_families,
            runtime_samples=args.runtime_samples,
        )
    report["status"] = "PASS"
    if args.output_json is not None:
        atomic_write_json(args.output_json.expanduser().resolve(), report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
