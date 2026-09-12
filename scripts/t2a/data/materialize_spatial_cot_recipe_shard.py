#!/usr/bin/env python3
"""Resolve one master-catalog work shard into render-ready edit recipes.

The master catalog is intentionally metadata-only.  This worker performs the
first expensive/recoverable step: exact dry-source resolution, deterministic
activity/motion/room construction, and AudioChat-style persistent-state edit
recipe construction.  It still does not render FOA or run the VAE.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import re
import sqlite3
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping

import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stable_audio_tools.data.source_assets import (  # noqa: E402
    ParquetAudioSourceIndex,
    resolve_scene_plan_assets,
)
from stable_audio_tools.data.spatial_edit_recipe import (  # noqa: E402
    build_edit_family,
    is_speech_source,
    stable_digest,
    validate_edit_family,
)
from stable_audio_tools.data.spatial_story import (  # noqa: E402
    SpatialStoryError,
    _metric_position,
)
from stable_audio_tools.data.t2a_artifacts import (  # noqa: E402
    atomic_write_json,
    atomic_write_jsonl,
)


SCHEMA = "stable_audio_tools.spatial_cot_recipe_shard"
VERSION = 1
_SILENT_PLAYBACK = re.compile(
    r"^dry playback window is effectively silent: rms=([^,]+), path=(.+)$"
)
_MAX_AUDIBLE_FALLBACKS = 256


class _CatalogReader:
    def __init__(self, root: Path, *, max_open_shards: int = 8):
        self.root = root
        self.max_open_shards = max(1, int(max_open_shards))
        self.source_db = sqlite3.connect(
            f"file:{(root / 'sources/index.sqlite').as_posix()}?mode=ro&immutable=1",
            uri=True,
        )
        self.family_db = sqlite3.connect(
            f"file:{(root / 'families/index.sqlite').as_posix()}?mode=ro&immutable=1",
            uri=True,
        )
        self.handles: OrderedDict[str, Any] = OrderedDict()

    def close(self) -> None:
        self.source_db.close()
        self.family_db.close()
        for handle in self.handles.values():
            handle.close()
        self.handles.clear()

    def _row(self, branch: str, shard: str, offset: int, length: int) -> dict[str, Any]:
        key = f"{branch}/{shard}"
        handle = self.handles.pop(key, None)
        if handle is None:
            handle = (self.root / branch / shard).open("rb")
        self.handles[key] = handle
        while len(self.handles) > self.max_open_shards:
            _, old = self.handles.popitem(last=False)
            old.close()
        handle.seek(int(offset))
        payload = handle.read(int(length))
        if len(payload) != int(length):
            raise RuntimeError(f"short catalog read: {key}@{offset}+{length}")
        return json.loads(payload)

    def families(self, split: str, work_shard: int) -> list[dict[str, Any]]:
        rows = self.family_db.execute(
            "SELECT shard, byte_offset, byte_length FROM families "
            "WHERE split=? AND work_shard=? ORDER BY family_rank",
            (str(split), int(work_shard)),
        ).fetchall()
        return [self._row("families", *row) for row in rows]

    def source(self, template_id: str) -> dict[str, Any]:
        row = self.source_db.execute(
            "SELECT shard, byte_offset, byte_length FROM sources WHERE template_id=?",
            (str(template_id),),
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown source template: {template_id}")
        return self._row("sources", *row)

    def source_candidates(
        self,
        *,
        split: str,
        kind: str,
        start_template_id: str,
        limit: int = 256,
    ) -> list[dict[str, Any]]:
        """Return a deterministic, wrapping slice of one split/kind pool."""

        requested = max(1, int(limit))
        columns = "template_id, shard, byte_offset, byte_length"
        rows = self.source_db.execute(
            f"SELECT {columns} FROM sources "
            "WHERE split=? AND kind=? AND template_id>=? "
            "ORDER BY template_id LIMIT ?",
            (str(split), str(kind), str(start_template_id), requested),
        ).fetchall()
        if len(rows) < requested:
            rows.extend(
                self.source_db.execute(
                    f"SELECT {columns} FROM sources "
                    "WHERE split=? AND kind=? AND template_id<? "
                    "ORDER BY template_id LIMIT ?",
                    (
                        str(split),
                        str(kind),
                        str(start_template_id),
                        requested - len(rows),
                    ),
                ).fetchall()
            )
        return [self._row("sources", row[1], row[2], row[3]) for row in rows]


def _position(position: Mapping[str, Any]) -> dict[str, Any]:
    azimuth, elevation, distance, quality = _metric_position(position)
    return {
        "azimuth_deg": round(float(azimuth), 3),
        "elevation_deg": round(max(-45.0, min(45.0, float(elevation))), 3),
        "distance_m": round(max(0.6, min(3.5, float(distance))), 3),
        "direction": position.get("direction"),
        "elevation": position.get("elevation"),
        "distance_label": position.get("distance_label"),
        "geometry_quality": (
            "exact" if quality >= 1.0 else "categorical_compiled_to_metric"
        ),
    }


def _normalize_source(
    source: Mapping[str, Any],
    *,
    rng: random.Random,
    duration_sec: float,
    force_motion: bool,
    source_id: str,
) -> dict[str, Any]:
    result = copy.deepcopy(dict(source))
    result["source_id"] = source_id
    path = Path(str((result.get("content") or {})["source_audio_path"]))
    audio = sf.info(str(path))
    dry_duration = min(duration_sec, float(audio.frames) / float(audio.samplerate))
    if dry_duration <= 0.05:
        raise ValueError(f"dry source is too short: {path}")

    # Exact activity windows, with non-zero onset represented often enough for
    # frame-accurate edit learning.  Speech keeps its full available utterance;
    # long sound/music clips receive a deterministic 2..10 second excerpt.
    if is_speech_source(result):
        window = dry_duration
    else:
        lower = min(2.0, dry_duration)
        window = rng.uniform(lower, dry_duration) if dry_duration > lower else dry_duration
    onset = rng.uniform(0.0, max(0.0, duration_sec - window))
    result["activity"] = {
        "onset_sec": round(onset, 6),
        "offset_sec": round(min(duration_sec, onset + window), 6),
        "quality": "paired_edit_exact",
    }

    original = sorted(
        ((result.get("motion") or {}).get("keyframes") or []),
        key=lambda item: float(item.get("t_norm", 0.0)),
    )
    start = _position((original[0].get("position") if original else {}) or {})
    if force_motion:
        delta = rng.choice((-1.0, 1.0)) * rng.uniform(50.0, 150.0)
        end = dict(start)
        end["azimuth_deg"] = round(
            ((float(start["azimuth_deg"]) + delta + 180.0) % 360.0) - 180.0,
            3,
        )
        end["elevation_deg"] = round(
            max(-40.0, min(40.0, float(start["elevation_deg"]) + rng.uniform(-20.0, 20.0))),
            3,
        )
        end["distance_m"] = round(
            max(0.6, min(3.5, float(start["distance_m"]) * rng.uniform(0.7, 1.3))),
            3,
        )
        keyframes = [
            {"t_norm": 0.0, "position": start},
            {"t_norm": 1.0, "position": end},
        ]
        motion_type = "linear"
    else:
        keyframes = [{"t_norm": 0.0, "position": start}]
        motion_type = "static"
    result["motion"] = {
        "type": motion_type,
        "time_basis": "activity_window",
        "interpolation": "linear_shortest_azimuth_arc",
        "timing_quality": "paired_edit_exact",
        "keyframes": keyframes,
    }
    result.setdefault("acoustics", {})["gain_db"] = round(rng.uniform(-3.0, 1.0), 3)
    return result


def _room(templates: list[Mapping[str, Any]], rng: random.Random) -> dict[str, Any]:
    inherited = copy.deepcopy(dict((templates[0].get("room") or {}) if templates else {}))
    inherited["dimensions_m"] = [
        round(rng.uniform(9.0, 12.0), 3),
        round(rng.uniform(9.0, 12.0), 3),
        round(rng.uniform(4.2, 5.2), 3),
    ]
    rt60 = inherited.get("rt60_s")
    inherited["rt60_s"] = round(
        max(0.15, min(1.2, float(rt60) if rt60 is not None else rng.uniform(0.25, 0.8))),
        4,
    )
    inherited["free_field"] = bool(inherited.get("free_field", False))
    inherited["quality"] = "paired_edit_deterministic_metric"
    return inherited


def _planner_caption(plan: Mapping[str, Any]) -> str:
    parts = []
    for source in ((plan.get("scene") or {}).get("sources") or []):
        event = source.get("event") or {}
        label = event.get("label") or event.get("category") or "sound"
        transcript = (source.get("content") or {}).get("transcript")
        if transcript and str(label).lower() == "speech":
            label = f"speech saying {json.dumps(str(transcript), ensure_ascii=False)}"
        activity = source["activity"]
        frames = source["motion"]["keyframes"]
        first = frames[0]["position"]
        if len(frames) == 1:
            spatial = (
                f"at azimuth {first['azimuth_deg']:.1f} degrees, elevation "
                f"{first['elevation_deg']:.1f} degrees, distance {first['distance_m']:.2f} meters"
            )
        else:
            last = frames[-1]["position"]
            spatial = (
                f"moving from ({first['azimuth_deg']:.1f}, {first['elevation_deg']:.1f}, "
                f"{first['distance_m']:.2f}m) to ({last['azimuth_deg']:.1f}, "
                f"{last['elevation_deg']:.1f}, {last['distance_m']:.2f}m)"
            )
        parts.append(
            f"{label}, active from {activity['onset_sec']:.2f}s to "
            f"{activity['offset_sec']:.2f}s, {spatial}"
        )
    return "Create a metric-3D FOA scene with " + "; ".join(parts) + "."


def _resolved_template_source(
    template: Mapping[str, Any],
    *,
    parquet: ParquetAudioSourceIndex,
    speech_cache: Path,
) -> dict[str, Any]:
    plan = {
        "scene": {"sources": [copy.deepcopy(template["source"])]},
    }
    resolved = resolve_scene_plan_assets(
        plan,
        parquet_index=parquet,
        speech_cache_root=speech_cache,
    )
    return resolved["scene"]["sources"][0]


def _build_one(
    spec: Mapping[str, Any],
    *,
    source_templates: list[Mapping[str, Any]],
    donor_templates: list[Mapping[str, Any]],
    base_source_refs: list[str] | None = None,
    donor_source_refs: list[str] | None = None,
    source_substitutions: list[Mapping[str, Any]] | None = None,
    parquet: ParquetAudioSourceIndex,
    speech_cache: Path,
    build_spec: Mapping[str, Any],
) -> dict[str, Any]:
    speech_bases = sum(
        str(item.get("kind") or "") == "speech" for item in source_templates
    )
    speech_donors = sum(
        str(item.get("kind") or "") == "speech" for item in donor_templates
    )
    if (
        speech_bases > 1
        or speech_donors > 1
        or (speech_bases and speech_donors)
    ):
        raise ValueError(
            "family catalog violates the at-most-one-speech policy: "
            f"base_speech={speech_bases}, donor_speech={speech_donors}"
        )
    declared_donor_kinds = list(spec.get("donor_source_kinds") or [])
    actual_donor_kinds = [str(item.get("kind") or "") for item in donor_templates]
    if declared_donor_kinds != actual_donor_kinds:
        raise ValueError(
            "catalog donor kinds disagree with referenced source templates: "
            f"declared={declared_donor_kinds} actual={actual_donor_kinds}"
        )
    rng = random.Random(int(spec["seed"]))
    duration = float(build_spec["audio"]["duration_sec"])
    base_resolved = [
        _resolved_template_source(item, parquet=parquet, speech_cache=speech_cache)
        for item in source_templates
    ]
    moving_slot = rng.randrange(len(base_resolved)) if spec.get("require_motion") else -1
    sources = [
        _normalize_source(
            source,
            rng=rng,
            duration_sec=duration,
            force_motion=index == moving_slot,
            source_id=f"s{index}",
        )
        for index, source in enumerate(base_resolved)
    ]
    plan = {
        "schema": "stable_audio_tools.spatial_scene_plan",
        "schema_version": "1.2-paired",
        "sample_id": str(spec["family_id"]),
        "audio": {
            "duration_sec": duration,
            "sample_rate": int(build_spec["audio"]["sample_rate"]),
            "spatial_format": "foa",
            "channel_layout": build_spec["audio"]["channel_layout"],
        },
        "mix": {"type": "single" if len(sources) == 1 else "mixture", "num_sources": len(sources)},
        "scene": {"room": _room(source_templates, rng), "sources": sources},
    }
    plan["caption"] = _planner_caption(plan)

    donors = []
    for donor_index, template in enumerate(donor_templates):
        source = _resolved_template_source(template, parquet=parquet, speech_cache=speech_cache)
        donors.append(
            _normalize_source(
                source,
                rng=rng,
                duration_sec=duration,
                force_motion=False,
                source_id=f"donor_{donor_index}",
            )
        )
    family = build_edit_family(
        plan,
        seed=int(spec["seed"]),
        turns=1 + int(build_spec["edit_distribution"]["transitions_per_family"]),
        sample_rate=int(build_spec["audio"]["sample_rate"]),
        max_sources=int(build_spec["family_distribution"]["max_sources"]),
        edit_types=list(spec["edit_types"]),
        donor_sources=donors,
        family_id=str(spec["family_id"]),
        source_loudness=build_spec["quality_control"]["source_loudness"],
    )
    family.update(
        {
            "split": spec["split"],
            "family_rank": int(spec["family_rank"]),
            "work_shard": int(spec["work_shard"]),
            "source_lineage": {
                "base_source_refs": list(
                    base_source_refs
                    if base_source_refs is not None
                    else spec["base_source_refs"]
                ),
                "donor_source_refs": list(
                    donor_source_refs
                    if donor_source_refs is not None
                    else spec["donor_source_refs"]
                ),
                "catalog_seed": int(spec["seed"]),
            },
        }
    )
    if source_substitutions:
        family["source_lineage"].update(
            {
                "requested_base_source_refs": list(spec["base_source_refs"]),
                "requested_donor_source_refs": list(spec["donor_source_refs"]),
                "source_substitutions": [dict(item) for item in source_substitutions],
                "substitution_policy": (
                    "deterministic_same_split_same_kind_after_dry_playback_qc"
                ),
            }
        )
    validate_edit_family(family, require_outputs=False)
    return family


def _template_audio_path(
    template: Mapping[str, Any],
    *,
    parquet: ParquetAudioSourceIndex,
    speech_cache: Path,
) -> str:
    source = _resolved_template_source(
        template,
        parquet=parquet,
        speech_cache=speech_cache,
    )
    path = (source.get("content") or {}).get("source_audio_path")
    if not path:
        raise RuntimeError(
            f"resolved source has no audio path: {template.get('template_id')}"
        )
    return str(Path(str(path)).expanduser().resolve())


def _build_with_audible_fallbacks(
    spec: Mapping[str, Any],
    *,
    source_templates: list[Mapping[str, Any]],
    donor_templates: list[Mapping[str, Any]],
    reader: _CatalogReader,
    parquet: ParquetAudioSourceIndex,
    speech_cache: Path,
    build_spec: Mapping[str, Any],
) -> dict[str, Any]:
    """Replace a rare quiet dry candidate without weakening audio QC.

    Source selection remains deterministic and split-disjoint. A replacement
    comes from the same source kind and split, must not duplicate another
    semantic/asset in the family, and is written into the exact lineage. The
    renderer and preencoder still enforce their independent final-state gates.
    """

    requested_refs = {
        "base": [str(item) for item in spec["base_source_refs"]],
        "donor": [str(item) for item in spec["donor_source_refs"]],
    }
    current_refs = {key: list(value) for key, value in requested_refs.items()}
    current_templates = {
        "base": list(source_templates),
        "donor": list(donor_templates),
    }
    rejected: dict[tuple[str, int], list[dict[str, Any]]] = {}
    candidate_pools: dict[tuple[str, int], list[dict[str, Any]]] = {}

    def substitutions() -> list[dict[str, Any]]:
        result = []
        for branch in ("base", "donor"):
            for index, replacement_ref in enumerate(current_refs[branch]):
                original_ref = requested_refs[branch][index]
                if replacement_ref == original_ref:
                    continue
                template = current_templates[branch][index]
                result.append(
                    {
                        "slot": f"{branch}_{index}",
                        "kind": str(template.get("kind") or ""),
                        "original_template_ref": original_ref,
                        "replacement_template_ref": replacement_ref,
                        "reason": "dry_playback_below_minimum_input_rms",
                        "minimum_input_rms": float(
                            build_spec["quality_control"]["source_loudness"][
                                "minimum_input_rms"
                            ]
                        ),
                        "rejected_candidates": list(rejected[(branch, index)]),
                    }
                )
        return result

    for _ in range(_MAX_AUDIBLE_FALLBACKS + 1):
        try:
            return _build_one(
                spec,
                source_templates=current_templates["base"],
                donor_templates=current_templates["donor"],
                base_source_refs=current_refs["base"],
                donor_source_refs=current_refs["donor"],
                source_substitutions=substitutions(),
                parquet=parquet,
                speech_cache=speech_cache,
                build_spec=build_spec,
            )
        except SpatialStoryError as error:
            match = _SILENT_PLAYBACK.match(str(error))
            if match is None:
                raise
            failed_rms = float(match.group(1))
            failed_path = str(Path(match.group(2)).expanduser().resolve())

        matching_slots: list[tuple[str, int]] = []
        for branch in ("base", "donor"):
            for index, template in enumerate(current_templates[branch]):
                if _template_audio_path(
                    template,
                    parquet=parquet,
                    speech_cache=speech_cache,
                ) == failed_path:
                    matching_slots.append((branch, index))
        if len(matching_slots) != 1:
            raise RuntimeError(
                "cannot uniquely map failed dry playback to a family source: "
                f"family={spec['family_id']} path={failed_path} "
                f"matches={matching_slots}"
            )
        slot = matching_slots[0]
        branch, index = slot
        failed_template = current_templates[branch][index]
        failed_ref = current_refs[branch][index]
        rejected.setdefault(slot, []).append(
            {
                "template_ref": failed_ref,
                "asset_id": str(failed_template.get("asset_id") or ""),
                "path": failed_path,
                "mono_rms": failed_rms,
            }
        )

        if slot not in candidate_pools:
            start = "src_" + stable_digest(
                "audible_fallback",
                str(spec["family_id"]),
                branch,
                index,
                requested_refs[branch][index],
                size=16,
            )
            candidate_pools[slot] = reader.source_candidates(
                split=str(spec["split"]),
                kind=str(failed_template.get("kind") or ""),
                start_template_id=start,
                limit=_MAX_AUDIBLE_FALLBACKS,
            )

        blocked_refs = {
            ref for values in current_refs.values() for ref in values
        }
        blocked_refs.update(
            item["template_ref"]
            for values in rejected.values()
            for item in values
        )
        blocked_assets = {
            str(template.get("asset_id") or "")
            for values in current_templates.values()
            for template in values
        }
        blocked_assets.update(
            item["asset_id"]
            for values in rejected.values()
            for item in values
        )
        blocked_semantics = {
            str(template.get("semantic_key") or "")
            for other_branch in ("base", "donor")
            for other_index, template in enumerate(current_templates[other_branch])
            if (other_branch, other_index) != slot
        }

        replacement = None
        for candidate in candidate_pools[slot]:
            candidate_ref = str(candidate["template_id"])
            if (
                candidate_ref in blocked_refs
                or str(candidate.get("asset_id") or "") in blocked_assets
                or str(candidate.get("semantic_key") or "") in blocked_semantics
            ):
                continue
            replacement = candidate
            break
        if replacement is None:
            raise RuntimeError(
                "no deterministic audible-fallback candidate remains: "
                f"family={spec['family_id']} slot={branch}_{index}"
            )
        current_templates[branch][index] = replacement
        current_refs[branch][index] = str(replacement["template_id"])

    raise RuntimeError(
        f"audible fallback limit exceeded for family {spec['family_id']}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-spec", type=Path, required=True)
    parser.add_argument("--catalog-root", type=Path, default=None)
    parser.add_argument("--split", choices=("train", "validation", "test"), required=True)
    parser.add_argument("--work-shard", type=int, required=True)
    parser.add_argument(
        "--max-family-rank-exclusive",
        type=int,
        default=None,
        help="Nested smoke/pilot boundary; omit for the full work shard.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.work_shard < 0:
        raise SystemExit("--work-shard must be non-negative")
    build_spec = json.loads(args.build_spec.resolve().read_text(encoding="utf-8"))
    catalog_root = Path(
        args.catalog_root or build_spec["storage"]["catalog_root"]
    ).expanduser().resolve()
    catalog_ready = json.loads(
        (catalog_root / "READY").read_text(encoding="utf-8")
    )
    if (
        int(catalog_ready.get("schema_version", -1)) != 5
        or catalog_ready.get("build_spec_fingerprint")
        != stable_digest(build_spec, size=32)
    ):
        raise SystemExit(
            f"catalog is stale or belongs to another build spec: {catalog_root}"
        )
    output = args.output_root.expanduser().resolve()
    if (output / "READY").is_file():
        print(f"[spatial-cot-recipes] cached READY {output}")
        return 0
    if output.exists() and any(output.iterdir()):
        names = {path.name for path in output.iterdir()}
        if names - {"shards", "details.json"}:
            raise SystemExit(
                f"unrecognized incomplete recipe output cannot resume: {output}"
            )
        details = output / "details.json"
        if details.is_file():
            previous = json.loads(details.read_text(encoding="utf-8"))
            if (
                previous.get("split") != args.split
                or int(previous.get("work_shard", -1)) != args.work_shard
            ):
                raise SystemExit(
                    f"incomplete recipe output belongs to another shard: {output}"
                )
        print(f"[spatial-cot-recipes] resuming incomplete output {output}")
    output.mkdir(parents=True, exist_ok=True)

    reader = _CatalogReader(catalog_root)
    parquet = ParquetAudioSourceIndex(
        build_spec["source_catalog"]["speech_parquet_index"]
    )
    speech_cache = Path(
        build_spec["source_catalog"]["speech_materialized_cache"]
    ).expanduser().resolve()
    try:
        family_specs = reader.families(args.split, args.work_shard)
        if args.max_family_rank_exclusive is not None:
            family_specs = [
                row
                for row in family_specs
                if int(row["family_rank"]) < args.max_family_rank_exclusive
            ]
        if not family_specs:
            raise SystemExit(
                f"catalog has no {args.split} work shard {args.work_shard}"
            )
        families = []
        for spec in family_specs:
            bases = [reader.source(ref) for ref in spec["base_source_refs"]]
            donors = [reader.source(ref) for ref in spec["donor_source_refs"]]
            families.append(
                _build_with_audible_fallbacks(
                    spec,
                    source_templates=bases,
                    donor_templates=donors,
                    reader=reader,
                    parquet=parquet,
                    speech_cache=speech_cache,
                    build_spec=build_spec,
                )
            )
    finally:
        reader.close()
        parquet.close()

    shard = output / "shards" / f"recipes-{args.split}-{args.work_shard:05d}.jsonl"
    count, digest = atomic_write_jsonl(shard, families)
    atomic_write_json(
        output / "details.json",
        {
            "schema": SCHEMA,
            "schema_version": VERSION,
            "catalog_root": str(catalog_root),
            "build_spec": str(args.build_spec.resolve()),
            "split": args.split,
            "work_shard": args.work_shard,
            "families": count,
            "states": sum(len(item["turns"]) for item in families),
            "recipe_sha256": digest,
        },
    )
    atomic_write_json(
        output / "READY",
        {
            "schema": SCHEMA,
            "schema_version": VERSION,
            "split": args.split,
            "work_shard": args.work_shard,
            "families": count,
            "states": sum(len(item["turns"]) for item in families),
            "recipe_sha256": digest,
        },
    )
    print(
        json.dumps(
            {"status": "READY", "output": str(output), "families": count},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
