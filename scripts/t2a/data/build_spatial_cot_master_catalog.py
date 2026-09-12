#!/usr/bin/env python3
"""Build the source-disjoint 1M/10K/2K Spatial-CoT master catalog.

This command writes metadata only.  It does not materialize TTS audio, render
FOA, or run the VAE.  Smoke and pilot views are nested deterministic subsets of
the same master family ranks, so no separately sampled trial dataset can drift
from production.
"""
from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import os
import random
import sqlite3
import sys
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stable_audio_tools.data.t2a_artifacts import atomic_write_json, atomic_write_jsonl
from stable_audio_tools.data.spatial_edit_recipe import (
    is_silent_source,
    is_speech_source,
    stable_digest,
)


SCHEMA = "stable_audio_tools.spatial_cot_master_catalog"
VERSION = 5


def _source_kind(source: Mapping[str, Any]) -> str:
    """Classify speech conservatively for the one-speech-per-state contract."""

    return "speech" if is_speech_source(source) else "sound_music"


def _semantic_key(source: Mapping[str, Any]) -> str:
    """Return the planner-visible identity used to avoid semantic no-op edits."""

    if _source_kind(source) == "speech":
        return "speech"
    event = source.get("event") or {}
    category = " ".join(
        str(event.get("category") or "sound").strip().lower().split()
    )
    label = " ".join(
        str(event.get("label") or category).strip().lower().split()
    )
    return f"{category}\x1f{label}"


def _digest(*values: Any, size: int = 16) -> str:
    payload = json.dumps(values, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.blake2b(payload.encode(), digest_size=size).hexdigest()


def _iter_plans(root: Path) -> Iterator[dict[str, Any]]:
    for shard in sorted((root / "shards").glob("*.jsonl")):
        with shard.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def _asset_identity(source: Mapping[str, Any]) -> tuple[str, str] | None:
    content = source.get("content") or {}
    direct = content.get("source_audio_path")
    if isinstance(direct, str) and direct:
        path = Path(direct).expanduser()
        if path.is_file():
            return "direct", os.path.normpath(str(path.resolve()))
    locator = content.get("source_locator") or {}
    if locator.get("type") == "parquet_source_id":
        dataset = locator.get("source_dataset")
        source_id = locator.get("source_id")
        if dataset and source_id:
            return "parquet", f"{dataset}:{source_id}"
    return None


def _split_for_asset(asset_id: str, config: Mapping[str, Any]) -> str:
    value = int(_digest("source-split", asset_id, size=8), 16) / float(2**64)
    train = float(config["train_fraction"])
    validation = float(config["validation_fraction"])
    if value < train:
        return "train"
    if value < train + validation:
        return "validation"
    return "test"


def _weighted_choice(rng: random.Random, weights: Mapping[Any, float]):
    keys = list(weights)
    cumulative = []
    total = 0.0
    for key in keys:
        weight = float(weights[key])
        if weight < 0:
            raise ValueError("weights must be non-negative")
        total += weight
        cumulative.append(total)
    if total <= 0:
        raise ValueError("at least one weight must be positive")
    return keys[bisect.bisect_left(cumulative, rng.random() * total)]


def _choose_distinct(
    rng: random.Random,
    pool: Sequence[dict[str, str]],
    count: int,
    *,
    excluded_assets: set[str],
    excluded_semantics: set[str] | None = None,
) -> list[dict[str, str]]:
    if not pool:
        raise RuntimeError("source pool is empty")
    selected = []
    attempts = 0
    while len(selected) < count and attempts < max(100, count * 100):
        candidate = pool[rng.randrange(len(pool))]
        attempts += 1
        if candidate["asset_id"] in excluded_assets:
            continue
        semantic_key = str(candidate["semantic_key"])
        if excluded_semantics is not None and semantic_key in excluded_semantics:
            continue
        excluded_assets.add(candidate["asset_id"])
        if excluded_semantics is not None:
            excluded_semantics.add(semantic_key)
        selected.append(candidate)
    if len(selected) != count:
        raise RuntimeError(
            f"could not draw {count} distinct assets from pool of {len(pool)}"
        )
    return selected


class _JsonlShardWriter:
    def __init__(self, root: Path, stem: str, rows_per_shard: int):
        self.root = root
        self.stem = stem
        self.rows_per_shard = int(rows_per_shard)
        self.shard_id = -1
        self.rows = 0
        self.total = 0
        self.handle = None
        self.relative = None

    def _next(self):
        if self.handle is not None:
            self.handle.flush()
            os.fsync(self.handle.fileno())
            self.handle.close()
        self.shard_id += 1
        self.rows = 0
        self.relative = Path("shards") / f"{self.stem}-{self.shard_id:05d}.jsonl"
        path = self.root / self.relative
        path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = path.open("wb")

    def write(self, row: Mapping[str, Any]) -> tuple[str, int, int]:
        if self.handle is None or self.rows >= self.rows_per_shard:
            self._next()
        payload = json.dumps(
            row, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        offset = self.handle.tell()
        self.handle.write(payload + b"\n")
        self.rows += 1
        self.total += 1
        return self.relative.as_posix(), offset, len(payload)

    def close(self):
        if self.handle is not None:
            self.handle.flush()
            os.fsync(self.handle.fileno())
            self.handle.close()
            self.handle = None

    @property
    def shard_count(self) -> int:
        return self.shard_id + 1


def _new_index(path: Path, ddl: str) -> tuple[sqlite3.Connection, Path]:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    connection = sqlite3.connect(temporary)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.executescript(ddl)
    return connection, temporary


def _publish_index(connection: sqlite3.Connection, temporary: Path, path: Path):
    connection.commit()
    connection.close()
    os.replace(temporary, path)


def _parse_counts(value: str | None, spec: Mapping[str, Any]) -> dict[str, int]:
    counts = {
        split: int(config["families"])
        for split, config in spec["splits"].items()
    }
    if value is None:
        return counts
    parsed = {}
    for item in value.split(","):
        name, raw_count = item.split("=", 1)
        parsed[name.strip()] = int(raw_count)
    if set(parsed) != set(counts) or any(value <= 0 for value in parsed.values()):
        raise ValueError("--counts must define positive train,validation,test values")
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-spec", type=Path, required=True)
    parser.add_argument("--scene-plan-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument(
        "--counts",
        default=None,
        help="smoke override, e.g. train=64,validation=8,test=4",
    )
    args = parser.parse_args()

    spec = json.loads(args.build_spec.expanduser().resolve().read_text(encoding="utf-8"))
    build_spec_fingerprint = stable_digest(spec, size=32)
    scene_root = (
        args.scene_plan_root
        or Path(spec["source_catalog"]["scene_plan_root"])
    ).expanduser().resolve()
    output_root = (
        args.output_root or Path(spec["storage"]["catalog_root"])
    ).expanduser().resolve()
    if not (scene_root / "READY").is_file():
        raise SystemExit(f"ScenePlan source store is not READY: {scene_root}")
    if output_root.exists() and any(output_root.iterdir()):
        raise SystemExit(f"catalog output must be new/empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    counts = _parse_counts(args.counts, spec)
    rows_per_shard = int(spec["sharding"]["catalog_rows_per_shard"])
    eligible = set(spec["source_catalog"]["eligible_dataset_ids"])
    reject_silence = bool(
        spec["source_catalog"].get("reject_semantic_silence", True)
    )

    source_root = output_root / "sources"
    source_root.mkdir(parents=True, exist_ok=True)
    source_writer = _JsonlShardWriter(source_root, "sources", rows_per_shard)
    source_index, source_index_tmp = _new_index(
        source_root / "index.sqlite",
        """
        CREATE TABLE sources(
          template_id TEXT PRIMARY KEY, asset_id TEXT NOT NULL, split TEXT NOT NULL,
          kind TEXT NOT NULL, shard TEXT NOT NULL, byte_offset INTEGER NOT NULL,
          byte_length INTEGER NOT NULL
        );
        CREATE INDEX source_split_kind ON sources(split, kind);
        """,
    )
    pools: dict[str, dict[str, list[dict[str, str]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    source_stats = Counter()
    for plan in _iter_plans(scene_root):
        if plan.get("dataset_id") not in eligible:
            continue
        for source_index_in_plan, source in enumerate(
            ((plan.get("scene") or {}).get("sources") or [])
        ):
            if reject_silence and is_silent_source(source):
                source_stats["skipped_semantic_silence"] += 1
                continue
            identity = _asset_identity(source)
            if identity is None:
                source_stats["skipped_unresolvable"] += 1
                continue
            locator_type, locator_value = identity
            asset_id = f"asset_{_digest(locator_type, locator_value)}"
            split = _split_for_asset(asset_id, spec["source_split"])
            kind = _source_kind(source)
            template_id = f"src_{_digest(plan.get('sample_id'), source_index_in_plan)}"
            record = {
                "schema": SCHEMA,
                "schema_version": VERSION,
                "record_type": "source_template",
                "template_id": template_id,
                "asset_id": asset_id,
                "asset_locator_type": locator_type,
                "split": split,
                "kind": kind,
                "semantic_key": _semantic_key(source),
                "dataset_id": plan.get("dataset_id"),
                "source_sample_id": plan.get("sample_id"),
                "source": source,
                "room": (plan.get("scene") or {}).get("room") or {},
            }
            shard, offset, length = source_writer.write(record)
            source_index.execute(
                "INSERT INTO sources VALUES(?,?,?,?,?,?,?)",
                (template_id, asset_id, split, kind, shard, offset, length),
            )
            compact = {
                "template_id": template_id,
                "asset_id": asset_id,
                "semantic_key": record["semantic_key"],
            }
            pools[split][kind].append(compact)
            pools[split]["all"].append(compact)
            source_stats[f"{split}_{kind}"] += 1
    source_writer.close()
    _publish_index(source_index, source_index_tmp, source_root / "index.sqlite")
    for split in counts:
        if not pools[split]["speech"] or not pools[split]["sound_music"]:
            raise RuntimeError(f"split {split} lacks speech or sound/music sources")

    family_root = output_root / "families"
    family_root.mkdir(parents=True, exist_ok=True)
    family_writer = _JsonlShardWriter(family_root, "families", rows_per_shard)
    family_index, family_index_tmp = _new_index(
        family_root / "index.sqlite",
        """
        CREATE TABLE families(
          split TEXT NOT NULL, family_rank INTEGER NOT NULL, family_id TEXT NOT NULL,
          work_shard INTEGER NOT NULL, shard TEXT NOT NULL, byte_offset INTEGER NOT NULL,
          byte_length INTEGER NOT NULL, PRIMARY KEY(split, family_rank), UNIQUE(family_id)
        );
        CREATE INDEX family_work_shard ON families(split, work_shard);
        """,
    )
    base_count_weights = {
        int(key): float(value)
        for key, value in spec["family_distribution"]["base_source_count"].items()
    }
    edit_weights = {
        str(key): float(value)
        for key, value in spec["edit_distribution"]["weights"].items()
    }
    speech_fraction = float(
        spec["family_distribution"]["speech_containing_fraction"]
    )
    motion_fraction = float(
        spec["family_distribution"]["motion_containing_fraction"]
    )
    max_sources = int(spec["family_distribution"]["max_sources"])
    max_speech_sources = int(
        spec["family_distribution"].get("max_speech_sources_per_state", 1)
    )
    if max_speech_sources != 1:
        raise ValueError(
            "this catalog builder currently requires max_speech_sources_per_state=1"
        )
    speech_donor_fraction = float(
        spec["edit_distribution"].get(
            "speech_donor_fraction_when_base_has_no_speech", 0.0
        )
    )
    if not 0.0 <= speech_donor_fraction <= 1.0:
        raise ValueError(
            "speech_donor_fraction_when_base_has_no_speech must lie in [0,1]"
        )
    families_per_work_shard = int(spec["sharding"]["families_per_latent_shard"])
    family_stats = Counter()
    subset_rows: dict[str, list[dict[str, Any]]] = {"smoke": [], "pilot": []}
    smoke_count = int(spec["sharding"]["smoke_families"])
    pilot_count = int(spec["sharding"]["pilot_families"])

    for split, count in counts.items():
        for rank in range(count):
            family_seed = int(_digest(spec["seed"], split, rank, size=8), 16)
            rng = random.Random(family_seed)
            source_count = int(_weighted_choice(rng, base_count_weights))
            speech_family = rng.random() < speech_fraction
            excluded: set[str] = set()
            excluded_semantics: set[str] = set()
            base_sources = []
            if speech_family:
                base_sources.extend(
                    _choose_distinct(
                        rng,
                        pools[split]["speech"],
                        1,
                        excluded_assets=excluded,
                        excluded_semantics=excluded_semantics,
                    )
                )
            remaining = source_count - len(base_sources)
            base_sources.extend(
                _choose_distinct(
                    rng,
                    pools[split]["sound_music"],
                    remaining,
                    excluded_assets=excluded,
                    excluded_semantics=excluded_semantics,
                )
            )

            operations = []
            active_sources = source_count
            donor_count = 0
            for _ in range(int(spec["edit_distribution"]["transitions_per_family"])):
                eligible_ops = dict(edit_weights)
                if active_sources <= 1:
                    eligible_ops.pop("remove_source", None)
                if active_sources >= max_sources:
                    eligible_ops.pop("add_source", None)
                operation = str(_weighted_choice(rng, eligible_ops))
                operations.append(operation)
                if operation == "add_source":
                    active_sources += 1
                    donor_count += 1
                elif operation == "remove_source":
                    active_sources -= 1
                elif operation == "replace_source":
                    donor_count += 1
            # A no-speech base may introduce one speech donor through add or
            # replace.  There is at most one such donor in the whole family;
            # all prior donors are non-speech and all later donors remain
            # non-speech, so no cumulative state can contain speech+speech.
            speech_donor_index = None
            if (
                donor_count
                and not speech_family
                and rng.random() < speech_donor_fraction
            ):
                speech_donor_index = rng.randrange(donor_count)
            donors: list[dict[str, str]] = []
            donor_kinds: list[str] = []
            for donor_index in range(donor_count):
                kind = (
                    "speech"
                    if donor_index == speech_donor_index
                    else "sound_music"
                )
                donors.extend(
                    _choose_distinct(
                        rng,
                        pools[split][kind],
                        1,
                        excluded_assets=excluded,
                        # Replacement must remain planner-visible. Spatial-only
                        # same-event edits are represented by move_source and do
                        # not need a new dry asset.
                        excluded_semantics=excluded_semantics,
                    )
                )
                donor_kinds.append(kind)
            family_id = f"spcot_{split}_{rank:07d}_{_digest(family_seed, size=6)}"
            row = {
                "schema": SCHEMA,
                "schema_version": VERSION,
                "record_type": "family_spec",
                "family_id": family_id,
                "split": split,
                "family_rank": rank,
                "work_shard": rank // families_per_work_shard,
                "seed": family_seed,
                "base_source_refs": [item["template_id"] for item in base_sources],
                "donor_source_refs": [item["template_id"] for item in donors],
                "donor_source_kinds": donor_kinds,
                "edit_types": operations,
                "require_motion": rng.random() < motion_fraction,
                "max_speech_sources_per_state": max_speech_sources,
                "unique_semantic_sources_per_family": True,
                "speech_donor_policy": (
                    "at_most_one_if_base_has_no_speech"
                ),
                "audio": spec["audio"],
            }
            shard, offset, length = family_writer.write(row)
            family_index.execute(
                "INSERT INTO families VALUES(?,?,?,?,?,?,?)",
                (split, rank, family_id, row["work_shard"], shard, offset, length),
            )
            family_stats[f"{split}_families"] += 1
            family_stats[f"{split}_base_sources_{source_count}"] += 1
            family_stats[f"{split}_speech_containing"] += int(speech_family)
            family_stats[f"{split}_speech_donor"] += int(
                speech_donor_index is not None
            )
            family_stats[f"{split}_motion_required"] += int(row["require_motion"])
            for operation in operations:
                family_stats[f"{split}_edit_{operation}"] += 1
            if split == "train" and rank < pilot_count:
                summary = {
                    "family_id": family_id,
                    "split": split,
                    "family_rank": rank,
                    "work_shard": row["work_shard"],
                }
                subset_rows["pilot"].append(summary)
                if rank < smoke_count:
                    subset_rows["smoke"].append(summary)
    family_writer.close()
    _publish_index(family_index, family_index_tmp, family_root / "index.sqlite")

    views = output_root / "views"
    views.mkdir(parents=True, exist_ok=True)
    for name, rows in subset_rows.items():
        atomic_write_jsonl(views / f"{name}.jsonl", rows)
    atomic_write_json(
        output_root / "stats.json",
        {"sources": dict(source_stats), "families": dict(family_stats)},
    )
    atomic_write_json(
        output_root / "details.json",
        {
            "schema": SCHEMA,
            "schema_version": VERSION,
            "build_spec": str(args.build_spec.expanduser().resolve()),
            "scene_plan_root": str(scene_root),
            "counts": counts,
            "source_disjoint": True,
            "max_speech_sources_per_state": max_speech_sources,
            "speech_pairing_policy": (
                "at_most_one_speech_per_state; a no-speech base may add_or_replace "
                "one speech donor"
            ),
            "speech_detection_policy": (
                "metadata_transcript_and_event_labels; hidden_unlabelled_background_"
                "speech_is_accepted_without_acoustic_screening"
            ),
            "semantic_pairing_policy": "all_base_and_donor_semantic_keys_are_distinct",
            "semantic_silence_policy": (
                "reject_explicit_no_sound_labels_at_source_catalog_build"
                if reject_silence
                else "disabled"
            ),
            "full_production_counts": args.counts is None,
            "build_spec_fingerprint": build_spec_fingerprint,
        },
    )
    atomic_write_json(
        output_root / "READY",
        {
            "schema": SCHEMA,
            "schema_version": VERSION,
            "families": sum(counts.values()),
            "counts": counts,
            "source_templates": source_writer.total,
            "source_shards": source_writer.shard_count,
            "family_shards": family_writer.shard_count,
            "smoke_families": len(subset_rows["smoke"]),
            "pilot_families": len(subset_rows["pilot"]),
            "build_spec_fingerprint": build_spec_fingerprint,
        },
    )
    print(
        json.dumps(
            {
                "status": "READY",
                "root": str(output_root),
                "counts": counts,
                "source_templates": source_writer.total,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
