#!/usr/bin/env python3
"""Build deterministic, fully-auditable Transfusion Editing pair indices.

Pilot mode selects a tiny representative panel.  Full mode selects exactly
1,000,000/20,000/5,000 unique P10 source rows for train/validation/test while
preserving fine source strata.  This script plans pairs only; target FOA and
target latent materialization is a separate fail-closed stage.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import os
import sqlite3
import sys
import time
import zlib
from array import array
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stable_audio_tools.data.sceneplan_transfusion_editing import (  # noqa: E402
    EDITING_INSTRUCTION_CONTRACT,
    EDITING_PAIR_CONTRACT,
    EDITING_PAIR_SEED,
    EDIT_OPERATIONS,
    OPERATION_FAMILY,
    OP_EVENT_ADD,
    OP_EVENT_REMOVE,
    OP_LINEAR_TO_STATIC,
    OP_RELOCATE,
    OP_STATIC_TO_LINEAR,
    build_event_addition,
    build_event_removal,
    build_linear_to_static,
    build_static_to_linear,
    build_stationary_relocation,
    canonical_json,
    deterministic_u64,
    eligible_source_ids,
    make_pair_id,
    sha256_json,
)
from stable_audio_tools.data.sceneplan_transfusion_editing_index import (  # noqa: E402
    P10EditingSourceResolver,
    SOURCE_RESOLUTION_CONTRACT,
    scene_domain,
    sha256_file,
)


PAIR_INDEX_SCHEMA = "sceneplan_transfusion_editing_pair_index"
PAIR_INDEX_SCHEMA_VERSION = 1
PAIR_GAIN_POLICY = (
    "fixed_member_corrections_nonboosting_peak_safe_target_master_v1"
)
SELECTION_POLICY = "p10_fine_stratum_largest_remainder_hash_rank_seed42_v1"
EDITING_AR_INPUT_CONTRACT = "source_foa_latent_plus_raw_edit_request_v2"
FULL_COUNTS = {"train": 1_000_000, "validation": 20_000, "test": 5_000}
SOURCE_INDICES = {
    split: Path(
        "/mnt/sdb/audio_dataset/sceneplan_v2_1p124m/revisions/"
        "speech_expansion_noalign_15s_v1/training_index"
    )
    / f"{split}.sqlite"
    for split in FULL_COUNTS
}
SHARED_CONTRACT = REPO_ROOT / (
    "docs/sceneplan_v2/p11_p10v11_shared_transfusion_v1_contract_20260904.md"
)
MUTATION_MODULE = REPO_ROOT / (
    "stable_audio_tools/data/sceneplan_transfusion_editing.py"
)
RESOLVER_MODULE = REPO_ROOT / (
    "stable_audio_tools/data/sceneplan_transfusion_editing_index.py"
)


def _digest_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _pack(value: Any) -> bytes:
    return zlib.compress(canonical_json(value).encode("utf-8"), level=9)


def _unpack(value: bytes) -> Any:
    return json.loads(zlib.decompress(value))


def _rank(*parts: Any, purpose: str) -> str:
    person = purpose.encode("ascii")[:16]
    return f"{deterministic_u64(EDITING_PAIR_SEED, *parts, person=person):016x}"


def _largest_remainder(
    counts: Mapping[tuple[Any, ...], int], total: int
) -> dict[tuple[Any, ...], int]:
    population = sum(int(value) for value in counts.values())
    if not 0 <= int(total) <= population:
        raise ValueError(f"invalid allocation total {total} for population {population}")
    allocations: dict[tuple[Any, ...], int] = {}
    remainders = []
    used = 0
    for key in sorted(counts):
        numerator = int(counts[key]) * int(total)
        quotient, remainder = divmod(numerator, population)
        allocations[key] = quotient
        used += quotient
        remainders.append((remainder, key))
    for _, key in sorted(remainders, key=lambda item: (-item[0], item[1]))[
        : int(total) - used
    ]:
        allocations[key] += 1
    if sum(allocations.values()) != int(total):
        raise AssertionError("largest-remainder allocation lost rows")
    return allocations


def _equal_family_counts(total: int) -> dict[str, int]:
    families = (
        "event_add_remove",
        "stationary_azimuth_change",
        "static_linear_toggle",
    )
    base, remainder = divmod(int(total), len(families))
    return {
        family: base + (index < remainder)
        for index, family in enumerate(families)
    }


def _create_stage(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.executescript(
        """
        CREATE TABLE candidates(
            source_ordinal INTEGER PRIMARY KEY,
            source_sample_id TEXT NOT NULL UNIQUE,
            source_count INTEGER NOT NULL,
            domain TEXT NOT NULL,
            latent_bucket_frames INTEGER NOT NULL,
            has_static INTEGER NOT NULL,
            has_linear INTEGER NOT NULL,
            selection_rank TEXT NOT NULL,
            operation_rank TEXT NOT NULL
        );
        CREATE INDEX candidate_stratum_rank ON candidates(
            source_count,domain,latent_bucket_frames,selection_rank,source_ordinal
        );
        CREATE TABLE selected(
            output_ordinal INTEGER PRIMARY KEY,
            source_ordinal INTEGER NOT NULL UNIQUE,
            operation_family TEXT NOT NULL,
            operation TEXT NOT NULL,
            FOREIGN KEY(source_ordinal) REFERENCES candidates(source_ordinal)
        );
        CREATE TABLE resolved_sources(
            output_ordinal INTEGER PRIMARY KEY,
            payload_zlib BLOB NOT NULL
        );
        CREATE TABLE donors(
            donor_id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_ordinal INTEGER NOT NULL,
            source_sample_id TEXT NOT NULL,
            source_id TEXT NOT NULL,
            kind TEXT NOT NULL,
            identity_hash TEXT NOT NULL,
            dry_samples INTEGER NOT NULL,
            duration_bin INTEGER NOT NULL,
            plan_source_zlib BLOB NOT NULL,
            recipe_source_zlib BLOB NOT NULL
        );
        CREATE INDEX donor_lookup ON donors(kind,duration_bin,donor_id);
        """
    )
    return connection


def _scan_candidates(
    resolver: P10EditingSourceResolver,
    connection: sqlite3.Connection,
) -> dict[tuple[int, str, int], int]:
    counts: Counter[tuple[int, str, int]] = Counter()
    pending = []
    started = time.time()
    for index, row in enumerate(resolver.iter_selection_fields(), start=1):
        stratum = (
            int(row["source_count"]),
            str(row["domain"]),
            int(row["latent_bucket_frames"]),
        )
        counts[stratum] += 1
        pending.append(
            (
                int(row["ordinal"]),
                str(row["sample_id"]),
                *stratum,
                int(bool(row["has_static"])),
                int(bool(row["has_linear"])),
                _rank(row["sample_id"], purpose="edit-select-v1"),
                _rank(row["sample_id"], purpose="edit-oper-v1"),
            )
        )
        if len(pending) >= 10_000:
            connection.executemany(
                "INSERT INTO candidates VALUES(?,?,?,?,?,?,?,?,?)", pending
            )
            connection.commit()
            pending.clear()
        if index % 200_000 == 0:
            print(
                canonical_json(
                    {
                        "stage": "candidate_scan",
                        "rows": index,
                        "elapsed_sec": round(time.time() - started, 1),
                    }
                ),
                flush=True,
            )
    if pending:
        connection.executemany(
            "INSERT INTO candidates VALUES(?,?,?,?,?,?,?,?,?)", pending
        )
        connection.commit()
    if sum(counts.values()) != resolver.row_count:
        raise RuntimeError("candidate scan did not cover the frozen source index")
    return dict(counts)


def _select_full(
    connection: sqlite3.Connection,
    counts: Mapping[tuple[int, str, int], int],
    target_count: int,
) -> None:
    quotas = _largest_remainder(counts, target_count)
    selected: list[tuple[int, str, str]] = []
    candidate_by_ordinal: dict[int, tuple[int, bool, bool, str]] = {}
    for stratum in sorted(quotas):
        source_count, domain, bucket = stratum
        quota = int(quotas[stratum])
        rows = connection.execute(
            """
            SELECT source_ordinal,has_static,has_linear,selection_rank,
                   operation_rank
            FROM candidates
            WHERE source_count=? AND domain=? AND latent_bucket_frames=?
            ORDER BY selection_rank,source_ordinal
            """,
            (source_count, domain, bucket),
        ).fetchall()
        if len(rows) != int(counts[stratum]):
            raise RuntimeError(f"stratum {stratum} candidate scan is incomplete")
        family_counts = _equal_family_counts(quota)
        relocation_needed = family_counts["stationary_azimuth_change"]
        relocation = sorted(
            (row for row in rows if bool(row[1])),
            key=lambda row: (str(row[4]), int(row[0])),
        )[:relocation_needed]
        if len(relocation) != relocation_needed:
            raise RuntimeError(f"stratum {stratum} lacks stationary relocation rows")
        used = {int(row[0]) for row in relocation}
        # Reserve every eligibility-constrained family before the unrestricted
        # event family. Otherwise a low-quota source-rank sample can contain no
        # toggle row even when the complete stratum has enough eligible rows.
        toggle_needed = family_counts["static_linear_toggle"]
        toggle = sorted(
            (
                row
                for row in rows
                if int(row[0]) not in used and (bool(row[1]) or bool(row[2]))
            ),
            key=lambda row: (str(row[4]), int(row[0])),
        )[:toggle_needed]
        if len(toggle) != toggle_needed:
            raise RuntimeError(f"stratum {stratum} lacks static/linear toggle rows")
        used.update(int(row[0]) for row in toggle)
        event = [row for row in rows if int(row[0]) not in used][
            : family_counts["event_add_remove"]
        ]
        if len(event) != family_counts["event_add_remove"]:
            raise RuntimeError(f"stratum {stratum} event selection is incomplete")
        selected_rows = [*relocation, *toggle, *event]
        if len(selected_rows) != quota:
            raise RuntimeError(f"stratum {stratum} selection is incomplete")
        for row in selected_rows:
            candidate_by_ordinal[int(row[0])] = (
                int(source_count),
                bool(row[1]),
                bool(row[2]),
                str(row[4]),
            )
        selected.extend(
            (int(row[0]), "stationary_azimuth_change", OP_RELOCATE)
            for row in relocation
        )
        # Sub-operation direction is assigned globally below.
        selected.extend(
            (int(row[0]), "static_linear_toggle", "__toggle__")
            for row in toggle
        )
        selected.extend(
            (int(row[0]), "event_add_remove", "__event__") for row in event
        )

    if len(selected) != int(target_count) or len({row[0] for row in selected}) != len(
        selected
    ):
        raise RuntimeError("full selection is not unique and complete")
    if len(candidate_by_ordinal) != int(target_count):
        raise RuntimeError("full selection candidate metadata is incomplete")
    # Balance add/remove and static/linear direction as closely as eligibility
    # allows.  Counts are deterministic and are not driven by source order.
    event_counts = Counter()
    toggle_counts = Counter()
    final = []
    for source_ordinal, family, operation in sorted(
        selected,
        key=lambda row: (candidate_by_ordinal[row[0]][3], row[0]),
    ):
        source_count, has_static, has_linear, _ = candidate_by_ordinal[
            source_ordinal
        ]
        if operation == "__event__":
            if source_count == 1:
                operation = OP_EVENT_ADD
            elif source_count == 4:
                operation = OP_EVENT_REMOVE
            else:
                operation = (
                    OP_EVENT_ADD
                    if event_counts[OP_EVENT_ADD] <= event_counts[OP_EVENT_REMOVE]
                    else OP_EVENT_REMOVE
                )
            event_counts[operation] += 1
        elif operation == "__toggle__":
            if has_static and has_linear:
                operation = (
                    OP_STATIC_TO_LINEAR
                    if toggle_counts[OP_STATIC_TO_LINEAR]
                    <= toggle_counts[OP_LINEAR_TO_STATIC]
                    else OP_LINEAR_TO_STATIC
                )
            elif has_static:
                operation = OP_STATIC_TO_LINEAR
            elif has_linear:
                operation = OP_LINEAR_TO_STATIC
            else:
                raise AssertionError("toggle row has no supported trajectory")
            toggle_counts[operation] += 1
        final.append((source_ordinal, family, operation))
    final.sort(key=lambda row: row[0])
    connection.executemany(
        "INSERT INTO selected VALUES(?,?,?,?)",
        [
            (output_ordinal, source_ordinal, family, operation)
            for output_ordinal, (source_ordinal, family, operation) in enumerate(final)
        ],
    )
    connection.commit()


def _select_pilot(
    resolver: P10EditingSourceResolver,
    connection: sqlite3.Connection,
    per_operation: int,
) -> None:
    heaps: dict[str, list[tuple[int, int, str, dict[str, Any]]]] = {
        operation: [] for operation in EDIT_OPERATIONS
    }
    for row in resolver.iter_selection_fields():
        plan = row["sceneplan"]
        legal = [
            operation
            for operation in EDIT_OPERATIONS
            if eligible_source_ids(plan, operation)
        ]
        if not legal:
            continue
        preferred = legal[
            deterministic_u64(
                EDITING_PAIR_SEED,
                resolver.expected_split,
                row["sample_id"],
                person=b"edit-pilot-v1",
            )
            % len(legal)
        ]
        rank = deterministic_u64(
            EDITING_PAIR_SEED,
            row["sample_id"],
            preferred,
            person=b"edit-pick-v1",
        )
        item = (-rank, -int(row["ordinal"]), str(row["sample_id"]), row)
        heap = heaps[preferred]
        if len(heap) < per_operation:
            heapq.heappush(heap, item)
        elif item > heap[0]:
            heapq.heapreplace(heap, item)
    selected = []
    for operation in EDIT_OPERATIONS:
        if len(heaps[operation]) != per_operation:
            raise RuntimeError(f"pilot lacks {operation} candidates")
        for neg_rank, neg_ordinal, sample_id, row in heaps[operation]:
            source_ordinal = -neg_ordinal
            selected.append((source_ordinal, operation, sample_id, row, -neg_rank))
    if len({item[0] for item in selected}) != len(selected):
        raise RuntimeError("pilot operation assignment reused a source row")
    for source_ordinal, operation, sample_id, row, rank in selected:
        connection.execute(
            "INSERT INTO candidates VALUES(?,?,?,?,?,?,?,?,?)",
            (
                source_ordinal,
                sample_id,
                int(row["source_count"]),
                str(row["domain"]),
                int(row["latent_bucket_frames"]),
                int(bool(row["has_static"])),
                int(bool(row["has_linear"])),
                f"{rank:016x}",
                _rank(sample_id, purpose="edit-oper-v1"),
            ),
        )
    ordered = sorted(selected, key=lambda item: (EDIT_OPERATIONS.index(item[1]), item[4]))
    connection.executemany(
        "INSERT INTO selected VALUES(?,?,?,?)",
        [
            (index, item[0], OPERATION_FAMILY[item[1]], item[1])
            for index, item in enumerate(ordered)
        ],
    )
    connection.commit()


def _resolved_payload(source) -> dict[str, Any]:
    return {
        key: getattr(source, key)
        for key in (
            "split",
            "source_ordinal",
            "source_sample_id",
            "model_num_samples",
            "latent_frames_valid",
            "latent_bucket_frames",
            "source_count",
            "domain",
            "sceneplan",
            "model_sceneplan_sha256",
            "model_sceneplan_path",
            "model_sceneplan_row",
            "render_recipe",
            "render_recipe_path",
            "render_recipe_sha256",
            "render_result",
            "source_manifest_path",
            "source_manifest_sha256",
            "source_foa_sha256",
            "source_latent_path",
            "source_latent_key",
            "source_latent_tensor_sha256",
            "source_latent_shard_sha256",
        )
    }


def _resolve_selected(
    resolver: P10EditingSourceResolver, connection: sqlite3.Connection
) -> dict[tuple[str, int], array]:
    donor_groups: dict[tuple[str, int], array] = defaultdict(lambda: array("q"))
    rows = connection.execute(
        "SELECT output_ordinal,source_ordinal FROM selected ORDER BY source_ordinal"
    ).fetchall()
    started = time.time()
    for index, (output_ordinal, source_ordinal) in enumerate(rows, start=1):
        source = resolver.resolve(int(source_ordinal))
        payload = _resolved_payload(source)
        connection.execute(
            "INSERT INTO resolved_sources VALUES(?,?)",
            (int(output_ordinal), _pack(payload)),
        )
        plan_by_id = {
            str(item["source_id"]): item for item in source.sceneplan["sources"]
        }
        for recipe_source in source.render_recipe["sources"]:
            source_id = str(recipe_source["source_id"])
            window = recipe_source["exact_source_sample_window"]
            dry_samples = int(window["dry_end_sample"]) - int(
                window["dry_start_sample"]
            )
            duration_bin = max(1, math.ceil(dry_samples / 44_100))
            cursor = connection.execute(
                """
                INSERT INTO donors(
                    source_ordinal,source_sample_id,source_id,kind,identity_hash,
                    dry_samples,duration_bin,plan_source_zlib,recipe_source_zlib
                ) VALUES(?,?,?,?,?,?,?,?,?)
                """,
                (
                    source.source_ordinal,
                    source.source_sample_id,
                    source_id,
                    str(recipe_source["kind"]),
                    str(recipe_source["asset_ref"]["identity_hash"]),
                    dry_samples,
                    duration_bin,
                    _pack(plan_by_id[source_id]),
                    _pack(recipe_source),
                ),
            )
            donor_groups[(str(recipe_source["kind"]), duration_bin)].append(
                int(cursor.lastrowid)
            )
        if index % 1000 == 0 or index == len(rows):
            connection.commit()
            print(
                canonical_json(
                    {
                        "stage": "source_resolution",
                        "rows": index,
                        "total": len(rows),
                        "elapsed_sec": round(time.time() - started, 1),
                    }
                ),
                flush=True,
            )
    connection.commit()
    return donor_groups


def _pick_donor(
    connection: sqlite3.Connection,
    groups: Mapping[tuple[str, int], Sequence[int]],
    *,
    pair_id: str,
    source_payload: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    model_num_samples = int(source_payload["model_num_samples"])
    old_has_speech = any(
        source["kind"] == "speech" for source in source_payload["sceneplan"]["sources"]
    )
    old_identities = {
        str(source["asset_ref"]["identity_hash"])
        for source in source_payload["render_recipe"]["sources"]
    }
    kinds = ("music", "sound") if old_has_speech else ("speech", "music", "sound")
    eligible_groups = sorted(
        key
        for key, values in groups.items()
        if key[0] in kinds
        and values
        and (int(key[1]) - 1) * 44_100 < model_num_samples
    )
    if not eligible_groups:
        raise RuntimeError(f"{pair_id}: no same-split addition donor group fits")
    group_start = deterministic_u64(pair_id, "donor-group", person=b"edit-donor-v1") % len(
        eligible_groups
    )
    for group_offset in range(len(eligible_groups)):
        group = eligible_groups[(group_start + group_offset) % len(eligible_groups)]
        donor_ids = groups[group]
        start = deterministic_u64(pair_id, group, person=b"edit-member-v1") % len(
            donor_ids
        )
        for member_offset in range(min(len(donor_ids), 1024)):
            donor_id = int(donor_ids[(start + member_offset) % len(donor_ids)])
            row = connection.execute(
                """
                SELECT source_ordinal,source_sample_id,source_id,kind,
                       identity_hash,dry_samples,plan_source_zlib,
                       recipe_source_zlib
                FROM donors WHERE donor_id=?
                """,
                (donor_id,),
            ).fetchone()
            if row is None:
                raise AssertionError("donor catalog id disappeared")
            if (
                str(row[1]) == str(source_payload["source_sample_id"])
                or str(row[4]) in old_identities
                or int(row[5]) > model_num_samples
            ):
                continue
            provenance = {
                "donor_id": donor_id,
                "source_ordinal": int(row[0]),
                "source_sample_id": str(row[1]),
                "source_id": str(row[2]),
                "kind": str(row[3]),
                "identity_hash": str(row[4]),
                "dry_samples": int(row[5]),
            }
            return _unpack(row[6]), _unpack(row[7]), provenance
    raise RuntimeError(f"{pair_id}: deterministic donor search exhausted")


PAIR_COLUMNS = (
    "pair_ordinal,pair_id,split,work_shard,row_in_shard,source_ordinal,"
    "source_sample_id,target_sample_id,donor_provenance_json,operation_family,"
    "operation,raw_edit_request,instruction_template_id,instruction_sha256,"
    "source_count,target_count,source_domain,target_domain,latent_bucket_frames,"
    "model_num_samples,latent_frames_valid,old_sceneplan_zlib,new_sceneplan_zlib,"
    "old_sceneplan_sha256,new_sceneplan_sha256,source_render_recipe_zlib,"
    "target_render_recipe_zlib,source_render_recipe_sha256,"
    "target_render_recipe_sha256,source_render_result_zlib,"
    "source_render_result_sha256,source_members_zlib,target_members_zlib,"
    "source_members_sha256,target_members_sha256,edited_source_ids_json,"
    "unchanged_source_ids_json,source_model_sceneplan_path,"
    "source_model_sceneplan_row,source_manifest_path,source_manifest_sha256,"
    "source_foa_sha256,source_latent_path,source_latent_key,source_latent_ref,"
    "source_latent_tensor_sha256,source_latent_shard_sha256,target_latent_path,"
    "target_latent_key,target_latent_ref,target_latent_tensor_sha256,"
    "target_latent_shard_sha256,target_foa_path,target_foa_sha256,"
    "target_render_result_sha256,pair_gain_policy,materialization_status,"
    "pair_record_sha256"
)


def _create_final(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.executescript(
        """
        CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL) WITHOUT ROWID;
        CREATE TABLE pairs(
            pair_ordinal INTEGER PRIMARY KEY,
            pair_id TEXT NOT NULL UNIQUE,
            split TEXT NOT NULL,
            work_shard INTEGER NOT NULL,
            row_in_shard INTEGER NOT NULL,
            source_ordinal INTEGER NOT NULL UNIQUE,
            source_sample_id TEXT NOT NULL UNIQUE,
            target_sample_id TEXT NOT NULL UNIQUE,
            donor_provenance_json TEXT,
            operation_family TEXT NOT NULL,
            operation TEXT NOT NULL,
            raw_edit_request TEXT NOT NULL,
            instruction_template_id TEXT NOT NULL,
            instruction_sha256 TEXT NOT NULL,
            source_count INTEGER NOT NULL,
            target_count INTEGER NOT NULL,
            source_domain TEXT NOT NULL,
            target_domain TEXT NOT NULL,
            latent_bucket_frames INTEGER NOT NULL,
            model_num_samples INTEGER NOT NULL,
            latent_frames_valid INTEGER NOT NULL,
            old_sceneplan_zlib BLOB NOT NULL,
            new_sceneplan_zlib BLOB NOT NULL,
            old_sceneplan_sha256 TEXT NOT NULL,
            new_sceneplan_sha256 TEXT NOT NULL,
            source_render_recipe_zlib BLOB NOT NULL,
            target_render_recipe_zlib BLOB NOT NULL,
            source_render_recipe_sha256 TEXT NOT NULL,
            target_render_recipe_sha256 TEXT NOT NULL,
            source_render_result_zlib BLOB NOT NULL,
            source_render_result_sha256 TEXT NOT NULL,
            source_members_zlib BLOB NOT NULL,
            target_members_zlib BLOB NOT NULL,
            source_members_sha256 TEXT NOT NULL,
            target_members_sha256 TEXT NOT NULL,
            edited_source_ids_json TEXT NOT NULL,
            unchanged_source_ids_json TEXT NOT NULL,
            source_model_sceneplan_path TEXT NOT NULL,
            source_model_sceneplan_row INTEGER NOT NULL,
            source_manifest_path TEXT NOT NULL,
            source_manifest_sha256 TEXT NOT NULL,
            source_foa_sha256 TEXT NOT NULL,
            source_latent_path TEXT NOT NULL,
            source_latent_key TEXT NOT NULL,
            source_latent_ref TEXT NOT NULL,
            source_latent_tensor_sha256 TEXT NOT NULL,
            source_latent_shard_sha256 TEXT NOT NULL,
            target_latent_path TEXT NOT NULL,
            target_latent_key TEXT NOT NULL,
            target_latent_ref TEXT NOT NULL,
            target_latent_tensor_sha256 TEXT,
            target_latent_shard_sha256 TEXT,
            target_foa_path TEXT,
            target_foa_sha256 TEXT,
            target_render_result_sha256 TEXT,
            pair_gain_policy TEXT NOT NULL,
            materialization_status TEXT NOT NULL,
            pair_record_sha256 TEXT NOT NULL
        );
        CREATE INDEX pairs_operation ON pairs(operation_family,operation);
        CREATE INDEX pairs_stratum ON pairs(
            source_count,source_domain,latent_bucket_frames
        );
        CREATE INDEX pairs_work_shard ON pairs(work_shard,row_in_shard);
        """
    )
    return connection


def _mutate_pairs(
    stage: sqlite3.Connection,
    final: sqlite3.Connection,
    donor_groups: Mapping[tuple[str, int], Sequence[int]],
    *,
    split: str,
    target_root: Path,
) -> tuple[Counter[str], Counter[tuple[int, str, int]]]:
    rows = stage.execute(
        """
        SELECT s.output_ordinal,s.operation_family,s.operation,r.payload_zlib
        FROM selected AS s JOIN resolved_sources AS r USING(output_ordinal)
        ORDER BY s.output_ordinal
        """
    ).fetchall()
    operation_counts: Counter[str] = Counter()
    strata: Counter[tuple[int, str, int]] = Counter()
    pending = []
    started = time.time()
    for index, (ordinal, family, operation, payload_zlib) in enumerate(rows, start=1):
        source = _unpack(payload_zlib)
        pair_id = make_pair_id(split, source["source_sample_id"], operation)
        legal_ids = eligible_source_ids(source["sceneplan"], operation)
        if not legal_ids:
            raise RuntimeError(
                f"{source['source_sample_id']}: assigned ineligible {operation}"
            )
        edited_source_id = legal_ids[
            deterministic_u64(pair_id, "edited-source", person=b"edit-source-v1")
            % len(legal_ids)
        ]
        donor_provenance = None
        if operation == OP_EVENT_ADD:
            donor_plan, donor_recipe, donor_provenance = _pick_donor(
                stage,
                donor_groups,
                pair_id=pair_id,
                source_payload=source,
            )
            pair = build_event_addition(
                source["sceneplan"],
                source["render_recipe"],
                donor_plan,
                donor_recipe,
                split=split,
                pair_id=pair_id,
            )
        elif operation == OP_EVENT_REMOVE:
            pair = build_event_removal(
                source["sceneplan"],
                source["render_recipe"],
                split=split,
                source_id=edited_source_id,
                pair_id=pair_id,
            )
        elif operation == OP_RELOCATE:
            pair = build_stationary_relocation(
                source["sceneplan"],
                source["render_recipe"],
                split=split,
                source_id=edited_source_id,
                pair_id=pair_id,
            )
        elif operation == OP_STATIC_TO_LINEAR:
            pair = build_static_to_linear(
                source["sceneplan"],
                source["render_recipe"],
                split=split,
                source_id=edited_source_id,
                pair_id=pair_id,
            )
        elif operation == OP_LINEAR_TO_STATIC:
            pair = build_linear_to_static(
                source["sceneplan"],
                source["render_recipe"],
                split=split,
                source_id=edited_source_id,
                pair_id=pair_id,
            )
        else:
            raise AssertionError(operation)
        if pair.operation_family != family:
            raise RuntimeError("staged operation family changed during mutation")

        work_shard, row_in_shard = divmod(int(ordinal), 1024)
        target_latent_path = (
            target_root
            / "materialized/latents"
            / split
            / f"latents-{split}-{work_shard:05d}.safetensors"
        )
        old_plan_sha = sha256_json(pair.old_sceneplan)
        new_plan_sha = sha256_json(pair.new_sceneplan)
        source_recipe_sha = sha256_json(pair.source_render_recipe)
        target_recipe_sha = sha256_json(pair.target_render_recipe)
        source_result_sha = sha256_json(source["render_result"])
        source_members = list(pair.source_members)
        target_members = list(pair.target_members)
        source_members_sha = sha256_json(source_members)
        target_members_sha = sha256_json(target_members)
        instruction_sha = _digest_text(pair.instruction)
        source_latent_ref = (
            f"{source['source_latent_path']}#{source['source_latent_key']}"
        )
        target_latent_ref = f"{target_latent_path}#{pair.new_sceneplan['sample_id']}"
        record_digest = sha256_json(
            {
                "pair_id": pair_id,
                "split": split,
                "operation": operation,
                "instruction_sha256": instruction_sha,
                "old_sceneplan_sha256": old_plan_sha,
                "new_sceneplan_sha256": new_plan_sha,
                "source_render_recipe_sha256": source_recipe_sha,
                "target_render_recipe_sha256": target_recipe_sha,
                "source_render_result_sha256": source_result_sha,
                "source_members_sha256": source_members_sha,
                "target_members_sha256": target_members_sha,
                "source_latent_ref": source_latent_ref,
                "source_latent_tensor_sha256": source[
                    "source_latent_tensor_sha256"
                ],
                "target_latent_ref": target_latent_ref,
                "pair_gain_policy": PAIR_GAIN_POLICY,
            }
        )
        pending.append(
            (
                int(ordinal),
                pair_id,
                split,
                work_shard,
                row_in_shard,
                int(source["source_ordinal"]),
                source["source_sample_id"],
                pair.new_sceneplan["sample_id"],
                canonical_json(donor_provenance) if donor_provenance else None,
                family,
                operation,
                pair.instruction,
                pair.instruction_template_id,
                instruction_sha,
                len(pair.old_sceneplan["sources"]),
                len(pair.new_sceneplan["sources"]),
                source["domain"],
                scene_domain(pair.new_sceneplan),
                int(source["latent_bucket_frames"]),
                int(source["model_num_samples"]),
                int(source["latent_frames_valid"]),
                _pack(pair.old_sceneplan),
                _pack(pair.new_sceneplan),
                old_plan_sha,
                new_plan_sha,
                _pack(pair.source_render_recipe),
                _pack(pair.target_render_recipe),
                source_recipe_sha,
                target_recipe_sha,
                _pack(source["render_result"]),
                source_result_sha,
                _pack(source_members),
                _pack(target_members),
                source_members_sha,
                target_members_sha,
                canonical_json(list(pair.edited_source_ids)),
                canonical_json(list(pair.unchanged_source_ids)),
                source["model_sceneplan_path"],
                int(source["model_sceneplan_row"]),
                source["source_manifest_path"],
                source["source_manifest_sha256"],
                source["source_foa_sha256"],
                source["source_latent_path"],
                source["source_latent_key"],
                source_latent_ref,
                source["source_latent_tensor_sha256"],
                source["source_latent_shard_sha256"],
                str(target_latent_path),
                pair.new_sceneplan["sample_id"],
                target_latent_ref,
                None,
                None,
                None,
                None,
                None,
                PAIR_GAIN_POLICY,
                "planned",
                record_digest,
            )
        )
        operation_counts[operation] += 1
        strata[
            (
                len(pair.old_sceneplan["sources"]),
                source["domain"],
                int(source["latent_bucket_frames"]),
            )
        ] += 1
        if len(pending) >= 1000:
            placeholders = ",".join("?" for _ in pending[0])
            final.executemany(
                f"INSERT INTO pairs({PAIR_COLUMNS}) VALUES({placeholders})", pending
            )
            final.commit()
            pending.clear()
        if index % 1000 == 0 or index == len(rows):
            print(
                canonical_json(
                    {
                        "stage": "pair_mutation",
                        "rows": index,
                        "total": len(rows),
                        "elapsed_sec": round(time.time() - started, 1),
                    }
                ),
                flush=True,
            )
    if pending:
        placeholders = ",".join("?" for _ in pending[0])
        final.executemany(
            f"INSERT INTO pairs({PAIR_COLUMNS}) VALUES({placeholders})", pending
        )
        final.commit()
    return operation_counts, strata


def _write_metadata(
    connection: sqlite3.Connection,
    *,
    split: str,
    mode: str,
    rows: int,
    source_index: Path,
    source_index_sha: str,
    target_root: Path,
    operations: Mapping[str, int],
    strata: Mapping[tuple[int, str, int], int],
) -> None:
    family_counts: Counter[str] = Counter()
    for operation, count in operations.items():
        family_counts[OPERATION_FAMILY[operation]] += int(count)
    values = {
        "schema": PAIR_INDEX_SCHEMA,
        "schema_version": str(PAIR_INDEX_SCHEMA_VERSION),
        "state": "planned_targets_not_materialized",
        "mode": mode,
        "split": split,
        "rows": str(rows),
        "seed": str(EDITING_PAIR_SEED),
        "editing_pair_contract": EDITING_PAIR_CONTRACT,
        "editing_instruction_contract": EDITING_INSTRUCTION_CONTRACT,
        "editing_ar_input_contract": EDITING_AR_INPUT_CONTRACT,
        "editing_ar_old_sceneplan_input": "false",
        "source_resolution_contract": SOURCE_RESOLUTION_CONTRACT,
        "pair_gain_policy": PAIR_GAIN_POLICY,
        "selection_policy": SELECTION_POLICY,
        "source_index_path": str(source_index),
        "source_index_sha256": source_index_sha,
        "target_root": str(target_root),
        "operation_counts_json": canonical_json(dict(sorted(operations.items()))),
        "operation_family_counts_json": canonical_json(
            dict(sorted(family_counts.items()))
        ),
        "source_strata_counts_json": canonical_json(
            {
                f"{key[0]}|{key[1]}|{key[2]}": count
                for key, count in sorted(strata.items())
            }
        ),
        "builder_path": str(Path(__file__).resolve()),
        "builder_sha256": sha256_file(Path(__file__).resolve()),
        "mutation_module_sha256": sha256_file(MUTATION_MODULE),
        "resolver_module_sha256": sha256_file(RESOLVER_MODULE),
        "shared_contract_path": str(SHARED_CONTRACT),
        "shared_contract_sha256": sha256_file(SHARED_CONTRACT),
    }
    connection.executemany(
        "INSERT INTO metadata(key,value) VALUES(?,?)", sorted(values.items())
    )
    connection.commit()


def _audit_final(connection: sqlite3.Connection, *, split: str, rows: int) -> None:
    count = int(connection.execute("SELECT COUNT(*) FROM pairs").fetchone()[0])
    if count != rows:
        raise RuntimeError(f"pair index row count differs: {count} != {rows}")
    checks = {
        "distinct_pair": "COUNT(DISTINCT pair_id)",
        "distinct_source": "COUNT(DISTINCT source_ordinal)",
        "distinct_target": "COUNT(DISTINCT target_sample_id)",
        "planned": "SUM(materialization_status='planned')",
        "same_frames": "SUM(latent_bucket_frames IN (432,648) AND latent_frames_valid>0 AND latent_frames_valid<=latent_bucket_frames)",
        "aligned_refs": "SUM(source_latent_ref=source_latent_path||'#'||source_latent_key AND target_latent_ref=target_latent_path||'#'||target_latent_key)",
    }
    for label, expression in checks.items():
        observed = int(
            connection.execute(f"SELECT {expression} FROM pairs").fetchone()[0]
        )
        if observed != rows:
            raise RuntimeError(f"final audit {label} failed: {observed} != {rows}")
    split_rows = int(
        connection.execute(
            "SELECT COUNT(*) FROM pairs WHERE split=?", (split,)
        ).fetchone()[0]
    )
    if split_rows != rows:
        raise RuntimeError(f"final audit split failed: {split_rows} != {rows}")
    if connection.execute(
        "SELECT COUNT(*) FROM pairs WHERE target_latent_tensor_sha256 IS NOT NULL"
    ).fetchone()[0]:
        raise RuntimeError("planned index unexpectedly claims materialized targets")


def build(args: argparse.Namespace) -> Path:
    split = str(args.split)
    source_index = Path(args.source_index or SOURCE_INDICES[split]).resolve(strict=True)
    target_root = Path(args.output_root).expanduser().resolve()
    try:
        target_root.relative_to("/mnt/sdb")
    except ValueError as error:
        raise ValueError("Editing data output_root must reside on /mnt/sdb") from error
    index_root = target_root / "pair_index"
    index_root.mkdir(parents=True, exist_ok=True)
    final_path = index_root / f"{split}.sqlite"
    if final_path.exists() and not args.replace:
        raise FileExistsError(f"refusing to replace existing pair index: {final_path}")
    stage_path = index_root / f".{split}.stage.{os.getpid()}.sqlite"
    temporary_path = index_root / f".{split}.tmp.{os.getpid()}.sqlite"
    for path in (stage_path, temporary_path):
        if path.exists():
            raise FileExistsError(f"temporary path already exists: {path}")
    started = time.time()
    stage = _create_stage(stage_path)
    final = _create_final(temporary_path)
    try:
        with P10EditingSourceResolver(
            source_index, expected_split=split
        ) as resolver:
            if args.mode == "full":
                counts = _scan_candidates(resolver, stage)
                _select_full(stage, counts, FULL_COUNTS[split])
                expected_rows = FULL_COUNTS[split]
            else:
                _select_pilot(resolver, stage, int(args.pilot_per_operation))
                expected_rows = len(EDIT_OPERATIONS) * int(args.pilot_per_operation)
            donor_groups = _resolve_selected(resolver, stage)
            operation_counts, strata = _mutate_pairs(
                stage,
                final,
                donor_groups,
                split=split,
                target_root=target_root,
            )
        source_index_sha = sha256_file(source_index)
        _write_metadata(
            final,
            split=split,
            mode=args.mode,
            rows=expected_rows,
            source_index=source_index,
            source_index_sha=source_index_sha,
            target_root=target_root,
            operations=operation_counts,
            strata=strata,
        )
        _audit_final(final, split=split, rows=expected_rows)
        final.execute("PRAGMA optimize")
        final.commit()
        final.close()
        final = None
        os.replace(temporary_path, final_path)
    finally:
        stage.close()
        if final is not None:
            final.close()
        stage_path.unlink(missing_ok=True)
        temporary_path.unlink(missing_ok=True)
    print(
        json.dumps(
            {
                "ok": True,
                "mode": args.mode,
                "split": split,
                "rows": expected_rows,
                "path": str(final_path),
                "sha256": sha256_file(final_path),
                "elapsed_sec": round(time.time() - started, 2),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return final_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=tuple(FULL_COUNTS), required=True)
    parser.add_argument("--mode", choices=("pilot", "full"), default="pilot")
    parser.add_argument("--pilot-per-operation", type=int, default=2)
    parser.add_argument("--source-index", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    if args.mode == "pilot" and int(args.pilot_per_operation) <= 0:
        raise ValueError("pilot_per_operation must be positive")
    build(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
