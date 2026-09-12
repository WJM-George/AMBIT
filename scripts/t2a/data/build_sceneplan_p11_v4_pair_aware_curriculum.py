#!/usr/bin/env python3
"""Interleave a P11-v4 curriculum while preserving exact batch-8 E pairs.

This builder is intentionally an ordering-only transform.  It does not create,
delete, or alter any training example.  Every non-ordinal SQLite row value must
match the frozen curriculum-v1 multiset exactly.

For each Editing pair identity, p0 and p1 contain the two opposite legal
numeric edits.  A pair-aware batch contains two *different* pair/paraphrase
instances together with two G and two U rows:

    G U E(a,-) E(a,+) G U E(b,-) E(b,+)

Consequently all 120 pair/paraphrase instances expose both directions once per
sequential epoch at the frozen batch size 8.  Pair-aware and remainder batches
are interleaved so no two full batches without paired Editing supervision are
adjacent and the epoch cannot end in an Editing-free suffix.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import time
from typing import Any, Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stable_audio_tools.data.sceneplan_p11_v4_curriculum import (  # noqa: E402
    P11_V4_CURRICULUM_CONTRACT,
    P11_V4_CURRICULUM_SCHEMA,
    P11_V4_CURRICULUM_VERSION,
)


DEFAULT_SOURCE = Path(
    "/mnt/sdb/audio_dataset/sceneplan_v2_1p124m/p11_single_turn_15s_v2/"
    "p11_v4_curriculum/"
    "p11_train_pilot90_curriculum_v7_seed42.sqlite"
)
DEFAULT_OUTPUT = Path(
    "/mnt/sdb/audio_dataset/sceneplan_v2_1p124m/p11_single_turn_15s_v2/"
    "p11_v4_curriculum/"
    "p11_train_pilot90_curriculum_v8_pair_aware_interleaved_batch8_seed42.sqlite"
)
ORDERING_CONTRACT = (
    "p11_v4_pair_aware_interleaved_batch8_diverse_instruction_surface_v6"
)
DDP8_ORDERING_CONTRACT = (
    "p11_v4_ddp8_rank_balanced_pair_aware_batch8_v7"
)
ROW_COLUMNS = (
    "ordinal,curriculum_id,task,family,view_id,template_id,"
    "base_manifest_ordinal,base_target_ordinal,sample_id,prompt,"
    "known_field_groups_json,target_sceneplan_zlib,edit_spec_json,"
    "evidence_transform_json,target_variant,pair_id,pair_label"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_digest(seed: int, *values: Any) -> bytes:
    payload = json.dumps(
        [seed, *values], ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).digest()


def _field_bytes(value: Any) -> bytes:
    if value is None:
        return b"n"
    if isinstance(value, bytes):
        payload = value
        tag = b"b"
    else:
        payload = str(value).encode("utf-8")
        tag = b"s"
    return tag + len(payload).to_bytes(8, "big") + payload


def _row_content_digest(rows: Iterable[Sequence[Any]]) -> str:
    row_digests: list[bytes] = []
    for row in rows:
        digest = hashlib.sha256()
        for value in row[1:]:
            digest.update(_field_bytes(value))
        row_digests.append(digest.digest())
    aggregate = hashlib.sha256()
    for digest in sorted(row_digests):
        aggregate.update(digest)
    return aggregate.hexdigest()


def _direction(edit_json: str) -> int:
    spec = json.loads(edit_json)
    operation = str(spec.get("operation") or "")
    if operation == "rotate_source":
        value = float(spec["delta_azimuth_deg"])
        if value == 0.0:
            raise ValueError("zero rotation is not a counterfactual edit")
        return -1 if value < 0.0 else 1
    if operation == "distance_source":
        value = float(spec["distance_factor"])
        if value == 1.0:
            raise ValueError("identity distance is not a counterfactual edit")
        return -1 if value < 1.0 else 1
    raise ValueError(f"unsupported counterfactual operation {operation!r}")


def _paraphrase(label: str) -> str:
    if "_p" not in label:
        raise ValueError(f"counterfactual label lacks paraphrase suffix: {label!r}")
    value = "p" + label.rsplit("_p", 1)[1]
    if value not in {"p0", "p1"}:
        raise ValueError(f"unexpected counterfactual paraphrase {value!r}")
    return value


def _rank_rows(rows: list[tuple[Any, ...]], *, seed: int, stream: str) -> list[tuple[Any, ...]]:
    return sorted(
        rows,
        key=lambda row: _stable_digest(seed, stream, str(row[1])),
    )


def _ordered_rows(
    rows: list[tuple[Any, ...]],
    *,
    base_scenes: int,
    seed: int,
    batch_size: int,
) -> tuple[list[tuple[Any, ...]], dict[str, Any]]:
    if batch_size != 8:
        raise ValueError("pair-aware curriculum v4 is frozen at batch size 8")
    task_rows: dict[str, list[tuple[Any, ...]]] = defaultdict(list)
    pair_instances: dict[str, dict[str, dict[int, tuple[Any, ...]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    pair_row_ids: set[str] = set()
    for row in rows:
        task = str(row[2])
        task_rows[task].append(row)
        pair_id, pair_label, edit_json = row[15], row[16], row[12]
        if task == "editing" and pair_id is not None:
            if edit_json is None or pair_label is None:
                raise RuntimeError("paired E row lacks edit spec or pair label")
            paraphrase = _paraphrase(str(pair_label))
            direction = _direction(str(edit_json))
            identity = str(row[1])
            if direction in pair_instances[str(pair_id)][paraphrase]:
                raise RuntimeError("duplicate pair/paraphrase/direction row")
            pair_instances[str(pair_id)][paraphrase][direction] = row
            pair_row_ids.add(identity)

    if set(task_rows) != {"generation", "understanding", "editing"}:
        raise RuntimeError(f"unexpected task set: {set(task_rows)}")
    expected_task_rows = base_scenes * 9
    if {key: len(value) for key, value in task_rows.items()} != {
        "generation": expected_task_rows,
        "understanding": expected_task_rows,
        "editing": expected_task_rows,
    }:
        raise RuntimeError("curriculum task balance changed before pair-aware ordering")
    expected_pair_identities = base_scenes * 2
    if len(pair_instances) != expected_pair_identities:
        raise RuntimeError(
            f"expected {expected_pair_identities} E pair identities, "
            f"got {len(pair_instances)}"
        )
    for pair_id, paraphrases in pair_instances.items():
        if set(paraphrases) != {"p0", "p1"}:
            raise RuntimeError(f"pair {pair_id!r} does not contain p0/p1")
        for paraphrase, directions in paraphrases.items():
            if set(directions) != {-1, 1}:
                raise RuntimeError(
                    f"pair {pair_id!r}/{paraphrase} lacks opposite directions"
                )

    pair_ids = sorted(
        pair_instances,
        key=lambda value: _stable_digest(seed, "pair-order", value),
    )
    generation = deque(_rank_rows(task_rows["generation"], seed=seed, stream="g"))
    understanding = deque(
        _rank_rows(task_rows["understanding"], seed=seed, stream="u")
    )
    exact_editing = _rank_rows(
        [row for row in task_rows["editing"] if str(row[1]) not in pair_row_ids],
        seed=seed,
        stream="e-exact",
    )
    if len(pair_row_ids) != base_scenes * 8 or len(exact_editing) != base_scenes:
        raise RuntimeError("paired/exact Editing split changed")

    pair_batch_entries: list[
        tuple[list[tuple[Any, ...]], list[str]]
    ] = []
    pair_count = len(pair_ids)
    for index, left_id in enumerate(pair_ids):
        # Cyclically shift p1 so one batch never groups both paraphrases of the
        # same identity; the reasoner therefore selects both explicit pairs.
        right_id = pair_ids[(index + 1) % pair_count]
        left = pair_instances[left_id]["p0"]
        right = pair_instances[right_id]["p1"]
        batch = [
            generation.popleft(),
            understanding.popleft(),
            left[-1],
            left[1],
            generation.popleft(),
            understanding.popleft(),
            right[-1],
            right[1],
        ]
        if len({str(batch[2][15]), str(batch[6][15])}) != 2:
            raise RuntimeError("pair-aware batch collapsed to one pair identity")
        pair_batch_entries.append(
            (batch, [f"{left_id}/p0", f"{right_id}/p1"])
        )

    # Preserve every remaining example once.  G/U alternate, with exact E
    # distributed through the tail.  No paired E example can enter drop_last.
    remainder: list[tuple[Any, ...]] = []
    exact_queue = deque(exact_editing)
    while generation or understanding or exact_queue:
        if generation:
            remainder.append(generation.popleft())
        if understanding:
            remainder.append(understanding.popleft())
        if exact_queue and (
            len(remainder) % batch_size in {2, 5} or not generation
        ):
            remainder.append(exact_queue.popleft())
    remainder_full_rows = len(remainder) // batch_size * batch_size
    remainder_batches = [
        remainder[start : start + batch_size]
        for start in range(0, remainder_full_rows, batch_size)
    ]
    remainder_tail = remainder[remainder_full_rows:]
    if len(remainder_batches) >= len(pair_batch_entries):
        raise RuntimeError("cannot interleave every remainder batch with paired E")

    # Begin with a pair-aware batch, alternate while remainder batches exist,
    # then finish with the remaining pair-aware batches.  This keeps every row
    # and exact pair intact while removing the old 26-batch Editing-free suffix.
    full_batches: list[list[tuple[Any, ...]]] = []
    pair_batches: list[dict[str, Any]] = []
    pair_batch_indices: set[int] = set()
    for index, (pair_batch, pair_instances_in_batch) in enumerate(
        pair_batch_entries
    ):
        pair_batch_index = len(full_batches)
        full_batches.append(pair_batch)
        pair_batch_indices.add(pair_batch_index)
        pair_batches.append(
            {
                "batch_index": pair_batch_index,
                "pair_instances": pair_instances_in_batch,
            }
        )
        if index < len(remainder_batches):
            full_batches.append(remainder_batches[index])

    ordered = [row for batch in full_batches for row in batch]
    ordered.extend(remainder_tail)
    consecutive_without_pair = 0
    max_consecutive_without_pair = 0
    for batch_index in range(len(full_batches)):
        if batch_index in pair_batch_indices:
            consecutive_without_pair = 0
        else:
            consecutive_without_pair += 1
            max_consecutive_without_pair = max(
                max_consecutive_without_pair, consecutive_without_pair
            )

    if len(ordered) != len(rows) or len({str(row[1]) for row in ordered}) != len(rows):
        raise RuntimeError("pair-aware ordering lost or duplicated curriculum rows")
    if _row_content_digest(ordered) != _row_content_digest(rows):
        raise RuntimeError("pair-aware ordering changed the row-content multiset")
    pair_batches_payload = json.dumps(
        pair_batches,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    pair_batches_preview = (
        pair_batches
        if len(pair_batches) <= 8
        else [*pair_batches[:4], *pair_batches[-4:]]
    )
    return ordered, {
        "pair_identity_count": len(pair_instances),
        "pair_paraphrase_instance_count": 2 * len(pair_instances),
        "pair_aware_full_batches": len(pair_batches),
        "remainder_full_batches": len(remainder_batches),
        "full_batches": len(full_batches),
        "pairs_per_pair_aware_batch": 2,
        "pair_aware_rows": len(pair_batches) * batch_size,
        "dropped_tail_rows": len(rows) % batch_size,
        "max_consecutive_full_batches_without_paired_edit": (
            max_consecutive_without_pair
        ),
        "last_pair_aware_full_batch_index": max(pair_batch_indices),
        "pair_batches_sha256": hashlib.sha256(pair_batches_payload).hexdigest(),
        "pair_batches_preview": pair_batches_preview,
    }


def _ddp_rank_balanced_rows(
    rows: list[tuple[Any, ...]],
    *,
    seed: int,
    world_size: int,
    local_batch_size: int,
) -> tuple[list[tuple[Any, ...]], dict[str, Any]]:
    """Lay out global superblocks for PyTorch's strided DistributedSampler.

    ``DistributedSampler(shuffle=False)`` gives rank ``r`` dataset ordinals
    ``r, r + world_size, ...``.  Merely making each *contiguous* batch of eight
    balanced therefore pins one task to a rank at world-size eight.  This
    transform writes each 64-row optimizer-step block in column-major rank
    order, so every rank receives one balanced local batch and complete
    counterfactual Editing pairs.
    """

    if world_size != 8 or local_batch_size != 8:
        raise ValueError("canonical DDP-aware P11-v4 ordering requires 8x8")
    global_batch_size = world_size * local_batch_size
    if len(rows) % global_batch_size:
        raise RuntimeError(
            "DDP-aware curriculum rows must be exactly divisible by global batch 64"
        )

    output: list[tuple[Any, ...]] = []
    rank_epoch_counts = [Counter() for _ in range(world_size)]
    local_task_min = {task: local_batch_size for task in (
        "generation", "understanding", "editing"
    )}
    local_task_max = {task: 0 for task in local_task_min}
    local_complete_pair_min = local_batch_size
    local_complete_pair_max = 0
    local_batches_with_all_tasks = 0
    local_batches_with_complete_pair = 0
    covered_pair_instances: set[str] = set()

    for global_step, start in enumerate(range(0, len(rows), global_batch_size)):
        block = rows[start : start + global_batch_size]
        by_task: dict[str, list[tuple[Any, ...]]] = defaultdict(list)
        pair_groups: dict[tuple[str, str], list[tuple[Any, ...]]] = defaultdict(list)
        exact_editing: list[tuple[Any, ...]] = []
        for row in block:
            task = str(row[2])
            by_task[task].append(row)
            if task != "editing":
                continue
            if row[15] is None:
                exact_editing.append(row)
                continue
            if row[12] is None or row[16] is None:
                raise RuntimeError("paired Editing row is missing spec/label")
            key = (str(row[15]), _paraphrase(str(row[16])))
            pair_groups[key].append(row)

        task_counts = Counter(str(row[2]) for row in block)
        layout = (
            task_counts["generation"],
            task_counts["understanding"],
            task_counts["editing"],
            sum(len(value) for value in pair_groups.values()),
            len(exact_editing),
        )
        if layout not in {
            (24, 24, 16, 16, 0),
            (20, 20, 24, 16, 8),
            (16, 16, 32, 32, 0),
        }:
            raise RuntimeError(
                f"unexpected global P11-v4 superblock composition {layout}"
            )
        for key, pair_rows in pair_groups.items():
            if len(pair_rows) != 2 or {
                _direction(str(row[12])) for row in pair_rows
            } != {-1, 1}:
                raise RuntimeError(
                    f"global superblock has incomplete Editing pair {key!r}"
                )

        rank_rows: list[list[tuple[Any, ...]]] = [
            [] for _ in range(world_size)
        ]
        rank_order = [
            (global_step + offset) % world_size for offset in range(world_size)
        ]
        ordered_pairs = sorted(
            pair_groups.items(),
            key=lambda item: _stable_digest(
                seed, "ddp-pair", global_step, item[0][0], item[0][1]
            ),
        )
        for pair_index, (key, pair_rows) in enumerate(ordered_pairs):
            rank = rank_order[pair_index % world_size]
            rank_rows[rank].extend(
                sorted(pair_rows, key=lambda row: _direction(str(row[12])))
            )
            covered_pair_instances.add(f"{key[0]}/{key[1]}")

        ordered_exact = sorted(
            exact_editing,
            key=lambda row: _stable_digest(
                seed, "ddp-exact-e", global_step, str(row[1])
            ),
        )
        if ordered_exact:
            if len(ordered_exact) != world_size:
                raise RuntimeError("exact Editing rows cannot be rank-balanced")
            for rank, row in zip(rank_order, ordered_exact):
                rank_rows[rank].append(row)

        if layout == (24, 24, 16, 16, 0):
            target_generation = [3] * world_size
        elif layout == (20, 20, 24, 16, 8):
            extra_generation_ranks = set(rank_order[: world_size // 2])
            target_generation = [
                3 if rank in extra_generation_ranks else 2
                for rank in range(world_size)
            ]
        else:
            target_generation = [2] * world_size
        target_understanding = [
            local_batch_size - len(rank_rows[rank]) - target_generation[rank]
            for rank in range(world_size)
        ]

        generation = deque(
            sorted(
                by_task["generation"],
                key=lambda row: _stable_digest(
                    seed, "ddp-g", global_step, str(row[1])
                ),
            )
        )
        understanding = deque(
            sorted(
                by_task["understanding"],
                key=lambda row: _stable_digest(
                    seed, "ddp-u", global_step, str(row[1])
                ),
            )
        )
        for rank in rank_order:
            for _ in range(target_generation[rank]):
                rank_rows[rank].append(generation.popleft())
            for _ in range(target_understanding[rank]):
                rank_rows[rank].append(understanding.popleft())
        if generation or understanding:
            raise RuntimeError("DDP task allocation left unassigned G/U rows")

        for rank in range(world_size):
            local = sorted(
                rank_rows[rank],
                key=lambda row: _stable_digest(
                    seed, "ddp-local-order", global_step, rank, str(row[1])
                ),
            )
            if len(local) != local_batch_size:
                raise RuntimeError("DDP rank received a non-canonical local batch")
            rank_rows[rank] = local
            counts = Counter(str(row[2]) for row in local)
            if set(counts) != {"generation", "understanding", "editing"}:
                raise RuntimeError(
                    f"DDP rank {rank} local batch lacks a task: {counts}"
                )
            local_batches_with_all_tasks += 1
            rank_epoch_counts[rank].update(counts)
            for task in local_task_min:
                local_task_min[task] = min(local_task_min[task], counts[task])
                local_task_max[task] = max(local_task_max[task], counts[task])

            local_groups: dict[tuple[str, str], set[int]] = defaultdict(set)
            local_paired_rows = 0
            for row in local:
                if str(row[2]) != "editing" or row[15] is None:
                    continue
                local_paired_rows += 1
                key = (str(row[15]), _paraphrase(str(row[16])))
                local_groups[key].add(_direction(str(row[12])))
            complete = sum(signs == {-1, 1} for signs in local_groups.values())
            if complete * 2 != local_paired_rows:
                raise RuntimeError(
                    f"DDP rank {rank} split an Editing pair across local batches"
                )
            if complete <= 0:
                raise RuntimeError("DDP local batch has no complete Editing pair")
            local_batches_with_complete_pair += 1
            local_complete_pair_min = min(local_complete_pair_min, complete)
            local_complete_pair_max = max(local_complete_pair_max, complete)

        # Column-major storage is the inverse of DistributedSampler's strided
        # rank view: rank r reads positions r, r+8, ..., r+56.
        for local_position in range(local_batch_size):
            for rank in range(world_size):
                output.append(rank_rows[rank][local_position])

    global_counts = Counter(str(row[2]) for row in rows)
    expected_per_rank = {
        task: count // world_size for task, count in global_counts.items()
    }
    if any(count % world_size for count in global_counts.values()):
        raise RuntimeError("global task rows are not exactly rank-divisible")
    if any(dict(counts) != expected_per_rank for counts in rank_epoch_counts):
        raise RuntimeError(
            "DDP epoch task balance differs across ranks: "
            f"expected={expected_per_rank}, observed={rank_epoch_counts}"
        )
    if len(output) != len(rows) or _row_content_digest(output) != _row_content_digest(rows):
        raise RuntimeError("DDP-aware ordering changed the row-content multiset")

    rank_task_counts = [
        {task: int(counts[task]) for task in sorted(expected_per_rank)}
        for counts in rank_epoch_counts
    ]
    local_batches = len(rows) // local_batch_size
    return output, {
        "world_size": world_size,
        "local_batch_size": local_batch_size,
        "global_batch_size": global_batch_size,
        "global_optimizer_steps": len(rows) // global_batch_size,
        "local_batches": local_batches,
        "local_batches_with_all_tasks": local_batches_with_all_tasks,
        "local_batches_with_complete_pair": local_batches_with_complete_pair,
        "local_task_count_min": local_task_min,
        "local_task_count_max": local_task_max,
        "local_complete_pair_instances_min": local_complete_pair_min,
        "local_complete_pair_instances_max": local_complete_pair_max,
        "rank_task_counts": rank_task_counts,
        "rank_task_counts_identical": True,
        "pair_paraphrase_instances_covered": len(covered_pair_instances),
        "distributed_sampler_contract": "strided_shuffle_false_drop_last_false_v1",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--ddp-world-size", type=int, choices=(1, 8), default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    source_path = args.source.expanduser().resolve(strict=True)
    output_path = args.output.expanduser().resolve()
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    source = sqlite3.connect(f"file:{source_path}?mode=ro&immutable=1", uri=True)
    source.execute("PRAGMA query_only=ON")
    metadata = dict(source.execute("SELECT key,value FROM metadata"))
    base_scenes = int(metadata.get("base_scenes", 0))
    expected_rows = base_scenes * 27
    if base_scenes <= 0:
        raise RuntimeError("source curriculum lacks a positive base_scenes count")
    required = {
        "schema": P11_V4_CURRICULUM_SCHEMA,
        "schema_version": str(P11_V4_CURRICULUM_VERSION),
        "contract": P11_V4_CURRICULUM_CONTRACT,
        "rows": str(expected_rows),
    }
    for key, expected in required.items():
        if metadata.get(key) != expected:
            raise RuntimeError(
                f"source curriculum metadata {key}={metadata.get(key)!r}, "
                f"expected {expected!r}"
            )
    rows = source.execute(f"SELECT {ROW_COLUMNS} FROM rows ORDER BY ordinal").fetchall()
    source.close()
    if len(rows) != expected_rows:
        raise RuntimeError(
            f"source curriculum has {len(rows)} rows, expected {expected_rows}"
        )
    source_content_sha = _row_content_digest(rows)
    ordered, prelayout_audit = _ordered_rows(
        rows,
        base_scenes=base_scenes,
        seed=args.seed,
        batch_size=args.batch_size,
    )
    ddp_audit = None
    ordering_contract = ORDERING_CONTRACT
    if args.ddp_world_size == 8:
        ordered, ddp_audit = _ddp_rank_balanced_rows(
            ordered,
            seed=args.seed,
            world_size=args.ddp_world_size,
            local_batch_size=args.batch_size,
        )
        ordering_contract = DDP8_ORDERING_CONTRACT

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".sqlite", dir=output_path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    destination = sqlite3.connect(temporary)
    completed = False
    started = time.perf_counter()
    try:
        destination.executescript(
            """
            PRAGMA journal_mode=OFF;
            PRAGMA synchronous=OFF;
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL) WITHOUT ROWID;
            CREATE TABLE rows (
                ordinal INTEGER PRIMARY KEY,
                curriculum_id TEXT NOT NULL UNIQUE,
                task TEXT NOT NULL,
                family TEXT NOT NULL,
                view_id TEXT NOT NULL,
                template_id TEXT NOT NULL,
                base_manifest_ordinal INTEGER NOT NULL,
                base_target_ordinal INTEGER NOT NULL,
                sample_id TEXT NOT NULL,
                prompt TEXT NOT NULL,
                known_field_groups_json TEXT NOT NULL,
                target_sceneplan_zlib BLOB NOT NULL,
                edit_spec_json TEXT,
                evidence_transform_json TEXT NOT NULL,
                target_variant INTEGER NOT NULL,
                pair_id TEXT,
                pair_label TEXT
            );
            CREATE INDEX rows_task_family ON rows(task,family);
            CREATE INDEX rows_base_scene ON rows(base_target_ordinal);
            CREATE INDEX rows_pair ON rows(pair_id);
            CREATE INDEX rows_template ON rows(template_id);
            """
        )
        rewritten = [(ordinal, *row[1:]) for ordinal, row in enumerate(ordered)]
        destination.executemany(
            "INSERT INTO rows VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rewritten
        )
        output_metadata = dict(metadata)
        output_metadata.update(
            {
                "ordering_contract": ordering_contract,
                "ordering_batch_size": str(args.batch_size),
                "ordering_seed": str(args.seed),
                "source_curriculum": str(source_path),
                "source_curriculum_sha256": _sha256_file(source_path),
                "source_row_content_multiset_sha256": source_content_sha,
                "row_content_multiset_sha256": _row_content_digest(rewritten),
                "pair_identity_count": str(
                    prelayout_audit["pair_identity_count"]
                ),
                "pair_paraphrase_instance_count": str(
                    prelayout_audit["pair_paraphrase_instance_count"]
                ),
                "builder": str(Path(__file__).resolve()),
                "builder_sha256": _sha256_file(Path(__file__).resolve()),
            }
        )
        if ddp_audit is None:
            output_metadata.update(
                {
                    "pair_aware_full_batches": str(
                        prelayout_audit["pair_aware_full_batches"]
                    ),
                    "remainder_full_batches": str(
                        prelayout_audit["remainder_full_batches"]
                    ),
                    "full_batches": str(prelayout_audit["full_batches"]),
                    "pairs_per_pair_aware_batch": str(
                        prelayout_audit["pairs_per_pair_aware_batch"]
                    ),
                    "max_consecutive_full_batches_without_paired_edit": str(
                        prelayout_audit[
                            "max_consecutive_full_batches_without_paired_edit"
                        ]
                    ),
                    "last_pair_aware_full_batch_index": str(
                        prelayout_audit["last_pair_aware_full_batch_index"]
                    ),
                }
            )
        else:
            output_metadata.update(
                {
                    "ordering_world_size": str(ddp_audit["world_size"]),
                    "ordering_global_batch_size": str(
                        ddp_audit["global_batch_size"]
                    ),
                    "global_optimizer_steps": str(
                        ddp_audit["global_optimizer_steps"]
                    ),
                    "ddp_local_batches": str(ddp_audit["local_batches"]),
                    "ddp_local_batches_with_all_tasks": str(
                        ddp_audit["local_batches_with_all_tasks"]
                    ),
                    "ddp_local_batches_with_complete_pair": str(
                        ddp_audit["local_batches_with_complete_pair"]
                    ),
                    "ddp_local_complete_pair_instances_min": str(
                        ddp_audit["local_complete_pair_instances_min"]
                    ),
                    "ddp_local_complete_pair_instances_max": str(
                        ddp_audit["local_complete_pair_instances_max"]
                    ),
                    "ddp_rank_task_counts_json": json.dumps(
                        ddp_audit["rank_task_counts"], sort_keys=True
                    ),
                    "ddp_rank_task_counts_identical": "true",
                    "distributed_sampler_contract": ddp_audit[
                        "distributed_sampler_contract"
                    ],
                    "prelayout_pair_aware_full_batches": str(
                        prelayout_audit["pair_aware_full_batches"]
                    ),
                    "prelayout_remainder_full_batches": str(
                        prelayout_audit["remainder_full_batches"]
                    ),
                }
            )
        destination.executemany(
            "INSERT INTO metadata(key,value) VALUES (?,?)",
            sorted(output_metadata.items()),
        )
        destination.commit()
        destination.execute("VACUUM")
        destination.commit()
        completed = True
    finally:
        destination.close()
        if not completed:
            temporary.unlink(missing_ok=True)
    os.replace(temporary, output_path)

    report = {
        "schema": "stable_audio_tools.p11_v4_pair_aware_curriculum_build",
        "schema_version": 5,
        "status": "BUILT",
        "ordering_contract": ordering_contract,
        "source": str(source_path),
        "source_sha256": _sha256_file(source_path),
        "output": str(output_path),
        "output_sha256": _sha256_file(output_path),
        "row_content_multiset_sha256": source_content_sha,
        "base_scenes": base_scenes,
        "rows": len(ordered),
        "task_counts": dict(Counter(str(row[2]) for row in ordered)),
        "audit": {
            "prelayout": prelayout_audit,
            "ddp": ddp_audit,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    report_path = output_path.with_suffix(".build.json")
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
