#!/usr/bin/env python3
"""Audit the exact P11-v4 row view produced by strided DDP sampling."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stable_audio_tools.configuration import load_config  # noqa: E402


CANONICAL_ORDERING = "p11_v4_ddp8_rank_balanced_pair_aware_batch8_v7"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _paraphrase(label: str) -> str:
    if "_p" not in label:
        raise ValueError(f"Editing pair label lacks a paraphrase suffix: {label!r}")
    value = "p" + label.rsplit("_p", 1)[1]
    if value not in {"p0", "p1"}:
        raise ValueError(f"unsupported Editing paraphrase {value!r}")
    return value


def _direction(spec_json: str) -> int:
    spec = json.loads(spec_json)
    operation = str(spec.get("operation") or "")
    if operation == "rotate_source":
        value = float(spec["delta_azimuth_deg"])
        if value == 0.0:
            raise ValueError("zero rotation is not a direction")
        return -1 if value < 0.0 else 1
    if operation == "distance_source":
        value = float(spec["distance_factor"])
        if value == 1.0:
            raise ValueError("identity distance is not a direction")
        return -1 if value < 1.0 else 1
    raise ValueError(f"unsupported paired Editing operation {operation!r}")


def _blocks(values: Iterable[tuple[Any, ...]], size: int) -> Iterable[list[tuple[Any, ...]]]:
    block: list[tuple[Any, ...]] = []
    for value in values:
        block.append(value)
        if len(block) == size:
            yield block
            block = []
    if block:
        yield block


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-config", type=Path, required=True)
    parser.add_argument("--curriculum", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--local-batch-size", type=int, default=8)
    parser.add_argument("--expect-ordering-contract")
    args = parser.parse_args()
    if args.world_size != 8 or args.local_batch_size != 8:
        raise ValueError("canonical P11-v4 DDP audit is frozen at 8x8")

    config_path = args.dataset_config.expanduser().resolve(strict=True)
    config = load_config(config_path)
    curriculum_path = (
        args.curriculum
        if args.curriculum is not None
        else Path(config["p11_v4_curriculum_path"])
    ).expanduser().resolve(strict=True)
    expected_rows = int(config["p11_v4_curriculum_expected_rows"])
    world_size = int(args.world_size)
    local_batch_size = int(args.local_batch_size)
    global_batch_size = world_size * local_batch_size

    connection = sqlite3.connect(
        f"file:{curriculum_path}?mode=ro&immutable=1", uri=True
    )
    metadata = dict(connection.execute("SELECT key,value FROM metadata"))
    ordering_contract = metadata.get("ordering_contract")
    config_ordering = config.get("p11_v4_curriculum_ordering_contract")
    failures: list[str] = []
    if ordering_contract != config_ordering:
        failures.append("dataset config and SQLite ordering contracts differ")
    if (
        args.expect_ordering_contract is not None
        and ordering_contract != args.expect_ordering_contract
    ):
        failures.append(
            f"ordering contract is {ordering_contract!r}, expected "
            f"{args.expect_ordering_contract!r}"
        )
    if expected_rows % global_batch_size:
        failures.append("dataset has a partial 64-row global optimizer step")

    rank_task_counts = [Counter() for _ in range(world_size)]
    local_task_min = {
        task: local_batch_size
        for task in ("generation", "understanding", "editing")
    }
    local_task_max = {task: 0 for task in local_task_min}
    local_batches = 0
    local_batches_with_all_tasks = 0
    local_batches_with_complete_pair = 0
    local_batches_with_split_pair = 0
    local_complete_pair_min = local_batch_size
    local_complete_pair_max = 0
    global_complete_pair_min = global_batch_size
    global_complete_pair_max = 0
    global_steps = 0
    observed_rows = 0
    expected_ordinal = 0
    global_task_counts: Counter[str] = Counter()

    cursor = connection.execute(
        """
        SELECT ordinal,task,pair_id,pair_label,edit_spec_json
        FROM rows ORDER BY ordinal
        """
    )
    for block in _blocks(cursor, global_batch_size):
        if len(block) != global_batch_size:
            failures.append(f"terminal global block has {len(block)} rows")
            break
        global_steps += 1
        global_groups: dict[tuple[str, str], set[int]] = defaultdict(set)
        for row in block:
            ordinal, task, pair_id, pair_label, edit_json = row
            if int(ordinal) != expected_ordinal:
                failures.append(
                    f"ordinal discontinuity {ordinal} != {expected_ordinal}"
                )
            expected_ordinal += 1
            observed_rows += 1
            global_task_counts[str(task)] += 1
            if str(task) == "editing" and pair_id is not None:
                global_groups[(str(pair_id), _paraphrase(str(pair_label)))].add(
                    _direction(str(edit_json))
                )
        global_complete = sum(
            signs == {-1, 1} for signs in global_groups.values()
        )
        global_complete_pair_min = min(global_complete_pair_min, global_complete)
        global_complete_pair_max = max(global_complete_pair_max, global_complete)

        for rank in range(world_size):
            local = [
                block[rank + world_size * position]
                for position in range(local_batch_size)
            ]
            local_batches += 1
            counts = Counter(str(row[1]) for row in local)
            rank_task_counts[rank].update(counts)
            if set(counts) == {"generation", "understanding", "editing"}:
                local_batches_with_all_tasks += 1
            for task in local_task_min:
                local_task_min[task] = min(local_task_min[task], counts[task])
                local_task_max[task] = max(local_task_max[task], counts[task])

            groups: dict[tuple[str, str], set[int]] = defaultdict(set)
            paired_rows = 0
            for _, task, pair_id, pair_label, edit_json in local:
                if str(task) != "editing" or pair_id is None:
                    continue
                paired_rows += 1
                groups[(str(pair_id), _paraphrase(str(pair_label)))].add(
                    _direction(str(edit_json))
                )
            complete = sum(signs == {-1, 1} for signs in groups.values())
            if complete:
                local_batches_with_complete_pair += 1
            if 2 * complete != paired_rows:
                local_batches_with_split_pair += 1
            local_complete_pair_min = min(local_complete_pair_min, complete)
            local_complete_pair_max = max(local_complete_pair_max, complete)
    connection.close()

    if observed_rows != expected_rows:
        failures.append(f"observed {observed_rows} rows, expected {expected_rows}")
    if global_complete_pair_min <= 0:
        failures.append("one or more global optimizer steps lack an E pair")
    if local_batches_with_all_tasks != local_batches:
        failures.append(
            f"only {local_batches_with_all_tasks}/{local_batches} local batches "
            "contain G/U/E"
        )
    if local_batches_with_complete_pair != local_batches:
        failures.append(
            f"only {local_batches_with_complete_pair}/{local_batches} local "
            "batches contain a complete E pair"
        )
    if local_batches_with_split_pair:
        failures.append(
            f"{local_batches_with_split_pair} local batches split an E pair"
        )

    expected_rank_counts = {
        task: count // world_size for task, count in global_task_counts.items()
    }
    if any(count % world_size for count in global_task_counts.values()):
        failures.append("global task counts are not rank-divisible")
    normalized_rank_counts = [dict(counts) for counts in rank_task_counts]
    if any(counts != expected_rank_counts for counts in normalized_rank_counts):
        failures.append("epoch task counts differ across DDP ranks")

    report = {
        "schema": "stable_audio_tools.p11_v4_ddp_sampler_audit",
        "schema_version": 1,
        "status": "PASS" if not failures else "FAIL",
        "dataset_config": str(config_path),
        "dataset_config_sha256": _sha256_file(config_path),
        "curriculum": str(curriculum_path),
        "curriculum_sha256": _sha256_file(curriculum_path),
        "ordering_contract": ordering_contract,
        "canonical_ordering_contract": CANONICAL_ORDERING,
        "world_size": world_size,
        "local_batch_size": local_batch_size,
        "global_batch_size": global_batch_size,
        "rows": observed_rows,
        "global_optimizer_steps": global_steps,
        "global_task_counts": dict(global_task_counts),
        "global_complete_pair_instances_min": global_complete_pair_min,
        "global_complete_pair_instances_max": global_complete_pair_max,
        "local_batches": local_batches,
        "local_batches_with_all_tasks": local_batches_with_all_tasks,
        "local_batches_with_complete_pair": local_batches_with_complete_pair,
        "local_batches_with_split_pair": local_batches_with_split_pair,
        "local_task_count_min": local_task_min,
        "local_task_count_max": local_task_max,
        "local_complete_pair_instances_min": local_complete_pair_min,
        "local_complete_pair_instances_max": local_complete_pair_max,
        "expected_rank_task_counts": expected_rank_counts,
        "rank_task_counts": normalized_rank_counts,
        "failures": failures,
        "script": str(Path(__file__).resolve()),
        "script_sha256": _sha256_file(Path(__file__).resolve()),
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
