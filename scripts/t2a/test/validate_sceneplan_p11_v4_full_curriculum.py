#!/usr/bin/env python3
"""Fail-closed audit for the canonical P11-v4 9.6M DDP8 curriculum."""

from __future__ import annotations

import argparse
from collections import Counter
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


SCHEMA = "stable_audio_tools.sceneplan_p11_v4_curriculum"
FULL_CONTRACT = "p10_v11_full9p6m_gue_executable_thought_v2"
SELECTION_CONTRACT = (
    "exact4p8m_plus_pairbalanced_g_multitarget1p6m_u_stress1p6m_"
    "e_pair1p6m_v2"
)
ORDERING_CONTRACT = "p11_v4_full9p6m_ddp8_batch8_supercycle_v2"
AUGMENTATION_SELECTOR_CONTRACT = "adjacent_pair_stable_hash_choose_one_v1"
WORLD_SIZE = 8
LOCAL_BATCH_SIZE = 8
GLOBAL_BATCH_SIZE = WORLD_SIZE * LOCAL_BATCH_SIZE
TASKS = ("generation", "understanding", "editing")
EXPECTED_COLUMNS = (
    "ordinal",
    "curriculum_id",
    "task",
    "family",
    "view_id",
    "template_id",
    "base_manifest_ordinal",
    "base_target_ordinal",
    "sample_id",
    "prompt",
    "known_field_groups_json",
    "target_sceneplan_zlib",
    "edit_spec_json",
    "evidence_transform_json",
    "target_variant",
    "pair_id",
    "pair_label",
)
PATTERNS = (
    {
        "tasks": {"generation": 3, "understanding": 3, "editing": 2},
        "numeric_edit_rows": 2,
    },
    {
        "tasks": {"generation": 3, "understanding": 2, "editing": 3},
        "numeric_edit_rows": 2,
    },
    {
        "tasks": {"generation": 2, "understanding": 3, "editing": 3},
        "numeric_edit_rows": 0,
    },
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_int(*values: Any) -> int:
    payload = json.dumps(
        values, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _augmentation_selected(position: int, *, seed: int) -> bool:
    position = int(position)
    selected_parity = _stable_int(
        int(seed), AUGMENTATION_SELECTOR_CONTRACT, position // 2
    ) % 2
    return position % 2 == selected_parity


def _blocks(values: Iterable[tuple[Any, ...]], size: int):
    block: list[tuple[Any, ...]] = []
    for value in values:
        block.append(value)
        if len(block) == size:
            yield block
            block = []
    if block:
        yield block


def _metadata_int(metadata: dict[str, str], key: str) -> int:
    try:
        return int(metadata[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"full curriculum metadata {key!r} is not an integer") from exc


def _numeric_pair(rows: list[tuple[Any, ...]]) -> bool:
    """Validate one complete P10-legal numeric E direction pair."""

    if len(rows) != 2:
        return False
    pair_ids = {str(row[3]) for row in rows}
    if len(pair_ids) != 1 or None in {row[3] for row in rows}:
        return False
    try:
        specs = [json.loads(str(row[5])) for row in rows]
    except (TypeError, json.JSONDecodeError):
        return False
    operations = {str(value.get("operation")) for value in specs}
    source_ids = {str(value.get("source_id")) for value in specs}
    if len(operations) != 1 or len(source_ids) != 1:
        return False
    operation = next(iter(operations))
    labels = {str(row[4]) for row in rows}
    if operation == "rotate_source":
        values = {float(value.get("delta_azimuth_deg")) for value in specs}
        return values == {-45.0, 45.0} and labels == {"az_-45", "az_+45"}
    if operation == "distance_source":
        values = {float(value.get("distance_factor")) for value in specs}
        return values == {0.75, 1.25} and labels == {
            "distance_0.75",
            "distance_1.25",
        }
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-config", type=Path)
    parser.add_argument("--curriculum", type=Path)
    parser.add_argument("--expected-rows", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--verify-source-hashes",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()
    if args.dataset_config is None and args.curriculum is None:
        raise ValueError("provide --dataset-config or --curriculum")

    config_path: Path | None = None
    config: dict[str, Any] | None = None
    if args.dataset_config is not None:
        config_path = args.dataset_config.expanduser().resolve(strict=True)
        config = load_config(config_path)
    curriculum_value = (
        args.curriculum
        if args.curriculum is not None
        else Path(str(config["p11_v4_curriculum_path"]))
    )
    curriculum = curriculum_value.expanduser().resolve(strict=True)

    connection = sqlite3.connect(
        f"file:{curriculum}?mode=ro&immutable=1", uri=True
    )
    connection.execute("PRAGMA query_only=ON")
    metadata = dict(connection.execute("SELECT key,value FROM metadata"))
    failures: list[str] = []

    required_metadata = {
        "schema": SCHEMA,
        "schema_version": "1",
        "full_scale_contract": FULL_CONTRACT,
        "selection_contract": SELECTION_CONTRACT,
        "augmentation_selector_contract": AUGMENTATION_SELECTOR_CONTRACT,
        "ordering_contract": ORDERING_CONTRACT,
        "ordering_world_size": str(WORLD_SIZE),
        "ordering_batch_size": str(LOCAL_BATCH_SIZE),
        "ordering_global_batch_size": str(GLOBAL_BATCH_SIZE),
        "ordering_seed": "42",
        "distributed_sampler_contract": (
            "strided_shuffle_false_drop_last_false_v1"
        ),
        "ddp_rank_task_counts_identical": "true",
        "max_consecutive_local_batches_without_complete_pair": "1",
        "eval_reserved_templates_present": "false",
        "heldout_sample_overlap": "0",
        "heldout_edit_prompt_exact_overlap": "0",
        "generation_single_target_for_underspecified_prompt": "false",
        "generation_targets_per_underspecified_prompt": "2",
        "generation_reference_sets_are_exhaustive": "false",
        "u_degradation_scope": "synthetic_train_representation_stress_only",
        "editing_uses_complete_p10_atomic_numeric_vocabulary": "true",
        "p10_checkpoint_sha256": (
            "be8c90cd1434bd71f73951531175c3674ff0f3173d5db591e2e1c476152ff59e"
        ),
        "p10_max_latent_frames": "648",
        "p10_source_count": "1-4",
        "p10_motion_profile": "static,linear",
    }
    for key, expected in required_metadata.items():
        if metadata.get(key) != expected:
            failures.append(
                f"metadata {key}={metadata.get(key)!r}, expected {expected!r}"
            )

    columns = tuple(
        str(row[1]) for row in connection.execute("PRAGMA table_info(rows)")
    )
    if columns != EXPECTED_COLUMNS:
        failures.append("rows table schema changed")

    actual_rows = int(connection.execute("SELECT COUNT(*) FROM rows").fetchone()[0])
    expected_rows = (
        int(args.expected_rows)
        if args.expected_rows is not None
        else int(config["p11_v4_curriculum_expected_rows"])
        if config is not None
        else _metadata_int(metadata, "rows")
    )
    if expected_rows <= 0 or expected_rows % 6 or expected_rows % GLOBAL_BATCH_SIZE:
        failures.append("expected row count is not a positive 6x/64x curriculum")
    if actual_rows != expected_rows or _metadata_int(metadata, "rows") != actual_rows:
        failures.append(
            f"curriculum rows are stale: actual={actual_rows}, expected={expected_rows}, "
            f"metadata={metadata.get('rows')!r}"
        )
    base_scenes = expected_rows // 6
    if base_scenes <= 0 or base_scenes % 32:
        failures.append("base-scene count does not form complete 32-scene supercycles")
    if _metadata_int(metadata, "base_scenes") != base_scenes:
        failures.append("base_scenes metadata does not match rows/6")
    if _metadata_int(metadata, "global_optimizer_steps") != expected_rows // 64:
        failures.append("global optimizer-step metadata is stale")

    if config is not None:
        config_expectations = {
            "p11_v4_curriculum_path": str(curriculum),
            "p11_v4_curriculum_expected_rows": expected_rows,
            "p11_v4_curriculum_contract": metadata.get("contract"),
            "p11_v4_curriculum_ordering_contract": ORDERING_CONTRACT,
            "p11_v4_curriculum_ordering_batch_size": LOCAL_BATCH_SIZE,
        }
        for key, expected in config_expectations.items():
            observed = config.get(key)
            if key == "p11_v4_curriculum_path":
                try:
                    observed = str(Path(str(observed)).expanduser().resolve(strict=True))
                except (OSError, TypeError):
                    pass
            if observed != expected:
                failures.append(
                    f"dataset config {key}={observed!r}, expected {expected!r}"
                )

    task_counts = dict(
        connection.execute("SELECT task,COUNT(*) FROM rows GROUP BY task")
    )
    expected_task_counts = {task: 2 * base_scenes for task in TASKS}
    if task_counts != expected_task_counts:
        failures.append(
            f"task counts {task_counts!r}, expected {expected_task_counts!r}"
        )
    family_counts = {
        f"{task}:{family}": int(count)
        for task, family, count in connection.execute(
            "SELECT task,family,COUNT(*) FROM rows GROUP BY task,family"
        )
    }
    expected_families = {
        "generation:exact_compatibility": base_scenes,
        "generation:generation_multitarget_posterior": base_scenes,
        "understanding:exact_compatibility": base_scenes,
        "understanding:understanding_train_evidence_stress": base_scenes,
        "editing:exact_compatibility": base_scenes,
        "editing:editing_numeric_delta_curriculum": base_scenes,
    }
    if family_counts != expected_families:
        failures.append(
            f"family counts {family_counts!r}, expected {expected_families!r}"
        )

    scene_groups, target_ordinals, manifest_ordinals = connection.execute(
        """
        SELECT COUNT(DISTINCT sample_id),
               COUNT(DISTINCT base_target_ordinal),
               COUNT(DISTINCT base_manifest_ordinal)
        FROM rows
        """
    ).fetchone()
    scene_size_counts = {
        int(size): int(count)
        for size, count in connection.execute(
            """
            SELECT scene_rows,COUNT(*) FROM (
                SELECT sample_id,COUNT(*) AS scene_rows
                FROM rows GROUP BY sample_id
            ) GROUP BY scene_rows ORDER BY scene_rows
            """
        )
    }
    scene_group_failures = int(
        connection.execute(
            """
            SELECT COUNT(*) FROM (
                SELECT sample_id
                FROM rows
                GROUP BY sample_id
                HAVING COUNT(*) NOT IN (4,8)
                    OR SUM(CASE WHEN task='generation' THEN 1 ELSE 0 END)
                       NOT IN (1,3)
                    OR SUM(CASE WHEN task='understanding' THEN 1 ELSE 0 END) != 2
                    OR SUM(CASE WHEN task='editing' THEN 1 ELSE 0 END)
                       NOT IN (1,3)
                    OR SUM(CASE WHEN task='generation' THEN 1 ELSE 0 END)
                       != SUM(CASE WHEN task='editing' THEN 1 ELSE 0 END)
                    OR SUM(CASE WHEN family='exact_compatibility'
                                      AND task='generation' THEN 1 ELSE 0 END) != 1
                    OR SUM(CASE WHEN family='exact_compatibility'
                                      AND task='understanding' THEN 1 ELSE 0 END) != 1
                    OR SUM(CASE WHEN family='exact_compatibility'
                                      AND task='editing' THEN 1 ELSE 0 END) != 1
                    OR SUM(CASE WHEN family='generation_multitarget_posterior'
                                THEN 1 ELSE 0 END) NOT IN (0,2)
                    OR SUM(CASE WHEN family='understanding_train_evidence_stress'
                                THEN 1 ELSE 0 END) != 1
                    OR SUM(CASE WHEN family='editing_numeric_delta_curriculum'
                                THEN 1 ELSE 0 END) NOT IN (0,2)
                    OR COUNT(DISTINCT base_target_ordinal) != 1
                    OR COUNT(DISTINCT CASE WHEN task='generation'
                                           THEN base_manifest_ordinal END) != 1
                    OR COUNT(DISTINCT CASE WHEN task='understanding'
                                           THEN base_manifest_ordinal END) != 1
                    OR COUNT(DISTINCT CASE WHEN task='editing'
                                           THEN base_manifest_ordinal END) != 1
            )
            """
        ).fetchone()[0]
    )
    expected_coverage = (base_scenes, base_scenes, 3 * base_scenes)
    observed_coverage = (
        int(scene_groups),
        int(target_ordinals),
        int(manifest_ordinals),
    )
    expected_scene_sizes = {4: base_scenes // 2, 8: base_scenes // 2}
    if (
        observed_coverage != expected_coverage
        or scene_size_counts != expected_scene_sizes
        or scene_group_failures
    ):
        failures.append(
            "base-scene coverage is incomplete/duplicated: "
            f"observed={observed_coverage!r}, expected={expected_coverage!r}, "
            f"scene_sizes={scene_size_counts!r}, "
            f"expected_scene_sizes={expected_scene_sizes!r}, "
            f"invalid_groups={scene_group_failures}"
        )

    selector_rows = 0
    selector_mismatches = 0
    selector_pair_counts: Counter[int] = Counter()
    selector_parity_counts: Counter[int] = Counter()
    for (position_value,) in connection.execute(
        "SELECT DISTINCT base_manifest_ordinal / 3 "
        "FROM rows WHERE family='generation_multitarget_posterior' "
        "ORDER BY base_manifest_ordinal / 3"
    ):
        position = int(position_value)
        selector_rows += 1
        selector_pair_counts[position // 2] += 1
        selector_parity_counts[position % 2] += 1
        if not _augmentation_selected(position, seed=42):
            selector_mismatches += 1
    selector_pair_failures = sum(
        int(count != 1) for count in selector_pair_counts.values()
    )
    expected_selector_rows = base_scenes // 2
    if (
        selector_rows != expected_selector_rows
        or len(selector_pair_counts) != expected_selector_rows
        or selector_pair_failures
        or selector_mismatches
    ):
        failures.append(
            "G/E augmentation selector is incomplete or source-order biased: "
            f"rows={selector_rows}/{expected_selector_rows}, "
            f"pairs={len(selector_pair_counts)}/{expected_selector_rows}, "
            f"pair_failures={selector_pair_failures}, "
            f"selector_mismatches={selector_mismatches}"
        )
    if base_scenes >= 10_000:
        parity_imbalance = abs(
            selector_parity_counts.get(0, 0)
            - selector_parity_counts.get(1, 0)
        )
        if parity_imbalance > base_scenes // 100:
            failures.append(
                "pair-balanced selector retained an unexpected parity imbalance: "
                f"{dict(selector_parity_counts)!r}"
            )

    stress_counts = dict(
        connection.execute(
            "SELECT view_id,COUNT(*) FROM rows "
            "WHERE family='understanding_train_evidence_stress' GROUP BY view_id"
        )
    )
    if len(stress_counts) != 8 or set(stress_counts.values()) != {base_scenes // 8}:
        failures.append(f"U stress views are not uniformly balanced: {stress_counts!r}")

    generation_group_failures = int(
        connection.execute(
            """
            SELECT COUNT(*) FROM (
                SELECT pair_id
                FROM rows
                WHERE family='generation_multitarget_posterior'
                GROUP BY pair_id
                HAVING pair_id IS NULL
                    OR COUNT(*) != 2
                    OR COUNT(DISTINCT prompt) != 1
                    OR COUNT(DISTINCT target_variant) != 2
                    OR COUNT(DISTINCT hex(target_sceneplan_zlib)) != 2
            )
            """
        ).fetchone()[0]
    )
    generation_groups = int(
        connection.execute(
            "SELECT COUNT(DISTINCT pair_id) FROM rows "
            "WHERE family='generation_multitarget_posterior'"
        ).fetchone()[0]
    )
    if generation_groups != base_scenes // 2 or generation_group_failures:
        failures.append(
            "G multitarget groups are incomplete/non-distinct: "
            f"groups={generation_groups}, failures={generation_group_failures}"
        )

    numeric_operation_counts = dict(
        connection.execute(
            """
            SELECT json_extract(edit_spec_json,'$.operation'),COUNT(*)
            FROM rows WHERE family='editing_numeric_delta_curriculum'
            GROUP BY json_extract(edit_spec_json,'$.operation')
            """
        )
    )
    expected_operation_counts = {
        "distance_source": base_scenes // 2,
        "rotate_source": base_scenes // 2,
    }
    if numeric_operation_counts != expected_operation_counts:
        failures.append(
            "numeric E operation counts changed: "
            f"{numeric_operation_counts!r} != {expected_operation_counts!r}"
        )

    rank_task_counts = [Counter() for _ in range(WORLD_SIZE)]
    local_batches = 0
    pair_batches = 0
    pair_free_gap = [0] * WORLD_SIZE
    max_pair_free_gap = [0] * WORLD_SIZE
    observed_rows = 0
    expected_ordinal = 0
    cursor = connection.execute(
        """
        SELECT ordinal,task,family,pair_id,pair_label,edit_spec_json
        FROM rows ORDER BY ordinal
        """
    )
    for step, block in enumerate(_blocks(cursor, GLOBAL_BATCH_SIZE)):
        if len(block) != GLOBAL_BATCH_SIZE:
            failures.append(f"terminal global block has {len(block)} rows")
            break
        pattern = PATTERNS[step % len(PATTERNS)]
        for row in block:
            if int(row[0]) != expected_ordinal:
                failures.append(
                    f"ordinal discontinuity {row[0]} != {expected_ordinal}"
                )
                break
            expected_ordinal += 1
            observed_rows += 1
        for rank in range(WORLD_SIZE):
            local = [
                block[rank + WORLD_SIZE * position]
                for position in range(LOCAL_BATCH_SIZE)
            ]
            local_batches += 1
            counts = Counter(str(row[1]) for row in local)
            rank_task_counts[rank].update(counts)
            if dict(counts) != pattern["tasks"]:
                failures.append(
                    f"step={step} rank={rank} task pattern {dict(counts)!r}, "
                    f"expected {pattern['tasks']!r}"
                )
            numeric = [
                row
                for row in local
                if str(row[2]) == "editing_numeric_delta_curriculum"
            ]
            if len(numeric) != pattern["numeric_edit_rows"]:
                failures.append(
                    f"step={step} rank={rank} has {len(numeric)} numeric E rows"
                )
            has_pair = bool(numeric) and _numeric_pair(numeric)
            if numeric and not has_pair:
                failures.append(f"step={step} rank={rank} split/invalid E pair")
            pair_batches += int(has_pair)
            pair_free_gap[rank] = 0 if has_pair else pair_free_gap[rank] + 1
            max_pair_free_gap[rank] = max(
                max_pair_free_gap[rank], pair_free_gap[rank]
            )
        if len(failures) >= 100:
            failures.append("stopped DDP scan after 100 failures")
            break
    connection.close()

    expected_global_steps = expected_rows // GLOBAL_BATCH_SIZE
    expected_local_batches = expected_global_steps * WORLD_SIZE
    expected_pair_batches = expected_local_batches * 2 // 3
    if observed_rows != expected_rows:
        failures.append(f"DDP scan observed {observed_rows}/{expected_rows} rows")
    if local_batches != expected_local_batches:
        failures.append(
            f"DDP scan observed {local_batches}/{expected_local_batches} local batches"
        )
    if pair_batches != expected_pair_batches:
        failures.append(
            f"complete E-pair batches {pair_batches}/{expected_pair_batches}"
        )
    if max(max_pair_free_gap, default=0) != 1:
        failures.append(f"max E-pair-free gaps changed: {max_pair_free_gap!r}")
    expected_rank = {task: base_scenes // 4 for task in TASKS}
    if any(dict(counts) != expected_rank for counts in rank_task_counts):
        failures.append(
            f"rank task counts are not identical {expected_rank!r}: "
            f"{[dict(value) for value in rank_task_counts]!r}"
        )

    source_hashes: dict[str, dict[str, Any]] = {}
    if args.verify_source_hashes:
        for label, path_key, hash_key in (
            ("manifest", "source_manifest", "source_manifest_sha256"),
            ("index", "source_index", "source_index_sha256"),
            ("heldout", "heldout_challenge", "heldout_challenge_sha256"),
            ("builder", "builder", "builder_sha256"),
        ):
            try:
                path = Path(metadata[path_key]).expanduser().resolve(strict=True)
                observed = _sha256_file(path)
                expected = metadata[hash_key]
                matches = observed == expected
                source_hashes[label] = {
                    "path": str(path),
                    "observed_sha256": observed,
                    "expected_sha256": expected,
                    "matches": matches,
                }
                if not matches:
                    failures.append(f"{label} source hash changed")
            except (KeyError, OSError) as exc:
                failures.append(f"cannot verify {label} source: {exc}")

    report = {
        "schema": "stable_audio_tools.sceneplan_p11_v4_full_curriculum_audit",
        "schema_version": 1,
        "status": "PASS" if not failures else "FAIL",
        "curriculum": str(curriculum),
        "curriculum_sha256": _sha256_file(curriculum),
        "dataset_config": str(config_path) if config_path is not None else None,
        "dataset_config_sha256": (
            _sha256_file(config_path) if config_path is not None else None
        ),
        "rows": actual_rows,
        "base_scenes": base_scenes,
        "global_optimizer_steps": expected_global_steps,
        "task_counts": task_counts,
        "family_counts": family_counts,
        "base_scene_coverage": {
            "distinct_sample_ids": int(scene_groups),
            "distinct_target_ordinals": int(target_ordinals),
            "distinct_manifest_ordinals": int(manifest_ordinals),
            "scene_size_counts": scene_size_counts,
            "invalid_contract_groups": scene_group_failures,
        },
        "augmentation_selector": {
            "contract": metadata.get("augmentation_selector_contract"),
            "selected_scenes": selector_rows,
            "distinct_adjacent_pairs": len(selector_pair_counts),
            "pair_count_failures": selector_pair_failures,
            "selector_mismatches": selector_mismatches,
            "selected_parity_counts": dict(selector_parity_counts),
        },
        "understanding_stress_view_counts": stress_counts,
        "generation_multitarget_groups": generation_groups,
        "numeric_edit_operation_counts": numeric_operation_counts,
        "local_batches": local_batches,
        "local_batches_with_complete_e_pair": pair_batches,
        "max_pair_free_gap_by_rank": max_pair_free_gap,
        "rank_task_counts": [dict(value) for value in rank_task_counts],
        "source_hashes": source_hashes,
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
