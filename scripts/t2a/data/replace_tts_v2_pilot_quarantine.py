#!/usr/bin/env python3
"""Replace calibrated TTS-v2 pilot quarantines from the frozen reserve pool.

The stable pilot sample ID and its dataset/duration/motion/room stratum are
preserved.  Only the dry donor and donor-dependent room/trajectory recipe are
changed.  Selection/QC history is archived before the atomic selection swap.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Iterable

import pyarrow.compute as pc
import pyarrow.parquet as pq


SCRIPT_DIR = Path(__file__).resolve().parent
import sys

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from render_tts_v2_pilot import (  # noqa: E402
    DATASET_ROOT,
    DEFAULT_OUTPUT,
    LEDGER,
    MAX_MODEL_SAMPLES,
    MODEL_SAMPLE_RATE,
    atomic_write_json,
    atomic_write_jsonl,
    deterministic_digest,
    iter_jsonl,
    room_recipe,
    trajectory,
)


DEFAULT_QC_MANIFEST = (
    DEFAULT_OUTPUT
    / "qc/asr_distil_large_v3_calibrated/speech_qc_manifest.jsonl"
)
TAIL_BY_ROOM = {
    "dry": 40 + round(0.05 * MODEL_SAMPLE_RATE),
    "moderate": 40 + round(0.12 * MODEL_SAMPLE_RATE),
    "reverberant": 40 + round(0.25 * MODEL_SAMPLE_RATE),
    "outdoor": 40,
}


def ensure_sdb(path: Path) -> None:
    resolved = path.expanduser().resolve(strict=False)
    try:
        resolved.relative_to("/mnt/sdb")
    except ValueError as error:
        raise ValueError(f"replacement output must remain on SDB: {resolved}") from error


def load_history(path: Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path)) if path.is_file() else []


def candidate_cell(row: dict[str, Any]) -> tuple[str, int]:
    duration = float(row["duration_sec"])
    if duration < 2.0:
        bucket = 0
    elif duration < 4.0:
        bucket = 1
    elif duration < 6.0:
        bucket = 2
    elif duration < 8.0:
        bucket = 3
    else:
        bucket = 4
    return str(row["source_dataset"]), bucket


def rebuild_slot(
    donor: dict[str, Any],
    current: dict[str, Any],
    *,
    round_number: int,
) -> dict[str, Any]:
    row = dict(donor)
    sample_id = str(current["sample_id"])
    motion = str(current["motion"])
    room_class = str(current["room_class"])
    source_samples = int(row["model_num_samples"])
    recipe_seed = int(
        deterministic_digest(
            20260814,
            "tts_pilot_replacement",
            sample_id,
            row["asset_id"],
            round_number,
        )[:16],
        16,
    )
    room = room_recipe(room_class, recipe_seed)
    render_tail = min(MAX_MODEL_SAMPLES - source_samples, TAIL_BY_ROOM[room_class])
    if render_tail < 40:
        raise RuntimeError(f"replacement {row['asset_id']} cannot preserve Pyroom delay")
    row.update(
        {
            "sample_id": sample_id,
            "partition": str(current["partition"]),
            "duration_bin": int(current["duration_bin"]),
            "motion": motion,
            "room_class": room_class,
            "recipe_seed": recipe_seed,
            "room": room,
            "trajectory": trajectory(
                motion,
                source_samples / MODEL_SAMPLE_RATE,
                room,
                recipe_seed ^ 0xA57D1C,
            ),
            "render_tail_samples": render_tail,
            "scene_num_samples": source_samples + render_tail,
            "replacement_round": round_number,
            "replaced_asset_id": current["asset_id"],
        }
    )
    return row


def choose_replacements(
    ledger: Path,
    slots: dict[tuple[str, int], deque[dict[str, Any]]],
    current_by_id: dict[str, dict[str, Any]],
    excluded_assets: set[str],
    round_number: int,
) -> tuple[dict[str, dict[str, Any]], Counter[str]]:
    columns = pq.ParquetFile(ledger).schema_arrow.names
    table = pq.read_table(
        ledger,
        columns=columns,
        filters=[("pool", "=", "reserve"), ("replacement_split", "=", "train")],
    ).sort_by([("selection_rank", "ascending")])
    replacements: dict[str, dict[str, Any]] = {}
    stats: Counter[str] = Counter()
    for donor in table.to_pylist():
        stats["scanned"] += 1
        asset_id = str(donor["asset_id"])
        if asset_id in excluded_assets:
            stats["excluded_history_or_current"] += 1
            continue
        cell = candidate_cell(donor)
        if not slots[cell]:
            stats["unneeded_cell"] += 1
            continue
        if int(donor["model_num_samples"]) > MAX_MODEL_SAMPLES - 40:
            stats["insufficient_tail_room"] += 1
            continue
        qc = slots[cell].popleft()
        sample_id = str(qc["sample_id"])
        current = current_by_id[sample_id]
        replacement = rebuild_slot(donor, current, round_number=round_number)
        if candidate_cell(replacement) != cell:
            raise RuntimeError("replacement duration cell changed")
        replacements[sample_id] = replacement
        excluded_assets.add(asset_id)
        stats["selected"] += 1
        if not any(slots.values()):
            break
    missing = {"|".join(map(str, key)): len(value) for key, value in slots.items() if value}
    if missing:
        raise RuntimeError(f"reserve cannot fill pilot quarantine strata: {missing}")
    return replacements, stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qc-manifest", type=Path, default=DEFAULT_QC_MANIFEST)
    parser.add_argument("--ledger", type=Path, default=LEDGER)
    parser.add_argument("--pilot-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    pilot_root = args.pilot_root.expanduser().resolve(strict=True)
    ensure_sdb(pilot_root)
    selection_path = pilot_root / "selection.jsonl"
    qc_path = args.qc_manifest.expanduser().resolve(strict=True)
    ledger = args.ledger.expanduser().resolve(strict=True)
    current = list(iter_jsonl(selection_path))
    qc_rows = list(iter_jsonl(qc_path))
    if len(current) != 2_000 or len(qc_rows) != 2_000:
        raise RuntimeError("replacement requires exactly 2,000 selection and QC rows")
    if any(row.get("status") == "error" for row in qc_rows):
        raise RuntimeError("QC has audit errors; repair the audit before donor replacement")
    quarantines = [row for row in qc_rows if row.get("status") != "pass"]
    if not quarantines:
        print(json.dumps({"ok": True, "replacements": 0}, indent=2))
        return 0

    current_by_id = {str(row["sample_id"]): row for row in current}
    if len(current_by_id) != 2_000:
        raise RuntimeError("current pilot sample IDs are not unique")
    qc_ids = {str(row["sample_id"]) for row in qc_rows}
    if qc_ids != set(current_by_id):
        raise RuntimeError("QC and frozen selection sample IDs differ")

    audit_root = DATASET_ROOT / "audit/tts_2k_replacements"
    ensure_sdb(audit_root)
    audit_root.mkdir(parents=True, exist_ok=True)
    history_path = audit_root / "quarantine_history.jsonl"
    history = load_history(history_path)
    round_number = 1 + max((int(row.get("round", 0)) for row in history), default=0)
    round_root = audit_root / f"round_{round_number:02d}"
    round_root.mkdir(parents=True, exist_ok=False)

    slots: dict[tuple[str, int], deque[dict[str, Any]]] = defaultdict(deque)
    for row in sorted(quarantines, key=lambda value: str(value["sample_id"])):
        slots[(str(row["source_dataset"]), int(row["strata"]["duration_bin"]))].append(row)
    excluded_assets = {str(row["asset_id"]) for row in current}
    excluded_assets.update(str(row["asset_id"]) for row in history if row.get("asset_id"))
    replacements, stats = choose_replacements(
        ledger,
        slots,
        current_by_id,
        excluded_assets,
        round_number,
    )
    next_selection = [replacements.get(str(row["sample_id"]), row) for row in current]
    next_selection.sort(key=lambda row: str(row["sample_id"]))
    if len({str(row["asset_id"]) for row in next_selection}) != 2_000:
        raise RuntimeError("replacement selection contains duplicate donor assets")

    shutil.copy2(selection_path, round_root / "selection_before.jsonl")
    shutil.copy2(qc_path, round_root / "speech_qc_before.jsonl")
    qc_summary = qc_path.with_name("speech_qc_summary.json")
    if qc_summary.is_file():
        shutil.copy2(qc_summary, round_root / "speech_qc_summary_before.json")
    additions = []
    qc_by_id = {str(row["sample_id"]): row for row in qc_rows}
    for sample_id, replacement in sorted(replacements.items()):
        old = current_by_id[sample_id]
        qc = qc_by_id[sample_id]
        additions.append(
            {
                "round": round_number,
                "sample_id": sample_id,
                "source_dataset": old["source_dataset"],
                "asset_id": old["asset_id"],
                "replacement_asset_id": replacement["asset_id"],
                "duration_bin": old["duration_bin"],
                "motion": old["motion"],
                "room_class": old["room_class"],
                "failure_reasons": qc.get("failure_reasons", []),
                "action": "quarantined_and_replaced_from_frozen_reserve",
            }
        )

    # Commit only after every stratum has a valid unique replacement.
    atomic_write_jsonl(selection_path, next_selection)
    atomic_write_jsonl(history_path, [*history, *additions])
    ready = pilot_root / "READY"
    ready.unlink(missing_ok=True)
    report = {
        "schema": "stable_audio_tools.tts_v2_pilot_replacement_round",
        "schema_version": 2,
        "round": round_number,
        "qc_manifest": str(qc_path),
        "replacement_count": len(replacements),
        "strata": {
            "|".join(map(str, key)): value
            for key, value in sorted(
                Counter(
                    (row["source_dataset"], int(row["duration_bin"]))
                    for row in replacements.values()
                ).items()
            )
        },
        "selection_stats": dict(stats),
        "history": str(history_path),
        "archive": str(round_root),
        "ready_removed_for_safe_rerender": not ready.exists(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    atomic_write_json(round_root / "report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
