#!/usr/bin/env python3
"""Audit full-corpus augmentation selection against frozen source ordering."""

from __future__ import annotations
import os

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import sys
from typing import Any, Mapping
import zlib


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.t2a.data.build_sceneplan_p11_v4_full_curriculum import (  # noqa: E402
    AUGMENTATION_SELECTOR_CONTRACT,
    _augmentation_selected,
)


DEFAULT_MANIFEST = Path(
    os.environ.get("AMBIT_DATA_ROOT", "data") + "/sceneplan_v2_1p124m/p11_single_turn_15s_v2/"
    "manifests/p11_train_4p8m_v6.sqlite"
)
THRESHOLDS = {
    "maximum_room_total_variation": 0.03,
    "maximum_source_kind_total_variation": 0.03,
    "maximum_mean_source_count_gap": 0.05,
    "maximum_mean_duration_sec_gap": 0.25,
    "maximum_any_speech_rate_gap": 0.03,
    "maximum_any_linear_rate_gap": 0.03,
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _new_stats() -> dict[str, Any]:
    return {
        "rows": 0,
        "source_count_sum": 0,
        "duration_sec_sum": 0.0,
        "any_speech": 0,
        "any_linear": 0,
        "room_counts": Counter(),
        "source_kind_counts": Counter(),
    }


def _add(stats: dict[str, Any], plan: Mapping[str, Any]) -> None:
    sources = list(plan["sources"])
    kinds = [str(source["kind"]) for source in sources]
    stats["rows"] += 1
    stats["source_count_sum"] += len(sources)
    stats["duration_sec_sum"] += float(plan["duration_sec"])
    stats["any_speech"] += int("speech" in kinds)
    stats["any_linear"] += int(
        any(str(source["trajectory"]["type"]) == "linear" for source in sources)
    )
    stats["room_counts"][str(plan["room"]["type"])] += 1
    stats["source_kind_counts"].update(kinds)


def _finalize(stats: Mapping[str, Any]) -> dict[str, Any]:
    rows = int(stats["rows"])
    if rows <= 0:
        raise RuntimeError("augmentation balance audit selected no rows")
    return {
        "rows": rows,
        "mean_source_count": float(stats["source_count_sum"]) / rows,
        "mean_duration_sec": float(stats["duration_sec_sum"]) / rows,
        "any_speech_rate": float(stats["any_speech"]) / rows,
        "any_linear_rate": float(stats["any_linear"]) / rows,
        "room_counts": dict(sorted(stats["room_counts"].items())),
        "source_kind_counts": dict(sorted(stats["source_kind_counts"].items())),
    }


def _total_variation(a: Mapping[str, int], b: Mapping[str, int]) -> float:
    keys = set(a) | set(b)
    a_total = sum(int(a.get(key, 0)) for key in keys)
    b_total = sum(int(b.get(key, 0)) for key in keys)
    if a_total <= 0 or b_total <= 0:
        return math.inf
    return 0.5 * sum(
        abs(int(a.get(key, 0)) / a_total - int(b.get(key, 0)) / b_total)
        for key in keys
    )


def _gaps(selected: Mapping[str, Any], held_out: Mapping[str, Any]) -> dict[str, float]:
    return {
        "room_total_variation": _total_variation(
            selected["room_counts"], held_out["room_counts"]
        ),
        "source_kind_total_variation": _total_variation(
            selected["source_kind_counts"], held_out["source_kind_counts"]
        ),
        "mean_source_count_gap": abs(
            float(selected["mean_source_count"])
            - float(held_out["mean_source_count"])
        ),
        "mean_duration_sec_gap": abs(
            float(selected["mean_duration_sec"])
            - float(held_out["mean_duration_sec"])
        ),
        "any_speech_rate_gap": abs(
            float(selected["any_speech_rate"])
            - float(held_out["any_speech_rate"])
        ),
        "any_linear_rate_gap": abs(
            float(selected["any_linear_rate"])
            - float(held_out["any_linear_rate"])
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--sample-pairs", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.seed != 42:
        raise ValueError("canonical full augmentation audit fixes seed 42")
    if args.sample_pairs < 1_000:
        raise ValueError("augmentation balance audit requires at least 1000 pairs")

    manifest = args.manifest.expanduser().resolve(strict=True)
    connection = sqlite3.connect(
        f"file:{manifest}?mode=ro&immutable=1", uri=True
    )
    connection.execute("PRAGMA query_only=ON")
    metadata = dict(connection.execute("SELECT key,value FROM metadata"))
    base_scenes = int(metadata.get("base_samples", -1))
    if base_scenes <= 0 or base_scenes % 2:
        raise RuntimeError("source manifest lacks an even positive base-scene count")
    total_pairs = base_scenes // 2
    sample_pairs = min(int(args.sample_pairs), total_pairs)
    pair_indices = [
        (index * (total_pairs - 1)) // max(1, sample_pairs - 1)
        for index in range(sample_pairs)
    ]
    if len(set(pair_indices)) != sample_pairs:
        raise RuntimeError("pair sampling produced duplicate source positions")

    balanced = {True: _new_stats(), False: _new_stats()}
    retired_even = {True: _new_stats(), False: _new_stats()}
    selected_parity_counts: Counter[int] = Counter()
    selector_pair_failures = 0
    try:
        for pair_index in pair_indices:
            selected_in_pair = 0
            for parity in (0, 1):
                position = 2 * pair_index + parity
                row = connection.execute(
                    "SELECT task,target_sceneplan_zlib FROM rows WHERE ordinal=?",
                    (3 * position,),
                ).fetchone()
                if row is None or str(row[0]) != "generation":
                    raise RuntimeError(
                        f"source manifest lacks canonical G row at position {position}"
                    )
                plan = json.loads(zlib.decompress(row[1]))
                selected = _augmentation_selected(position, seed=args.seed)
                selected_in_pair += int(selected)
                if selected:
                    selected_parity_counts[parity] += 1
                _add(balanced[selected], plan)
                _add(retired_even[parity == 0], plan)
            selector_pair_failures += int(selected_in_pair != 1)
    finally:
        connection.close()

    balanced_final = {
        "selected": _finalize(balanced[True]),
        "held_out": _finalize(balanced[False]),
    }
    retired_final = {
        "selected": _finalize(retired_even[True]),
        "held_out": _finalize(retired_even[False]),
    }
    balanced_gaps = _gaps(
        balanced_final["selected"], balanced_final["held_out"]
    )
    retired_gaps = _gaps(retired_final["selected"], retired_final["held_out"])
    gates = {
        "exactly_one_selected_per_sampled_pair": selector_pair_failures == 0,
        "both_source_parities_selected": set(selected_parity_counts) == {0, 1},
        "room_balance": balanced_gaps["room_total_variation"]
        <= THRESHOLDS["maximum_room_total_variation"],
        "source_kind_balance": balanced_gaps["source_kind_total_variation"]
        <= THRESHOLDS["maximum_source_kind_total_variation"],
        "source_count_balance": balanced_gaps["mean_source_count_gap"]
        <= THRESHOLDS["maximum_mean_source_count_gap"],
        "duration_balance": balanced_gaps["mean_duration_sec_gap"]
        <= THRESHOLDS["maximum_mean_duration_sec_gap"],
        "speech_balance": balanced_gaps["any_speech_rate_gap"]
        <= THRESHOLDS["maximum_any_speech_rate_gap"],
        "motion_balance": balanced_gaps["any_linear_rate_gap"]
        <= THRESHOLDS["maximum_any_linear_rate_gap"],
        "strict_room_bias_reduction_vs_retired_even": balanced_gaps[
            "room_total_variation"
        ]
        < retired_gaps["room_total_variation"],
        "strict_kind_bias_reduction_vs_retired_even": balanced_gaps[
            "source_kind_total_variation"
        ]
        < retired_gaps["source_kind_total_variation"],
    }
    report: dict[str, Any] = {
        "schema": "stable_audio_tools.p11_v4_full_augmentation_balance_audit",
        "schema_version": 1,
        "status": "PASS" if all(gates.values()) else "FAIL",
        "manifest": str(manifest),
        "manifest_sha256": _sha256_file(manifest),
        "base_scenes": base_scenes,
        "sample_pairs": sample_pairs,
        "seed": args.seed,
        "selector_contract": AUGMENTATION_SELECTOR_CONTRACT,
        "selected_parity_counts": dict(selected_parity_counts),
        "selector_pair_failures": selector_pair_failures,
        "thresholds": THRESHOLDS,
        "pair_balanced_selector": {
            **balanced_final,
            "gaps": balanced_gaps,
        },
        "retired_fixed_even_counterfactual": {
            **retired_final,
            "gaps": retired_gaps,
        },
        "gates": gates,
        "script": str(Path(__file__).resolve()),
        "script_sha256": _sha256_file(Path(__file__).resolve()),
    }
    unhashed = json.dumps(
        report, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    report["report_sha256_without_self"] = hashlib.sha256(unhashed).hexdigest()
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
