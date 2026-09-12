#!/usr/bin/env python3
"""Build the deterministic dry-source Instruct A2T pilot for ScenePlan v2.

The pilot intentionally excludes speech.  Speech transcript and speaker
metadata remain authoritative in the frozen LibriTTS/HiFiTTS ledger.  Music
and sound are split between deliberately weak/generic legacy labels and more
descriptive labels so the model is tested on the semantic gap that P7
exposed, rather than only on easy examples.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

import pyarrow.compute as pc
import pyarrow.parquet as pq


DATASET_ROOT = Path("/mnt/sdb/audio_dataset/sceneplan_v2_1p124m")
DEFAULT_CATALOG = (
    DATASET_ROOT / "source_catalog/nonspeech/nonspeech_signal_catalog.parquet"
)
DEFAULT_OUTPUT = DATASET_ROOT / "audit/a2t_pilot_100"
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?")


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def normalized_label(value: Any) -> str:
    return " ".join(WORD_RE.findall(str(value or "").casefold()))


def is_generic(kind: str, description: str) -> bool:
    label = normalized_label(description)
    words = label.split()
    if kind == "music":
        return label in {
            "music",
            "background music",
            "instrumental music",
            "musical instrument",
        }
    return len(words) <= 2 or label in {
        "animal",
        "vehicle",
        "car",
        "engine",
        "outside",
        "water",
        "wind",
        "noise",
        "sound effect",
    }


def take_rows(
    rows: list[dict[str, Any]],
    *,
    count: int,
    unique_labels: bool,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen_labels: set[str] = set()
    for row in rows:
        label = normalized_label(row["description"])
        if unique_labels and label in seen_labels:
            continue
        path = Path(str(row["dry_audio_path"]))
        if not path.is_file():
            continue
        output.append(row)
        seen_labels.add(label)
        if len(output) == count:
            return output
    raise RuntimeError(
        f"could select only {len(output)}/{count} rows "
        f"(unique_labels={unique_labels})"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--per-kind", type=int, default=50)
    args = parser.parse_args()
    if args.per_kind <= 0 or args.per_kind % 2:
        raise ValueError("--per-kind must be a positive even number")

    catalog = args.catalog.expanduser().resolve(strict=True)
    output = args.output_root.expanduser().resolve(strict=False)
    try:
        output.relative_to("/mnt/sdb")
    except ValueError as error:
        raise ValueError(f"A2T pilot must persist on SDB: {output}") from error

    table = pq.read_table(
        catalog,
        columns=[
            "asset_id",
            "source_id",
            "source_dataset",
            "kind",
            "description",
            "dry_audio_path",
            "source_audio_sha256",
            "native_sample_rate_hz",
            "native_num_samples",
            "model_num_samples",
            "duration_sec",
            "eligible",
            "selection_rank",
        ],
    )
    table = table.filter(pc.equal(table["eligible"], True)).sort_by(
        [("selection_rank", "ascending")]
    )
    rows = table.to_pylist()
    half = args.per_kind // 2
    selected: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for kind in ("music", "sound"):
        kind_rows = [row for row in rows if row["kind"] == kind]
        generic = [row for row in kind_rows if is_generic(kind, row["description"])]
        detailed = [row for row in kind_rows if not is_generic(kind, row["description"])]
        generic_selected = take_rows(
            generic,
            count=half,
            # Twenty-five identical "Music" labels are useful: they directly
            # measure whether audio, rather than legacy text, drives the A2T.
            unique_labels=kind != "music",
        )
        detailed_selected = take_rows(detailed, count=half, unique_labels=True)
        for group, values in (
            ("generic_legacy_label", generic_selected),
            ("descriptive_legacy_label", detailed_selected),
        ):
            for row in values:
                selected.append(
                    {
                        "pilot_ordinal": len(selected),
                        "pilot_group": group,
                        "annotation_id": f"sha256:{row['source_audio_sha256']}",
                        "asset_id": str(row["asset_id"]),
                        "source_id": str(row["source_id"]),
                        "source_dataset": str(row["source_dataset"]),
                        "kind": kind,
                        "raw_label": str(row["description"]),
                        "audio_path": str(Path(row["dry_audio_path"]).resolve(strict=True)),
                        "source_audio_sha256": str(row["source_audio_sha256"]),
                        "native_sample_rate_hz": int(row["native_sample_rate_hz"]),
                        "native_num_samples": int(row["native_num_samples"]),
                        "model_num_samples": int(row["model_num_samples"]),
                        "duration_sec": float(row["duration_sec"]),
                        "selection_rank": str(row["selection_rank"]),
                    }
                )
        counts[f"{kind}|generic_legacy_label"] = len(generic_selected)
        counts[f"{kind}|descriptive_legacy_label"] = len(detailed_selected)

    expected = 2 * args.per_kind
    if (
        len(selected) != expected
        or len({row["asset_id"] for row in selected}) != expected
        or len({row["source_audio_sha256"] for row in selected}) != expected
    ):
        raise RuntimeError("A2T pilot count, asset uniqueness, or audio uniqueness changed")
    if {row["kind"] for row in selected} != {"music", "sound"}:
        raise RuntimeError("A2T pilot unexpectedly contains speech or another kind")

    selection_jsonl = "".join(stable_json(row) + "\n" for row in selected)
    instruct_input = "".join(
        stable_json(
            {
                "id": row["annotation_id"],
                "audio_path": row["audio_path"],
                "kind": row["kind"],
                "source_audio_sha256": row["source_audio_sha256"],
                "asset_id": row["asset_id"],
                "source_dataset": row["source_dataset"],
                "raw_label": row["raw_label"],
            }
        )
        + "\n"
        for row in selected
    )
    summary = {
        "schema": "stable_audio_tools.sceneplan_a2t_pilot_selection",
        "schema_version": 2,
        "speech_a2t": False,
        "annotation_model_family": "Qwen3-Omni-Instruct",
        "catalog": str(catalog),
        "rows": len(selected),
        "music": sum(row["kind"] == "music" for row in selected),
        "sound": sum(row["kind"] == "sound" for row in selected),
        "counts": counts,
        "selection_policy": (
            "deterministic_selection_rank_25_generic_25_descriptive_per_kind"
        ),
        "selection_jsonl": str(output / "selection.jsonl"),
        "instruct_input_jsonl": str(output / "instruct_input.jsonl"),
    }
    atomic_text(output / "selection.jsonl", selection_jsonl)
    atomic_text(output / "instruct_input.jsonl", instruct_input)
    atomic_text(
        output / "selection_instruct_summary.json",
        json.dumps(summary, indent=2) + "\n",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
