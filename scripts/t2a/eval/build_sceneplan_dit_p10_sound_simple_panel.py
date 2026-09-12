#!/usr/bin/env python3
"""Freeze a frequency-backed five-row simple-sound P10 comparison panel.

The panel is deliberately small and interpretable.  Every row is an existing
single-source P9 test example, has no spoken-language background, and has an
original dataset label that agrees with the registry-driven semantic text.
The same rows and noise seeds are then shared by r5 and r6 checkpoint runs.
"""

from __future__ import annotations
import os

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from scripts.t2a.eval.build_sceneplan_dit_p10_eval_contract import (
    DATASET_ROOT,
    _read_p9_rows,
    _reference_map,
    _sha256_file,
    _source_map,
    _spoken_language_map,
    _write_json,
    _write_jsonl,
)


DEFAULT_OUTPUT = Path(
    os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/archives//p10_pre_v11_20260831/sceneplan_dit_fail/"
    "sceneplan_dit_v2_r6_300m/evaluation/"
    "p10_sound_simple_5way_v1"
)
R5_PRIOR_CONTRACT = Path(
    os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/archives//p10_pre_v11_20260831/sceneplan_dit_fail/"
    "sceneplan_dit_v2_r5_300m/evaluation/"
    "p10_ckpt_10k_20k_30k_40k_50k_speaker_v1/EVAL_CONTRACT.json"
)
R6_PRIOR_CONTRACT = Path(
    os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/archives//p10_pre_v11_20260831/sceneplan_dit_fail/"
    "sceneplan_dit_v2_r6_300m/evaluation/"
    "p10_r6_10k_20k_effect_audit_v1/EVAL_CONTRACT.json"
)


SELECTED = (
    {
        "panel_id": "sound_01",
        "demo_name": "steady engine idle",
        "event_family": "vehicle_engine",
        "sample_id": "spv2_test_no_speech_1_0000208",
        "raw_terms": (r"\bengine\b", r"\bvehicle\b", r"\bcar\b"),
        "description_terms": (r"\bengine\b", r"\bidles?\b"),
    },
    {
        "panel_id": "sound_02",
        "demo_name": "cat meowing",
        "event_family": "cat_meow",
        "sample_id": "spv2_test_no_speech_1_0000624",
        "raw_terms": (r"\bcat\b", r"\bmeowing\b"),
        "description_terms": (r"\bcat\b", r"\bmeows?\b"),
    },
    {
        "panel_id": "sound_03",
        "demo_name": "mechanical keyboard typing",
        "event_family": "keyboard_typing",
        "sample_id": "spv2_test_no_speech_1_0000204",
        "raw_terms": (r"\btyping\b", r"\btypewriter\b"),
        "description_terms": (r"\btypes?\b", r"\bkeyboard\b"),
    },
    {
        "panel_id": "sound_04",
        "demo_name": "toilet flushing",
        "event_family": "toilet_flush",
        "sample_id": "spv2_test_no_speech_1_0000670",
        "raw_terms": (r"\btoilet\b", r"\bflush\b"),
        "description_terms": (r"\btoilet\b", r"\bflushes\b"),
    },
    {
        "panel_id": "sound_05",
        "demo_name": "electronic police siren",
        "event_family": "siren_alarm",
        "sample_id": "spv2_test_no_speech_1_0000256",
        "raw_terms": (r"\bsiren\b",),
        "description_terms": (r"\bsiren\b", r"\balarm\b"),
    },
)


# These are overlapping, reader-facing prevalence families rather than a
# forced partition.  Counts are used to show what is common in the source
# universe and why the demo anchors were selected.
FREQUENCY_PATTERNS = {
    "vehicle_engine": (
        r"\bvehicle\b", r"\bcar\b", r"\bengine\b", r"\btruck\b",
        r"\bbus\b", r"\bmotorcycle\b", r"\btrain\b", r"\bboat\b",
    ),
    "water_liquid": (
        r"\bwater\b", r"\brain\b", r"\bstream\b", r"\briver\b",
        r"\bocean\b", r"\bwave", r"\bsplash", r"\btoilet\b",
    ),
    "bird": (r"\bbird\b", r"\bchirp", r"\bpigeon\b", r"\bowl\b"),
    "wind": (r"\bwind\b", r"\brustl"),
    "fire_crackle": (r"\bfire\b", r"\bcrackle", r"\bburning\b"),
    "dog_bark": (r"\bdog\b", r"\bbark", r"\bbow-wow\b"),
    "siren_alarm": (r"\bsiren\b", r"\balarm\b", r"\bbuzzer\b"),
    "cat_meow": (r"\bcat\b", r"\bmeow"),
    "footsteps": (r"\bfootsteps?\b", r"\bwalking\b", r"\bwalk\b"),
    "keyboard_typing": (r"\bkeyboard\b", r"\btyping\b", r"\btypewriter\b"),
    "toilet_flush": (r"\btoilet\b", r"\bflush"),
}


def _matches_any(text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _checkpoint(contract: dict[str, Any], step: int) -> dict[str, Any]:
    matches = [row for row in contract["checkpoints"] if int(row["step"]) == step]
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one checkpoint at step {step}")
    result = dict(matches[0])
    path = Path(result["path"]).resolve(strict=True)
    if path.stat().st_size != int(result["bytes"]):
        raise RuntimeError(f"checkpoint byte size changed: {path}")
    return result


def _frequency_summary(universe: list[dict[str, Any]]) -> dict[str, Any]:
    sound_rows = [row for row in universe if row["kind"] == "sound"]
    total_refs = sum(int(row["sceneplan_reference_count"]) for row in sound_rows)
    families: dict[str, Any] = {}
    for family, patterns in FREQUENCY_PATTERNS.items():
        chosen = [
            row
            for row in sound_rows
            if _matches_any(str(row.get("raw_label") or ""), patterns)
        ]
        train = [row for row in chosen if row["split"] == "train"]
        refs = sum(int(row["sceneplan_reference_count"]) for row in chosen)
        train_refs = sum(int(row["sceneplan_reference_count"]) for row in train)
        families[family] = {
            "unique_sources_all_splits": len(chosen),
            "sceneplan_references_all_splits": refs,
            "reference_share_all_splits": refs / total_refs,
            "unique_sources_train": len(train),
            "sceneplan_references_train": train_refs,
        }
    return {
        "sound_unique_sources": len(sound_rows),
        "sound_sceneplan_references": total_refs,
        "source_dataset_unique_counts": dict(
            sorted(Counter(str(row["source_dataset"]) for row in sound_rows).items())
        ),
        "families_overlap": True,
        "families": families,
    }


def _make_contract(
    *,
    prior: dict[str, Any],
    checkpoints: list[dict[str, Any]],
    panel_path: Path,
    index_path: Path,
    panel: list[dict[str, Any]],
) -> dict[str, Any]:
    sampling = dict(prior["sampling"])
    return {
        "schema": "stable_audio_tools.sceneplan_dit_p10_simple_sound_eval_contract",
        "schema_version": 1,
        "status": "FROZEN_SIMPLE_SOUND_PANEL_V1",
        "purpose": "matched simple-sound checkpoint and VAE-ceiling comparison",
        "test_set": {
            "index": str(index_path.resolve()),
            "index_sha256": prior["test_set"]["index_sha256"],
            "all_rows": 4_000,
            "evaluation_rows": len(panel),
            "domain_counts": {"sound": len(panel)},
            "panel": str(panel_path.resolve()),
            "panel_sha256": _sha256_file(panel_path),
            "single_source_only": True,
            "source_disjoint": True,
            "spoken_language_background_rows": 0,
        },
        "checkpoints": checkpoints,
        "sampling": sampling,
        "comparison_rules": {
            "same_panel_across_all_systems": True,
            "same_noise_seed_per_sample_across_dit_checkpoints": True,
            "reference_and_vae_reconstruction_are_deterministic": True,
            "raw_metrics_use_four_channel_foa": True,
            "listening_only_uses_fixed_virtual_stereo_decode": True,
            "training_must_remain_paused": True,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    index_path = DATASET_ROOT / "training_index/test.sqlite"
    p9_rows = {row["sample_id"]: row for row in _read_p9_rows(index_path)}
    references = _reference_map()
    sources = _source_map()
    spoken = _spoken_language_map()
    universe_path = (
        DATASET_ROOT
        / "source_annotations/nonspeech_instruct_v2/source_universe.parquet"
    )
    universe_columns = [
        "source_audio_sha256", "split", "kind", "source_dataset", "source_id",
        "raw_label", "dry_audio_path", "sceneplan_reference_count",
    ]
    universe = pq.read_table(universe_path, columns=universe_columns).to_pylist()
    universe_by_hash = {
        str(row["source_audio_sha256"]): row for row in universe
    }

    panel: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    for panel_index, spec in enumerate(SELECTED, start=1):
        sample_id = str(spec["sample_id"])
        if sample_id not in p9_rows or sample_id not in references or sample_id not in sources:
            raise RuntimeError(f"selected sample provenance is incomplete: {sample_id}")
        row = dict(p9_rows[sample_id])
        row.update(references[sample_id])
        row.update(sources[sample_id])
        source_hash = str(row["source_audio_sha256"])
        if source_hash not in universe_by_hash or source_hash not in spoken:
            raise RuntimeError(f"selected source annotation is missing: {source_hash}")
        source_row = universe_by_hash[source_hash]
        raw_label = str(source_row["raw_label"] or "")
        description = str(row["semantic_text"])
        source = row["scene_plan"]["sources"][0]
        activity = source["activity"]
        activity_fraction = (
            float(activity["offset_sec"]) - float(activity["onset_sec"])
        ) / float(row["duration_sec"])
        gates = {
            "single_source_sound": (
                row["domain"] == "sound" and len(row["scene_plan"]["sources"]) == 1
            ),
            "spoken_language_background_false": not bool(spoken[source_hash]),
            "raw_label_event_match": _matches_any(raw_label, spec["raw_terms"]),
            "description_event_match": _matches_any(
                description, spec["description_terms"]
            ),
            "activity_fraction_at_least_0p90": activity_fraction >= 0.90,
            "reference_exists": Path(row["reference_foa_path"]).is_file(),
            "dry_source_exists": Path(str(source_row["dry_audio_path"])).is_file(),
        }
        if not all(gates.values()):
            raise RuntimeError(f"{sample_id}: simple-sound gates failed: {gates}")
        row.update(
            {
                "panel_index": panel_index,
                "panel_id": str(spec["panel_id"]),
                "demo_name": str(spec["demo_name"]),
                "event_family": str(spec["event_family"]),
                "raw_label": raw_label,
                "source_dataset": str(source_row["source_dataset"]),
                "source_split": str(source_row["split"]),
                "source_sceneplan_reference_count": int(
                    source_row["sceneplan_reference_count"]
                ),
                "spoken_language_background": False,
                "activity_fraction": activity_fraction,
            }
        )
        panel.append(row)
        gate_rows.append(
            {
                "panel_id": row["panel_id"],
                "sample_id": sample_id,
                "demo_name": row["demo_name"],
                "event_family": row["event_family"],
                "source_dataset": row["source_dataset"],
                "raw_label": raw_label,
                "semantic_text": description,
                "room_type": row["room_type"],
                "motion_type": row["motion_type"],
                "activity_fraction": activity_fraction,
                "gates": gates,
                "pass": all(gates.values()),
            }
        )

    if len({row["sample_id"] for row in panel}) != 5:
        raise RuntimeError("simple-sound panel sample IDs are not unique")
    if len({row["source_audio_sha256"] for row in panel}) != 5:
        raise RuntimeError("simple-sound panel source hashes are not unique")
    if Counter(row["motion_type"] for row in panel) != Counter(
        {"static": 3, "linear": 2}
    ):
        raise RuntimeError("simple-sound panel motion mix changed")

    panel_path = output_root / "SOUND_SIMPLE_PANEL_5.jsonl"
    _write_jsonl(panel_path, panel)
    frequency = _frequency_summary(universe)
    audit = {
        "schema": "stable_audio_tools.sceneplan_dit_p10_simple_sound_selection_audit",
        "schema_version": 1,
        "status": "PASS",
        "selection_policy": (
            "frequency-backed recognizable event families; frozen P9 test rows; "
            "single source; raw-label/A2T agreement; no spoken language; >=90% activity"
        ),
        "panel_rows": len(panel),
        "all_gates_pass": all(row["pass"] for row in gate_rows),
        "panel_sha256": _sha256_file(panel_path),
        "frequency_basis": frequency,
        "rows": gate_rows,
    }
    _write_json(output_root / "SELECTION_AUDIT.json", audit)

    r5_prior = _load_json(R5_PRIOR_CONTRACT)
    r6_prior = _load_json(R6_PRIOR_CONTRACT)
    r5_contract = _make_contract(
        prior=r5_prior,
        checkpoints=[_checkpoint(r5_prior, 50_000)],
        panel_path=panel_path,
        index_path=index_path,
        panel=panel,
    )
    r6_contract = _make_contract(
        prior=r6_prior,
        checkpoints=[
            _checkpoint(r6_prior, 10_000),
            _checkpoint(r6_prior, 20_000),
        ],
        panel_path=panel_path,
        index_path=index_path,
        panel=panel,
    )
    _write_json(output_root / "EVAL_CONTRACT_R5.json", r5_contract)
    _write_json(output_root / "EVAL_CONTRACT_R6.json", r6_contract)
    summary = {
        "status": "PASS",
        "output_root": str(output_root),
        "panel": str(panel_path),
        "panel_sha256": _sha256_file(panel_path),
        "rows": len(panel),
        "demo_names": [row["demo_name"] for row in panel],
        "motion_counts": dict(Counter(row["motion_type"] for row in panel)),
        "room_counts": dict(Counter(row["room_type"] for row in panel)),
        "spoken_language_background_rows": 0,
    }
    _write_json(output_root / "BUILD_SUMMARY.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
