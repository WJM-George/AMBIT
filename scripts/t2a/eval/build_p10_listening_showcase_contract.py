#!/usr/bin/env python3
"""Freeze a six-scene P10 listening showcase from the disjoint full test set.

The showcase intentionally covers two capabilities that the balanced
single-source benchmark cannot demonstrate:

* three static scenes with Speech on the listener's left and Music on the right;
* three scenes whose unique formal Speech source follows a linear trajectory.

Only the already-frozen 100k checkpoint is retained in the derived contract.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any


DEFAULT_SOURCE = Path(
    "/mnt/sdb/audio_dataset/sceneplan_v2_1p124m/revisions/"
    "speech_expansion_noalign_15s_v1/evaluation/"
    "p10_v9_full_test_8000_ckpt20k_100k_v1"
)
DEFAULT_OUTPUT = Path(
    "/mnt/sdb/audio_dataset/sceneplan_v2_1p124m/revisions/"
    "speech_expansion_noalign_15s_v1/evaluation/"
    "p10_v9_100k_listening_showcase_v1"
)

SELECTION = {
    "speech_left_music_right": (
        "full_0001119",  # rear-left narrator; front-right industrial electronic music
        "full_0005497",  # left narrator; right heavy-metal guitar music
        "full_0005618",  # left-above narrator; rear-right chiptune music
    ),
    "dynamic_speech": (
        "full_0000690",  # isolated speech, rear-left toward front-left
        "full_0001420",  # speech plus sound, front-right toward rear-left
        "full_0005555",  # speech plus music, rear-right toward front-left
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _speech(row: dict[str, Any]) -> dict[str, Any]:
    values = [source for source in row["scene_plan"]["sources"] if source["kind"] == "speech"]
    if len(values) != 1:
        raise RuntimeError(f"{row['panel_id']} must contain exactly one formal Speech source")
    return values[0]


def _validate_left_right(row: dict[str, Any]) -> None:
    speech = _speech(row)
    music = [source for source in row["scene_plan"]["sources"] if source["kind"] == "music"]
    if not music:
        raise RuntimeError(f"{row['panel_id']} has no Music source")
    if len(row["scene_plan"]["sources"]) != 2 or len(music) != 1:
        raise RuntimeError(f"{row['panel_id']} must contain only one Speech and one Music source")
    if speech["trajectory"]["type"] != "static":
        raise RuntimeError(f"{row['panel_id']} Speech is not static")
    speech_azimuth = float(speech["trajectory"]["position"]["azimuth_deg"])
    # Frozen ScenePlan convention: positive azimuth is listener-left,
    # negative azimuth is listener-right.
    if speech_azimuth < 45.0:
        raise RuntimeError(f"{row['panel_id']} Speech is not on the left: {speech_azimuth}")
    right_music = [
        source
        for source in music
        if source["trajectory"]["type"] == "static"
        and float(source["trajectory"]["position"]["azimuth_deg"]) <= -45.0
    ]
    if not right_music:
        raise RuntimeError(f"{row['panel_id']} has no static right-side Music source")


def _validate_dynamic(row: dict[str, Any]) -> None:
    speech = _speech(row)
    trajectory = speech["trajectory"]
    if trajectory["type"] != "linear":
        raise RuntimeError(f"{row['panel_id']} Speech is not linear")
    start = float(trajectory["start"]["azimuth_deg"])
    end = float(trajectory["end"]["azimuth_deg"])
    extent = abs((end - start + 180.0) % 360.0 - 180.0)
    if extent < 90.0:
        raise RuntimeError(f"{row['panel_id']} Speech motion is too small: {extent}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-eval-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    source = args.source_eval_root.expanduser().resolve(strict=True)
    output = args.output_root.expanduser().resolve()
    source_contract_path = source / "EVAL_CONTRACT.json"
    source_contract = json.loads(source_contract_path.read_text(encoding="utf-8"))
    panel_name = str(source_contract["test_set"]["panel_filename"])
    source_panel_path = source / panel_name
    rows = [
        json.loads(line)
        for line in source_panel_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    by_id = {str(row["panel_id"]): row for row in rows}

    selected: list[dict[str, Any]] = []
    for showcase_domain, panel_ids in SELECTION.items():
        for panel_id in panel_ids:
            row = copy.deepcopy(by_id[panel_id])
            row["source_eval_domain"] = row["domain"]
            row["domain"] = showcase_domain
            if showcase_domain == "speech_left_music_right":
                _validate_left_right(row)
            else:
                _validate_dynamic(row)
            selected.append(row)

    panel_path = output / "showcase_6.jsonl"
    _atomic_text(
        panel_path,
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in selected),
    )

    checkpoints = [
        row for row in source_contract["checkpoints"] if int(row["step"]) == 100_000
    ]
    if len(checkpoints) != 1:
        raise RuntimeError("source contract must expose exactly one 100k checkpoint")

    contract = copy.deepcopy(source_contract)
    contract.update(
        {
            "schema": "stable_audio_tools.sceneplan_dit_p10_listening_showcase_contract",
            "schema_version": 1,
            "status": "FROZEN_100K_LISTENING_SHOWCASE_V1",
            "purpose": "qualitative 100k native-FOA listening showcase",
            "checkpoints": checkpoints,
            "source_full_test_contract": str(source_contract_path),
            "source_full_test_contract_sha256": _sha256(source_contract_path),
            "selection": {
                "speech_left_music_right": list(SELECTION["speech_left_music_right"]),
                "dynamic_speech": list(SELECTION["dynamic_speech"]),
                "all_rows_are_from_content_disjoint_frozen_test": True,
            },
        }
    )
    contract["sampling"]["common_noise_seed_namespace"] = (
        "sceneplan-p10-v9-100k-listening-showcase-v1-20260830"
    )
    contract["test_set"].update(
        {
            "evaluation_rows": len(selected),
            "evaluation_subset": "three left-Speech/right-Music plus three dynamic-Speech scenes",
            "domain_counts": {key: len(value) for key, value in SELECTION.items()},
            "panel_filename": panel_path.name,
            "panel_sha256": _sha256(panel_path),
            "single_source_domain_pure": False,
        }
    )
    contract_path = output / "EVAL_CONTRACT.json"
    _atomic_text(
        contract_path,
        json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    summary = {
        "status": "PASS",
        "rows": len(selected),
        "domain_counts": contract["test_set"]["domain_counts"],
        "checkpoint_step": 100_000,
        "contract": str(contract_path),
        "contract_sha256": _sha256(contract_path),
        "panel": str(panel_path),
        "panel_sha256": _sha256(panel_path),
    }
    _atomic_text(
        output / "BUILD_SUMMARY.json",
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
