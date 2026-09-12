#!/usr/bin/env python3
"""Freeze six disjoint-test scenes for final P10 spatial listening.

Coverage is explicit rather than cherry-picked after hearing outputs: Speech
left/right crossed with Music/Sound on the opposite side, plus one long Speech
left-to-right and one long Speech right-to-left trajectory.  The checkpoint is
selected only by the quantitative balanced-panel evaluation.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.t2a.eval.sceneplan_dit_p10_panel_common import sha256_file
from stable_audio_tools.data.model_sceneplan import compile_model_semantic_caption_v2


DEFAULT_SOURCE = Path(
    "/mnt/sdb/audio_dataset/sceneplan_v2_1p124m/revisions/"
    "speech_expansion_noalign_15s_v1/evaluation/"
    "p10_v9_full_test_8000_ckpt20k_100k_v1"
)
DEFAULT_MODEL_CONFIG = REPO_ROOT / (
    "stable_audio_tools/configs/model_configs/txt2audio/t2a/dit/"
    "qwen35_0p8b_300m_model_sceneplan_44_"
    "soundexp_noalign_15s_resume_cosine_40k.json"
)
SELECTION = {
    "speech_left_music_right": "full_0006243",
    "speech_right_music_left": "full_0006303",
    "speech_left_sound_right": "full_0001018",
    "speech_right_sound_left": "full_0006447",
    "dynamic_speech_left_to_right": "full_0001550",
    "dynamic_speech_right_to_left": "full_0007560",
}


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _speech(row: dict[str, Any]) -> dict[str, Any]:
    sources = [
        source for source in row["scene_plan"]["sources"]
        if source["kind"] == "speech"
    ]
    if len(sources) != 1:
        raise RuntimeError(f"{row['panel_id']}: expected exactly one Speech source")
    return sources[0]


def _azimuth(source: dict[str, Any]) -> float:
    trajectory = source["trajectory"]
    if trajectory["type"] != "static":
        raise RuntimeError("opposite-side showcase sources must be static")
    return float(trajectory["position"]["azimuth_deg"])


def _validate_static(row: dict[str, Any], label: str) -> dict[str, Any]:
    speech = _speech(row)
    background_kind = "music" if "music" in label else "sound"
    backgrounds = [
        source for source in row["scene_plan"]["sources"]
        if source["kind"] == background_kind
    ]
    if len(row["scene_plan"]["sources"]) != 2 or len(backgrounds) != 1:
        raise RuntimeError(f"{row['panel_id']}: expected one Speech + one {background_kind}")
    speech_azimuth = _azimuth(speech)
    background_azimuth = _azimuth(backgrounds[0])
    speech_left = "speech_left" in label
    # Dataset convention: positive azimuth is listener-left; negative is right.
    if speech_left and not (speech_azimuth >= 40.0 and background_azimuth <= -40.0):
        raise RuntimeError(f"{row['panel_id']}: left/right contract failed")
    if not speech_left and not (speech_azimuth <= -40.0 and background_azimuth >= 40.0):
        raise RuntimeError(f"{row['panel_id']}: right/left contract failed")
    return {
        "speech_azimuth_deg": speech_azimuth,
        f"{background_kind}_azimuth_deg": background_azimuth,
        "speech_activity": speech["activity"],
    }


def _validate_dynamic(row: dict[str, Any], label: str) -> dict[str, Any]:
    speech = _speech(row)
    trajectory = speech["trajectory"]
    if trajectory["type"] != "linear":
        raise RuntimeError(f"{row['panel_id']}: Speech must be linear")
    start = float(trajectory["start"]["azimuth_deg"])
    end = float(trajectory["end"]["azimuth_deg"])
    extent = abs((end - start + 180.0) % 360.0 - 180.0)
    left_to_right = "left_to_right" in label
    if extent < 150.0:
        raise RuntimeError(f"{row['panel_id']}: Speech trajectory too small: {extent}")
    if left_to_right and not (start > 0.0 > end):
        raise RuntimeError(f"{row['panel_id']}: not left-to-right")
    if not left_to_right and not (start < 0.0 < end):
        raise RuntimeError(f"{row['panel_id']}: not right-to-left")
    return {
        "speech_start_azimuth_deg": start,
        "speech_end_azimuth_deg": end,
        "speech_trajectory_extent_deg": extent,
        "speech_activity": speech["activity"],
        "background_kinds": [
            source["kind"] for source in row["scene_plan"]["sources"]
            if source["kind"] != "speech"
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-eval-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--checkpoint-step", type=int, required=True)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    args = parser.parse_args()

    source = args.source_eval_root.expanduser().resolve(strict=True)
    output = args.output_root.expanduser().resolve()
    checkpoint = args.checkpoint_path.expanduser().resolve(strict=True)
    model_config = args.model_config.expanduser().resolve(strict=True)
    source_contract_path = source / "EVAL_CONTRACT.json"
    source_contract = json.loads(source_contract_path.read_text(encoding="utf-8"))
    panel_path = source / source_contract["test_set"]["panel_filename"]
    rows = [json.loads(line) for line in panel_path.read_text(encoding="utf-8").splitlines() if line]
    by_id = {row["panel_id"]: row for row in rows}

    selected = []
    facts = {}
    for label, panel_id in SELECTION.items():
        row = copy.deepcopy(by_id[panel_id])
        row["source_eval_domain"] = row["domain"]
        row["domain"] = label
        fact = (
            _validate_dynamic(row, label)
            if label.startswith("dynamic_")
            else _validate_static(row, label)
        )
        compiled = compile_model_semantic_caption_v2(row["scene_plan"])
        row["model_prompt_text"] = compiled["text"]
        row["model_event_regions"] = compiled["event_regions"]
        row["model_speech_regions"] = compiled["speech_regions"]
        facts[label] = {"panel_id": panel_id, **fact}
        selected.append(row)

    output.mkdir(parents=True, exist_ok=True)
    output_panel = output / "spatial_showcase_6.jsonl"
    _atomic_text(
        output_panel,
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in selected
        ),
    )
    contract = copy.deepcopy(source_contract)
    contract.update(
        {
            "schema": "stable_audio_tools.p10_v11_spatial_listening_showcase_contract",
            "schema_version": 1,
            "status": f"FROZEN_{args.checkpoint_step // 1000}K_SPATIAL_SHOWCASE_V2",
            "purpose": "quantitatively selected checkpoint spatial listening showcase",
            "source_full_test_contract": str(source_contract_path),
            "source_full_test_contract_sha256": sha256_file(source_contract_path),
            "checkpoints": [
                {
                    "step": int(args.checkpoint_step),
                    "path": str(checkpoint),
                    "bytes": checkpoint.stat().st_size,
                    "sha256": sha256_file(checkpoint),
                }
            ],
            "selection": {
                "precommitted_panel_ids": SELECTION,
                "facts": facts,
                "all_rows_from_content_disjoint_frozen_test": True,
            },
        }
    )
    contract["sampling"].update(
        {
            "model_config": str(model_config),
            "model_config_sha256": sha256_file(model_config),
            "semantic_caption_compiler_version": 2,
            "common_noise_seed_namespace": "p10-v11-final-spatial-showcase-v2-20260831",
            "inference_batch_size": 1,
        }
    )
    contract["test_set"].update(
        {
            "evaluation_rows": len(selected),
            "evaluation_subset": "four opposite-side mixtures plus two long dynamic Speech scenes",
            "domain_counts": {label: 1 for label in SELECTION},
            "panel_filename": output_panel.name,
            "panel_sha256": sha256_file(output_panel),
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
        "checkpoint_step": int(args.checkpoint_step),
        "rows": len(selected),
        "domain_counts": contract["test_set"]["domain_counts"],
        "semantic_caption_compiler_version": 2,
        "selection_facts": facts,
        "contract": str(contract_path),
        "panel": str(output_panel),
    }
    _atomic_text(
        output / "BUILD_SUMMARY.json",
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
