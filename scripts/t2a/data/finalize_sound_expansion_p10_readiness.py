#!/usr/bin/env python3
"""Fail-closed data-readiness gate for the Sound-expansion P10 revision."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

from stable_audio_tools.configuration import load_config, validate_training_configs


REPO = Path(__file__).resolve().parents[3]
DEFAULT_MODEL = REPO / (
    "stable_audio_tools/configs/model_configs/txt2audio/t2a/dit/"
    "qwen35_0p8b_300m_model_sceneplan_44_preflight_candidate.json"
)
EXPECTED = {"train": 1_100_000, "validation": 20_000, "test": 4_000}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision-root", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args()

    root = args.revision_root.expanduser().resolve(strict=True)
    try:
        root.relative_to(Path("/mnt/sdb/audio_dataset/sceneplan_v2_1p124m/revisions"))
    except ValueError as error:
        raise RuntimeError("revision must remain below the frozen dataset revisions root") from error
    marker_path = root / "FROZEN_P9.json"
    summary_path = root / "P0_P9_SUMMARY.json"
    marker = read_json(marker_path)
    summary = read_json(summary_path)
    if not (
        marker.get("schema") == "stable_audio_tools.sceneplan_v2_p9_marker"
        and marker.get("state") == "P9_complete_frozen_waiting_for_user_acceptance"
        and marker.get("dataset_revision") == "sound_expansion_v1"
        and marker.get("p10_training_started") is False
        and marker.get("p11_training_started") is False
        and summary.get("status") == "PASS"
        and all(
            value.get("status") == "PASS"
            for value in (summary.get("stages") or {}).values()
        )
    ):
        raise RuntimeError("P0-P9 revision summary/marker is not an all-pass gate")
    freeze_path = Path(marker["freeze_manifest"]).resolve(strict=True)
    if sha256(freeze_path) != marker["freeze_manifest_sha256"]:
        raise RuntimeError("freeze-manifest checksum changed")
    freeze = read_json(freeze_path)

    configs = {row["split"]: row for row in freeze["p10_dataset_configs"]}
    if set(configs) != set(EXPECTED):
        raise RuntimeError("P10 dataset-config split coverage changed")
    loaded_configs = {}
    for split, expected_rows in EXPECTED.items():
        entry = configs[split]
        path = Path(entry["path"]).resolve(strict=True)
        if sha256(path) != entry["sha256"] or int(entry["rows"]) != expected_rows:
            raise RuntimeError(f"{split}: frozen P10 dataset config changed")
        value = load_config(path)
        if int(value.get("expected_num_samples", -1)) != expected_rows:
            raise RuntimeError(f"{split}: P10 config row contract changed")
        loaded_configs[split] = (path, value)
    train = loaded_configs["train"][1]
    if not (
        train.get("require_speech_timing") is True
        and int(train.get("expected_speech_timing_rows", -1)) == 500_000
        and sha256(Path(train["speech_timing_index_path"]).resolve(strict=True))
        == train["speech_timing_index_sha256"]
    ):
        raise RuntimeError("train speech timing contract is incomplete")

    model_path = args.model_config.expanduser().resolve(strict=True)
    model = load_config(model_path)
    geometry = validate_training_configs(model, train)
    if geometry.get("latent_channels") != 64 or geometry.get("latent_length") != 432:
        raise RuntimeError("P10 resolved latent geometry changed")
    if model["training"]["sceneplan_speech_duration_loss"].get("enabled") is not True:
        raise RuntimeError("new P10 candidate lost speech duration supervision")
    sound_loss = model["training"]["sceneplan_sound_temporal_difference_loss"]
    if not (sound_loss.get("enabled") is True and sound_loss.get("scope") == "sound_only"):
        raise RuntimeError("new P10 candidate lost Sound temporal-difference supervision")

    audits = {}
    audit_root = root / "audit/conditioning_v3"
    for split, expected_rows in EXPECTED.items():
        path = audit_root / f"sceneplan_44_{split}_audit.json"
        value = read_json(path)
        expected_checked = 20_000 if split == "train" else expected_rows
        if not (
            value.get("ok") is True
            and int(value.get("index_rows", -1)) == expected_rows
            and int(value.get("rows_checked", -1)) == expected_checked
            and int(value.get("observed_max_tokens", 513)) <= 512
            and value.get("local_condition", {}).get("event_tracks") == 4
            and value.get("local_condition", {}).get("trajectory_tracks") == 4
            and value.get("local_condition", {}).get("gain_db_used") is False
        ):
            raise RuntimeError(f"{split}: conditioning audit is not P10-ready")
        if split != "train" and value.get("selection") != "all":
            raise RuntimeError(f"{split}: complete conditioning audit required")
        audits[split] = {"path": str(path), "sha256": sha256(path)}

    unit = subprocess.run(
        [str(REPO / ".venv/bin/python"), str(REPO / "scripts/t2a/train/test_sceneplan_dit_p10_44.py")],
        cwd=REPO,
        check=False,
        capture_output=True,
        text=True,
    )
    if unit.returncode or "OK" not in unit.stdout + unit.stderr:
        raise RuntimeError(f"P10 4+4 unit tests failed:\n{unit.stdout}\n{unit.stderr}")

    receipt = {
        "schema": "stable_audio_tools.sound_expansion_p10_data_readiness",
        "schema_version": 1,
        "status": "PASS",
        "scope": "P0-P9 data ready; P10 training not started",
        "revision_root": str(root),
        "frozen_p9_marker": str(marker_path),
        "frozen_p9_marker_sha256": sha256(marker_path),
        "p0_p9_summary": str(summary_path),
        "p0_p9_summary_sha256": sha256(summary_path),
        "model_config": str(model_path),
        "model_config_sha256": sha256(model_path),
        "train_dataset_config": str(loaded_configs["train"][0]),
        "validation_dataset_config": str(loaded_configs["validation"][0]),
        "test_dataset_config": str(loaded_configs["test"][0]),
        "resolved_geometry": geometry,
        "conditioning_audits": audits,
        "speech_duration_supervision": True,
        "sound_temporal_difference_supervision": True,
        "semantic_conditioning": "Qwen cross-attention",
        "structured_conditioning": "4 event tracks plus 4 trajectory tracks",
        "unit_tests": "PASS",
        "p10_training_started": False,
        "next_gate": "user approval, then new ten-sample overfit/throughput preflight before full P10",
    }
    path = root / "P10_DATA_READY.json"
    atomic_json(path, receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
