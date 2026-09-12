#!/usr/bin/env python3
"""Resolve and validate spatial T2A configs without loading model weights."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from stable_audio_tools.configuration import load_config, validate_t2a_config


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_ROOT = (
    REPO_ROOT / "stable_audio_tools" / "configs" / "model_configs" / "txt2audio"
)
DATA_ROOT = (
    REPO_ROOT / "stable_audio_tools" / "configs" / "dataset_configs" / "vae_v2_dataset"
)

DEFAULT_EXPERIMENTS = (
    (
        CONFIG_ROOT
        / "t2a"
        / "spatial_cot"
        / "qwen35_0p8b_spatial_chat_500m.json",
        DATA_ROOT / "t2a_spatial_cot_1m_families.json",
    ),
)


def validate_one(model_path: Path, dataset_path: Path | None) -> dict:
    model_config = load_config(model_path)
    dataset_config = load_config(dataset_path) if dataset_path is not None else None
    summary = validate_t2a_config(model_config, dataset_config)
    return {
        "model_config": str(model_path),
        "dataset_config": str(dataset_path) if dataset_path is not None else None,
        **summary,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", type=Path)
    parser.add_argument("--dataset-config", type=Path)
    args = parser.parse_args()

    if args.model_config is None and args.dataset_config is not None:
        parser.error("--dataset-config requires --model-config")

    experiments = (
        ((args.model_config, args.dataset_config),)
        if args.model_config is not None
        else DEFAULT_EXPERIMENTS
    )
    for model_path, dataset_path in experiments:
        result = validate_one(model_path, dataset_path)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    print(f"validated {len(experiments)} T2A config(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
