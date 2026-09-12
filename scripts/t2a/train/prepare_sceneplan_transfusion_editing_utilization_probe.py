#!/usr/bin/env python3
"""Build a frozen real-pair dataset config for Editing DiT utilization probes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stable_audio_tools.configuration import load_config, validate_training_configs  # noqa: E402
from stable_audio_tools.data.sceneplan_transfusion_editing_index import sha256_file  # noqa: E402

from prepare_sceneplan_transfusion_editing_full import (  # noqa: E402
    DEFAULT_MODEL_CONFIG,
    DEFAULT_ROOT,
    LATEST_AR_INPUT_CONTRACT,
    _atomic_json,
    _dataset_config,
    _index_report,
)


EXPECTED_ROWS = {"validation": 20_000, "test": 5_000}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--source-split", choices=sorted(EXPECTED_ROWS), default="validation")
    parser.add_argument("--short-batch-size", type=int, required=True)
    parser.add_argument("--long-batch-size", type=int, required=True)
    args = parser.parse_args()
    if args.short_batch_size <= 0 or args.long_batch_size <= 0:
        raise ValueError("probe batch sizes must be positive")

    root = args.root.expanduser().resolve(strict=True)
    model_path = args.model_config.expanduser().resolve(strict=True)
    source_split = str(args.source_split)
    report = _index_report(
        root / "training_index" / f"{source_split}.sqlite",
        split=source_split,
        expected_rows=EXPECTED_ROWS[source_split],
    )
    model = load_config(model_path)
    diffusion = model["model"]["diffusion"]
    if not (
        diffusion["input_concat_ids"] == ["sceneplan_44", "source_foa_latent"]
        and int(diffusion["config"]["input_concat_dim"]) == 320
        and model["model"]["conditioning"]["pre_encoded_keys"]
        == ["source_foa_latent"]
    ):
        raise RuntimeError("utilization probe is not using the 384-channel edit route")
    scheduler = model["training"]["optimizer_configs"]["diffusion"]["scheduler"]
    if scheduler != {
        "type": "CosineAnnealingLR",
        "config": {"T_max": 30_000, "eta_min": 0.000002},
    }:
        raise RuntimeError(
            "utilization probe scheduler retained incompatible inherited keys: "
            f"{scheduler}"
        )

    # Treat the frozen evaluation split as a training loader only for a short,
    # no-checkpoint throughput probe.  This avoids waiting for the 1M train
    # materialization before sizing the real model.
    config = _dataset_config(report, split="train")
    config["_task"] = (
        "Real-pair Editing DiT utilization probe; no checkpoint is retained."
    )
    config["_probe_source_split"] = source_split
    config["_per_rank_short_batch_size"] = int(args.short_batch_size)
    config["_editing_ar_input_contract"] = LATEST_AR_INPUT_CONTRACT
    config["_editing_ar_old_sceneplan_input"] = False
    config["datasets"][0]["id"] = (
        f"sceneplan_transfusion_editing_v1_{source_split}_utilization_probe"
    )
    config["length_bucket_batching"] = {
        "enabled": True,
        "long_batch_size": int(args.long_batch_size),
        "seed": 42,
    }
    validate_training_configs(model, config)

    stem = (
        f"{source_split}_as_train_b{args.short_batch_size}"
        f"_l{args.long_batch_size}"
    )
    output_root = root / "contracts" / "utilization_probe"
    dataset_path = output_root / f"{stem}.json"
    audit_path = output_root / f"{stem}.audit.json"
    _atomic_json(dataset_path, config)
    audit = {
        "schema": "sceneplan_transfusion_editing_dit_utilization_probe",
        "schema_version": 1,
        "status": "PASS",
        "source_split": source_split,
        "rows": report["rows"],
        "short_rows": report["short_rows"],
        "long_rows": report["long_rows"],
        "per_rank_short_batch_size": int(args.short_batch_size),
        "per_rank_long_batch_size": int(args.long_batch_size),
        "editing_ar_inputs": ["source_foa_latent", "raw_edit_request"],
        "old_sceneplan_input": False,
        "editing_dit_frame_channels": 384,
        "model_config": str(model_path),
        "model_config_sha256": sha256_file(model_path),
        "training_index": report,
        "dataset_config": str(dataset_path.resolve()),
        "dataset_config_sha256": sha256_file(dataset_path),
    }
    _atomic_json(audit_path, audit)
    print(json.dumps({**audit, "audit_path": str(audit_path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
