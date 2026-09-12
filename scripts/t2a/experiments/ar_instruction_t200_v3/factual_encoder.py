"""Load the factual-scene CLAP encoder as a frozen AR dependency.

The original factual checkpoint, readout head and optimizer states remain
intact. This adapter does not manufacture a legacy CLAP training contract.
Loading an encoder establishes provenance and compatibility, not AR quality.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from stable_audio_tools.models.sceneplan_transfusion_editing_clap44 import (
    CLAP44Config, EditingCLAP44,
)
from stable_audio_tools.models.sceneplan_transfusion_editing_clap44_io import file_sha256

SCHEMA = "editing_clap44_factual50k_warmstart_three_rank_v1"


def audit_factual_preflight(identity, preflight_path):
    """Bind the actual factual-training index to the AR train/validation split."""
    preflight_path = Path(preflight_path).resolve(strict=True)
    preflight = json.loads(preflight_path.read_text())
    if preflight.get("status") != "PASS":
        raise ValueError("AR requires the completed full-data preflight")
    indices = preflight["indices"]
    if [indices[s]["rows"] for s in ("train", "validation", "test")] != [1000000, 20000, 5000]:
        raise ValueError("AR split sizes differ from the frozen 1M/20k/5k contract")
    contract = identity["contract"]
    if contract["schema"] != SCHEMA:
        raise ValueError("this adapter only accepts the factual-scene training schema")
    data = contract["data"]
    if (Path(data["index"]).resolve() != Path(indices["train"]["path"]).resolve()
            or data["index_sha256"] != indices["train"]["sha256"]
            or data["rows"] != indices["train"]["rows"]):
        raise ValueError("factual CLAP and AR have different training data")
    # The frozen index itself is checked by the native dataset reader. The
    # test entry above is metadata only: no test rows or audio are opened.
    return {
        "preflight": str(preflight_path), "preflight_sha256": file_sha256(preflight_path),
        "train_index_sha256": data["index_sha256"],
        "validation_index_sha256": indices["validation"]["sha256"],
        "test_rows_opened": 0,
        "contract_kept_in_its_original_schema": True,
    }


def load_factual_encoder(path, *, expected_sha256, preflight_path, device="cpu"):
    path = Path(path).resolve(strict=True)
    before = path.stat()
    digest = file_sha256(path)
    if digest != expected_sha256:
        raise RuntimeError("factual CLAP differs from the selected checkpoint")
    manifest_path = path.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text())
    contract_path = path.parent / "TRAIN_CONTRACT.json"
    contract = json.loads(contract_path.read_text())
    if (manifest["schema"] != SCHEMA or contract["schema"] != SCHEMA
            or Path(manifest["checkpoint"]).resolve() != path
            or manifest["sha256"] != digest
            or manifest["contract_sha256"] != file_sha256(contract_path)):
        raise RuntimeError("factual CLAP manifest/contract identity mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if payload["schema"] != SCHEMA or payload["contract"] != contract:
        raise RuntimeError("factual CLAP payload and sidecar contract disagree")
    for key in ("step", "new_updates", "initialization_step", "epoch", "next_batch"):
        if payload[key] != manifest[key]:
            raise RuntimeError(f"factual CLAP manifest disagrees on {key}")
    if (payload["new_updates"] <= 0 or payload["step"] != payload["new_updates"]
            or payload["initialization_step"] != contract["initialization"]["step"]
            or contract.get("independent_test_used") is not False
            or contract.get("VAE_and_Qwen_frozen") is not True
            or contract.get("raw_edit_requests_consumed") is not False):
        raise ValueError("unexpected factual CLAP update/data-role contract")
    if not contract.get("source_sha256"):
        raise ValueError("factual CLAP is missing source provenance")
    for filename, expected in contract["source_sha256"].items():
        if file_sha256(filename) != expected:
            raise RuntimeError(f"factual CLAP bound source changed: {filename}")
    if set(payload["encoder"]) != {"model", "optimizer", "scheduler"} or "model" not in payload["readout"]:
        raise ValueError("factual checkpoint is missing its encoder/readout training state")
    # Preserve adapter initialization RNG when comparing old/new encoders.
    rng_before = torch.get_rng_state().clone()
    with torch.random.fork_rng(devices=[]):
        model = EditingCLAP44(CLAP44Config(**contract["native_config"]["model"]))
    model.load_state_dict(payload["encoder"]["model"], strict=True)
    for name, value in model.state_dict().items():
        if not bool(torch.isfinite(value).all()) or not torch.equal(value, payload["encoder"]["model"][name]):
            raise ValueError(f"non-finite or inexact encoder weight: {name}")
    if not torch.equal(rng_before, torch.get_rng_state()):
        raise RuntimeError("factual encoder loading perturbed CPU RNG")
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise RuntimeError("factual CLAP checkpoint changed while loading")
    identity = {
        "path": str(path), "sha256": digest, "schema": SCHEMA,
        "step": int(payload["new_updates"]), "new_updates": int(payload["new_updates"]),
        "initialization_step": int(payload["initialization_step"]),
        "contract": contract, "contract_sha256": manifest["contract_sha256"],
        "manifest_sha256": file_sha256(manifest_path),
        "component": "encoder.model", "encoder_state_loaded_exactly": True,
        "CPU_RNG_unchanged": True, "readout_preserved_in_original_checkpoint": True,
        "CLAP_optimizer_and_scheduler_not_imported_into_AR": True,
        "quality_gate_passed": False,
    }
    identity["AR_data_binding"] = audit_factual_preflight(identity, preflight_path)
    return model.to(device).eval().requires_grad_(False), identity
