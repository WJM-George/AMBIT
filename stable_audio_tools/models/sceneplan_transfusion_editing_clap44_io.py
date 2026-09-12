"""Checkpoint provenance for CLAP44; loading never implies quality approval."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from .sceneplan_transfusion_editing_clap44 import CLAP44_CONTRACT, CLAP44Config, EditingCLAP44


def file_sha256(path: str | Path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def load_clap44_checkpoint(path: str | Path, *, expected_sha256: str | None = None, device="cpu", verify_sources: bool = True):
    path = Path(path).resolve(strict=True)
    before = path.stat()
    digest = file_sha256(path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise RuntimeError("CLAP44 checkpoint SHA256 differs from the selected artifact")
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    contract = payload["contract"]
    if contract.get("schema") != CLAP44_CONTRACT or contract.get("m2d_used") is not False or contract.get("test_split_used_for_training_or_selection") is not False:
        raise ValueError("foreign or unapproved CLAP44 checkpoint contract")
    sidecar = path.parent / "TRAIN_CONTRACT.json"
    if not sidecar.is_file() or json.loads(sidecar.read_text()) != contract:
        raise RuntimeError("CLAP44 checkpoint and training contract disagree")
    if int(payload["step"]) <= 0:
        raise ValueError("CLAP44 checkpoint has no completed training steps")
    if verify_sources:
        if not contract.get("source_sha256"):
            raise RuntimeError("CLAP44 checkpoint has no source provenance")
        for filename, expected in contract["source_sha256"].items():
            if file_sha256(filename) != expected:
                raise RuntimeError(f"CLAP44 source changed: {filename}")
    # Loading a frozen dependency must not perturb the random initialization
    # of AR adapters in otherwise matched feature ablations.
    with torch.random.fork_rng(devices=[]):
        model = EditingCLAP44(CLAP44Config(**contract["config"]["model"]))
    model.load_state_dict(payload["model"], strict=True)
    if any(not bool(torch.isfinite(x).all()) for x in model.state_dict().values()):
        raise ValueError("CLAP44 checkpoint contains non-finite weights")
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise RuntimeError("CLAP44 checkpoint changed while loading")
    return model.to(device).eval().requires_grad_(False), {
        "path": str(path), "sha256": digest, "step": int(payload["step"]),
        "contract": contract, "quality_gate_passed": False,
    }
