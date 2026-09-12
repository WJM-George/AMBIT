#!/usr/bin/env python3
"""Prove full-state resume for revision-6 variable-bucket P10 training."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any

import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def checkpoint_at(root: Path, step: int) -> Path:
    matches = sorted((root / "checkpoints").glob(f"*step={step}.ckpt"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one step-{step} checkpoint, found {len(matches)}")
    return matches[0].resolve(strict=True)


def scalar(value: Any) -> int:
    return int(torch.as_tensor(value).detach().cpu())


def receipt(path: Path, step: int) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    if int(checkpoint.get("global_step", -1)) != step:
        raise RuntimeError(f"wrong global step in {path}")
    fit = checkpoint.get("loops", {}).get("fit_loop", {}).get("state_dict")
    if not isinstance(fit, dict):
        raise RuntimeError(f"{path}: no fit-loop state")
    loaders = fit.get("combined_loader")
    if not isinstance(loaders, list) or len(loaders) != 1:
        raise RuntimeError(f"{path}: no unique loader state")
    loader = loaders[0]
    sampler = loader.get("batch_sampler_state") or {}
    if not (
        loader.get("schema") == "stable_audio_tools.resumable_dataloader"
        and int(loader.get("version", -1)) == 2
        and int(loader.get("batches_yielded", -1)) == step
        and int(loader.get("dataset_items", -1)) == 1_600_000
        and int(loader.get("epoch_batches", 0)) > step
        and sampler.get("schema")
        == "stable_audio_tools.sceneplan_bucket_batch_sampler"
        and int(sampler.get("version", -1)) == 1
        and sampler.get("local_batch_sizes") == {432: 72, 648: 48}
        and sampler.get("bucket_counts") == {"432": 1_200_000, "648": 400_000}
        and int(sampler.get("num_replicas", -1)) == 8
        and int(sampler.get("rank", -1)) == 0
    ):
        raise RuntimeError(f"invalid revision-6 loader cursor at step {step}: {loader}")
    optimizer_steps = []
    for optimizer in checkpoint.get("optimizer_states", []):
        for state in optimizer.get("state", {}).values():
            if isinstance(state, dict) and "step" in state:
                optimizer_steps.append(scalar(state["step"]))
    if not optimizer_steps or min(optimizer_steps) != step:
        raise RuntimeError(f"optimizer state is not at step {step}")
    schedulers = checkpoint.get("lr_schedulers") or []
    scheduler_steps = [int(value.get("last_epoch", -1)) for value in schedulers]
    if not scheduler_steps or any(value != step for value in scheduler_steps):
        raise RuntimeError(f"scheduler state is not at step {step}")
    state = checkpoint.get("state_dict") or {}
    ema = {
        key: scalar(state[key])
        for key in ("diffusion_ema.step", "conditioner_ema.step")
        if key in state
    }
    if ema != {"diffusion_ema.step": step, "conditioner_ema.step": step}:
        raise RuntimeError(f"EMA state is not at step {step}: {ema}")
    return {
        "path": str(path),
        "sha256": sha256(path),
        "global_step": step,
        "data_cursor": loader,
        "optimizer_step_min": min(optimizer_steps),
        "optimizer_step_max": max(optimizer_steps),
        "scheduler_steps": scheduler_steps,
        "ema_steps": ema,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_root.expanduser().resolve(strict=True)
    raw = (root / "logs/train.log").read_text(encoding="utf-8", errors="replace")
    gates = [
        json.loads(value)
        for value in re.findall(r"SAT_TRAINING_GATE_RESULT=(\{[^\r\n]+\})", raw)
    ]
    if len(gates) != 2 or [int(gate.get("global_step", -1)) for gate in gates] != [10, 20]:
        raise RuntimeError("resume preflight did not emit step-10 and step-20 gates")
    for gate, initial, final in zip(gates, (0, 10), (10, 20)):
        if not (
            gate.get("status") == "PASS"
            and int(gate.get("optimizer_state_step_max", -1)) == final
            and int(gate.get("ema_initial", {}).get("diffusion_ema", -1)) == initial
            and int(gate.get("ema_initial", {}).get("conditioner_ema", -1)) == initial
            and int(gate.get("ema_final", {}).get("diffusion_ema", -1)) == final
            and int(gate.get("ema_final", {}).get("conditioner_ema", -1)) == final
            and all(
                math.isfinite(float(value))
                for value in (gate.get("objectives") or {}).values()
            )
        ):
            raise RuntimeError(f"invalid resume health gate: {gate}")
    rng = [
        int(value)
        for value in re.findall(
            r"\[rng\] global_rank=0 global_step=(\d+) training_seed=", raw
        )
    ]
    if rng != [0, 10]:
        raise RuntimeError(f"scratch/resume RNG boundaries changed: {rng}")
    if "Restored all states from the checkpoint" not in raw:
        raise RuntimeError("Lightning did not acknowledge a full-state restore")
    step10 = receipt(checkpoint_at(root, 10), 10)
    step20 = receipt(checkpoint_at(root, 20), 20)
    if (
        int(step20["data_cursor"]["batches_yielded"])
        - int(step10["data_cursor"]["batches_yielded"])
        != 10
    ):
        raise RuntimeError("resumed loader did not advance exactly ten batches")
    report = {
        "schema": "stable_audio_tools.sceneplan_44_revision6_resume_gate",
        "schema_version": 1,
        "status": "PASS",
        "dataset_contract_revision": 6,
        "run_root": str(root),
        "scratch_stage": step10,
        "resumed_stage": step20,
        "health_gates": gates,
        "proof": {
            "model": "Lightning full-state restore acknowledgement",
            "optimizer": "step 10 to 20",
            "scheduler": "step 10 to 20",
            "ema": "both EMA counters 10 to 20",
            "data": "revision-6 bucket sampler and batch cursor 10 to 20",
            "rng": "rank-aware boundary at global steps 0 and 10",
        },
    }
    atomic_json(root / "RESUME_GATE.json", report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

