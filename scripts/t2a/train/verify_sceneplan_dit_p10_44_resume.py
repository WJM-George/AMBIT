#!/usr/bin/env python3
"""Fail-closed proof that P10 restores every state and the next data row."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any

import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
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
        raise RuntimeError(
            f"expected exactly one step-{step} checkpoint, found {len(matches)}"
        )
    return matches[0].resolve(strict=True)


def scalar(value: Any) -> int:
    return int(torch.as_tensor(value).detach().cpu())


def checkpoint_receipt(path: Path, expected_step: int) -> dict[str, Any]:
    checkpoint = torch.load(
        path, map_location="cpu", mmap=True, weights_only=False
    )
    if int(checkpoint.get("global_step", -1)) != expected_step:
        raise RuntimeError(f"wrong global_step in {path}")

    fit_loop = dict(checkpoint.get("loops", {}).get("fit_loop", {}))
    # Lightning stores custom fit-loop state under fit_loop.state_dict; the
    # neighbouring flattened keys are progress trackers, not dataloader
    # state.  Fail closed on exactly one resumable-loader receipt.
    fit_state = fit_loop.get("state_dict")
    if not isinstance(fit_state, dict):
        raise RuntimeError(f"checkpoint has no fit-loop state_dict: {path}")
    loader_states = fit_state.get("combined_loader")
    if not isinstance(loader_states, list) or len(loader_states) != 1:
        raise RuntimeError(f"checkpoint has no unique P10 loader state: {path}")
    loader = loader_states[0]
    if not (
        loader.get("schema") == "stable_audio_tools.resumable_dataloader"
        and loader.get("version") == 1
        and int(loader.get("batches_yielded", -1)) == expected_step
        and int(loader.get("dataset_items", -1)) == 1_100_000
        and int(loader.get("epoch_batches", 0)) > expected_step
    ):
        raise RuntimeError(f"invalid data cursor at step {expected_step}: {loader}")

    optimizer_steps = []
    for optimizer in checkpoint.get("optimizer_states", []):
        for state in optimizer.get("state", {}).values():
            if isinstance(state, dict) and "step" in state:
                optimizer_steps.append(scalar(state["step"]))
    if not optimizer_steps or min(optimizer_steps) != expected_step:
        raise RuntimeError(
            f"optimizer state did not reach step {expected_step}: "
            f"min={min(optimizer_steps) if optimizer_steps else None}, "
            f"max={max(optimizer_steps) if optimizer_steps else None}"
        )

    schedulers = checkpoint.get("lr_schedulers")
    if not isinstance(schedulers, list) or not schedulers:
        raise RuntimeError("checkpoint has no scheduler state")
    scheduler_steps = [int(value.get("last_epoch", -1)) for value in schedulers]
    if any(step != expected_step for step in scheduler_steps):
        raise RuntimeError(
            f"scheduler state did not reach step {expected_step}: {scheduler_steps}"
        )

    state_dict = checkpoint.get("state_dict", {})
    ema_steps = {
        name: scalar(value)
        for name, value in state_dict.items()
        if name in {"diffusion_ema.step", "conditioner_ema.step"}
    }
    if ema_steps != {
        "diffusion_ema.step": expected_step,
        "conditioner_ema.step": expected_step,
    }:
        raise RuntimeError(
            f"EMA state did not reach step {expected_step}: {ema_steps}"
        )

    return {
        "path": str(path),
        "sha256": sha256(path),
        "global_step": expected_step,
        "data_cursor": dict(loader),
        "optimizer_step_min": min(optimizer_steps),
        "optimizer_step_max": max(optimizer_steps),
        "scheduler_steps": scheduler_steps,
        "ema_steps": ema_steps,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()

    run_root = args.run_root.expanduser().resolve(strict=True)
    contract_path = run_root / "SCENEPLAN_44_RUN_CONTRACT.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if not (
        contract.get("mode") == "resume"
        and contract.get("schema_version", 0) >= 7
        and contract.get("warm_start") is False
        and contract.get("training_schedule", {}).get("max_steps") == 20
        and contract.get("training_schedule", {}).get("checkpoint_every") == 10
    ):
        raise RuntimeError("not a frozen P10 resume-preflight contract")

    train_log = run_root / "logs/train.log"
    raw = train_log.read_text(encoding="utf-8", errors="replace")
    resume_launcher_log = run_root / "logs/resume_load_launcher.log"
    launcher_raw = resume_launcher_log.read_text(
        encoding="utf-8", errors="replace"
    )
    gates = [
        json.loads(value)
        for value in re.findall(
            r"SAT_TRAINING_GATE_RESULT=(\{[^\r\n]+\})", raw
        )
    ]
    if len(gates) != 2 or [gate.get("global_step") for gate in gates] != [10, 20]:
        raise RuntimeError(
            "resume preflight must emit exactly the step-10 and step-20 health gates"
        )
    for index, (gate, initial, final) in enumerate(
        zip(gates, (0, 10), (10, 20)), start=1
    ):
        if not (
            gate.get("status") == "PASS"
            and gate.get("metric_observations") == 10
            and gate.get("optimizer_state_step_max") == final
            and gate.get("ema_initial", {}).get("diffusion_ema") == initial
            and gate.get("ema_initial", {}).get("conditioner_ema") == initial
            and gate.get("ema_final", {}).get("diffusion_ema") == final
            and gate.get("ema_final", {}).get("conditioner_ema") == final
            and all(
                math.isfinite(float(value))
                for value in gate.get("objectives", {}).values()
            )
        ):
            raise RuntimeError(f"invalid health gate for resume stage {index}: {gate}")

    rng_steps = [
        int(value)
        for value in re.findall(
            r"\[rng\] global_rank=0 global_step=(\d+) training_seed=", raw
        )
    ]
    if rng_steps != [0, 10]:
        raise RuntimeError(f"scratch/resume RNG boundary changed: {rng_steps}")
    # The shell launcher owns checkpoint selection, while Lightning owns the
    # restore acknowledgement.  They are intentionally recorded in separate
    # logs, so require both independent pieces of evidence instead of looking
    # for the launcher line in Python's train.log.
    if not (
        re.search(r"resuming from .*step=10\.ckpt", launcher_raw)
        and "Restored all states from the checkpoint" in raw
    ):
        raise RuntimeError("log does not prove a full step-10 checkpoint restore")

    step10 = checkpoint_receipt(checkpoint_at(run_root, 10), 10)
    step20 = checkpoint_receipt(checkpoint_at(run_root, 20), 20)
    if step20["data_cursor"]["batches_yielded"] - step10["data_cursor"][
        "batches_yielded"
    ] != 10:
        raise RuntimeError("resumed data cursor did not advance exactly ten batches")

    report = {
        "schema": "stable_audio_tools.sceneplan_44_resume_gate",
        "schema_version": 1,
        "ok": True,
        "run_root": str(run_root),
        "contract": str(contract_path),
        "contract_sha256": sha256(contract_path),
        "train_log": str(train_log),
        "train_log_sha256": sha256(train_log),
        "resume_launcher_log": str(resume_launcher_log),
        "resume_launcher_log_sha256": sha256(resume_launcher_log),
        "scratch_stage": step10,
        "resumed_stage": step20,
        "health_gates": gates,
        "proof": {
            "model_state": "restored by Lightning strict checkpoint restore",
            "optimizer_state": "step 10 -> step 20",
            "scheduler_state": "last_epoch 10 -> 20",
            "ema_state": "both EMA counters 10 -> 20",
            "rng_state": "checkpoint restore plus rank-aware seed at global_step 10",
            "data_cursor": "batch 10 -> next unconsumed batch -> batch 20",
        },
    }
    output = run_root / "RESUME_GATE.json"
    atomic_json(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
