#!/usr/bin/env python3
"""Immutable run identity and crash-safe resume resolution for Editing Joint.

The formal Joint trainer publishes named 5K checkpoints before updating its
``LATEST.json`` pointer.  This module treats the named checkpoint payloads as
the durable source of truth, binds every artifact to one absolute run directory
and UUID, and repairs only the narrow pointer-publication crash window.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import sys
from typing import Any, Mapping
import uuid

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stable_audio_tools.data.sceneplan_transfusion_editing_index import (  # noqa: E402
    sha256_file,
)


RUN_IDENTITY_SCHEMA = "sceneplan_transfusion_editing_joint_run_identity"
RUN_IDENTITY_SCHEMA_VERSION = 1
RUN_IDENTITY_NAME = "RUN_ID.json"
RUN_ID_PATTERN = re.compile(r"[0-9a-f]{32}")
RUN_CONTRACT_SCHEMA = "sceneplan_transfusion_editing_joint_full_run"
RUN_CONTRACT_SCHEMA_VERSION = 2
RUN_CONTRACT_NAME = "RUN_CONTRACT.json"
CHECKPOINT_SCHEMA = "sceneplan_transfusion_editing_joint_full_checkpoint"
CHECKPOINT_SCHEMA_VERSION = 2
LATEST_SCHEMA = "sceneplan_transfusion_editing_joint_latest_checkpoint"
LATEST_SCHEMA_VERSION = 1
FINAL_SCHEMA = "sceneplan_transfusion_editing_joint_final"
FINAL_SCHEMA_VERSION = 1
RNG_SCHEMA = "sceneplan_transfusion_editing_joint_rank_rng"
RNG_SCHEMA_VERSION = 1
CHECKPOINT_PATTERN = re.compile(r"step-(\d{8})\.pt")
DEFAULT_MAX_STEPS = 25_000
DEFAULT_SAVE_EVERY = 5_000
FORMAL_WORLD_SIZE = 5
EDITING_AR_CONTRACT = "p10v11_shared_blocks_audio_reference_editing_ar_v4"
EDITING_JOINT_DATASET_CONTRACT = (
    "source_audio_instruction_to_canonical_new_plan_plus_aligned_rf_v2"
)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _exclusive_json(path: Path, value: Mapping[str, Any]) -> None:
    """Create an immutable small artifact without an overwrite race."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        payload = (
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
        remaining = memoryview(payload)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise OSError("could not write joint Editing run identity")
            remaining = remaining[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def validate_run_identity(
    run_dir: Path, *, identity_path: Path | None = None
) -> dict[str, Any]:
    run_dir = run_dir.expanduser().resolve()
    path = (
        run_dir / RUN_IDENTITY_NAME
        if identity_path is None
        else identity_path.expanduser().resolve(strict=True)
    )
    if path.resolve(strict=True) != run_dir / RUN_IDENTITY_NAME:
        raise RuntimeError("joint Editing run identity escaped its run directory")
    value = json.loads(path.read_text(encoding="utf-8"))
    run_id = str(value.get("run_id", ""))
    if not (
        set(value) == {"schema", "schema_version", "run_dir", "run_id"}
        and value.get("schema") == RUN_IDENTITY_SCHEMA
        and int(value.get("schema_version", -1)) == RUN_IDENTITY_SCHEMA_VERSION
        and value.get("run_dir") == str(run_dir)
        and RUN_ID_PATTERN.fullmatch(run_id) is not None
    ):
        raise RuntimeError("joint Editing run identity changed")
    return value


def ensure_run_identity(run_dir: Path) -> dict[str, Any]:
    """Load an existing identity, or mint one only for an unclaimed run."""

    run_dir = run_dir.expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / RUN_IDENTITY_NAME
    if path.exists():
        return validate_run_identity(run_dir, identity_path=path)

    unexpected: list[Path] = []
    for child in run_dir.iterdir():
        if child.name == ".editing-joint-full.lock":
            continue
        if child.name in {"checkpoints", "logs"} and child.is_dir():
            if not any(child.iterdir()):
                continue
        unexpected.append(child)
    if unexpected:
        names = ", ".join(sorted(child.name for child in unexpected))
        raise RuntimeError(
            "refusing to mint a joint Editing run_id over existing artifacts: "
            f"{names}"
        )
    value = {
        "schema": RUN_IDENTITY_SCHEMA,
        "schema_version": RUN_IDENTITY_SCHEMA_VERSION,
        "run_dir": str(run_dir),
        "run_id": uuid.uuid4().hex,
    }
    _exclusive_json(path, value)
    return validate_run_identity(run_dir, identity_path=path)


def validate_run_contract_identity(
    run_dir: Path,
    identity: Mapping[str, Any],
    *,
    max_steps: int,
    save_every: int,
) -> tuple[dict[str, Any], Path, str]:
    run_dir = run_dir.expanduser().resolve()
    expected_path = run_dir / RUN_CONTRACT_NAME
    path = expected_path.resolve(strict=True)
    if path != expected_path:
        raise RuntimeError("joint Editing run contract escaped its run directory")
    identity_path = (run_dir / RUN_IDENTITY_NAME).resolve(strict=True)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not (
        value.get("schema") == RUN_CONTRACT_SCHEMA
        and int(value.get("schema_version", -1)) == RUN_CONTRACT_SCHEMA_VERSION
        and value.get("run_dir") == str(run_dir)
        and value.get("run_id") == identity.get("run_id")
        and value.get("run_identity")
        == {"path": str(identity_path), "sha256": sha256_file(identity_path)}
        and value.get("editing_ar_contract") == EDITING_AR_CONTRACT
        and value.get("joint_dataset_contract")
        == EDITING_JOINT_DATASET_CONTRACT
        and int(value.get("max_steps", -1)) == int(max_steps)
        and int(value.get("save_every", -1)) == int(save_every)
        and int(value.get("world_size", -1)) == FORMAL_WORLD_SIZE
    ):
        raise RuntimeError("joint Editing run contract identity changed")
    return value, path, sha256_file(path)


def validate_rng_inventory(value: Any, *, world_size: int) -> None:
    if (
        int(world_size) <= 0
        or not isinstance(value, list)
        or len(value) != int(world_size)
    ):
        raise RuntimeError("joint Editing checkpoint rank RNG inventory changed")
    if not all(isinstance(record, Mapping) for record in value):
        raise RuntimeError("joint Editing checkpoint rank RNG inventory changed")
    try:
        ranks = [int(record.get("rank", -1)) for record in value]
    except (TypeError, ValueError):
        ranks = []
    if ranks != list(range(int(world_size))):
        raise RuntimeError("joint Editing checkpoint rank RNG inventory changed")
    for record in value:
        numpy_state = record.get("numpy_random_state")
        cpu_state = record.get("torch_cpu_rng_state")
        cuda_state = record.get("torch_cuda_rng_state")
        try:
            valid = (
                record.get("schema") == RNG_SCHEMA
                and int(record.get("schema_version", -1)) == RNG_SCHEMA_VERSION
                and isinstance(record.get("python_random_state"), tuple)
                and isinstance(numpy_state, Mapping)
                and numpy_state.get("bit_generator") == "MT19937"
                and isinstance(numpy_state.get("keys"), torch.Tensor)
                and numpy_state["keys"].dtype == torch.int64
                and numpy_state["keys"].ndim == 1
                and int(numpy_state["keys"].numel()) == 624
                and 0 <= int(numpy_state.get("position", -1)) <= 624
                and int(numpy_state.get("has_gauss", -1)) in (0, 1)
                and isinstance(cpu_state, torch.Tensor)
                and cpu_state.dtype == torch.uint8
                and cpu_state.ndim == 1
                and isinstance(cuda_state, torch.Tensor)
                and cuda_state.dtype == torch.uint8
                and cuda_state.ndim == 1
            )
        except (AttributeError, TypeError, ValueError):
            valid = False
        if not valid:
            raise RuntimeError(
                "joint Editing checkpoint rank RNG inventory changed"
            )


def _load_named_candidate(
    path: Path,
    *,
    expected_step: int,
    run_dir: Path,
    identity: Mapping[str, Any],
    run_contract: Mapping[str, Any],
    run_contract_path: Path,
    run_contract_sha256: str,
) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"invalid joint Editing named checkpoint: {path}") from exc
    world_size = int(run_contract.get("world_size", -1))
    if not (
        isinstance(payload, dict)
        and payload.get("schema") == CHECKPOINT_SCHEMA
        and int(payload.get("schema_version", -1)) == CHECKPOINT_SCHEMA_VERSION
        and payload.get("contract") == EDITING_AR_CONTRACT
        and payload.get("joint_dataset_contract")
        == EDITING_JOINT_DATASET_CONTRACT
        and int(payload.get("global_step", -1)) == int(expected_step)
        and int(payload.get("epoch", -1)) >= 0
        and int(payload.get("batch_in_epoch", -1)) > 0
        and payload.get("run_dir") == str(run_dir)
        and payload.get("run_id") == identity.get("run_id")
        and payload.get("run_contract") == run_contract
        and payload.get("run_contract_path") == str(run_contract_path)
        and payload.get("run_contract_sha256") == run_contract_sha256
        and world_size == FORMAL_WORLD_SIZE
        and isinstance(payload.get("diffusion_state_dict"), dict)
        and isinstance(payload.get("editing_ar_specific_state_dict"), dict)
        and isinstance(payload.get("optimizer"), dict)
        and isinstance(payload.get("scheduler"), dict)
    ):
        raise RuntimeError(f"joint Editing checkpoint lineage changed: {path}")
    validate_rng_inventory(payload.get("rng_states_by_rank"), world_size=world_size)
    return payload


def latest_record(
    *,
    run_dir: Path,
    identity: Mapping[str, Any],
    run_contract_path: Path,
    run_contract_sha256: str,
    checkpoint: Path,
    checkpoint_sha256: str,
    step: int,
) -> dict[str, Any]:
    return {
        "schema": LATEST_SCHEMA,
        "schema_version": LATEST_SCHEMA_VERSION,
        "run_dir": str(run_dir),
        "run_id": identity["run_id"],
        "run_contract_path": str(run_contract_path),
        "run_contract_sha256": run_contract_sha256,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "global_step": int(step),
    }


def _quarantine_latest(path: Path, *, reason: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = path.parent / f"recovery-quarantine-{stamp}-{os.getpid()}"
    destination.mkdir(parents=False, exist_ok=False)
    moved = destination / path.name
    shutil.move(str(path), str(moved))
    _atomic_json(
        destination / "RECOVERY.json",
        {"reason": reason, "source": str(path), "preserved_as": str(moved)},
    )
    print(
        f"[editing-joint-lineage] preserved invalid LATEST at {moved}",
        file=sys.stderr,
    )
    return moved


def _validate_final(
    path: Path,
    *,
    run_dir: Path,
    identity: Mapping[str, Any],
    run_contract_path: Path,
    run_contract_sha256: str,
    checkpoint: Path,
    checkpoint_sha256: str,
    max_steps: int,
) -> None:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not (
        value.get("schema") == FINAL_SCHEMA
        and int(value.get("schema_version", -1)) == FINAL_SCHEMA_VERSION
        and value.get("event") == "complete"
        and int(value.get("step", -1)) == int(max_steps)
        and value.get("run_dir") == str(run_dir)
        and value.get("run_id") == identity.get("run_id")
        and value.get("run_contract_path") == str(run_contract_path)
        and value.get("run_contract_sha256") == run_contract_sha256
        and Path(str(value.get("checkpoint", ""))).expanduser().resolve(strict=True)
        == checkpoint
        and value.get("checkpoint_sha256") == checkpoint_sha256
        and value.get("shared_transformer_same_object") is True
        and value.get("old_sceneplan_exposed") is False
        and value.get("source_semantic_mode") == "m2d_audio_caption_aux"
        and value.get("source_caption_exposed_as_model_input") is False
    ):
        raise RuntimeError("completed joint Editing FINAL identity changed")


def resolve_resume(
    run_dir: Path,
    *,
    requested_checkpoint: Path | None = None,
    max_steps: int = DEFAULT_MAX_STEPS,
    save_every: int = DEFAULT_SAVE_EVERY,
) -> tuple[str, str | None]:
    """Resolve the highest verified named checkpoint without rollback.

    A missing or stale ``LATEST.json`` is repaired only after every named
    checkpoint through the highest observed step has passed payload, run-id,
    run-directory, run-contract, RNG-inventory, and file-hash validation.
    """

    run_dir = run_dir.expanduser().resolve()
    if int(max_steps) != DEFAULT_MAX_STEPS or int(save_every) != DEFAULT_SAVE_EVERY:
        raise RuntimeError("formal joint Editing resume schedule changed")
    identity = ensure_run_identity(run_dir)
    if (run_dir / RUN_CONTRACT_NAME).exists():
        validate_run_contract_identity(
            run_dir,
            identity,
            max_steps=max_steps,
            save_every=save_every,
        )
    checkpoint_dir = run_dir / "checkpoints"
    if not checkpoint_dir.exists():
        if requested_checkpoint is not None or (run_dir / "FINAL.json").exists():
            raise RuntimeError("joint Editing resume artifacts have no checkpoints")
        return "FRESH", None

    resolved_checkpoint_dir = checkpoint_dir.resolve(strict=True)
    if resolved_checkpoint_dir != checkpoint_dir:
        raise RuntimeError("joint Editing checkpoint directory escaped its run")
    checkpoint_dir = resolved_checkpoint_dir
    raw_named = sorted(checkpoint_dir.glob("step-*.pt"))
    for path in raw_named:
        if CHECKPOINT_PATTERN.fullmatch(path.name) is None:
            raise RuntimeError(f"non-canonical joint Editing checkpoint: {path}")
    other_pt = sorted(
        path
        for path in checkpoint_dir.glob("*.pt")
        if not path.name.startswith("step-")
    )
    if other_pt:
        raise RuntimeError(f"unexpected joint Editing checkpoint: {other_pt[0]}")

    if not raw_named:
        if (checkpoint_dir / "LATEST.json").exists() or requested_checkpoint is not None:
            raise RuntimeError("joint Editing resume pointer has no named checkpoint")
        if (run_dir / "FINAL.json").exists():
            raise RuntimeError("joint Editing FINAL has no named checkpoint")
        return "FRESH", None

    run_contract, run_contract_path, run_contract_sha = validate_run_contract_identity(
        run_dir,
        identity,
        max_steps=max_steps,
        save_every=save_every,
    )
    named: dict[int, tuple[Path, str]] = {}
    for path in raw_named:
        match = CHECKPOINT_PATTERN.fullmatch(path.name)
        assert match is not None
        step = int(match.group(1))
        if not (0 < step <= int(max_steps) and step % int(save_every) == 0):
            raise RuntimeError(f"joint Editing checkpoint step is out of contract: {path}")
        if step in named:
            raise RuntimeError(f"ambiguous joint Editing checkpoint step: {step}")
        resolved = path.resolve(strict=True)
        if resolved.parent != checkpoint_dir:
            raise RuntimeError(f"joint Editing checkpoint escaped its run: {path}")
        _load_named_candidate(
            resolved,
            expected_step=step,
            run_dir=run_dir,
            identity=identity,
            run_contract=run_contract,
            run_contract_path=run_contract_path,
            run_contract_sha256=run_contract_sha,
        )
        named[step] = (resolved, sha256_file(resolved))

    highest_step = max(named)
    expected_through_highest = list(range(save_every, highest_step + 1, save_every))
    if sorted(named) != expected_through_highest:
        missing = sorted(set(expected_through_highest) - set(named))
        raise RuntimeError(
            "joint Editing checkpoint history has missing completed intervals: "
            f"{missing}"
        )
    highest, highest_sha = named[highest_step]

    final_path = run_dir / "FINAL.json"
    if final_path.exists() and highest_step != int(max_steps):
        raise RuntimeError(
            "completed joint Editing run cannot roll back to a preterminal checkpoint"
        )

    if requested_checkpoint is not None:
        requested = requested_checkpoint.expanduser().resolve(strict=True)
        if requested != highest:
            raise RuntimeError(
                "RESUME_CHECKPOINT cannot bypass the highest verified joint candidate"
            )

    expected_latest = latest_record(
        run_dir=run_dir,
        identity=identity,
        run_contract_path=run_contract_path,
        run_contract_sha256=run_contract_sha,
        checkpoint=highest,
        checkpoint_sha256=highest_sha,
        step=highest_step,
    )
    latest_path = checkpoint_dir / "LATEST.json"
    if latest_path.exists():
        try:
            if latest_path.resolve(strict=True) != latest_path:
                raise RuntimeError(
                    "LATEST escaped its checkpoint directory; rollback is forbidden"
                )
            observed_latest = json.loads(latest_path.read_text(encoding="utf-8"))
            if observed_latest != expected_latest:
                claimed_steps: list[int] = []
                try:
                    claimed_steps.append(int(observed_latest.get("global_step", -1)))
                except (TypeError, ValueError):
                    pass
                checkpoint_name = Path(
                    str(observed_latest.get("checkpoint", ""))
                ).name
                checkpoint_match = CHECKPOINT_PATTERN.fullmatch(checkpoint_name)
                if checkpoint_match is not None:
                    claimed_steps.append(int(checkpoint_match.group(1)))
                if any(step > highest_step for step in claimed_steps):
                    raise RuntimeError(
                        "LATEST claims a newer joint Editing step than the highest "
                        "verified named checkpoint; rollback is forbidden"
                    )
                raise RuntimeError("LATEST is stale or inconsistent")
        except Exception as exc:  # noqa: BLE001
            if "rollback is forbidden" in str(exc):
                raise
            _quarantine_latest(latest_path, reason=f"{type(exc).__name__}: {exc}")
            _atomic_json(latest_path, expected_latest)
    else:
        _atomic_json(latest_path, expected_latest)

    if final_path.exists():
        resolved_final_path = final_path.resolve(strict=True)
        if resolved_final_path != final_path:
            raise RuntimeError("joint Editing FINAL escaped its run directory")
        _validate_final(
            resolved_final_path,
            run_dir=run_dir,
            identity=identity,
            run_contract_path=run_contract_path,
            run_contract_sha256=run_contract_sha,
            checkpoint=highest,
            checkpoint_sha256=highest_sha,
            max_steps=max_steps,
        )
        return "COMPLETE", None
    return "RESUME", str(highest)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    ensure = subparsers.add_parser("ensure-identity")
    ensure.add_argument("--run-dir", type=Path, required=True)
    resolve = subparsers.add_parser("resolve-resume")
    resolve.add_argument("--run-dir", type=Path, required=True)
    resolve.add_argument("--requested-checkpoint", type=Path)
    resolve.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    resolve.add_argument("--save-every", type=int, default=DEFAULT_SAVE_EVERY)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.command == "ensure-identity":
        value = ensure_run_identity(args.run_dir)
        print(value["run_id"])
        return 0
    action, checkpoint = resolve_resume(
        args.run_dir,
        requested_checkpoint=args.requested_checkpoint,
        max_steps=int(args.max_steps),
        save_every=int(args.save_every),
    )
    print(action)
    print(checkpoint or "")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
