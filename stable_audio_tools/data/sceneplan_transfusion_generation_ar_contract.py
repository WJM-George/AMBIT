"""Shared provenance helpers for the canonical Generation-AR pipeline.

Keep checkpoint scheduling and continuation-lineage validation here so the
trainer, selector, evaluator, and post-train runner cannot silently implement
different interpretations of the same run contract.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


PARENT_LINEAGE_CONTRACT = "selected_completed_generation_ar_parent_v1"
CHECKPOINT_SELECTION_SCHEMA = (
    "stable_audio_tools.sceneplan_transfusion_generation_ar_checkpoint_selection"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def stage_checkpoint_steps(
    *,
    stage_start_step: int,
    requested_steps: int,
    save_every_steps: int,
    policy: str,
) -> tuple[int, ...]:
    """Return checkpoints produced by this stage, excluding its parent."""

    start = int(stage_start_step)
    end = int(requested_steps)
    interval = int(save_every_steps)
    if not 0 <= start < end or interval <= 0:
        raise ValueError("invalid staged checkpoint interval")
    if policy == "half_epoch_and_end":
        midpoint = start + (end - start + 1) // 2
        return (midpoint, end) if midpoint != end else (end,)
    if policy != "interval":
        raise ValueError("invalid checkpoint policy")
    values = tuple(range(start + interval, end + 1, interval))
    if not values or values[-1] != end:
        values = (*values, end)
    return values


def expected_checkpoint_steps(contract: dict[str, Any]) -> list[int]:
    """Validate and return the only schedule accepted for a full run."""

    requested_steps = int(contract.get("requested_steps", -1))
    policy = str(contract.get("checkpoint_policy", ""))
    schema_version = int(contract.get("schema_version", -1))
    stage_start_step = (
        int(contract.get("stage_start_step", 0)) if schema_version >= 3 else 0
    )
    if requested_steps <= 0:
        raise RuntimeError("Generation AR requested-step contract is invalid")
    if not 0 <= stage_start_step < requested_steps:
        raise RuntimeError("Generation AR stage start is invalid")
    if policy == "half_epoch_and_end":
        return list(
            stage_checkpoint_steps(
                stage_start_step=stage_start_step,
                requested_steps=requested_steps,
                save_every_steps=max(1, requested_steps - stage_start_step),
                policy=policy,
            )
        )
    if policy != "interval":
        raise RuntimeError("Generation AR checkpoint policy is invalid")

    steps_per_epoch = int(contract.get("steps_per_epoch", -1))
    epochs = int(contract.get("epochs_requested", -1))
    save_every = int(contract.get("save_every_steps", -1))
    if (
        steps_per_epoch <= 0
        or steps_per_epoch % 2 != 0
        or not 5 <= epochs <= 10
        or requested_steps != epochs * steps_per_epoch
        or stage_start_step % steps_per_epoch != 0
        or stage_start_step // steps_per_epoch >= epochs
        or save_every != steps_per_epoch // 2
        or (requested_steps - stage_start_step) % save_every != 0
    ):
        raise RuntimeError(
            "Generation AR multi-epoch schedule must cover 5-10 exact epochs "
            "and save every half epoch"
        )
    return list(
        stage_checkpoint_steps(
            stage_start_step=stage_start_step,
            requested_steps=requested_steps,
            save_every_steps=save_every,
            policy=policy,
        )
    )


def expected_selection_candidate_steps(contract: dict[str, Any]) -> list[int]:
    checkpoints = expected_checkpoint_steps(contract)
    parent = contract.get("parent_lineage")
    if parent is None:
        return checkpoints
    if not isinstance(parent, dict):
        raise RuntimeError("Generation AR parent lineage is invalid")
    parent_step = int(parent.get("checkpoint_step", -1))
    if (
        parent.get("contract") != PARENT_LINEAGE_CONTRACT
        or parent_step != int(contract.get("stage_start_step", -2))
        or parent_step <= 0
    ):
        raise RuntimeError("Generation AR parent checkpoint step is invalid")
    return [parent_step, *checkpoints]


def _read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def validate_parent_lineage_artifacts(contract: dict[str, Any]) -> Path | None:
    """Fail closed if any artifact proving the continuation parent changed."""

    parent = contract.get("parent_lineage")
    if parent is None:
        return None
    if not isinstance(parent, dict):
        raise RuntimeError("Generation AR parent lineage is not an object")

    parent_run = Path(str(parent.get("run_dir", ""))).resolve(strict=True)
    parent_contract_path = Path(str(parent.get("run_contract", ""))).resolve(
        strict=True
    )
    parent_final_path = Path(str(parent.get("final", ""))).resolve(strict=True)
    parent_selection_path = Path(str(parent.get("selection_manifest", ""))).resolve(
        strict=True
    )
    parent_checkpoint = Path(str(parent.get("checkpoint", ""))).resolve(strict=True)
    parent_contract = _read_json_object(parent_contract_path)
    parent_final = _read_json_object(parent_final_path)
    parent_selection = _read_json_object(parent_selection_path)

    contract_sha = sha256_file(parent_contract_path)
    final_sha = sha256_file(parent_final_path)
    checkpoint_sha = sha256_file(parent_checkpoint)
    parent_step = int(parent.get("checkpoint_step", -1))
    if (
        parent.get("contract") != PARENT_LINEAGE_CONTRACT
        or parent_contract_path != parent_run / "RUN_CONTRACT.json"
        or parent_final_path != parent_run / "FINAL.json"
        or parent_checkpoint.parent.parent != parent_run
        or parent.get("run_contract_sha256") != contract_sha
        or parent.get("run_contract_canonical_sha256")
        != canonical_sha256(parent_contract)
        or parent.get("final_sha256") != final_sha
        or parent.get("selection_manifest_sha256") != sha256_file(parent_selection_path)
        or parent.get("checkpoint_sha256") != checkpoint_sha
        or parent_step != int(contract.get("stage_start_step", -2))
        or parent_step <= 0
        or parent_contract.get("mode") != "full"
        or int(parent_contract.get("requested_steps", -1)) != parent_step
        or parent_final.get("event") != "complete"
        or parent_final.get("mode") != "full"
        or int(parent_final.get("step", -1)) != parent_step
        or parent_selection.get("schema") != CHECKPOINT_SELECTION_SCHEMA
        or parent_selection.get("status") != "COMPLETE"
        or Path(str(parent_selection.get("training_run_dir", ""))).resolve()
        != parent_run
        or parent_selection.get("training_run_contract_sha256") != contract_sha
        or parent_selection.get("training_final_sha256") != final_sha
        or Path(str(parent_selection.get("selected_checkpoint", ""))).resolve()
        != parent_checkpoint
        or parent_selection.get("selected_checkpoint_sha256") != checkpoint_sha
        or int(parent_selection.get("selected_checkpoint_step", -1)) != parent_step
    ):
        raise RuntimeError("Generation AR parent lineage artifact mismatch")
    return parent_checkpoint


__all__ = [
    "CHECKPOINT_SELECTION_SCHEMA",
    "PARENT_LINEAGE_CONTRACT",
    "canonical_json_bytes",
    "canonical_sha256",
    "expected_checkpoint_steps",
    "expected_selection_candidate_steps",
    "sha256_file",
    "stage_checkpoint_steps",
    "validate_parent_lineage_artifacts",
]
