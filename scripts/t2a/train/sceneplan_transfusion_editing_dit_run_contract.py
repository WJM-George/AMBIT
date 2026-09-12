#!/usr/bin/env python3
"""Freeze and validate the formal Editing-DiT training lineage.

This module is deliberately Editing-only.  The generic trainer embeds the
contract only when ``SAT_EDITING_RUN_CONTRACT_PATH`` is explicitly exported by
the formal Editing launcher.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stable_audio_tools.data.sceneplan_transfusion_editing_index import (  # noqa: E402
    sha256_file,
)
from stable_audio_tools.data.sceneplan_transfusion_editing_dataset import (  # noqa: E402
    verify_editing_source_latent_shards,
    verify_editing_target_latent_shards,
)
from scripts.t2a.train.prepare_sceneplan_transfusion_editing_full import (  # noqa: E402
    CURRENT_SHARED_CONTRACT,
    INDEX_BUILD_SHARED_CONTRACT_SHA256,
)


SCHEMA = "sceneplan_transfusion_editing_dit_training_run_contract"
SCHEMA_VERSION = 2
CONTRACT_NAME = "TRAIN_RUN_CONTRACT.json"
VISIBLE_GPUS = os.environ.get("CUDA_VISIBLE_DEVICES", "").replace(" ", "")
PHYSICAL_GPUS = [int(item) for item in VISIBLE_GPUS.split(",") if item]
WORLD_SIZE = max(len(PHYSICAL_GPUS), 1)
SOURCE_PATHS = (
    "pyproject.toml",
    "uv.lock",
    "train.py",
    "scripts/t2a/train/sceneplan_transfusion_editing_dit_run_contract.py",
    "scripts/t2a/train/prepare_sceneplan_transfusion_editing_full.py",
    "scripts/t2a/train/run_sceneplan_transfusion_editing_full_chain_5gpu.sh",
    "scripts/t2a/train/run_sceneplan_transfusion_editing_dit_full_5gpu.sh",
    "scripts/t2a/train/run_t2a_common_8gpu.sh",
    "stable_audio_tools/configuration.py",
    "stable_audio_tools/data/dataset.py",
    "stable_audio_tools/data/model_sceneplan.py",
    "stable_audio_tools/data/resumable_dataloader.py",
    "stable_audio_tools/data/sceneplan_bucket_sampler.py",
    "stable_audio_tools/data/sceneplan_transfusion_editing.py",
    "stable_audio_tools/data/sceneplan_transfusion_editing_dataset.py",
    "stable_audio_tools/data/sceneplan_transfusion_editing_index.py",
    "stable_audio_tools/data/text_conditioning.py",
    "stable_audio_tools/inference/generation.py",
    "stable_audio_tools/inference/sampling.py",
    "stable_audio_tools/models/blocks.py",
    "stable_audio_tools/models/conditioners.py",
    "stable_audio_tools/models/diffusion.py",
    "stable_audio_tools/models/dit.py",
    "stable_audio_tools/models/factory.py",
    "stable_audio_tools/models/pretransforms.py",
    "stable_audio_tools/models/sceneplan_transfusion_editing_provenance.py",
    "stable_audio_tools/models/transformer.py",
    "stable_audio_tools/models/utils.py",
    "stable_audio_tools/training/diffusion.py",
    "stable_audio_tools/training/distributed.py",
    "stable_audio_tools/training/ema.py",
    "stable_audio_tools/training/factory.py",
    "stable_audio_tools/training/utils.py",
)
CHECKPOINT_PATTERN = re.compile(r"epoch=(\d+)-step=(\d+)\.ckpt$")
LATENT_AUDIT_KEYS = {
    role: (
        f"{role}_latent_shards",
        f"{role}_pair_rows",
        f"{role}_latent_shard_inventory_sha256",
        f"{role}_latent_shards_exhaustively_verified",
    )
    for role in ("source", "target")
}


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _artifact(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


def _source_sha256() -> dict[str, str]:
    return {
        relative: sha256_file((REPO_ROOT / relative).resolve(strict=True))
        for relative in SOURCE_PATHS
    }


def _latent_audit(
    record: Mapping[str, Any], *, role: str, split: str, expected_rows: int
) -> dict[str, Any]:
    if role not in LATENT_AUDIT_KEYS:
        raise ValueError("Editing-DiT latent audit role changed")
    shard_key, rows_key, digest_key, verified_key = LATENT_AUDIT_KEYS[role]
    audit = {key: record.get(key) for key in LATENT_AUDIT_KEYS[role]}
    digest = str(audit[digest_key] or "")
    if not (
        audit[verified_key] is True
        and int(audit[rows_key] or -1) == int(expected_rows)
        and int(audit[shard_key] or -1) > 0
        and re.fullmatch(r"[0-9a-f]{64}", digest) is not None
    ):
        raise RuntimeError(
            f"Editing-DiT {split} {role}-latent audit is incomplete"
        )
    return {
        shard_key: int(audit[shard_key]),
        rows_key: int(audit[rows_key]),
        digest_key: digest,
        verified_key: True,
    }


def build_contract(
    *,
    run_dir: Path,
    preflight_path: Path,
    model_config: Path,
    train_dataset_config: Path,
    validation_dataset_config: Path,
    p10_checkpoint: Path,
    max_steps: int,
    checkpoint_every: int,
    batch_size: int,
    long_batch_size: int,
    num_workers: int,
    training_seed: int,
) -> dict[str, Any]:
    run_dir = run_dir.expanduser().resolve()
    preflight_path = preflight_path.expanduser().resolve(strict=True)
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    expected_steps = list(range(checkpoint_every, max_steps + 1, checkpoint_every))
    train_index = dict(preflight.get("indices", {}).get("train") or {})
    validation_index = dict(preflight.get("indices", {}).get("validation") or {})
    shared_contract = dict(preflight.get("shared_contract") or {})
    train_source_audit = _latent_audit(
        train_index, role="source", split="train", expected_rows=1_000_000
    )
    validation_source_audit = _latent_audit(
        validation_index,
        role="source",
        split="validation",
        expected_rows=20_000,
    )
    train_target_audit = _latent_audit(
        train_index, role="target", split="train", expected_rows=1_000_000
    )
    validation_target_audit = _latent_audit(
        validation_index,
        role="target",
        split="validation",
        expected_rows=20_000,
    )
    if not (
        preflight.get("schema")
        == "sceneplan_transfusion_editing_full_training_preflight"
        and int(preflight.get("schema_version", -1)) == 2
        and preflight.get("status") == "PASS"
        and preflight.get("latest_route", {}).get("editing_ar_inputs")
        == ["source_foa_latent", "raw_edit_request"]
        and preflight.get("latest_route", {}).get("old_sceneplan_input") is False
        and preflight.get("split_disjointness", {}).get("status") == "PASS"
        and all(preflight.get("split_disjointness", {}).get("checks", {}).values())
        and int(
            preflight.get("latest_route", {}).get(
                "editing_dit_frame_channels", -1
            )
        )
        == 384
        and int(train_index.get("rows", -1)) == 1_000_000
        and int(validation_index.get("rows", -1)) == 20_000
        and max_steps == 30_000
        and checkpoint_every == 5_000
        and expected_steps == [5_000, 10_000, 15_000, 20_000, 25_000, 30_000]
        and batch_size == 72
        and long_batch_size == 48
        and num_workers == 12
        and training_seed == 42
        and Path(shared_contract.get("path", "")).resolve()
        == CURRENT_SHARED_CONTRACT.resolve(strict=True)
        and shared_contract.get("sha256")
        == sha256_file(CURRENT_SHARED_CONTRACT.resolve(strict=True))
        and shared_contract.get("index_build_sha256")
        == INDEX_BUILD_SHARED_CONTRACT_SHA256
        and shared_contract.get("index_build_role")
        == "historical_pair_construction_provenance"
        and shared_contract.get("current_role") == "active_execution_contract"
    ):
        raise RuntimeError("formal Editing-DiT inputs do not match the frozen route")

    model_artifact = _artifact(model_config)
    p10_artifact = _artifact(p10_checkpoint)
    train_config_artifact = _artifact(train_dataset_config)
    validation_config_artifact = _artifact(validation_dataset_config)
    if not (
        Path(str(preflight.get("model_config", ""))).resolve()
        == Path(model_artifact["path"])
        and preflight.get("model_config_sha256") == model_artifact["sha256"]
        and Path(str(preflight.get("canonical_p10_checkpoint", ""))).resolve()
        == Path(p10_artifact["path"])
        and preflight.get("canonical_p10_checkpoint_sha256")
        == p10_artifact["sha256"]
        and preflight.get("dataset_configs", {}).get("train")
        == train_config_artifact
        and preflight.get("dataset_configs", {}).get("validation")
        == validation_config_artifact
    ):
        raise RuntimeError("Editing-DiT preflight artifact binding changed")

    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": "FROZEN",
        "run_dir": str(run_dir),
        "latest_route": {
            "editing_ar_inputs": ["source_foa_latent", "raw_edit_request"],
            "editing_ar_target": "complete_new_sceneplan",
            "old_sceneplan_input": False,
            "editing_dit_frame_input": [
                "noisy_target_64",
                "complete_new_sceneplan_control_256",
                "clean_source_foa_latent_64",
            ],
            "editing_dit_frame_channels": 384,
        },
        "preflight": _artifact(preflight_path),
        "shared_contract": shared_contract,
        "model_config": model_artifact,
        "dataset_configs": {
            "train": train_config_artifact,
            "validation": validation_config_artifact,
        },
        "indices": {"train": train_index, "validation": validation_index},
        "source_latent_integrity": {
            "policy": "hash_every_distinct_external_source_shard_once_per_gate_v1",
            "train": train_source_audit,
            "validation": validation_source_audit,
        },
        "target_latent_integrity": {
            "policy": "hash_every_distinct_external_target_shard_once_per_gate_v1",
            "train": train_target_audit,
            "validation": validation_target_audit,
        },
        "p10_checkpoint": p10_artifact,
        "training": {
            "max_steps": max_steps,
            "checkpoint_every": checkpoint_every,
            "candidate_steps": expected_steps,
            "checkpoint_retention": "all_six_5k_candidates_plus_last",
            "short_batch_size_per_gpu": batch_size,
            "long_batch_size_per_gpu": long_batch_size,
            "num_workers_per_rank": num_workers,
            "world_size": WORLD_SIZE,
            "physical_gpus": PHYSICAL_GPUS,
            "cuda_visible_devices": VISIBLE_GPUS,
            "cuda_device_order": "PCI_BUS_ID",
            "training_seed": training_seed,
            "accumulate_grad_batches": 1,
            "strategy": "ddp_static",
            "gradient_clip_val": 1.0,
            "save_top_k": -1,
            "save_on_exception": True,
            "validation_every": 1_000,
            "limit_validation_batches": 64,
            "logger": "none",
            "durable_validation_marker": "SAT_EDITING_VALIDATION",
        },
        "source_sha256": _source_sha256(),
    }


def validate_contract(
    contract_path: Path,
    *,
    expected_run_dir: Path | None = None,
    verify_latent_shards: bool = True,
) -> tuple[dict[str, Any], str]:
    path = contract_path.expanduser().resolve(strict=True)
    value = json.loads(path.read_text(encoding="utf-8"))
    run_dir = Path(str(value.get("run_dir", ""))).expanduser().resolve()
    if not (
        value.get("schema") == SCHEMA
        and int(value.get("schema_version", -1)) == SCHEMA_VERSION
        and value.get("status") == "FROZEN"
        and path == run_dir / CONTRACT_NAME
        and (expected_run_dir is None or run_dir == expected_run_dir.resolve())
        and value.get("latest_route")
        == {
            "editing_ar_inputs": ["source_foa_latent", "raw_edit_request"],
            "editing_ar_target": "complete_new_sceneplan",
            "old_sceneplan_input": False,
            "editing_dit_frame_input": [
                "noisy_target_64",
                "complete_new_sceneplan_control_256",
                "clean_source_foa_latent_64",
            ],
            "editing_dit_frame_channels": 384,
        }
    ):
        raise RuntimeError("Editing-DiT training contract identity changed")
    training = dict(value.get("training") or {})
    if training != {
        "max_steps": 30_000,
        "checkpoint_every": 5_000,
        "candidate_steps": [5_000, 10_000, 15_000, 20_000, 25_000, 30_000],
        "checkpoint_retention": "all_six_5k_candidates_plus_last",
        "short_batch_size_per_gpu": 72,
        "long_batch_size_per_gpu": 48,
        "num_workers_per_rank": 12,
        "world_size": WORLD_SIZE,
        "physical_gpus": PHYSICAL_GPUS,
        "cuda_visible_devices": VISIBLE_GPUS,
        "cuda_device_order": "PCI_BUS_ID",
        "training_seed": 42,
        "accumulate_grad_batches": 1,
        "strategy": "ddp_static",
        "gradient_clip_val": 1.0,
        "save_top_k": -1,
        "save_on_exception": True,
        "validation_every": 1_000,
        "limit_validation_batches": 64,
        "logger": "none",
        "durable_validation_marker": "SAT_EDITING_VALIDATION",
    }:
        raise RuntimeError("Editing-DiT training schedule contract changed")

    shared_contract = dict(value.get("shared_contract") or {})
    if not (
        Path(shared_contract.get("path", "")).resolve()
        == CURRENT_SHARED_CONTRACT.resolve(strict=True)
        and shared_contract.get("sha256")
        == sha256_file(CURRENT_SHARED_CONTRACT.resolve(strict=True))
        and shared_contract.get("index_build_sha256")
        == INDEX_BUILD_SHARED_CONTRACT_SHA256
        and shared_contract.get("index_build_role")
        == "historical_pair_construction_provenance"
        and shared_contract.get("current_role") == "active_execution_contract"
    ):
        raise RuntimeError("Editing-DiT shared-contract lineage changed")

    artifact_groups = [
        value.get("preflight"),
        value.get("shared_contract"),
        value.get("model_config"),
        value.get("p10_checkpoint"),
        *(dict(value.get("dataset_configs") or {}).values()),
    ]
    for artifact in artifact_groups:
        if not isinstance(artifact, Mapping):
            raise RuntimeError("Editing-DiT contract artifact record is missing")
        artifact_path = Path(str(artifact.get("path", ""))).resolve(strict=True)
        if artifact.get("sha256") != sha256_file(artifact_path):
            raise RuntimeError(f"Editing-DiT contract artifact changed: {artifact_path}")
    if value.get("source_sha256") != _source_sha256():
        raise RuntimeError("Editing-DiT training implementation changed")

    preflight = json.loads(
        Path(str(value["preflight"]["path"])).read_text(encoding="utf-8")
    )
    if not (
        preflight.get("schema")
        == "sceneplan_transfusion_editing_full_training_preflight"
        and int(preflight.get("schema_version", -1)) == 2
        and preflight.get("status") == "PASS"
        and preflight.get("split_disjointness", {}).get("status") == "PASS"
        and preflight.get("split_disjointness", {}).get("checks")
        == {
            "train_vs_test": True,
            "train_vs_validation": True,
            "validation_vs_test": True,
        }
        and preflight.get("indices", {}).get("train") == value.get("indices", {}).get("train")
        and preflight.get("indices", {}).get("validation")
        == value.get("indices", {}).get("validation")
        and preflight.get("dataset_configs", {}).get("train")
        == value.get("dataset_configs", {}).get("train")
        and preflight.get("dataset_configs", {}).get("validation")
        == value.get("dataset_configs", {}).get("validation")
        and preflight.get("model_config") == value["model_config"]["path"]
        and preflight.get("model_config_sha256") == value["model_config"]["sha256"]
        and preflight.get("canonical_p10_checkpoint")
        == value["p10_checkpoint"]["path"]
        and preflight.get("canonical_p10_checkpoint_sha256")
        == value["p10_checkpoint"]["sha256"]
        and preflight.get("shared_contract") == value.get("shared_contract")
    ):
        raise RuntimeError("Editing-DiT contract no longer matches its preflight")
    for split in ("train", "validation"):
        index = dict(value.get("indices", {}).get(split) or {})
        expected_rows = 1_000_000 if split == "train" else 20_000
        stored_source_audit = _latent_audit(
            index,
            role="source",
            split=split,
            expected_rows=expected_rows,
        )
        stored_target_audit = _latent_audit(
            index,
            role="target",
            split=split,
            expected_rows=expected_rows,
        )
        index_path = Path(str(index.get("path", ""))).resolve(strict=True)
        marker_path = Path(str(index.get("marker_path", ""))).resolve(strict=True)
        if not (
            index.get("sha256") == sha256_file(index_path)
            and index.get("marker_sha256") == sha256_file(marker_path)
        ):
            raise RuntimeError(f"Editing-DiT frozen {split} index changed")
        integrity = dict(value.get("source_latent_integrity") or {})
        if not (
            integrity.get("policy")
            == "hash_every_distinct_external_source_shard_once_per_gate_v1"
            and integrity.get(split) == stored_source_audit
        ):
            raise RuntimeError(
                f"Editing-DiT {split} source-latent integrity binding changed"
            )
        target_integrity = dict(value.get("target_latent_integrity") or {})
        if not (
            target_integrity.get("policy")
            == "hash_every_distinct_external_target_shard_once_per_gate_v1"
            and target_integrity.get(split) == stored_target_audit
        ):
            raise RuntimeError(
                f"Editing-DiT {split} target-latent integrity binding changed"
            )
        if verify_latent_shards:
            observed_source_audit = verify_editing_source_latent_shards(
                index_path
            )
            if observed_source_audit != stored_source_audit:
                raise RuntimeError(
                    f"Editing-DiT {split} source-latent shards changed"
                )
            observed_target_audit = verify_editing_target_latent_shards(
                index_path
            )
            if observed_target_audit != stored_target_audit:
                raise RuntimeError(
                    f"Editing-DiT {split} target-latent shards changed"
                )
    return value, sha256_file(path)


def validate_checkpoint_lineage(
    checkpoint_path: Path,
    *,
    contract_path: Path,
    expected_step: int | None = None,
    validated_contract: Mapping[str, Any] | None = None,
    validated_contract_sha256: str | None = None,
) -> tuple[int, dict[str, Any]]:
    import torch

    if validated_contract is None or validated_contract_sha256 is None:
        contract, contract_sha = validate_contract(contract_path)
    else:
        contract = dict(validated_contract)
        contract_sha = str(validated_contract_sha256)
    path = checkpoint_path.expanduser().resolve(strict=True)
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(payload, dict):
        raise RuntimeError(f"invalid Editing-DiT checkpoint payload: {path}")
    step = int(payload.get("global_step", -1))
    if not (
        0 < step <= int(contract["training"]["max_steps"])
        and (expected_step is None or step == int(expected_step))
        and payload.get("editing_run_contract") == contract
        and payload.get("editing_run_contract_sha256") == contract_sha
        and Path(str(payload.get("editing_run_contract_path", ""))).resolve()
        == contract_path.resolve()
        and payload.get("model_config") is not None
    ):
        raise RuntimeError(f"Editing-DiT checkpoint lineage changed: {path}")
    return step, payload


def _quarantine_last(last: Path, *, reason: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = last.parent / f"recovery-quarantine-{stamp}-{os.getpid()}"
    destination.mkdir(parents=False, exist_ok=False)
    moved = destination / last.name
    shutil.move(str(last), str(moved))
    (destination / "RECOVERY.json").write_text(
        json.dumps(
            {"reason": reason, "source": str(last), "preserved_as": str(moved)},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"[editing-dit-lineage] preserved unusable last.ckpt at {moved}",
        file=sys.stderr,
    )
    return moved


def resolve_resume(contract_path: Path) -> tuple[str, str | None]:
    contract, contract_sha = validate_contract(contract_path)
    run_dir = Path(contract["run_dir"])
    checkpoint_dir = run_dir / "checkpoints"
    if not checkpoint_dir.exists():
        return "FRESH", None
    named: dict[int, Path] = {}
    for path in checkpoint_dir.glob("*.ckpt"):
        if path.name == "last.ckpt":
            continue
        match = CHECKPOINT_PATTERN.fullmatch(path.name)
        if match is None:
            raise RuntimeError(f"non-canonical Editing-DiT checkpoint: {path}")
        step = int(match.group(2))
        if step in named:
            raise RuntimeError(f"ambiguous Editing-DiT checkpoint step {step}")
        observed, _ = validate_checkpoint_lineage(
            path,
            contract_path=contract_path,
            expected_step=step,
            validated_contract=contract,
            validated_contract_sha256=contract_sha,
        )
        named[observed] = path.resolve(strict=True)

    last = checkpoint_dir / "last.ckpt"
    last_step: int | None = None
    if last.exists():
        try:
            last_step, _ = validate_checkpoint_lineage(
                last,
                contract_path=contract_path,
                validated_contract=contract,
                validated_contract_sha256=contract_sha,
            )
        except Exception as exc:
            if not named:
                raise RuntimeError(
                    "last.ckpt is invalid and no verified named checkpoint exists"
                ) from exc
            _quarantine_last(last, reason=f"{type(exc).__name__}: {exc}")
            last_step = None

    expected = list(contract["training"]["candidate_steps"])
    max_observed = max([*named, *([] if last_step is None else [last_step])], default=0)
    missing_completed_intervals = [
        step for step in expected if step <= max_observed and step not in named
    ]
    if missing_completed_intervals:
        # A terminal last.ckpt can be published before its named peer.  Replaying
        # exactly 25K->30K is safe and reconstructs the complete candidate set.
        if (
            missing_completed_intervals == [30_000]
            and last_step == 30_000
            and 25_000 in named
        ):
            _quarantine_last(last, reason="terminal named 30K checkpoint missing")
            return "RESUME", str(named[25_000])
        raise RuntimeError(
            "Editing-DiT checkpoint history has missing completed intervals: "
            f"{missing_completed_intervals}"
        )
    if all(step in named for step in expected):
        return "COMPLETE", None

    if last_step is not None and last_step >= max(named, default=0):
        return "RESUME", str(last.resolve(strict=True))
    if named:
        return "RESUME", str(named[max(named)])
    return "FRESH", None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--run-dir", type=Path, required=True)
    prepare.add_argument("--preflight", type=Path, required=True)
    prepare.add_argument("--model-config", type=Path, required=True)
    prepare.add_argument("--train-dataset-config", type=Path, required=True)
    prepare.add_argument("--validation-dataset-config", type=Path, required=True)
    prepare.add_argument("--p10-checkpoint", type=Path, required=True)
    prepare.add_argument("--max-steps", type=int, required=True)
    prepare.add_argument("--checkpoint-every", type=int, required=True)
    prepare.add_argument("--batch-size", type=int, required=True)
    prepare.add_argument("--long-batch-size", type=int, default=48)
    prepare.add_argument("--num-workers", type=int, required=True)
    prepare.add_argument("--training-seed", type=int, required=True)

    validate = subparsers.add_parser("validate")
    validate.add_argument("--contract", type=Path, required=True)
    validate.add_argument("--run-dir", type=Path)

    resolve = subparsers.add_parser("resolve-resume")
    resolve.add_argument("--contract", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.command == "prepare":
        run_dir = args.run_dir.expanduser().resolve()
        value = build_contract(
            run_dir=run_dir,
            preflight_path=args.preflight,
            model_config=args.model_config,
            train_dataset_config=args.train_dataset_config,
            validation_dataset_config=args.validation_dataset_config,
            p10_checkpoint=args.p10_checkpoint,
            max_steps=int(args.max_steps),
            checkpoint_every=int(args.checkpoint_every),
            batch_size=int(args.batch_size),
            long_batch_size=int(args.long_batch_size),
            num_workers=int(args.num_workers),
            training_seed=int(args.training_seed),
        )
        output = run_dir / CONTRACT_NAME
        if output.exists():
            existing = json.loads(output.read_text(encoding="utf-8"))
            if existing != value:
                raise RuntimeError("existing Editing-DiT training contract changed")
        else:
            _atomic_json(output, value)
        # build_contract consumed a freshly emitted preflight whose source
        # and target shards were exhaustively hashed.  The immediately following
        # resolve-resume process performs the independent live recheck; avoid
        # rereading the same external shards twice inside this prepare call.
        validate_contract(
            output, expected_run_dir=run_dir, verify_latent_shards=False
        )
        print(str(output.resolve()))
        return 0
    if args.command == "validate":
        _, digest = validate_contract(
            args.contract,
            expected_run_dir=args.run_dir,
        )
        print(digest)
        return 0
    action, checkpoint = resolve_resume(args.contract)
    print(action)
    print(checkpoint or "")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
