#!/usr/bin/env python3
"""Select the best Generation AR checkpoint on complete 32K validation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

import torch
from torch import distributed as dist


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.t2a.test.evaluate_sceneplan_transfusion_generation_ar_8k import (  # noqa: E402
    _atomic_json,
    _distributed,
    _teacher_forced_metrics,
)
from stable_audio_tools.data.sceneplan_transfusion_generation_ar_contract import (  # noqa: E402
    canonical_sha256 as _canonical_sha256,
    expected_checkpoint_steps as _expected_full_checkpoint_steps,
    expected_selection_candidate_steps as _expected_selection_candidate_steps,
    sha256_file as _sha256_file,
    validate_parent_lineage_artifacts as _validate_parent_lineage_artifacts,
)
from stable_audio_tools.data.model_sceneplan_codec_v4 import (  # noqa: E402
    ModelScenePlanCodecV4,
)
from stable_audio_tools.data.sceneplan_transfusion_generation_ar_dataset import (  # noqa: E402
    GenerationARSQLiteDataset,
    manifest_summary,
)
from stable_audio_tools.models.sceneplan_transfusion_generation_ar import (  # noqa: E402
    GENERATION_AR_CONTRACT,
    load_p10v11_generation_ar,
)


CODEC_PATH = Path(
    "/mnt/sdb/audio_dataset/sceneplan_v2_1p124m/p11_single_turn_15s_v2/"
    "model_sceneplan_codec_v4"
)
EXPECTED_TRAIN_MANIFEST_SHA256 = (
    "e1921b167a09135ae3c3bdf28ecac509f7175a387cb9c0f54b5d7a3608a1799c"
)
EXPECTED_VALIDATION_MANIFEST_SHA256 = (
    "697113f9c38f190cb7e54bf8863c3e3b78dfa77de76daeb9b1ea7c035fa6dd4c"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, action="append")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _validate_completed_run(
    *, run_dir: Path, codec, p10_report
) -> tuple[dict[str, Any], dict[str, Any], Path, Path]:
    contract_path = (run_dir / "RUN_CONTRACT.json").resolve(strict=True)
    final_path = (run_dir / "FINAL.json").resolve(strict=True)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    final = json.loads(final_path.read_text(encoding="utf-8"))
    requested_steps = int(contract.get("requested_steps", -1))
    expected_checkpoint_steps = _expected_full_checkpoint_steps(contract)
    expected_candidate_steps = _expected_selection_candidate_steps(contract)
    if (
        contract.get("contract") != GENERATION_AR_CONTRACT
        or int(contract.get("schema_version", -1)) not in (2, 3)
        or contract.get("mode") != "full"
        or int(contract.get("seed", -1)) != 42
        or int(contract.get("world_size", -1)) != 3
        or contract.get("cuda_visible_devices", "").replace(" ", "") != "0,1,2"
    ):
        raise RuntimeError("Generation AR training run contract mismatch")
    train_metadata = dict(contract.get("train_manifest", {}).get("metadata") or {})
    validation_metadata = dict(
        contract.get("validation_manifest", {}).get("metadata") or {}
    )
    coverage = dict(contract.get("training_row_coverage") or {})
    if (
        int(train_metadata.get("rows", -1)) != 1_600_000
        or int(validation_metadata.get("rows", -1)) != 32_000
        or contract.get("train_manifest_sha256") != EXPECTED_TRAIN_MANIFEST_SHA256
        or contract.get("validation_manifest_sha256")
        != EXPECTED_VALIDATION_MANIFEST_SHA256
        or list(contract.get("checkpoint_steps") or ()) != expected_checkpoint_steps
        or (
            int(contract.get("schema_version", -1)) >= 3
            and list(contract.get("selection_candidate_steps") or ())
            != expected_candidate_steps
        )
        or int(coverage.get("manifest_rows", -1)) != 1_600_000
        or int(coverage.get("unique_rows_per_epoch", -1)) != 1_600_000
        or int(coverage.get("dropped_rows_per_epoch", -1)) != 0
        or int(coverage.get("duplicated_rows_per_epoch", -1)) != 0
        or bool(coverage.get("drop_last", True))
    ):
        raise RuntimeError("Generation AR exact-data/checkpoint contract mismatch")
    if (
        requested_steps <= 0
        or final.get("event") != "complete"
        or final.get("mode") != "full"
        or int(final.get("step", -1)) != requested_steps
    ):
        raise RuntimeError("Generation AR training run is not complete")
    if contract.get("codec_fingerprint") != codec.fingerprint:
        raise RuntimeError("checkpoint selection codec mismatch")
    if contract.get("p10_load") != p10_report.as_dict():
        raise RuntimeError("checkpoint selection P10-v11 mismatch")
    _validate_parent_lineage_artifacts(contract)
    source_snapshot = dict(contract.get("source_snapshot") or {})
    snapshot_path = Path(str(source_snapshot.get("path", ""))).resolve(strict=True)
    if snapshot_path != (REPO_ROOT / "SOURCE_SNAPSHOT_MANIFEST.json").resolve(
        strict=True
    ) or _sha256_file(snapshot_path) != source_snapshot.get("sha256"):
        raise RuntimeError("checkpoint selection source snapshot mismatch")
    snapshot_manifest = json.loads(snapshot_path.read_text(encoding="utf-8"))
    if (
        snapshot_manifest.get("tree_sha256_excluding_this_manifest")
        != source_snapshot.get("tree_sha256_excluding_manifest")
        or snapshot_manifest.get("status") != "IMMUTABLE_FOR_GENERATION_AR_RUN"
    ):
        raise RuntimeError("checkpoint selection source snapshot lineage mismatch")
    train_path = Path(contract["train_manifest"]["path"]).resolve(strict=True)
    validation_path = Path(contract["validation_manifest"]["path"]).resolve(strict=True)
    if (
        _sha256_file(train_path) != EXPECTED_TRAIN_MANIFEST_SHA256
        or _sha256_file(validation_path) != EXPECTED_VALIDATION_MANIFEST_SHA256
    ):
        raise RuntimeError("training or validation manifest changed since training")
    for relative, expected in dict(contract.get("source_sha256") or {}).items():
        source = (REPO_ROOT / str(relative)).resolve(strict=True)
        if _sha256_file(source) != str(expected):
            raise RuntimeError(f"training source changed: {relative}")
    return contract, final, contract_path, final_path


def main() -> int:
    args = _parse_args()
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("batch size must be positive and workers non-negative")
    rank, _local_rank, world_size, device = _distributed()
    torch.manual_seed(42 + rank)
    run_dir = args.run_dir.expanduser().resolve(strict=True)
    codec = ModelScenePlanCodecV4(CODEC_PATH)
    model, p10_report = load_p10v11_generation_ar(
        pad_id=codec.pad_id,
        verify_sha256=rank == 0,
        activation_checkpointing=False,
    )
    contract, _final, contract_path, final_path = _validate_completed_run(
        run_dir=run_dir, codec=codec, p10_report=p10_report
    )
    requested_steps = int(contract["requested_steps"])
    default_steps = tuple(_expected_selection_candidate_steps(contract))
    parent_checkpoint = _validate_parent_lineage_artifacts(contract)
    parent_step = (
        int(contract["parent_lineage"]["checkpoint_step"])
        if parent_checkpoint is not None
        else None
    )
    default_checkpoints = [
        (
            parent_checkpoint
            if parent_checkpoint is not None and step == parent_step
            else (run_dir / "checkpoints" / f"step_{step:08d}.pt").resolve(strict=True)
        )
        for step in default_steps
    ]
    checkpoints = (
        [path.expanduser().resolve(strict=True) for path in args.checkpoint]
        if args.checkpoint
        else default_checkpoints
    )
    if len(checkpoints) < 2:
        raise ValueError("selection requires at least two checkpoint candidates")
    if set(checkpoints) != set(default_checkpoints) or len(checkpoints) != len(
        default_checkpoints
    ):
        raise RuntimeError(
            "selection must cover every contracted candidate exactly once"
        )
    expected_checkpoint_dir = (run_dir / "checkpoints").resolve(strict=True)
    if any(
        path.parent != expected_checkpoint_dir
        and (parent_checkpoint is None or path != parent_checkpoint)
        for path in checkpoints
    ):
        raise RuntimeError("selection checkpoint is outside the completed lineage")
    parent_contract = (
        json.loads(
            Path(str(contract["parent_lineage"]["run_contract"])).read_text(
                encoding="utf-8"
            )
        )
        if parent_checkpoint is not None
        else None
    )

    validation_path = Path(contract["validation_manifest"]["path"]).resolve(strict=True)
    validation_rows = int(manifest_summary(validation_path)["metadata"]["rows"])
    assigned = tuple(range(rank, validation_rows, world_size))
    dataset = GenerationARSQLiteDataset(
        validation_path, split="validation", row_ordinals=assigned
    )
    model.p10_dit.to(device=device, dtype=torch.bfloat16)
    model.prompt_conditioner.to(device=device, dtype=torch.bfloat16)
    model.ar_adapter.to(device=device, dtype=torch.float32)
    model.eval()

    candidates: list[dict[str, Any]] = []
    seen_steps: set[int] = set()
    for checkpoint in checkpoints:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        step = int(state.get("global_step", -1))
        is_parent = parent_checkpoint is not None and checkpoint == parent_checkpoint
        expected_contract = parent_contract if is_parent else contract
        if (
            state.get("contract") != GENERATION_AR_CONTRACT
            or state.get("run_contract") != expected_contract
            or step <= 0
            or step > requested_steps
            or step in seen_steps
            or step not in default_steps
            or (is_parent and step != parent_step)
        ):
            raise RuntimeError(f"invalid selection candidate: {checkpoint}")
        seen_steps.add(step)
        model.load_trainable_state_dict(state["ar_adapter"])
        sha256 = [_sha256_file(checkpoint) if rank == 0 else None]
        dist.broadcast_object_list(sha256, src=0, device=device)
        metrics = _teacher_forced_metrics(
            model,
            dataset,
            codec,
            device=device,
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
        )
        if int(metrics["sequences"]) != validation_rows:
            raise RuntimeError("validation selection did not cover all rows")
        candidate = {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": str(sha256[0]),
            "step": step,
            "lineage_role": (
                "parent"
                if is_parent
                else (
                    "continuation_stage"
                    if parent_checkpoint is not None
                    else "current_run"
                )
            ),
            "validation": metrics,
        }
        candidates.append(candidate)
        if rank == 0:
            print(
                json.dumps(
                    {"event": "checkpoint_validation_complete", **candidate},
                    sort_keys=True,
                ),
                flush=True,
            )

    selected = min(
        candidates,
        key=lambda row: (
            float(row["validation"]["loss"]),
            -float(row["validation"]["token_accuracy"]),
            -int(row["step"]),
        ),
    )
    output_path = (
        args.output.expanduser().resolve(strict=False)
        if args.output is not None
        else run_dir / "CHECKPOINT_SELECTION.json"
    )
    if rank == 0:
        selection = {
            "schema": "stable_audio_tools.sceneplan_transfusion_generation_ar_checkpoint_selection",
            "schema_version": 1,
            "status": "COMPLETE",
            "selection_contract": "full_32k_validation_minimum_token_ce_v1",
            "selection_rule": [
                "minimum_validation_loss",
                "maximum_token_accuracy_tiebreak",
                "latest_step_tiebreak",
            ],
            "training_run_dir": str(run_dir),
            "training_run_contract": str(contract_path),
            "training_run_contract_sha256": _sha256_file(contract_path),
            "training_run_contract_canonical_sha256": _canonical_sha256(contract),
            "training_final": str(final_path),
            "training_final_sha256": _sha256_file(final_path),
            "training_final_step": requested_steps,
            "validation_manifest": manifest_summary(validation_path),
            "validation_manifest_sha256": _sha256_file(validation_path),
            "validation_rows": validation_rows,
            "world_size": world_size,
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "candidate_policy": (
                "selected_parent_plus_every_stage_half_epoch_full_32k_validation"
                if parent_checkpoint is not None
                else (
                    "every_half_epoch_full_32k_validation"
                    if contract.get("checkpoint_policy") == "interval"
                    else "half_epoch_and_epoch_end_only"
                )
            ),
            "candidates": candidates,
            "selected_checkpoint": selected["checkpoint"],
            "selected_checkpoint_sha256": selected["checkpoint_sha256"],
            "selected_checkpoint_step": selected["step"],
            "selected_lineage_role": selected["lineage_role"],
            "selected_validation": selected["validation"],
            "source_sha256": {
                str(Path(__file__).resolve().relative_to(REPO_ROOT)): _sha256_file(
                    Path(__file__).resolve()
                ),
                "scripts/t2a/test/evaluate_sceneplan_transfusion_generation_ar_8k.py": _sha256_file(
                    REPO_ROOT
                    / "scripts/t2a/test/evaluate_sceneplan_transfusion_generation_ar_8k.py"
                ),
                "stable_audio_tools/data/sceneplan_transfusion_generation_ar_contract.py": _sha256_file(
                    REPO_ROOT
                    / "stable_audio_tools/data/sceneplan_transfusion_generation_ar_contract.py"
                ),
            },
        }
        _atomic_json(output_path, selection)
        print(
            json.dumps(
                {
                    "event": "checkpoint_selection_complete",
                    "output": str(output_path),
                    "selected_checkpoint": selected["checkpoint"],
                    "selected_step": selected["step"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    dist.barrier()
    dataset.close()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
