from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sqlite3
import zlib

import numpy as np
import pytest

_EVALUATOR_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts/t2a/test/evaluate_sceneplan_transfusion_generation_ar_8k.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "sceneplan_transfusion_generation_ar_8k_evaluator", _EVALUATOR_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
_EVALUATOR = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_EVALUATOR)
_canonical_json_bytes = _EVALUATOR._canonical_json_bytes
_expected_full_checkpoint_steps = _EVALUATOR._expected_full_checkpoint_steps
_expected_selection_candidate_steps = _EVALUATOR._expected_selection_candidate_steps
_open_prediction_shard = _EVALUATOR._open_prediction_shard
_validate_training_checkpoint = _EVALUATOR._validate_training_checkpoint
_write_prediction = _EVALUATOR._write_prediction
from stable_audio_tools.models.sceneplan_transfusion_generation_ar import (
    GENERATION_AR_CONTRACT,
)


def _plan(sample_id: str = "row_0") -> dict:
    return {
        "sample_id": sample_id,
        "duration_sec": 2.0,
        "room": {"type": "dry"},
        "sources": [
            {
                "source_id": "source_0",
                "kind": "sound",
                "description": "a bell",
                "activity": {"onset_sec": 0.0, "offset_sec": 2.0},
                "trajectory": {
                    "type": "static",
                    "position": {
                        "azimuth_deg": 0.0,
                        "elevation_deg": 0.0,
                        "distance_m": 1.0,
                    },
                },
                "gain_db": 0.0,
            }
        ],
    }


class _Codec:
    bos_id = 1
    eos_id = 2

    @staticmethod
    def allowed_next_ids(prefix):
        return {(1,): {3}, (1, 3): {2}}.get(tuple(prefix), set())

    @staticmethod
    def decode(tokens, *, sample_id):
        if list(tokens) != [1, 3, 2]:
            raise ValueError("bad model sequence")
        return _plan(sample_id)


class _Report:
    @staticmethod
    def as_dict():
        return {"p10": "frozen"}


def _row() -> dict:
    canonical = _canonical_json_bytes(_plan())
    return {
        "ordinal": 0,
        "sample_id": "row_0",
        "template_id": "test/0",
        "source_count": 1,
        "target_token_count": 3,
        "target_token_ids_u16le": np.asarray([1, 3, 2], dtype="<u2").tobytes(),
        "target_sceneplan_zlib": zlib.compress(canonical),
        "target_sceneplan_sha256": hashlib.sha256(canonical).hexdigest(),
    }


def test_prediction_writer_keeps_parse_failure_token_metrics(tmp_path: Path) -> None:
    shard = _open_prediction_shard(
        tmp_path / "rank.sqlite", rank=0, world_size=3, run_contract_sha256="abc"
    )
    _write_prediction(shard, _row(), [1, 2], None, 0.1, _Codec())
    shard.commit()
    status, metrics_json = shard.execute(
        "SELECT status,metrics_json FROM predictions"
    ).fetchone()
    shard.close()
    assert status == "parse_error"
    metrics = json.loads(metrics_json)
    assert metrics["parse_rate"] == 0.0
    assert metrics["token_sequence_exact"] == 0.0
    assert "grammar_legal" in metrics

    resumed = _open_prediction_shard(
        tmp_path / "rank.sqlite", rank=0, world_size=3, run_contract_sha256="abc"
    )
    assert resumed.execute("SELECT COUNT(*) FROM predictions").fetchone()[0] == 0
    assert (
        resumed.execute(
            "SELECT status FROM prediction_attempt_history ORDER BY attempt_id"
        ).fetchone()[0]
        == "parse_error"
    )
    resumed.close()


def test_prediction_writer_fails_closed_on_target_corruption(tmp_path: Path) -> None:
    shard = _open_prediction_shard(
        tmp_path / "rank.sqlite", rank=0, world_size=3, run_contract_sha256="abc"
    )
    row = _row()
    row["target_sceneplan_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="target ScenePlan SHA256 mismatch"):
        _write_prediction(shard, row, [1, 3, 2], None, 0.1, _Codec())
    assert shard.execute("SELECT COUNT(*) FROM predictions").fetchone()[0] == 0
    shard.close()


def test_checkpoint_gate_requires_completed_run_and_selection_for_intermediate(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    checkpoint = checkpoint_dir / "step_00000010.pt"
    checkpoint.write_bytes(b"checkpoint")
    contract = {
        "schema_version": 2,
        "contract": GENERATION_AR_CONTRACT,
        "mode": "full",
        "seed": 42,
        "world_size": 3,
        "cuda_visible_devices": "0,1,2",
        "codec_fingerprint": "codec",
        "p10_load": _Report.as_dict(),
        "requested_steps": 10,
        "checkpoint_policy": "half_epoch_and_end",
        "checkpoint_steps": [5, 10],
        "source_sha256": {},
        "train_manifest": {"metadata": {"rows": "1600000"}},
        "train_manifest_sha256": _EVALUATOR.EXPECTED_MANIFEST_SHA256["train"],
        "validation_manifest": {"metadata": {"rows": "32000"}},
        "validation_manifest_sha256": _EVALUATOR.EXPECTED_MANIFEST_SHA256["validation"],
        "training_row_coverage": {
            "unique_rows_per_epoch": 1_600_000,
            "dropped_rows_per_epoch": 0,
            "duplicated_rows_per_epoch": 0,
            "drop_last": False,
        },
    }
    (run_dir / "RUN_CONTRACT.json").write_text(json.dumps(contract), encoding="utf-8")
    (run_dir / "FINAL.json").write_text(
        json.dumps({"event": "complete", "mode": "full", "step": 10}),
        encoding="utf-8",
    )
    codec = type("Codec", (), {"fingerprint": "codec"})()
    proof = _validate_training_checkpoint(
        checkpoint=checkpoint,
        checkpoint_sha256=hashlib.sha256(b"checkpoint").hexdigest(),
        state={
            "contract": GENERATION_AR_CONTRACT,
            "run_contract": contract,
            "global_step": 10,
        },
        codec=codec,
        p10_report=_Report(),
        selection_manifest=None,
    )
    assert proof["training_final_step"] == 10

    with pytest.raises(RuntimeError, match="selection-manifest"):
        _validate_training_checkpoint(
            checkpoint=checkpoint,
            checkpoint_sha256=hashlib.sha256(b"checkpoint").hexdigest(),
            state={
                "contract": GENERATION_AR_CONTRACT,
                "run_contract": contract,
                "global_step": 5,
            },
            codec=codec,
            p10_report=_Report(),
            selection_manifest=None,
        )


def test_checkpoint_gate_accepts_only_validated_multiepoch_winner(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    steps = list(range(5, 51, 5))
    checkpoints = {}
    for step in steps:
        checkpoint = checkpoint_dir / f"step_{step:08d}.pt"
        checkpoint.write_bytes(f"checkpoint-{step}".encode("ascii"))
        checkpoints[step] = checkpoint
    selected_checkpoint = checkpoints[25]
    validation = tmp_path / "validation.sqlite"
    validation.write_bytes(b"immutable validation fixture")
    contract = {
        "schema_version": 2,
        "contract": GENERATION_AR_CONTRACT,
        "mode": "full",
        "seed": 42,
        "world_size": 3,
        "cuda_visible_devices": "0,1,2",
        "codec_fingerprint": "codec",
        "p10_load": _Report.as_dict(),
        "epochs_requested": 5,
        "requested_steps": 50,
        "steps_per_epoch": 10,
        "checkpoint_policy": "interval",
        "save_every_steps": 5,
        "checkpoint_steps": steps,
        "source_sha256": {},
        "train_manifest": {"metadata": {"rows": "1600000"}},
        "train_manifest_sha256": _EVALUATOR.EXPECTED_MANIFEST_SHA256["train"],
        "validation_manifest": {"metadata": {"rows": "32000"}},
        "validation_manifest_sha256": _EVALUATOR.EXPECTED_MANIFEST_SHA256["validation"],
        "training_row_coverage": {
            "unique_rows_per_epoch": 1_600_000,
            "dropped_rows_per_epoch": 0,
            "duplicated_rows_per_epoch": 0,
            "drop_last": False,
        },
    }
    contract_path = run_dir / "RUN_CONTRACT.json"
    final_path = run_dir / "FINAL.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    final_path.write_text(
        json.dumps({"event": "complete", "mode": "full", "step": 50}),
        encoding="utf-8",
    )
    candidates = []
    for step, checkpoint in checkpoints.items():
        candidates.append(
            {
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": hashlib.sha256(
                    checkpoint.read_bytes()
                ).hexdigest(),
                "step": step,
                "validation": {
                    "loss": 0.1 + abs(step - 25) / 100.0,
                    "token_accuracy": 0.8,
                    "sequences": 32000,
                },
            }
        )
    selected_metrics = next(
        row["validation"] for row in candidates if row["step"] == 25
    )
    selection = {
        "schema": "stable_audio_tools.sceneplan_transfusion_generation_ar_checkpoint_selection",
        "status": "COMPLETE",
        "selection_contract": "full_32k_validation_minimum_token_ce_v1",
        "candidate_policy": "every_half_epoch_full_32k_validation",
        "training_run_dir": str(run_dir),
        "training_final_step": 50,
        "training_final_sha256": hashlib.sha256(final_path.read_bytes()).hexdigest(),
        "training_run_contract_sha256": hashlib.sha256(
            contract_path.read_bytes()
        ).hexdigest(),
        "validation_manifest": {"path": str(validation)},
        "validation_manifest_sha256": hashlib.sha256(
            validation.read_bytes()
        ).hexdigest(),
        "candidates": candidates,
        "selected_checkpoint": str(selected_checkpoint),
        "selected_checkpoint_sha256": hashlib.sha256(
            selected_checkpoint.read_bytes()
        ).hexdigest(),
        "selected_checkpoint_step": 25,
        "selected_validation": selected_metrics,
        "source_sha256": {},
    }
    selection_path = run_dir / "CHECKPOINT_SELECTION.json"
    selection_path.write_text(json.dumps(selection), encoding="utf-8")
    codec = type("Codec", (), {"fingerprint": "codec"})()
    proof = _validate_training_checkpoint(
        checkpoint=selected_checkpoint,
        checkpoint_sha256=hashlib.sha256(selected_checkpoint.read_bytes()).hexdigest(),
        state={
            "contract": GENERATION_AR_CONTRACT,
            "run_contract": contract,
            "global_step": 25,
        },
        codec=codec,
        p10_report=_Report(),
        selection_manifest=selection_path,
    )
    assert proof["selection"]["selected_checkpoint_step"] == 25


def test_evaluator_accepts_only_contracted_continuation_candidates() -> None:
    contract = {
        "schema_version": 3,
        "checkpoint_policy": "interval",
        "epochs_requested": 10,
        "requested_steps": 100,
        "steps_per_epoch": 10,
        "stage_start_step": 50,
        "save_every_steps": 5,
        "parent_lineage": {
            "contract": "selected_completed_generation_ar_parent_v1",
            "checkpoint_step": 50,
        },
    }
    assert _expected_full_checkpoint_steps(contract) == list(range(55, 101, 5))
    assert _expected_selection_candidate_steps(contract) == [
        50,
        *range(55, 101, 5),
    ]
