from __future__ import annotations

import os
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import pytest
import torch

from stable_audio_tools.models.sceneplan_transfusion_generation_ar import (
    GENERATION_AR_CONTRACT,
)
from stable_audio_tools.data.model_sceneplan_codec_v4 import ModelScenePlanCodecV4
from stable_audio_tools.data.sceneplan_transfusion_generation_ar_contract import (
    validate_parent_lineage_artifacts,
)


_TRAINER_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts/t2a/train/train_sceneplan_transfusion_generation_ar.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "sceneplan_transfusion_generation_ar_trainer", _TRAINER_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
_TRAINER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_TRAINER)
_QUEUE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts/t2a/train/run_sceneplan_transfusion_generation_ar_10ep_continuation.py"
)
_QUEUE_SPEC = importlib.util.spec_from_file_location(
    "sceneplan_transfusion_generation_ar_continuation_queue", _QUEUE_PATH
)
assert _QUEUE_SPEC is not None and _QUEUE_SPEC.loader is not None
_QUEUE = importlib.util.module_from_spec(_QUEUE_SPEC)
_QUEUE_SPEC.loader.exec_module(_QUEUE)
_CODEC_PATH = Path(
    os.environ.get("AMBIT_DATA_ROOT", "data") + "/sceneplan_v2_1p124m/p11_single_turn_15s_v2/"
    "model_sceneplan_codec_v4"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_stage_local_cosine_continues_without_lr_jump() -> None:
    assert _TRAINER._cosine_floor_multiplier(
        0, schedule_steps=50, warmup_steps=0
    ) == pytest.approx(1.0)
    assert _TRAINER._cosine_floor_multiplier(
        25, schedule_steps=50, warmup_steps=0
    ) == pytest.approx(0.55)
    assert _TRAINER._cosine_floor_multiplier(
        50, schedule_steps=50, warmup_steps=0
    ) == pytest.approx(0.1)
    assert _TRAINER._cosine_floor_multiplier(
        0, schedule_steps=50, warmup_steps=5
    ) == pytest.approx(0.2)


def test_parameter_contract_allows_only_adapter_updates(monkeypatch) -> None:
    class FrozenExternalPrompt(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.auxiliary = torch.nn.Linear(3, 4).requires_grad_(False)
            self.__dict__["model"] = torch.nn.Linear(5, 6).requires_grad_(False)

    model = torch.nn.Module()
    model.p10_dit = torch.nn.Linear(2, 3).requires_grad_(False)
    model.prompt_conditioner = FrozenExternalPrompt()
    model.ar_adapter = torch.nn.Linear(4, 5)
    monkeypatch.setattr(
        _TRAINER, "EXPECTED_QWEN_TEXT_BACKBONE_PARAMETER_COUNT", 36
    )
    contract = _TRAINER._parameter_contract(model)
    assert contract == {
        "parameter_freeze_contract": "only_discrete_ar_adapter_trainable_v1",
        "trainable_parameter_names": ["ar_adapter.weight", "ar_adapter.bias"],
        "trainable_parameter_count": 25,
        "frozen_p10_parameter_count": 9,
        "frozen_qwen_parameter_count": 36,
        "frozen_qwen_registered_auxiliary_parameter_count": 16,
        "qwen_backbone_storage_contract": (
            "external_frozen_text_backbone_excluded_from_generation_checkpoint_v1"
        ),
    }

    model.p10_dit.weight.requires_grad_(True)
    with pytest.raises(RuntimeError, match="parameter-freeze contract changed"):
        _TRAINER._parameter_contract(model)

    model.p10_dit.weight.requires_grad_(False)
    model.prompt_conditioner.model.weight.requires_grad_(True)
    with pytest.raises(RuntimeError, match="parameter-freeze contract changed"):
        _TRAINER._parameter_contract(model)


def test_continuation_uses_observed_safe_high_utilization_settings() -> None:
    assert _QUEUE.SNAPSHOT_ROOT == _QUEUE_PATH.resolve().parents[3]
    assert _QUEUE.TRAIN_BATCH_SIZE_PER_RANK == 64
    assert _QUEUE.TRAIN_NUM_WORKERS_PER_RANK == 8
    assert _QUEUE.ONLINE_VALIDATION_EVERY_STEPS == 4167


def test_continuation_saves_only_new_half_epochs() -> None:
    assert _TRAINER._stage_checkpoint_steps(
        stage_start_step=50,
        requested_steps=100,
        save_every_steps=5,
        policy="interval",
    ) == tuple(range(55, 101, 5))
    assert _TRAINER._normalise_resume_position(
        epoch=4, batch_in_epoch=10, batches_per_epoch=10
    ) == (5, 0)


def test_queue_tree_hash_and_metric_rewind_are_deterministic(tmp_path: Path) -> None:
    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "a.txt").write_text("a\n", encoding="utf-8")
    (tree / "z.txt").write_text("z\n", encoding="utf-8")
    (tree / "SOURCE_SNAPSHOT_MANIFEST.json").write_text("ignored", encoding="utf-8")
    expected = hashlib.sha256()
    for name in ("a.txt", "z.txt"):
        path = tree / name
        expected.update(f"{_sha256(path)}  ./{name}\n".encode("utf-8"))
    assert _QUEUE._snapshot_tree_sha256(tree) == expected.hexdigest()

    run = tmp_path / "run"
    run.mkdir()
    metrics = run / "metrics.jsonl"
    metrics.write_text(
        "".join(
            json.dumps({"event": "train", "step": step}) + "\n"
            for step in (41680, 45837, 46000)
        ),
        encoding="utf-8",
    )
    original_run_dir = _QUEUE.RUN_DIR
    try:
        _QUEUE.RUN_DIR = run
        _QUEUE._rewind_metrics_to_checkpoint(45837)
    finally:
        _QUEUE.RUN_DIR = original_run_dir
    retained = [json.loads(line) for line in metrics.read_text().splitlines()]
    assert [row["step"] for row in retained] == [41680, 45837]
    assert len(list((run / "queue/attempt_history").glob("*.jsonl"))) == 1


def test_constrained_decoder_exposes_all_four_source_counts() -> None:
    codec = ModelScenePlanCodecV4(_CODEC_PATH)
    prefix = [codec.bos_id]
    for token in (
        "<duration_frames>",
        "<frame_001>",
        "<room>",
        "<room_dry>",
        "<num_sources>",
    ):
        token_id = codec.token_to_id[token]
        assert token_id in codec.allowed_next_ids(prefix)
        prefix.append(token_id)
    assert codec.allowed_next_ids(prefix) == {
        codec.token_to_id[f"<num_sources_{count}>"] for count in range(1, 5)
    }


def test_extension_parent_is_selected_completed_terminal_checkpoint(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "parent"
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    checkpoint = checkpoint_dir / "step_00000050.pt"
    checkpoint.write_bytes(b"parent checkpoint")
    contract = {
        "contract": GENERATION_AR_CONTRACT,
        "mode": "full",
        "seed": 42,
        "world_size": 3,
        "batch_size_per_rank": 64,
        "gradient_accumulation": 1,
        "epochs_requested": 5,
        "requested_steps": 50,
        "steps_per_epoch": 10,
        "train_manifest_sha256": "train-sha",
        "validation_manifest_sha256": "validation-sha",
        "codec_fingerprint": "codec",
        "p10_load": {"checkpoint": "frozen"},
        "training_row_coverage": {
            "unique_rows_per_epoch": 1_600_000,
            "dropped_rows_per_epoch": 0,
            "duplicated_rows_per_epoch": 0,
            "drop_last": False,
        },
    }
    final = {"event": "complete", "mode": "full", "step": 50}
    contract_path = run_dir / "RUN_CONTRACT.json"
    final_path = run_dir / "FINAL.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    final_path.write_text(json.dumps(final), encoding="utf-8")
    selection = {
        "schema": "stable_audio_tools.sceneplan_transfusion_generation_ar_checkpoint_selection",
        "status": "COMPLETE",
        "training_run_dir": str(run_dir.resolve()),
        "training_run_contract_sha256": _sha256(contract_path),
        "training_final_sha256": _sha256(final_path),
        "selected_checkpoint": str(checkpoint.resolve()),
        "selected_checkpoint_sha256": _sha256(checkpoint),
        "selected_checkpoint_step": 50,
    }
    selection_path = run_dir / "CHECKPOINT_SELECTION.json"
    selection_path.write_text(json.dumps(selection), encoding="utf-8")
    state = {
        "contract": GENERATION_AR_CONTRACT,
        "run_contract": contract,
        "global_step": 50,
        "epoch": 4,
        "batch_in_epoch": 10,
        "optimizer": {"param_groups": [{"lr": 3e-5}]},
    }

    lineage = _TRAINER._validate_extension_parent(
        checkpoint=checkpoint,
        state=state,
        selection_manifest=selection_path,
        target_epochs=10,
        steps_per_epoch=10,
        batches_per_epoch=10,
        train_manifest_sha256="train-sha",
        validation_manifest_sha256="validation-sha",
        codec_fingerprint="codec",
        p10_load={"checkpoint": "frozen"},
        world_size=3,
        batch_size=64,
        gradient_accumulation=1,
    )
    assert (
        validate_parent_lineage_artifacts(
            {"stage_start_step": 50, "parent_lineage": lineage}
        )
        == checkpoint.resolve()
    )
    assert lineage["checkpoint_step"] == 50
    assert lineage["completed_epochs"] == 5
    assert math.isclose(lineage["terminal_learning_rate"], 3e-5)

    selection["selected_checkpoint_sha256"] = "0" * 64
    selection_path.write_text(json.dumps(selection), encoding="utf-8")
    with pytest.raises(RuntimeError, match="parent selection mismatch"):
        _TRAINER._validate_extension_parent(
            checkpoint=checkpoint,
            state=state,
            selection_manifest=selection_path,
            target_epochs=10,
            steps_per_epoch=10,
            batches_per_epoch=10,
            train_manifest_sha256="train-sha",
            validation_manifest_sha256="validation-sha",
            codec_fingerprint="codec",
            p10_load={"checkpoint": "frozen"},
            world_size=3,
            batch_size=64,
            gradient_accumulation=1,
        )
