from __future__ import annotations

import json
from pathlib import Path
import random

import numpy as np
import pytest
import torch

from scripts.t2a.train import (
    sceneplan_transfusion_editing_joint_run_contract as lineage,
)
from scripts.t2a.train import (
    train_sceneplan_transfusion_editing_ar_joint_full as trainer,
)
from stable_audio_tools.data.sceneplan_transfusion_editing_index import sha256_file


def _run_contract(root: Path) -> tuple[dict, dict, Path]:
    identity = lineage.ensure_run_identity(root)
    identity_path = (root / lineage.RUN_IDENTITY_NAME).resolve(strict=True)
    contract = {
        "schema": lineage.RUN_CONTRACT_SCHEMA,
        "schema_version": lineage.RUN_CONTRACT_SCHEMA_VERSION,
        "run_dir": str(root.resolve()),
        "run_id": identity["run_id"],
        "run_identity": {
            "path": str(identity_path),
            "sha256": sha256_file(identity_path),
        },
        "editing_ar_contract": lineage.EDITING_AR_CONTRACT,
        "joint_dataset_contract": lineage.EDITING_JOINT_DATASET_CONTRACT,
        "max_steps": lineage.DEFAULT_MAX_STEPS,
        "save_every": lineage.DEFAULT_SAVE_EVERY,
        "world_size": 5,
    }
    contract_path = root / lineage.RUN_CONTRACT_NAME
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    return identity, contract, contract_path.resolve(strict=True)


def _rng_states() -> list[dict]:
    numpy_state = np.random.get_state()
    return [
        {
            "schema": lineage.RNG_SCHEMA,
            "schema_version": lineage.RNG_SCHEMA_VERSION,
            "rank": rank,
            "python_random_state": random.getstate(),
            "numpy_random_state": {
                "bit_generator": str(numpy_state[0]),
                "keys": torch.from_numpy(
                    np.asarray(numpy_state[1], dtype=np.uint32).astype(np.int64)
                ),
                "position": int(numpy_state[2]),
                "has_gauss": int(numpy_state[3]),
                "cached_gaussian": float(numpy_state[4]),
            },
            "torch_cpu_rng_state": torch.get_rng_state(),
            # The formal resolver requires each GPU rank's local CUDA state.
            # A tiny CPU tensor is sufficient to exercise serialization here.
            "torch_cuda_rng_state": torch.zeros(8, dtype=torch.uint8),
        }
        for rank in range(5)
    ]


def _write_candidate(root: Path, *, step: int) -> Path:
    identity = lineage.validate_run_identity(root)
    contract_path = (root / lineage.RUN_CONTRACT_NAME).resolve(strict=True)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    path = root / "checkpoints" / f"step-{step:08d}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": lineage.CHECKPOINT_SCHEMA,
            "schema_version": lineage.CHECKPOINT_SCHEMA_VERSION,
            "contract": lineage.EDITING_AR_CONTRACT,
            "joint_dataset_contract": lineage.EDITING_JOINT_DATASET_CONTRACT,
            "run_dir": str(root.resolve()),
            "run_id": identity["run_id"],
            "run_contract_path": str(contract_path),
            "run_contract_sha256": sha256_file(contract_path),
            "run_contract": contract,
            "global_step": step,
            "epoch": 0,
            "batch_in_epoch": step,
            "diffusion_state_dict": {},
            "editing_ar_specific_state_dict": {},
            "optimizer": {},
            "scheduler": {},
            "rng_states_by_rank": _rng_states(),
        },
        path,
    )
    return path.resolve(strict=True)


def test_joint_run_identity_is_immutable_and_absolute(tmp_path: Path) -> None:
    assert lineage.EDITING_AR_CONTRACT == trainer.EDITING_AR_CONTRACT
    assert (
        lineage.EDITING_JOINT_DATASET_CONTRACT
        == trainer.EDITING_JOINT_DATASET_CONTRACT
    )
    root = tmp_path / "run"
    root.mkdir()
    (root / ".editing-joint-full.lock").touch()
    first = lineage.ensure_run_identity(root)
    assert first == lineage.ensure_run_identity(root)
    assert Path(first["run_dir"]) == root.resolve()

    copied = tmp_path / "copied"
    copied.mkdir()
    (copied / lineage.RUN_IDENTITY_NAME).write_text(
        json.dumps(first), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="identity changed"):
        lineage.validate_run_identity(copied)


@pytest.mark.parametrize("location", ["run_contract", "checkpoint"])
@pytest.mark.parametrize(
    ("run_field", "checkpoint_field", "legacy_value"),
    [
        (
            "editing_ar_contract",
            "contract",
            "p10v11_shared_blocks_audio_reference_editing_ar_v3",
        ),
        (
            "joint_dataset_contract",
            "joint_dataset_contract",
            "source_audio_instruction_to_new_plan_plus_aligned_rf_v1",
        ),
    ],
)
def test_joint_resolver_rejects_legacy_random_slot_training(
    tmp_path: Path,
    location: str,
    run_field: str,
    checkpoint_field: str,
    legacy_value: str,
) -> None:
    _, contract, contract_path = _run_contract(tmp_path)
    checkpoint = _write_candidate(tmp_path, step=5_000)
    if location == "run_contract":
        contract[run_field] = legacy_value
        contract_path.write_text(json.dumps(contract), encoding="utf-8")
        message = "run contract identity changed"
    else:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        payload[checkpoint_field] = legacy_value
        torch.save(payload, checkpoint)
        message = "checkpoint lineage changed"
    with pytest.raises(RuntimeError, match=message):
        lineage.resolve_resume(tmp_path)
    assert not (tmp_path / "checkpoints/LATEST.json").exists()


def test_joint_resolver_repairs_missing_and_stale_latest(tmp_path: Path) -> None:
    identity, _, contract_path = _run_contract(tmp_path)
    step_5k = _write_candidate(tmp_path, step=5_000)
    assert lineage.resolve_resume(tmp_path) == ("RESUME", str(step_5k))
    latest_path = tmp_path / "checkpoints/LATEST.json"
    first_latest = json.loads(latest_path.read_text(encoding="utf-8"))
    assert first_latest["global_step"] == 5_000
    assert first_latest["run_id"] == identity["run_id"]
    assert first_latest["run_contract_sha256"] == sha256_file(contract_path)

    step_10k = _write_candidate(tmp_path, step=10_000)
    assert lineage.resolve_resume(tmp_path) == ("RESUME", str(step_10k))
    repaired = json.loads(latest_path.read_text(encoding="utf-8"))
    assert repaired["global_step"] == 10_000
    assert len(list((tmp_path / "checkpoints").glob(
        "recovery-quarantine-*/LATEST.json"
    ))) == 1


def test_joint_resolver_rejects_gaps_rollbacks_and_cross_run_payloads(
    tmp_path: Path,
) -> None:
    gap = tmp_path / "gap"
    _run_contract(gap)
    step_5k = _write_candidate(gap, step=5_000)
    _write_candidate(gap, step=15_000)
    with pytest.raises(RuntimeError, match="missing completed intervals"):
        lineage.resolve_resume(gap)

    source = tmp_path / "source"
    _run_contract(source)
    foreign_candidate = _write_candidate(source, step=5_000)
    target = tmp_path / "target"
    _run_contract(target)
    target_candidate = target / "checkpoints/step-00005000.pt"
    target_candidate.parent.mkdir(parents=True)
    target_candidate.write_bytes(foreign_candidate.read_bytes())
    with pytest.raises(RuntimeError, match="lineage changed"):
        lineage.resolve_resume(target)

    rollback = tmp_path / "rollback"
    _run_contract(rollback)
    old = _write_candidate(rollback, step=5_000)
    _write_candidate(rollback, step=10_000)
    with pytest.raises(RuntimeError, match="highest verified"):
        lineage.resolve_resume(rollback, requested_checkpoint=old)
    assert step_5k.name == "step-00005000.pt"

    pointer_ahead = tmp_path / "pointer-ahead"
    identity, _, contract_path = _run_contract(pointer_ahead)
    current = _write_candidate(pointer_ahead, step=5_000)
    latest_path = pointer_ahead / "checkpoints/LATEST.json"
    latest_path.write_text(
        json.dumps(
            lineage.latest_record(
                run_dir=pointer_ahead.resolve(),
                identity=identity,
                run_contract_path=contract_path,
                run_contract_sha256=sha256_file(contract_path),
                checkpoint=current.with_name("step-00010000.pt"),
                checkpoint_sha256="0" * 64,
                step=10_000,
            )
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="rollback is forbidden"):
        lineage.resolve_resume(pointer_ahead)
    assert latest_path.exists()
    assert not list(
        latest_path.parent.glob("recovery-quarantine-*/LATEST.json")
    )

    final_ahead = tmp_path / "final-ahead"
    _run_contract(final_ahead)
    _write_candidate(final_ahead, step=5_000)
    (final_ahead / "FINAL.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="cannot roll back"):
        lineage.resolve_resume(final_ahead)
    assert not (final_ahead / "checkpoints/LATEST.json").exists()


def test_joint_resolver_accepts_only_a_lineage_bound_complete_run(
    tmp_path: Path,
) -> None:
    identity, _, contract_path = _run_contract(tmp_path)
    candidates = [
        _write_candidate(tmp_path, step=step)
        for step in range(5_000, 25_001, 5_000)
    ]
    assert lineage.resolve_resume(tmp_path) == (
        "RESUME",
        str(candidates[-1]),
    )
    final = {
        "schema": lineage.FINAL_SCHEMA,
        "schema_version": lineage.FINAL_SCHEMA_VERSION,
        "event": "complete",
        "step": 25_000,
        "run_dir": str(tmp_path.resolve()),
        "run_id": identity["run_id"],
        "run_contract_path": str(contract_path),
        "run_contract_sha256": sha256_file(contract_path),
        "checkpoint": str(candidates[-1]),
        "checkpoint_sha256": sha256_file(candidates[-1]),
        "shared_transformer_same_object": True,
        "old_sceneplan_exposed": False,
        "source_semantic_mode": "m2d_audio_caption_aux",
        "source_caption_exposed_as_model_input": False,
    }
    (tmp_path / "FINAL.json").write_text(json.dumps(final), encoding="utf-8")
    assert lineage.resolve_resume(tmp_path) == ("COMPLETE", None)

    final["run_id"] = "0" * 32
    (tmp_path / "FINAL.json").write_text(json.dumps(final), encoding="utf-8")
    with pytest.raises(RuntimeError, match="FINAL identity changed"):
        lineage.resolve_resume(tmp_path)


def test_joint_rank_rng_round_trip_is_weights_only_safe(tmp_path: Path) -> None:
    random.seed(123)
    np.random.seed(456)
    torch.manual_seed(789)
    state = trainer._capture_rank_rng_state(rank=2, device=None)
    path = tmp_path / "rng.pt"
    torch.save(state, path)
    loaded = torch.load(path, map_location="cpu", weights_only=True)

    expected = (random.random(), np.random.random(4), torch.rand(4))
    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)
    trainer._restore_rank_rng_state(loaded, rank=2, device=None)
    observed = (random.random(), np.random.random(4), torch.rand(4))
    assert observed[0] == expected[0]
    np.testing.assert_array_equal(observed[1], expected[1])
    torch.testing.assert_close(observed[2], expected[2], rtol=0, atol=0)

    with pytest.raises(RuntimeError, match="rank 1"):
        trainer._restore_rank_rng_state(loaded, rank=1, device=None)


def test_joint_rng_gather_requires_one_ordered_record_per_rank(monkeypatch) -> None:
    states = _rng_states()

    monkeypatch.setattr(
        trainer,
        "_capture_rank_rng_state",
        lambda *, rank, device: states[rank],
    )

    def fake_all_gather(output, _local) -> None:
        output[:] = [states[index] for index in (4, 2, 0, 3, 1)]

    monkeypatch.setattr(trainer.dist, "all_gather_object", fake_all_gather)
    gathered = trainer._gather_rank_rng_states(
        rank=3, world_size=5, device=torch.device("cpu")
    )
    assert [record["rank"] for record in gathered] == [0, 1, 2, 3, 4]
