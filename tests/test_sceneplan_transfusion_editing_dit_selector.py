from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3
from unittest import mock
import zlib

import pytest
import torch

from scripts.t2a.eval import select_sceneplan_transfusion_editing_dit_checkpoint as selector
from scripts.t2a.train import sceneplan_transfusion_editing_dit_run_contract as lineage
from scripts.t2a.train import prepare_sceneplan_transfusion_editing_full as full_preflight


REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def baseline_reader_sources(tmp_path: Path, monkeypatch):
    relative = "scripts/t2a/eval/select_sceneplan_transfusion_editing_dit_checkpoint.py"
    archived = "scripts/t2a/eval/frozen_sources/editing_dit_selector_11b57706.py.txt"
    for name in (relative, archived):
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((REPO_ROOT / name).read_bytes())
    monkeypatch.setattr(selector, "REPO_ROOT", tmp_path)
    expected = {relative: _sha(tmp_path / relative), "other.py": "a" * 64}
    recorded = {**expected, relative: _sha(tmp_path / archived)}
    return recorded, expected, tmp_path / relative, tmp_path / archived


def test_baseline_reader_accepts_current_sources(baseline_reader_sources) -> None:
    _, expected, _, _ = baseline_reader_sources
    assert selector._selection_source_identity_matches(expected, expected)


def test_baseline_reader_proves_only_known_reader_repair(baseline_reader_sources) -> None:
    recorded, expected, _, _ = baseline_reader_sources
    assert recorded != expected
    assert selector._selection_source_identity_matches(recorded, expected)


@pytest.mark.parametrize("change", ["unknown_producer", "other_source", "missing_source"])
def test_baseline_reader_rejects_other_source_changes(
    baseline_reader_sources, change: str
) -> None:
    recorded, expected, current, _ = baseline_reader_sources
    relative = str(current.relative_to(selector.REPO_ROOT))
    if change == "unknown_producer":
        recorded[relative] = "b" * 64
    elif change == "other_source":
        recorded["other.py"] = "b" * 64
    else:
        del recorded["other.py"]
    assert not selector._selection_source_identity_matches(recorded, expected)


@pytest.mark.parametrize("remove", [False, True])
def test_baseline_reader_requires_exact_archived_producer(
    baseline_reader_sources, remove: bool
) -> None:
    recorded, expected, _, archive = baseline_reader_sources
    if remove:
        archive.unlink()
    else:
        archive.write_text(archive.read_text() + "\n# changed snapshot\n")
    assert not selector._selection_source_identity_matches(recorded, expected)


@pytest.mark.parametrize(
    ("before", "after"),
    [
        ("CONFIDENCE = 0.99", "CONFIDENCE = 0.95"),
        ("return mapping, digest", "return {}, digest"),
        ("and selected_gate_valid", "and True"),
    ],
)
def test_baseline_reader_does_not_authorize_gate_or_sampler_changes(
    baseline_reader_sources, before: str, after: str
) -> None:
    recorded, expected, current, _ = baseline_reader_sources
    source = current.read_text()
    assert source.count(before) == 1
    current.write_text(source.replace(before, after))
    expected[str(current.relative_to(selector.REPO_ROOT))] = _sha(current)
    assert not selector._selection_source_identity_matches(recorded, expected)


def test_baseline_reader_rejects_source_changed_after_hashing(
    baseline_reader_sources,
) -> None:
    recorded, expected, current, _ = baseline_reader_sources
    current.write_text(current.read_text() + "\n# changed after hashing\n")
    assert not selector._selection_source_identity_matches(recorded, expected)


def _tiny_lineage_contract(tmp_path: Path, monkeypatch) -> tuple[Path, dict]:
    monkeypatch.setattr(lineage, "_source_sha256", lambda: {})
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    shared_contract_path = tmp_path / "shared-contract.md"
    shared_contract_path.write_text("current Editing contract", encoding="utf-8")
    monkeypatch.setattr(lineage, "CURRENT_SHARED_CONTRACT", shared_contract_path)
    shared_contract = {
        "path": str(shared_contract_path.resolve()),
        "sha256": _sha(shared_contract_path),
        "index_build_sha256": lineage.INDEX_BUILD_SHARED_CONTRACT_SHA256,
        "index_build_role": "historical_pair_construction_provenance",
        "current_role": "active_execution_contract",
    }
    assets = {}
    for name in ("model.json", "p10.ckpt", "train.json", "validation.json"):
        path = tmp_path / name
        path.write_text(name, encoding="utf-8")
        assets[name] = {"path": str(path.resolve()), "sha256": _sha(path)}
    indices = {}
    source_audits = {}
    target_audits = {}
    for split in ("train", "validation"):
        path = tmp_path / f"{split}.sqlite"
        marker = tmp_path / f"{split}.sqlite.frozen.json"
        path.write_bytes(split.encode("ascii"))
        marker.write_text(split + "-marker", encoding="utf-8")
        rows = 1_000_000 if split == "train" else 20_000
        source_audits[split] = {
            "source_latent_shards": 3 if split == "train" else 2,
            "source_pair_rows": rows,
            "source_latent_shard_inventory_sha256": hashlib.sha256(
                f"{split}-source-inventory".encode("utf-8")
            ).hexdigest(),
            "source_latent_shards_exhaustively_verified": True,
        }
        target_audits[split] = {
            "target_latent_shards": 5 if split == "train" else 4,
            "target_pair_rows": rows,
            "target_latent_shard_inventory_sha256": hashlib.sha256(
                f"{split}-target-inventory".encode("utf-8")
            ).hexdigest(),
            "target_latent_shards_exhaustively_verified": True,
        }
        indices[split] = {
            "path": str(path.resolve()),
            "sha256": _sha(path),
            "marker_path": str(marker.resolve()),
            "marker_sha256": _sha(marker),
            "rows": rows,
            **source_audits[split],
            **target_audits[split],
        }
    monkeypatch.setattr(
        lineage,
        "verify_editing_source_latent_shards",
        lambda path: dict(source_audits[Path(path).stem]),
    )
    monkeypatch.setattr(
        lineage,
        "verify_editing_target_latent_shards",
        lambda path: dict(target_audits[Path(path).stem]),
    )
    preflight = {
        "schema": "sceneplan_transfusion_editing_full_training_preflight",
        "schema_version": 2,
        "status": "PASS",
        "split_disjointness": {
            "status": "PASS",
            "checks": {
                "train_vs_test": True,
                "train_vs_validation": True,
                "validation_vs_test": True,
            },
        },
        "indices": indices,
        "dataset_configs": {
            "train": assets["train.json"],
            "validation": assets["validation.json"],
        },
        "model_config": assets["model.json"]["path"],
        "model_config_sha256": assets["model.json"]["sha256"],
        "canonical_p10_checkpoint": assets["p10.ckpt"]["path"],
        "canonical_p10_checkpoint_sha256": assets["p10.ckpt"]["sha256"],
        "shared_contract": shared_contract,
    }
    preflight_path = tmp_path / "PREFLIGHT.json"
    _write_json(preflight_path, preflight)
    contract = {
        "schema": lineage.SCHEMA,
        "schema_version": lineage.SCHEMA_VERSION,
        "status": "FROZEN",
        "run_dir": str(run_dir.resolve()),
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
        "preflight": {
            "path": str(preflight_path.resolve()),
            "sha256": _sha(preflight_path),
        },
        "shared_contract": shared_contract,
        "model_config": assets["model.json"],
        "dataset_configs": {
            "train": assets["train.json"],
            "validation": assets["validation.json"],
        },
        "indices": indices,
        "source_latent_integrity": {
            "policy": "hash_every_distinct_external_source_shard_once_per_gate_v1",
            "train": source_audits["train"],
            "validation": source_audits["validation"],
        },
        "target_latent_integrity": {
            "policy": "hash_every_distinct_external_target_shard_once_per_gate_v1",
            "train": target_audits["train"],
            "validation": target_audits["validation"],
        },
        "p10_checkpoint": assets["p10.ckpt"],
        "training": {
            "max_steps": 30_000,
            "checkpoint_every": 5_000,
            "candidate_steps": [5_000, 10_000, 15_000, 20_000, 25_000, 30_000],
            "checkpoint_retention": "all_six_5k_candidates_plus_last",
            "short_batch_size_per_gpu": 72,
            "long_batch_size_per_gpu": 48,
            "num_workers_per_rank": 12,
            "world_size": 5,
            "physical_gpus": [3, 4, 5, 6, 7],
            "cuda_visible_devices": "3,4,5,6,7",
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
        },
        "source_sha256": {},
    }
    contract_path = run_dir / lineage.CONTRACT_NAME
    _write_json(contract_path, contract)
    lineage.validate_contract(contract_path, expected_run_dir=run_dir)
    return contract_path, contract


def test_source_latent_audit_contract_is_fail_closed() -> None:
    complete = {
        "source_latent_shards": 4,
        "source_pair_rows": 20_000,
        "source_latent_shard_inventory_sha256": "a" * 64,
        "source_latent_shards_exhaustively_verified": True,
    }
    assert selector._source_latent_audit_complete(
        complete, expected_rows=20_000
    )
    for key in complete:
        changed = dict(complete)
        changed.pop(key)
        assert not selector._source_latent_audit_complete(
            changed, expected_rows=20_000
        )
    target = {
        "target_latent_shards": 20,
        "target_pair_rows": 20_000,
        "target_latent_shard_inventory_sha256": "b" * 64,
        "target_latent_shards_exhaustively_verified": True,
    }
    assert selector._target_latent_audit_complete(
        target, expected_rows=20_000
    )
    for key in target:
        changed = dict(target)
        changed.pop(key)
        assert not selector._target_latent_audit_complete(
            changed, expected_rows=20_000
        )


def test_lineage_rechecks_live_source_shard_inventory(
    tmp_path: Path, monkeypatch
) -> None:
    contract_path, _ = _tiny_lineage_contract(tmp_path, monkeypatch)
    original = lineage.verify_editing_source_latent_shards

    def changed(path: Path) -> dict:
        value = dict(original(path))
        value["source_latent_shard_inventory_sha256"] = "f" * 64
        return value

    monkeypatch.setattr(lineage, "verify_editing_source_latent_shards", changed)
    with pytest.raises(RuntimeError, match="source-latent shards changed"):
        lineage.validate_contract(contract_path)


def test_lineage_rechecks_live_target_shard_inventory(
    tmp_path: Path, monkeypatch
) -> None:
    contract_path, _ = _tiny_lineage_contract(tmp_path, monkeypatch)
    original = lineage.verify_editing_target_latent_shards

    def changed(path: Path) -> dict:
        value = dict(original(path))
        value["target_latent_shard_inventory_sha256"] = "e" * 64
        return value

    monkeypatch.setattr(lineage, "verify_editing_target_latent_shards", changed)
    with pytest.raises(RuntimeError, match="target-latent shards changed"):
        lineage.validate_contract(contract_path)


def _formal_layout() -> list[tuple[int, int, str]]:
    operations = (
        "event_addition",
        "event_removal",
        "linear_to_static",
        "static_to_linear",
        "stationary_spatial_relocation",
    )
    return [
        (ordinal, 432 if ordinal < 15_000 else 648, operations[ordinal % 5])
        for ordinal in range(20_000)
    ]


def test_rank_assignment_covers_every_validation_row_without_tail_drop() -> None:
    rows = _formal_layout()
    assignments = [selector._rank_ordinals(rows, rank)[0] for rank in range(5)]
    flattened = [ordinal for assignment in assignments for ordinal in assignment]
    assert len(flattened) == 20_000
    assert len(set(flattened)) == 20_000
    assert set(flattened) == set(range(20_000))
    for assignment in assignments:
        assert len(assignment) == 4_000
        assert sum(ordinal < 15_000 for ordinal in assignment) == 3_000
        assert sum(ordinal >= 15_000 for ordinal in assignment) == 1_000


def test_candidate_discovery_requires_all_six_formal_steps(tmp_path: Path) -> None:
    assert selector.CHECKPOINT_RETENTION == "all_6_named_candidates_plus_last_copy"
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    expected = []
    for index, step in enumerate(range(5_000, 30_001, 5_000)):
        path = checkpoint_dir / f"epoch={index}-step={step}.ckpt"
        path.write_bytes(b"candidate")
        expected.append(path.resolve())
    # A recovered exception checkpoint is audit-only and must not replace or
    # invalidate one of the six registered interval candidates.
    (checkpoint_dir / "epoch=3-step=1234.ckpt").write_bytes(b"exception")
    (checkpoint_dir / "last.ckpt").write_bytes(b"candidate")
    candidates, last = selector._discover_candidates(
        tmp_path, expected_max_step=30_000, checkpoint_every=5_000
    )
    assert [step for step, _ in candidates] == list(range(5_000, 30_001, 5_000))
    assert [path for _, path in candidates] == expected
    assert last == (checkpoint_dir / "last.ckpt").resolve()

    expected[4].unlink()
    with pytest.raises(RuntimeError, match="incomplete or ambiguous"):
        selector._discover_candidates(
            tmp_path, expected_max_step=30_000, checkpoint_every=5_000
        )


@pytest.mark.parametrize(
    "forbidden",
    [
        "old_sceneplan",
        "old-plan",
        "source_sceneplan",
        "source-plan",
        "previous_sceneplan",
        "previous-plan",
    ],
)
def test_selector_rejects_old_plan_aliases(forbidden: str) -> None:
    with pytest.raises(RuntimeError, match="old-plan"):
        selector._reject_old_plan_metadata([{"pair_id": "p", forbidden: {}}])


def test_paired_gate_uses_positive_one_sided_99pct_lower_bound() -> None:
    values = torch.full((100,), 0.25, dtype=torch.float64)
    with mock.patch.object(selector, "_all_reduce", side_effect=lambda raw, _: raw):
        positive = selector._paired_stats(values, torch.device("cpu"))
        negative = selector._paired_stats(-values, torch.device("cpu"))
    assert positive["pass"] is True
    assert positive["one_sided_lower_confidence_bound"] > 0
    assert negative["pass"] is False
    assert negative["one_sided_lower_confidence_bound"] < 0


def test_matrix_summary_is_row_weighted_and_preserves_strata() -> None:
    matrix = torch.tensor([[1.0, 3.0], [2.0, 4.0]], dtype=torch.float64)
    operations = ["a", "b"]
    buckets = [432, 648]
    with (
        mock.patch.object(selector, "_all_reduce", side_effect=lambda raw, _: raw),
        mock.patch.object(selector, "_broadcast", side_effect=lambda raw, **_: raw),
        mock.patch.object(selector.dist, "get_rank", return_value=0),
    ):
        report = selector._matrix_summary(
            matrix,
            operations=operations,
            buckets=buckets,
            timesteps=(0.1, 0.9),
            device=torch.device("cpu"),
        )
    assert report["mean"] == pytest.approx(2.5)
    assert report["by_timestep"] == pytest.approx({"0.1": 1.5, "0.9": 3.5})
    assert report["by_operation"]["a"]["mean"] == pytest.approx(2.0)
    assert report["by_operation"]["b"]["mean"] == pytest.approx(3.0)
    assert report["by_latent_bucket"]["432"]["rows"] == 1
    assert report["by_latent_bucket"]["648"]["rows"] == 1


def test_promotion_gate_requires_both_latent_buckets() -> None:
    passed = {"pass": True}
    failed = {"pass": False}
    report = {
        "overall": passed,
        "by_operation": {name: passed for name in selector.EXPECTED_OPERATIONS},
        "by_latent_bucket": {"432": passed, "648": failed},
    }
    assert selector._paired_report_pass(report) is False
    report["by_latent_bucket"]["648"] = passed
    assert selector._paired_report_pass(report) is True


@pytest.mark.parametrize("initial", [0, 2_500, 12_345, 27_500])
def test_completed_training_gate_accepts_exact_resume_segment(initial: int) -> None:
    final = 30_000
    segment = final - initial
    value = {
        "status": "PASS",
        "initial_global_step": initial,
        "global_step": final,
        "optimizer_events": segment,
        "metric_observations": segment,
        "optimizer_state_step_max": final,
        "ema_initial": {
            "diffusion_ema": initial,
            "conditioner_ema": initial,
        },
        "ema_final": {
            "diffusion_ema": final,
            "conditioner_ema": final,
        },
        "ema_advances": {
            "diffusion_ema": segment,
            "conditioner_ema": segment,
        },
        "distributed_health": {
            "optimizer_events_min": segment,
            "optimizer_events_max": segment,
            "optimizer_state_step_min": final,
            "optimizer_state_step_max": final,
            "ema_advance_min": segment,
            "ema_advance_max": segment,
            "metric_observations_min": segment,
            "metric_observations_max": segment,
        },
    }
    assert selector._completed_training_gate_valid(value, expected_max_step=final)


def test_completed_training_gate_rejects_zero_length_or_partial_segment() -> None:
    value = {
        "status": "PASS",
        "initial_global_step": 30_000,
        "global_step": 30_000,
        "optimizer_events": 0,
        "metric_observations": 0,
        "optimizer_state_step_max": 30_000,
        "ema_initial": {"diffusion_ema": 30_000, "conditioner_ema": 30_000},
        "ema_final": {"diffusion_ema": 30_000, "conditioner_ema": 30_000},
        "ema_advances": {"diffusion_ema": 0, "conditioner_ema": 0},
        "distributed_health": {
            "optimizer_events_min": 0,
            "optimizer_events_max": 0,
            "optimizer_state_step_min": 30_000,
            "optimizer_state_step_max": 30_000,
            "ema_advance_min": 0,
            "ema_advance_max": 0,
            "metric_observations_min": 0,
            "metric_observations_max": 0,
        },
    }
    assert not selector._completed_training_gate_valid(
        value, expected_max_step=30_000
    )
    value["initial_global_step"] = 2_500
    value["optimizer_events"] = 27_499
    assert not selector._completed_training_gate_valid(
        value, expected_max_step=30_000
    )


def _completed_log_fixture(run_dir: Path) -> Path:
    # These are verbatim trainer events from the completed 30K run.  The shell
    # launch summary went to the enclosing service log before tee was started.
    fixture = REPO_ROOT / (
        "tests/fixtures/sceneplan_transfusion_editing_dit_completed_30000.log"
    )
    path = run_dir / "logs/train.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(fixture.read_text(encoding="utf-8"), encoding="utf-8")
    return path


def test_completed_run_audit_uses_bound_contract_without_shell_headers(
    tmp_path: Path, monkeypatch
) -> None:
    contract_path, contract = _tiny_lineage_contract(tmp_path, monkeypatch)
    log = _completed_log_fixture(contract_path.parent)
    text = log.read_text()
    assert "batch/GPU=" not in text
    assert "checkpoint_every=" not in text
    report = selector._audit_completed_training(
        contract_path.parent,
        expected_max_step=30_000,
        checkpoint_every=5_000,
        training_run_contract=contract,
        training_run_contract_sha256=_sha(contract_path),
    )
    assert report["training_gate"]["global_step"] == 30_000
    assert report["visible_local_ranks"] == list(range(5))
    assert report["durable_validation_log"]["steps"] == list(range(1_000, 30_001, 1_000))
    assert report["launch_settings_evidence"]["sha256"] == _sha(contract_path)
    assert report["log_sha256"] == _sha(log)


@pytest.mark.parametrize(
    "key,value",
    [("max_steps", 29_000), ("checkpoint_every", 1_000),
     ("short_batch_size_per_gpu", 64), ("long_batch_size_per_gpu", 32),
     ("num_workers_per_rank", 8), ("world_size", 4),
     ("accumulate_grad_batches", 2), ("gradient_clip_val", 0.5),
     ("save_top_k", 1)],
)
def test_completed_run_audit_rejects_changed_launch_settings(
    tmp_path: Path, monkeypatch, key: str, value: int | float
) -> None:
    contract_path, contract = _tiny_lineage_contract(tmp_path, monkeypatch)
    _completed_log_fixture(contract_path.parent)
    contract["training"][key] = value
    _write_json(contract_path, contract)
    with pytest.raises(RuntimeError, match="frozen launch contract changed"):
        selector._audit_completed_training(
            contract_path.parent,
            expected_max_step=30_000,
            checkpoint_every=5_000,
            training_run_contract=contract,
            training_run_contract_sha256=_sha(contract_path),
        )


def test_completed_run_audit_rejects_contract_modified_after_validation(
    tmp_path: Path, monkeypatch
) -> None:
    contract_path, contract = _tiny_lineage_contract(tmp_path, monkeypatch)
    _completed_log_fixture(contract_path.parent)
    validated_sha = _sha(contract_path)
    contract_path.write_text(contract_path.read_text() + "\n")
    with pytest.raises(RuntimeError, match="frozen launch contract changed"):
        selector._audit_completed_training(
            contract_path.parent,
            expected_max_step=30_000,
            checkpoint_every=5_000,
            training_run_contract=contract,
            training_run_contract_sha256=validated_sha,
        )


@pytest.mark.parametrize(
    "remove_marker",
    ["SAT_TRAINING_GATE_RESULT=", "SAT_EDITING_VALIDATION=",
     "LOCAL_RANK: 4", "[p10-data]", "`Trainer.fit` stopped:"],
)
def test_completed_run_audit_still_requires_actual_execution_evidence(
    tmp_path: Path, monkeypatch, remove_marker: str
) -> None:
    contract_path, contract = _tiny_lineage_contract(tmp_path, monkeypatch)
    log = _completed_log_fixture(contract_path.parent)
    log.write_text("\n".join(
        line for line in log.read_text().splitlines()
        if remove_marker not in line
    ) + "\n")
    with pytest.raises(RuntimeError, match="gates"):
        selector._audit_completed_training(
            contract_path.parent,
            expected_max_step=30_000,
            checkpoint_every=5_000,
            training_run_contract=contract,
            training_run_contract_sha256=_sha(contract_path),
        )


def test_marked_json_events_parse_formal_bucket_records() -> None:
    text = (
        'prefix [p10-data] {"batches_per_epoch":3124,'
        '"bucket_counts":{"432":750000,"648":250000},'
        '"long_batch_size":48,"rank":3,"short_batch_size":72,'
        '"world_size":5}\n'
    )
    assert selector._marked_json_events(text, "[p10-data]") == [
        {
            "batches_per_epoch": 3_124,
            "bucket_counts": {"432": 750_000, "648": 250_000},
            "long_batch_size": 48,
            "rank": 3,
            "short_batch_size": 72,
            "world_size": 5,
        }
    ]


def test_full_launchers_cannot_fall_back_to_unvalidated_last_checkpoint() -> None:
    full_chain = (
        REPO_ROOT
        / "scripts/t2a/train/run_sceneplan_transfusion_editing_full_chain_5gpu.sh"
    ).read_text(encoding="utf-8")
    materialization = (
        REPO_ROOT
        / "scripts/t2a/data/run_sceneplan_transfusion_editing_materialization_5gpu.sh"
    ).read_text(encoding="utf-8")
    dit = (
        REPO_ROOT
        / "scripts/t2a/train/run_sceneplan_transfusion_editing_dit_full_5gpu.sh"
    ).read_text(encoding="utf-8")
    assert "export CUDA_DEVICE_ORDER=PCI_BUS_ID" in materialization
    assert "unset CUDA_VISIBLE_DEVICES" in materialization
    assert "materialization workers are already active" in materialization
    assert "flock -n 9" in materialization
    assert ') >>"$LOG" 2>&1 &' in materialization
    assert "set CUDA_VISIBLE_DEVICES to the GPUs for this job" in full_chain
    assert "full-chain.lock" in full_chain
    assert "materialized_complete_frozen" in full_chain
    assert "--replace" not in full_chain
    assert "run_sceneplan_transfusion_editing_dit_full_5gpu.sh" in full_chain
    joint = (
        REPO_ROOT
        / "scripts/t2a/train/run_sceneplan_transfusion_editing_ar_joint_full_5gpu.sh"
    ).read_text(encoding="utf-8")
    assert "export SAVE_TOP_K=-1" in dit
    assert "export CUDA_DEVICE_ORDER=PCI_BUS_ID" in dit
    assert "formal utilization contract requires BATCH_SIZE=72" in dit
    assert "editing_dit_utilization_benchmark" in dit
    assert ")==319314304" in dit
    assert "319360528" not in dit
    assert "run_sceneplan_transfusion_editing_dit_checkpoint_selection_5gpu.sh" in dit
    assert "verified six-candidate inventory" in dit
    assert "resolve-resume --contract" in dit
    assert "SAT_EDITING_RUN_CONTRACT_PATH" in dit
    assert "SAT_EDITING_VALIDATION_JSON_LOG=1" in dit
    assert 'exec 7>"$LOCK_ROOT/training-chain.lock"' in dit
    assert "flock -n 7" in dit
    assert "EDITING_TRAIN_CHAIN_LOCK_FD=7" in dit
    assert 'f"{role}_latent_shards_exhaustively_verified"' in dit
    assert 'f"{role}_latent_shard_inventory_sha256"' in dit
    assert (
        "scripts/t2a/train/run_sceneplan_transfusion_editing_full_chain_5gpu.sh"
        in lineage.SOURCE_PATHS
    )
    assert (
        "scripts/t2a/train/run_sceneplan_transfusion_editing_full_chain_5gpu.sh"
        in selector.AUDITED_SOURCE_PATHS
    )
    assert "export CUDA_DEVICE_ORDER=PCI_BUS_ID" in joint
    assert 'exec 6>"$RUN_DIR/.editing-joint-full.lock"' in joint
    assert "flock -n 6" in joint
    assert 'inherited_lock="$(readlink "/proc/$$/fd/7"' in joint
    assert 'exec 7>"$GLOBAL_CHAIN_LOCK"' in joint
    assert joint.count('--validate-every "$VALIDATE_EVERY"') == 1
    assert "checkpoints/last.ckpt" not in joint
    assert "--checkpoint-selection \"$DIT_CHECKPOINT_SELECTION\"" in joint
    assert "BASE_DIT_CHECKPOINT cannot bypass the validated selection" in joint


def test_joint_selector_supplies_complete_dit_lineage_to_loader(
    tmp_path: Path, monkeypatch
) -> None:
    from scripts.t2a.eval import (
        select_sceneplan_transfusion_editing_joint_checkpoint as joint_selector,
    )

    contract_path = tmp_path / "TRAIN_RUN_CONTRACT.json"
    contract_path.write_text("{}\n", encoding="utf-8")
    contract = {"schema": "tiny-editing-dit-lineage"}
    digest = "a" * 64
    captured: dict = {}

    monkeypatch.setattr(
        joint_selector,
        "validate_dit_training_run_contract",
        lambda *args, **kwargs: (contract, digest),
    )

    def fake_loader(*args, **kwargs):
        captured.update(kwargs)
        return {"loaded": True}

    monkeypatch.setattr(joint_selector, "_load_dit_candidate", fake_loader)
    result = joint_selector._load_promoted_base_dit_candidate(
        object(),
        tmp_path / "step.ckpt",
        selection_summary={
            "training_run_contract": {
                "path": str(contract_path),
                "sha256": digest,
                "canonical_sha256": lineage.canonical_sha256(contract),
            }
        },
        selection_value={
            "training_run": {"run_dir": str(tmp_path)},
            "selected_checkpoint_step": 5_000,
        },
        resolved_model_config={"model": "tiny"},
    )
    assert result == {"loaded": True}
    assert captured["training_run_contract"] == contract
    assert captured["training_run_contract_path"] == contract_path.resolve()
    assert captured["training_run_contract_sha256"] == digest
    assert captured["step"] == 5_000


def test_lineage_resolver_accepts_only_embedded_contract(
    tmp_path: Path, monkeypatch
) -> None:
    contract_path, contract = _tiny_lineage_contract(tmp_path, monkeypatch)
    checkpoint_dir = contract_path.parent / "checkpoints"
    checkpoint_dir.mkdir()
    candidate = checkpoint_dir / "epoch=0-step=5000.ckpt"
    torch.save(
        {
            "global_step": 5_000,
            "model_config": {},
            "editing_run_contract": contract,
            "editing_run_contract_path": str(contract_path.resolve()),
            "editing_run_contract_sha256": _sha(contract_path),
        },
        candidate,
    )
    assert lineage.resolve_resume(contract_path) == (
        "RESUME",
        str(candidate.resolve()),
    )

    foreign = dict(contract)
    foreign["run_dir"] = str((tmp_path / "foreign").resolve())
    torch.save(
        {
            "global_step": 10_000,
            "model_config": {},
            "editing_run_contract": foreign,
            "editing_run_contract_path": str(contract_path.resolve()),
            "editing_run_contract_sha256": _sha(contract_path),
        },
        checkpoint_dir / "epoch=0-step=10000.ckpt",
    )
    with pytest.raises(RuntimeError, match="lineage changed"):
        lineage.resolve_resume(contract_path)


def test_corrupt_last_is_preserved_and_falls_back_to_verified_candidate(
    tmp_path: Path, monkeypatch
) -> None:
    contract_path, contract = _tiny_lineage_contract(tmp_path, monkeypatch)
    checkpoint_dir = contract_path.parent / "checkpoints"
    checkpoint_dir.mkdir()
    candidate = checkpoint_dir / "epoch=0-step=5000.ckpt"
    payload = {
        "global_step": 5_000,
        "model_config": {},
        "editing_run_contract": contract,
        "editing_run_contract_path": str(contract_path.resolve()),
        "editing_run_contract_sha256": _sha(contract_path),
    }
    torch.save(payload, candidate)
    (checkpoint_dir / "last.ckpt").write_bytes(b"not-a-checkpoint")

    assert lineage.resolve_resume(contract_path) == (
        "RESUME",
        str(candidate.resolve()),
    )
    assert not (checkpoint_dir / "last.ckpt").exists()
    preserved = list(checkpoint_dir.glob("recovery-quarantine-*/last.ckpt"))
    assert len(preserved) == 1
    assert preserved[0].read_bytes() == b"not-a-checkpoint"


def test_generic_checkpoint_embedding_is_explicitly_editing_opt_in() -> None:
    trainer = (REPO_ROOT / "train.py").read_text(encoding="utf-8")
    assert 'os.environ.get("SAT_EDITING_RUN_CONTRACT_PATH")' in trainer
    assert 'int(value.get("schema_version", -1)) == 2' in trainer
    assert lineage.SCHEMA_VERSION == 2
    assert 'checkpoint["editing_run_contract"]' in trainer
    assert 'checkpoint["editing_run_contract_sha256"]' in trainer
    assert "if self.editing_run_contract is not None" in trainer


def _write_identity_index(path: Path, *, token: str, parent: str) -> None:
    member = {
        "identity_hash": hashlib.sha256(f"member-{token}".encode()).hexdigest(),
        "asset_ref": {"asset_id": f"asset-{token}", "parent_asset_id": parent},
    }
    blob = zlib.compress(json.dumps([member]).encode("utf-8"))
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            "CREATE TABLE pairs("
            "source_sample_id TEXT,source_foa_sha256 TEXT,"
            "source_latent_tensor_sha256 TEXT,old_sceneplan_sha256 TEXT,"
            "source_members_sha256 TEXT,raw_edit_request TEXT,"
            "source_members_zlib BLOB,target_members_zlib BLOB)"
        )
        connection.execute(
            "INSERT INTO pairs VALUES(?,?,?,?,?,?,?,?)",
            (
                f"sample-{token}",
                hashlib.sha256(f"foa-{token}".encode()).hexdigest(),
                hashlib.sha256(f"latent-{token}".encode()).hexdigest(),
                hashlib.sha256(f"plan-{token}".encode()).hexdigest(),
                hashlib.sha256(f"members-{token}".encode()).hexdigest(),
                f"instruction-{token}",
                blob,
                blob,
            ),
        )
        connection.commit()
    finally:
        connection.close()


def test_split_audit_detects_parent_asset_leakage(tmp_path: Path) -> None:
    paths = {split: tmp_path / f"{split}.sqlite" for split in ("train", "validation", "test")}
    _write_identity_index(paths["train"], token="train", parent="parent-train")
    _write_identity_index(paths["validation"], token="validation", parent="parent-validation")
    _write_identity_index(paths["test"], token="test", parent="parent-test")
    reports = {split: {"path": str(path)} for split, path in paths.items()}
    result = full_preflight._split_disjointness_audit(reports)
    assert result["status"] == "PASS"
    assert all(result["checks"].values())

    paths["test"].unlink()
    _write_identity_index(paths["test"], token="test", parent="parent-validation")
    with pytest.raises(RuntimeError, match="identity leakage"):
        full_preflight._split_disjointness_audit(reports)


def test_preflight_separates_historical_and_current_shared_contract(
    tmp_path: Path, monkeypatch
) -> None:
    current = tmp_path / "shared-contract.md"
    current.write_text("new active Editing route", encoding="utf-8")
    monkeypatch.setattr(full_preflight, "CURRENT_SHARED_CONTRACT", current)
    metadata = {
        "shared_contract_path": str(current.resolve()),
        "shared_contract_sha256": (
            full_preflight.INDEX_BUILD_SHARED_CONTRACT_SHA256
        ),
    }
    report = full_preflight._index_build_shared_contract_provenance(
        metadata, split="validation"
    )
    assert report["index_build_sha256"] != report["current_sha256"]
    assert report["current_sha256"] == _sha(current)

    metadata["shared_contract_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="historical shared-contract"):
        full_preflight._index_build_shared_contract_provenance(
            metadata, split="validation"
        )
