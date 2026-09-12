from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sqlite3
import sys
from types import SimpleNamespace

import pytest
import torch

_EVALUATOR_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts/t2a/test/evaluate_sceneplan_transfusion_generation_ar_p10_audio_8k.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "sceneplan_transfusion_generation_ar_p10_audio_evaluator", _EVALUATOR_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
_EVALUATOR = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _EVALUATOR
_SPEC.loader.exec_module(_EVALUATOR)
AUDIO_CLOSURE_CONTRACT = _EVALUATOR.AUDIO_CLOSURE_CONTRACT
PlanEvaluationRow = _EVALUATOR.PlanEvaluationRow
_bundle_fingerprints = _EVALUATOR._bundle_fingerprints
_duration_group = _EVALUATOR._duration_group
_insert_audio_result = _EVALUATOR._insert_audio_result
_open_audio_shard = _EVALUATOR._open_audio_shard
_select_exact_anchors = _EVALUATOR._select_exact_anchors
_stable_seed = _EVALUATOR._stable_seed
_spatial_pair_metrics = _EVALUATOR._spatial_pair_metrics
_summarize_audio_records = _EVALUATOR._summarize_audio_records
_validate_release_document = _EVALUATOR._validate_release_document
validate_completed_audio_output = _EVALUATOR.validate_completed_audio_output


def test_distributed_uses_cpu_group_for_long_tail_completion(monkeypatch) -> None:
    calls: dict[str, object] = {}
    completion_group = object()

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "3")
    monkeypatch.setattr(_EVALUATOR.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(_EVALUATOR.torch.cuda, "set_device", lambda rank: None)
    monkeypatch.setattr(
        _EVALUATOR.dist,
        "init_process_group",
        lambda **kwargs: calls.setdefault("init", kwargs),
    )

    def fake_new_group(**kwargs):
        calls["completion"] = kwargs
        return completion_group

    monkeypatch.setattr(_EVALUATOR.dist, "new_group", fake_new_group)

    rank, local_rank, world_size, device, group = _EVALUATOR._distributed()

    assert (rank, local_rank, world_size) == (0, 0, 3)
    assert device == torch.device("cuda", 0)
    assert group is completion_group
    assert calls["init"]["backend"] == "nccl"
    assert calls["completion"]["backend"] == "gloo"
    assert (
        calls["completion"]["timeout"].total_seconds()
        == _EVALUATOR.COMPLETION_BARRIER_TIMEOUT_SECONDS
    )


def _plan_row(
    ordinal: int,
    *,
    source_count: int = 1,
    template_id: str = "test/0",
    room: str = "dry",
    motion: str = "static",
    exact: bool = True,
) -> PlanEvaluationRow:
    target = b'{"sample_id":"row"}' + str(ordinal).encode()
    prediction = target if exact else target + b"-different"
    import hashlib

    return PlanEvaluationRow(
        ordinal=ordinal,
        sample_id=f"row_{ordinal}",
        template_id=template_id,
        source_count=source_count,
        room_type=room,
        motion_signature=motion,
        manifest_latent_frames=10,
        target_sceneplan_sha256=hashlib.sha256(target).hexdigest(),
        prediction_sceneplan_sha256=hashlib.sha256(prediction).hexdigest(),
        target_sceneplan_bytes=target,
        prediction_sceneplan_bytes=prediction,
        target_sceneplan={},
        prediction_sceneplan={},
        target_token_ids=(1, 2),
        prediction_token_ids=(1, 2),
    )


def _metrics_for_nonexact() -> dict:
    return {
        "target": {"finite_foa": True, "shape_exact": True},
        "prediction": {"finite_foa": True, "shape_exact": True},
        "predicted_target": {
            "waveform_mae": 0.2,
            "waveform_rmse": 0.3,
            "waveform_cosine": 0.4,
            "spectral": {
                "w_log_magnitude_l1": 0.5,
                "w_magnitude_cosine": 0.6,
                "w_spectral_convergence": 0.7,
            },
        },
        "predicted_target_foa": {"spherical_error_mean_deg": 8.0},
        "predicted_target_spatial": {
            "angular_error_mean_deg": 6.0,
            "direction_cosine": 0.9,
            "diffuseness_mae": 0.05,
        },
        "prediction_plan_foa": {"spherical_error_mean_deg": 9.0},
        "target_plan_foa": {"spherical_error_mean_deg": 7.0},
        "prediction_plan_activity": {
            "temporal_iou": 0.8,
            "onset_abs_error_sec": 0.1,
            "offset_abs_error_sec": 0.2,
        },
    }


def _record(
    ordinal: int,
    *,
    exact: bool,
    anchor: bool,
    status: str,
) -> dict:
    target_hash = f"target-{ordinal}"
    prediction_hash = target_hash if exact else f"prediction-{ordinal}"
    metrics = (
        {
            "target_repeat_waveform_exact": True,
            "prediction_target_waveform_exact": True,
        }
        if anchor
        else (_metrics_for_nonexact() if not exact else {})
    )
    return {
        "ordinal": ordinal,
        "sample_id": f"row_{ordinal}",
        "template_id": "test/0",
        "source_count": 1,
        "plan_exact": exact,
        "exact_anchor": anchor,
        "status": status,
        "error": None,
        "target_sceneplan_sha256": target_hash,
        "prediction_sceneplan_sha256": prediction_hash,
        "target_bundle_sha256": "bundle" if exact else "target-bundle",
        "prediction_bundle_sha256": "bundle" if exact else "prediction-bundle",
        "target_render_input_sha256": "render" if exact else "target-render",
        "prediction_render_input_sha256": "render" if exact else "prediction-render",
        "render_seed": 1,
        "length_group": "same_duration_and_latent_length",
        "target_model_num_samples": 10240,
        "prediction_model_num_samples": 10240,
        "target_latent_frames": 10,
        "prediction_latent_frames": 10,
        "target_foa_sha256": "audio" if anchor else None,
        "prediction_foa_sha256": "audio" if anchor else None,
        "target_repeat_foa_sha256": "audio" if anchor else None,
        "metrics": metrics,
        "render_sec": 1.0,
    }


def test_render_input_fingerprint_is_stable_and_sensitive() -> None:
    metadata = {
        "sample_id": "row_0",
        "prompt": {"input_ids": torch.tensor([1, 2], dtype=torch.long)},
        "sceneplan_44": {
            "source_event_frame_ids": torch.tensor([[1, 1]], dtype=torch.int8)
        },
    }
    bundle = SimpleNamespace(
        task="generation",
        sample_id="row_0",
        plan_token_ids=torch.tensor([3, 4]),
        sceneplan={"sample_id": "row_0"},
        renderer_caption={"text": "bell"},
        p10_metadata=metadata,
        model_num_samples=2048,
        latent_frames_valid=2,
        execution_contract="external",
        editing_contract=None,
        preserves_unedited_waveform=False,
        localized_editing=False,
    )
    first = _bundle_fingerprints(bundle)
    second = _bundle_fingerprints(bundle)
    assert first == second

    changed = copy.copy(bundle)
    changed.p10_metadata = copy.deepcopy(metadata)
    changed.p10_metadata["prompt"]["input_ids"][1] = 9
    assert (
        _bundle_fingerprints(changed)["render_input_sha256"]
        != first["render_input_sha256"]
    )


def test_exact_anchor_selection_is_deterministic_and_stratified() -> None:
    rows = [
        _plan_row(
            index,
            source_count=index % 4 + 1,
            template_id=f"test/{index % 2}",
            room=("dry" if index % 2 == 0 else "outdoor"),
            motion=("static" if index % 3 else "linear"),
        )
        for index in range(40)
    ]
    rows.append(_plan_row(40, exact=False))
    first = _select_exact_anchors(rows, count=32, seed=42)
    second = _select_exact_anchors(list(reversed(rows)), count=32, seed=42)
    assert first == second
    assert len(first) == 32
    assert 40 not in first
    assert {rows[index].source_count for index in first} == {1, 2, 3, 4}


def test_duration_groups_preserve_same_latent_duration_mismatch() -> None:
    assert (
        _duration_group(
            target_samples=1024,
            prediction_samples=1024,
            target_frames=1,
            prediction_frames=1,
        )
        == "same_duration_and_latent_length"
    )
    assert (
        _duration_group(
            target_samples=1000,
            prediction_samples=1001,
            target_frames=1,
            prediction_frames=1,
        )
        == "same_latent_length_duration_mismatch"
    )
    assert (
        _duration_group(
            target_samples=1024,
            prediction_samples=2048,
            target_frames=1,
            prediction_frames=2,
        )
        == "duration_and_latent_length_mismatch"
    )


def test_spatial_pair_metric_is_exact_for_identical_directional_foa() -> None:
    samples = 4 * 1024
    time = torch.linspace(0.0, 20.0, samples)
    w = torch.sin(time)
    # A coherent x-facing plane-wave-like fixture in WYZX ordering.
    foa = torch.stack((w, torch.zeros_like(w), torch.zeros_like(w), w))
    metrics = _spatial_pair_metrics(foa, foa, hop=1024)
    assert metrics["direction_cosine"] == pytest.approx(1.0, abs=1.0e-6)
    assert metrics["angular_error_mean_deg"] == pytest.approx(0.0, abs=1.0e-4)
    assert metrics["diffuseness_mae"] == 0.0


def test_summary_passes_only_when_all_integrity_gates_hold() -> None:
    records = [
        _record(0, exact=True, anchor=True, status="exact_anchor_rendered"),
        _record(1, exact=True, anchor=False, status="exact_input_equivalent"),
        _record(2, exact=False, anchor=False, status="nonexact_pair_rendered"),
    ]
    summary = _summarize_audio_records(
        records,
        expected_rows=3,
        expected_anchor_ordinals=[0],
        inputs_unchanged=True,
    )
    assert summary["status"] == "PASS"
    assert summary["coverage"]["exact_inferred_rows"] == 1
    assert summary["quality_metrics"]["same_latent_length"]["waveform_rmse"] == 0.3

    failed = copy.deepcopy(records)
    failed[0]["metrics"]["target_repeat_waveform_exact"] = False
    assert (
        _summarize_audio_records(
            failed,
            expected_rows=3,
            expected_anchor_ordinals=[0],
            inputs_unchanged=True,
        )["status"]
        == "FAIL"
    )
    failed = copy.deepcopy(records)
    failed[2]["status"] = "system_error"
    assert (
        _summarize_audio_records(
            failed,
            expected_rows=3,
            expected_anchor_ordinals=[0],
            inputs_unchanged=True,
        )["status"]
        == "FAIL"
    )


def test_audio_shard_resume_contract_is_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "rank.sqlite"
    connection = _open_audio_shard(
        path, rank=0, world_size=3, run_contract_sha256="abc"
    )
    connection.close()
    reopened = _open_audio_shard(
        path, rank=0, world_size=3, run_contract_sha256="abc"
    )
    metadata = dict(reopened.execute("SELECT key,value FROM metadata"))
    reopened.close()
    assert metadata["contract"] == AUDIO_CLOSURE_CONTRACT
    with pytest.raises(RuntimeError, match="contract mismatch"):
        _open_audio_shard(path, rank=0, world_size=3, run_contract_sha256="changed")


def test_audio_shard_archives_and_retries_system_errors(tmp_path: Path) -> None:
    path = tmp_path / "rank.sqlite"
    connection = _open_audio_shard(
        path, rank=0, world_size=3, run_contract_sha256="abc"
    )
    failed = _record(0, exact=False, anchor=False, status="system_error")
    failed["error"] = "CUDA out of memory"
    failed["metrics_json"] = json.dumps({"system_error": failed["error"]})
    del failed["metrics"]
    _insert_audio_result(connection, failed)
    connection.commit()
    connection.close()

    resumed = _open_audio_shard(
        path, rank=0, world_size=3, run_contract_sha256="abc"
    )
    assert resumed.execute("SELECT COUNT(*) FROM audio_results").fetchone()[0] == 0
    archived = resumed.execute(
        "SELECT status,error FROM audio_attempt_history ORDER BY attempt_id"
    ).fetchone()
    assert archived == ("system_error", "CUDA out of memory")
    resumed.close()


def test_completed_audio_output_is_recomputed_read_only(tmp_path: Path) -> None:
    source_root = tmp_path / "frozen_source"
    source_root.mkdir()
    frozen_source = source_root / "evaluator.py"
    frozen_source.write_text("frozen = True\n", encoding="utf-8")
    plan_dir = tmp_path / "plan"
    plan_dir.mkdir()
    identity_files: dict[str, Path] = {}
    for name in ("RUN_CONTRACT.json", "SUMMARY.json", "TEACHER_FORCED.json"):
        path = plan_dir / name
        path.write_text(f"plan fixture {name}\n", encoding="utf-8")
        identity_files[name] = path
    prediction = plan_dir / "rank_000.sqlite"
    prediction.write_bytes(b"prediction fixture")
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint fixture")
    manifest = tmp_path / "test.sqlite"
    manifest.write_bytes(b"test manifest fixture")
    source_index = tmp_path / "test.index.sqlite"
    source_index.write_bytes(b"source index fixture")

    def sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    plan_identity = {
        "path": str(plan_dir),
        "run_contract": str(identity_files["RUN_CONTRACT.json"]),
        "run_contract_sha256": sha256(identity_files["RUN_CONTRACT.json"]),
        "summary": str(identity_files["SUMMARY.json"]),
        "summary_sha256": sha256(identity_files["SUMMARY.json"]),
        "teacher_forced": str(identity_files["TEACHER_FORCED.json"]),
        "teacher_forced_sha256": sha256(identity_files["TEACHER_FORCED.json"]),
        "prediction_shard_sha256": {str(prediction): sha256(prediction)},
        "generation_ar_checkpoint": str(checkpoint),
        "generation_ar_checkpoint_sha256": sha256(checkpoint),
        "generation_ar_checkpoint_step": 30,
        "test_manifest": str(manifest),
        "test_manifest_sha256": sha256(manifest),
        "test_source_index": str(source_index),
        "test_source_index_sha256": sha256(source_index),
    }
    p10_identity = {
        "release_id": "p10-test",
        "sampling": {"steps": 100},
        "files": {},
        "codec": {"files": {}},
        "qwen": {"critical_files": {}},
    }
    output = tmp_path / "audio"
    (output / "audio_metrics").mkdir(parents=True)
    run_contract = {
        "schema": _EVALUATOR.AUDIO_CLOSURE_SCHEMA + ".run_contract",
        "schema_version": 1,
        "contract": AUDIO_CLOSURE_CONTRACT,
        "rows": 3,
        "world_size": 3,
        "cuda_visible_devices": "0,1,2",
        "root_seed": 42,
        "same_seed_pairing": True,
        "requested_exact_anchors": 32,
        "exact_anchor_ordinals": [0],
        "plan_evaluation": plan_identity,
        "p10": p10_identity,
        "source": {
            "evaluator.py": {
                "bytes": frozen_source.stat().st_size,
                "sha256": sha256(frozen_source),
            }
        },
        "retained_audio": False,
    }
    contract_path = output / "RUN_CONTRACT.json"
    contract_path.write_text(json.dumps(run_contract), encoding="utf-8")
    contract_canonical_sha = _EVALUATOR._canonical_sha256(run_contract)
    records = [
        _record(0, exact=True, anchor=True, status="exact_anchor_rendered"),
        _record(1, exact=False, anchor=False, status="nonexact_pair_rendered"),
        _record(2, exact=False, anchor=False, status="nonexact_pair_rendered"),
    ]
    shard_paths: list[Path] = []
    for rank, record in enumerate(records):
        shard_path = output / "audio_metrics" / f"rank_{rank:03d}.sqlite"
        shard = _open_audio_shard(
            shard_path,
            rank=rank,
            world_size=3,
            run_contract_sha256=contract_canonical_sha,
        )
        stored = dict(record)
        stored["metrics_json"] = json.dumps(stored.pop("metrics"))
        _insert_audio_result(shard, stored)
        shard.execute(
            "INSERT OR REPLACE INTO metadata(key,value) VALUES ('rows','1')"
        )
        shard.execute(
            "INSERT OR REPLACE INTO metadata(key,value) VALUES ('status','COMPLETE')"
        )
        shard.commit()
        shard.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        shard.close()
        shard_paths.append(shard_path)

    unchanged = _EVALUATOR._verify_run_inputs_unchanged(
        run_contract, source_root=source_root
    )
    summary_core = _summarize_audio_records(
        records,
        expected_rows=3,
        expected_anchor_ordinals=[0],
        inputs_unchanged=True,
    )
    report = {
        "schema": _EVALUATOR.AUDIO_CLOSURE_SCHEMA,
        "schema_version": 1,
        "contract": AUDIO_CLOSURE_CONTRACT,
        "run_contract": str(contract_path),
        "run_contract_sha256": sha256(contract_path),
        "run_contract_canonical_sha256": contract_canonical_sha,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_step": 30,
        "p10_release_id": "p10-test",
        "sampling": {"steps": 100},
        "input_reverification": unchanged,
        **summary_core,
    }
    report["report_sha256_without_self"] = _EVALUATOR._canonical_sha256(report)
    summary_path = output / "SUMMARY.json"
    summary_path.write_text(json.dumps(report), encoding="utf-8")
    before = [sha256(path) for path in shard_paths]

    validated = validate_completed_audio_output(
        output,
        plan_evaluation_dir=plan_dir,
        test_manifest=manifest,
        expected_rows=3,
        expected_world_size=3,
        source_root=source_root,
    )
    assert validated["status"] == "PASS"
    assert [sha256(path) for path in shard_paths] == before

    wrong_source_root = tmp_path / "wrong_source"
    wrong_source_root.mkdir()
    (wrong_source_root / "evaluator.py").write_text(
        "frozen = False\n", encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="input changed during run"):
        validate_completed_audio_output(
            output,
            plan_evaluation_dir=plan_dir,
            test_manifest=manifest,
            expected_rows=3,
            expected_world_size=3,
            source_root=wrong_source_root,
        )

    report["coverage"]["rows"] = 2
    report.pop("report_sha256_without_self")
    report["report_sha256_without_self"] = _EVALUATOR._canonical_sha256(report)
    summary_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(RuntimeError, match="does not match stored shards"):
        validate_completed_audio_output(
            output,
            plan_evaluation_dir=plan_dir,
            test_manifest=manifest,
            expected_rows=3,
            expected_world_size=3,
            source_root=source_root,
        )

    report["coverage"]["rows"] = 3
    report.pop("report_sha256_without_self")
    report["report_sha256_without_self"] = _EVALUATOR._canonical_sha256(report)
    summary_path.write_text(json.dumps(report), encoding="utf-8")
    for shard_path, ordinal in zip(shard_paths[1:], (2, 1)):
        connection = sqlite3.connect(shard_path)
        connection.execute("UPDATE audio_results SET ordinal=?", (ordinal,))
        connection.commit()
        connection.close()
    with pytest.raises(RuntimeError, match="rank ownership mismatch"):
        validate_completed_audio_output(
            output,
            plan_evaluation_dir=plan_dir,
            test_manifest=manifest,
            expected_rows=3,
            expected_world_size=3,
            source_root=source_root,
        )


def test_release_document_rejects_sampler_override() -> None:
    release_path = (
        Path(__file__).resolve().parents[1]
        / "artifacts/releases/P10_SCENEPLAN_DIT_V11_150K_RELEASE.json"
    )
    import json

    release = json.loads(release_path.read_text(encoding="utf-8"))
    paths = _validate_release_document(release, release_path=release_path)
    assert paths["checkpoint"].name == "epoch=48-step=150000.ckpt"
    changed = copy.deepcopy(release)
    changed["canonical_inference"]["steps"] = 50
    with pytest.raises(RuntimeError, match="frozen canonical"):
        _validate_release_document(changed, release_path=release_path)


def test_stable_seed_is_reproducible_and_row_specific() -> None:
    assert _stable_seed(42, "a") == _stable_seed(42, "a")
    assert _stable_seed(42, "a") != _stable_seed(42, "b")
