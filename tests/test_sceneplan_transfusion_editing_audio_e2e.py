from __future__ import annotations

import inspect
import json
from pathlib import Path
import sqlite3

import pytest
import torch

from scripts.t2a.eval import (
    evaluate_sceneplan_transfusion_editing_audio_end_to_end as audio_e2e,
)
from scripts.t2a.eval import (
    select_sceneplan_transfusion_editing_joint_checkpoint as joint_selector,
)
from scripts.t2a.train import (
    sceneplan_transfusion_editing_joint_run_contract as joint_lineage,
    train_sceneplan_transfusion_editing_ar_joint_full as joint_trainer,
)
from stable_audio_tools.models.sceneplan_transfusion_editing_pipeline import (
    EXPECTED_JOINT_SELECTION_SOURCE_PATHS,
    EXPECTED_JOINT_TRAINING_SOURCE_PATHS,
    ScenePlanTransfusionEditingPipeline,
    load_sceneplan_transfusion_editing_pipeline,
)
from stable_audio_tools.data.sceneplan_transfusion_editing_index import sha256_file


REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_joint_resume_artifacts(
    root: Path, *, include_final: bool
) -> Path:
    identity = joint_lineage.ensure_run_identity(root)
    identity_path = (root / joint_lineage.RUN_IDENTITY_NAME).resolve(strict=True)
    run_contract = {
        "schema": joint_lineage.RUN_CONTRACT_SCHEMA,
        "schema_version": joint_lineage.RUN_CONTRACT_SCHEMA_VERSION,
        "run_dir": str(root.resolve()),
        "run_id": identity["run_id"],
        "run_identity": {
            "path": str(identity_path),
            "sha256": sha256_file(identity_path),
        },
        "max_steps": 25_000,
        "save_every": 5_000,
        "world_size": 5,
    }
    contract_path = (root / joint_lineage.RUN_CONTRACT_NAME).resolve()
    contract_path.write_text(json.dumps(run_contract), encoding="utf-8")
    checkpoint = root / "checkpoints/step-00025000.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"joint-checkpoint")
    (checkpoint.parent / "LATEST.json").write_text(
        json.dumps(
            joint_lineage.latest_record(
                run_dir=root.resolve(),
                identity=identity,
                run_contract_path=contract_path.resolve(strict=True),
                run_contract_sha256=sha256_file(contract_path),
                checkpoint=checkpoint.resolve(),
                checkpoint_sha256=sha256_file(checkpoint),
                step=25_000,
            )
        ),
        encoding="utf-8",
    )
    if include_final:
        (root / "FINAL.json").write_text(
            json.dumps(
                {
                    "schema": joint_lineage.FINAL_SCHEMA,
                    "schema_version": joint_lineage.FINAL_SCHEMA_VERSION,
                    "event": "complete",
                    "step": 25_000,
                    "run_dir": str(root.resolve()),
                    "run_id": identity["run_id"],
                    "run_contract_path": str(contract_path.resolve(strict=True)),
                    "run_contract_sha256": sha256_file(contract_path),
                    "checkpoint": str(checkpoint.resolve()),
                    "checkpoint_sha256": sha256_file(checkpoint),
                    "shared_transformer_same_object": True,
                    "old_sceneplan_exposed": False,
                    "source_semantic_mode": "m2d_audio_caption_aux",
                    "source_caption_exposed_as_model_input": False,
                }
            ),
            encoding="utf-8",
        )
    return checkpoint.resolve()


def test_joint_max_step_resume_recovers_only_a_missing_final(tmp_path: Path) -> None:
    checkpoint = _write_joint_resume_artifacts(tmp_path, include_final=False)
    assert joint_trainer._completed_resume_action(
        tmp_path, resume_path=checkpoint, global_step=25_000
    ) == "finalize"

    _write_joint_resume_artifacts(tmp_path, include_final=True)
    assert joint_trainer._completed_resume_action(
        tmp_path, resume_path=checkpoint, global_step=25_000
    ) == "reuse"


def test_joint_max_step_resume_rejects_inconsistent_artifacts(tmp_path: Path) -> None:
    checkpoint = _write_joint_resume_artifacts(tmp_path, include_final=True)
    final_path = tmp_path / "FINAL.json"
    final = json.loads(final_path.read_text(encoding="utf-8"))
    final["old_sceneplan_exposed"] = True
    final_path.write_text(json.dumps(final), encoding="utf-8")
    with pytest.raises(RuntimeError, match="FINAL is inconsistent"):
        joint_trainer._completed_resume_action(
            tmp_path, resume_path=checkpoint, global_step=25_000
        )

    final["old_sceneplan_exposed"] = False
    final_path.write_text(json.dumps(final), encoding="utf-8")
    latest_path = tmp_path / "checkpoints/LATEST.json"
    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    latest["checkpoint_sha256"] = "0" * 64
    latest_path.write_text(json.dumps(latest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="LATEST/checkpoint disagree"):
        joint_trainer._completed_resume_action(
            tmp_path, resume_path=checkpoint, global_step=25_000
        )

    # Restore the valid resume pointer, then prove FINAL cannot claim a
    # different checkpoint identity from the already-published interval file.
    _write_joint_resume_artifacts(tmp_path, include_final=True)
    final = json.loads(final_path.read_text(encoding="utf-8"))
    final["checkpoint_sha256"] = "f" * 64
    final_path.write_text(json.dumps(final), encoding="utf-8")
    with pytest.raises(RuntimeError, match="FINAL is inconsistent"):
        joint_trainer._completed_resume_action(
            tmp_path, resume_path=checkpoint, global_step=25_000
        )


def test_joint_finalization_never_rewrites_the_interval_checkpoint() -> None:
    source = inspect.getsource(joint_trainer._finalize_joint_run)
    assert "_published_checkpoint_identity" in source
    assert "_save_checkpoint(" not in source


def test_joint_provenance_source_allowlists_cannot_drift() -> None:
    assert set(joint_trainer.JOINT_TRAINING_SOURCE_PATHS) == (
        EXPECTED_JOINT_TRAINING_SOURCE_PATHS
    )
    assert set(joint_selector.AUDITED_SOURCE_PATHS) == (
        EXPECTED_JOINT_SELECTION_SOURCE_PATHS
    )
    assert joint_selector.EXPECTED_JOINT_TRAINING_SOURCE_PATHS == (
        EXPECTED_JOINT_TRAINING_SOURCE_PATHS
    )
    lineage_source = (
        "scripts/t2a/train/"
        "sceneplan_transfusion_editing_joint_run_contract.py"
    )
    assert lineage_source in joint_trainer.JOINT_TRAINING_SOURCE_PATHS
    assert lineage_source in joint_selector.AUDITED_SOURCE_PATHS


def _plan(source_id: str = "source_2") -> dict:
    duration = round(4 * 1024 / 44_100, 6)
    return {
        "sample_id": "sample",
        "duration_sec": duration,
        "room": {"type": "dry"},
        "sources": [
            {
                "source_id": source_id,
                "kind": "sound",
                "description": "a test sound",
                "activity": {"onset_sec": 0.0, "offset_sec": duration},
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


def test_sparse_persistent_source_slot_activity_is_not_reindexed() -> None:
    plan = _plan("source_2")
    mask = audio_e2e._plan_activity_mask(
        plan, ["source_2"], frames=4, samples=4 * 1024
    )
    assert mask.tolist() == [True, True, True, True]
    assert not audio_e2e._plan_activity_mask(
        plan, ["source_0"], frames=4, samples=4 * 1024
    ).any()


def test_target_plan_demixer_recovers_two_known_foa_sources() -> None:
    plan = _plan("source_0")
    second = _plan("source_2")["sources"][0]
    second["trajectory"]["position"]["azimuth_deg"] = 90.0
    plan["sources"].append(second)
    source_ids, decoder, active, condition = audio_e2e._plan_demix_operator(
        plan, frames=4, samples=4 * 1024, ridge=1.0e-6
    )
    assert source_ids == ["source_0", "source_2"]
    assert active.all()
    assert max(condition.values()) < 5.0
    samples = 4 * 1024
    time = torch.arange(samples, dtype=torch.float32)
    stems = torch.stack((torch.sin(time * 0.011), torch.cos(time * 0.017)))
    steering = torch.tensor(
        [
            [2.0**-0.5, 2.0**-0.5],
            [0.0, 1.0],
            [0.0, 0.0],
            [1.0, 0.0],
        ]
    )
    audio = steering @ stems
    recovered = audio_e2e._apply_plan_demix(
        audio,
        decoder=decoder,
        active=active,
        frames=4,
        samples=samples,
    )
    assert torch.mean((recovered - stems) ** 2) < 1.0e-8


@pytest.mark.parametrize(
    "key",
    [
        "old_sceneplan",
        "old-plan",
        "source_sceneplan",
        "source-plan",
        "previous_sceneplan",
        "previous-plan",
    ],
)
def test_real_audio_model_call_rejects_every_old_plan_alias(key: str) -> None:
    with pytest.raises(RuntimeError, match="leaked"):
        audio_e2e._reject_model_key_leak({key: {}}, where="unit test")


def test_calibration_selection_is_exact_deterministic_5x2x100() -> None:
    rows = []
    ordinal = 0
    for operation in audio_e2e.OPERATIONS:
        for bucket in (432, 648):
            for index in range(101):
                rows.append(
                    {
                        "pair_ordinal": ordinal,
                        "pair_id": f"{operation}-{bucket}-{index}",
                        "operation": operation,
                        "latent_bucket_frames": bucket,
                    }
                )
                ordinal += 1
    selected, summary = audio_e2e._select_ordinals(rows, phase="calibration")
    repeated, repeated_summary = audio_e2e._select_ordinals(
        list(reversed(rows)), phase="calibration"
    )
    assert selected == repeated
    assert summary == repeated_summary
    assert len(selected) == 1_000
    assert len(set(selected)) == 1_000
    assert set(summary["by_operation_bucket"].values()) == {100}


def _quality_rows() -> list[dict]:
    rows = []
    ordinal = 0
    for operation in audio_e2e.OPERATIONS:
        for bucket in (432, 648):
            for _ in range(100):
                metrics = {
                    name: 0.9 if name.startswith("plan_") else 0.5
                    for name in audio_e2e.HIGHER_BETTER_SPECS
                    if name not in audio_e2e.AGGREGATE_ONLY_METRICS
                }
                metrics["plan_grammar_legal"] = 1.0
                metrics["unchanged_demix_total_sources"] = 4
                metrics["unchanged_demix_eligible_sources"] = 4
                metrics["unchanged_demix_eligibility_fraction"] = 1.0
                if operation not in {"event_addition", "event_removal"}:
                    # W is invariant for purely spatial edits, so normalized
                    # source->target W progress is either undefined or can be
                    # numerically unstable around a near-zero denominator.
                    metrics["audio_raw_w_progress"] = -100.0
                metrics.update(
                    {name: 0.1 for name in audio_e2e.LOWER_BETTER_SPECS}
                )
                rows.append(
                    {
                        "schema": audio_e2e.SCHEMA,
                        "schema_version": audio_e2e.SCHEMA_VERSION,
                        "status": "ok",
                        "pair_ordinal": ordinal,
                        "pair_id": f"pair-{ordinal:06d}",
                        "operation": operation,
                        "latent_bucket_frames": bucket,
                        "independent_content_metric_contract": (
                            audio_e2e.INDEPENDENT_CONTENT_METRIC_CONTRACT
                        ),
                        "model_input_contract": dict(
                            audio_e2e.MODEL_INPUT_CONTRACT
                        ),
                        "metrics": metrics,
                    }
                )
                ordinal += 1
    return rows


def test_calibration_thresholds_are_frozen_then_enforced() -> None:
    rows = _quality_rows()
    summaries = audio_e2e._all_metric_summaries(rows)
    thresholds, calibration_checks = audio_e2e._calibration_thresholds(summaries)
    assert calibration_checks and all(calibration_checks.values())
    assert set(thresholds["audio_raw_w_progress"]["by_operation"]) == {
        "event_removal",
    }
    assert set(thresholds["audio_raw_w_progress"]) == {
        "by_operation",
        "by_operation_bucket",
    }
    assert set(thresholds["audio_raw_w_progress"]["by_operation_bucket"]) == {
        "432:event_removal",
        "648:event_removal",
    }
    assert set(thresholds["unchanged_preservation_budget_ratio"]) == {
        "overall",
        "by_operation",
        "by_latent_bucket",
    }
    assert set(thresholds["latent_foa_progress"]) == {"by_operation", "by_operation_bucket"}
    assert set(thresholds["latent_foa_progress"]["by_operation"]) == set(audio_e2e.REFERENCE_DERIVED_OPERATIONS)
    assert set(thresholds["doa_spatial_progress"]["by_operation"]) == set(
        audio_e2e.SPATIAL_OPERATIONS
    )
    test_checks = audio_e2e._test_threshold_checks(summaries, thresholds)
    assert test_checks and all(test_checks.values())

    for row in rows:
        if row["operation"] == "event_removal":
            row["metrics"]["latent_foa_progress"] = -0.5
    degraded = audio_e2e._all_metric_summaries(rows)
    degraded_checks = audio_e2e._test_threshold_checks(degraded, thresholds)
    assert degraded_checks["latent_foa_progress:by_operation:event_removal"] is False

    semantic_rows = _quality_rows()
    for row in semantic_rows:
        if row["operation"] == "event_removal":
            row["metrics"]["independent_clap_edit_text_progress"] = -0.5
            row["metrics"]["removed_speech_recall_progress"] = -0.5
    semantic_summaries = audio_e2e._all_metric_summaries(semantic_rows)
    semantic_checks = audio_e2e._test_threshold_checks(
        semantic_summaries, thresholds
    )
    assert semantic_checks[
        "independent_clap_edit_text_progress:by_operation:event_removal"
    ] is False
    assert semantic_checks[
        "removed_speech_recall_progress:by_operation:event_removal"
    ] is False

    added_speech_rows = _quality_rows()
    for row in added_speech_rows:
        if row["operation"] == "event_addition":
            row["metrics"]["speech_addition_wer_progress"] = -0.5
            row["metrics"]["speech_addition_cer_progress"] = -0.5
    added_speech_checks = audio_e2e._test_threshold_checks(
        audio_e2e._all_metric_summaries(added_speech_rows), thresholds
    )
    assert added_speech_checks[
        "speech_addition_wer_progress:by_operation:event_addition"
    ] is False
    assert added_speech_checks[
        "speech_addition_cer_progress:by_operation:event_addition"
    ] is False

    for row in rows:
        if row["operation"] == "linear_to_static":
            row["metrics"]["unchanged_preservation_budget_ratio"] = 2.0
    degraded_preservation = audio_e2e._all_metric_summaries(rows)
    preservation_checks = audio_e2e._test_threshold_checks(
        degraded_preservation, thresholds
    )
    assert preservation_checks[
        "unchanged_preservation_budget_ratio:by_operation:linear_to_static"
    ] is False

    coverage_rows = _quality_rows()
    retained = 0
    for row in coverage_rows:
        if row["operation"] == "event_removal":
            retained += 1
            if retained > 10:
                row["metrics"]["change_direction_cosine_foa"] = None
    coverage_summaries = audio_e2e._all_metric_summaries(coverage_rows)
    coverage_checks = audio_e2e._test_threshold_checks(
        coverage_summaries, thresholds
    )
    assert coverage_checks[
        "change_direction_cosine_foa:by_operation:event_removal"
    ] is False


def test_calibration_rejects_vacuous_noop_edit_effects() -> None:
    rows = _quality_rows()
    for row in rows:
        for name in audio_e2e.HIGHER_BETTER_SPECS:
            if not name.startswith("plan_") and name != "unchanged_demix_si_sdr_delta_vs_copy_db":
                row["metrics"][name] = 1.0e-9
    _, checks = audio_e2e._calibration_thresholds(
        audio_e2e._all_metric_summaries(rows)
    )
    assert all(checks[f"latent_foa_progress:by_operation:{operation}"] is False
               for operation in audio_e2e.REFERENCE_DERIVED_OPERATIONS)
    assert checks["change_temporal_iou:overall:overall"] is False
    assert checks[
        "audio_raw_foa_progress:by_operation:stationary_spatial_relocation"
    ] is False


def test_calibration_replay_cannot_hide_a_failed_derived_quality_gate() -> None:
    assert not audio_e2e._calibration_checks_are_closed(
        {"forged_only": True},
        {"latent_foa_progress:overall:overall": False},
    )
    assert audio_e2e._calibration_checks_are_closed(
        {"structural": True, "latent_foa_progress:overall:overall": True},
        {"latent_foa_progress:overall:overall": True},
    )


@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_formal_audio_batch_sizes_are_sidecar_certified(batch_size: int) -> None:
    calibration = audio_e2e._batch_size_sidecar_value(
        phase="calibration", batch_size=batch_size
    )
    test = audio_e2e._batch_size_sidecar_value(
        phase="test", batch_size=batch_size
    )
    assert calibration["certified_batch_sizes_per_rank"] == [1, 2, 4]
    assert calibration["selected_batch_size_per_rank"] == batch_size
    assert test["selected_batch_size_per_rank"] == batch_size
    assert audio_e2e._require_matching_calibration_batch_size(
        {"batch_size_per_rank": batch_size}, test_batch_size=batch_size
    ) == batch_size


@pytest.mark.parametrize("batch_size", [0, 3, 5])
def test_uncertified_or_mismatched_audio_batch_size_is_rejected(
    batch_size: int,
) -> None:
    with pytest.raises(ValueError, match="certified"):
        audio_e2e._batch_size_sidecar_value(
            phase="calibration", batch_size=batch_size
        )
    with pytest.raises(RuntimeError, match="differs"):
        audio_e2e._require_matching_calibration_batch_size(
            {"batch_size_per_rank": 2}, test_batch_size=batch_size
        )
    with pytest.raises(RuntimeError, match="differs"):
        audio_e2e._require_matching_calibration_batch_size(
            {"batch_size_per_rank": 1}, test_batch_size=2
        )


def test_batch_size_sidecar_is_path_and_sha_bound(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    value = audio_e2e._batch_size_sidecar_value(phase="test", batch_size=2)
    path = run_dir / "BATCH_SIZE.json"
    audio_e2e._atomic_json(path, value)
    record = {
        "path": str(path.resolve(strict=True)),
        "sha256": sha256_file(path),
        "value": value,
    }
    assert audio_e2e._validated_batch_size_record(
        record,
        phase="test",
        batch_size=2,
        expected_parent=run_dir,
    ) == record

    outside = tmp_path / "BATCH_SIZE.json"
    audio_e2e._atomic_json(outside, value)
    outside_record = {
        "path": str(outside.resolve(strict=True)),
        "sha256": sha256_file(outside),
        "value": value,
    }
    with pytest.raises(RuntimeError, match="certification changed"):
        audio_e2e._validated_batch_size_record(
            outside_record,
            phase="test",
            batch_size=2,
            expected_parent=run_dir,
        )


def test_final_report_cannot_forge_replayed_derivations() -> None:
    structural = {"route": True}
    checks = {"route": True, "quality": True}
    summaries = {"metric": {"overall": {"mean": 0.5}}}
    thresholds = {"metric": {"overall": {"overall": {"direction": "higher"}}}}
    final = {
        "structural": structural,
        "checks": checks,
        "metric_summaries": summaries,
        "thresholds": thresholds,
    }
    audio_e2e._require_replayed_final_derivations(
        final,
        structural=structural,
        checks=checks,
        summaries=summaries,
        thresholds=thresholds,
    )
    forged = {**final, "checks": {"route": True, "quality": True, "forged": True}}
    with pytest.raises(RuntimeError, match="derived checks"):
        audio_e2e._require_replayed_final_derivations(
            forged,
            structural=structural,
            checks=checks,
            summaries=summaries,
            thresholds=thresholds,
        )
    with pytest.raises(RuntimeError, match="derived checks"):
        audio_e2e._require_replayed_final_derivations(
            final,
            structural=structural,
            checks={**checks, "quality": False},
            summaries=summaries,
            thresholds=thresholds,
        )


def test_calibration_and_test_reject_high_metric_abstention() -> None:
    good_rows = _quality_rows()
    good_summaries = audio_e2e._all_metric_summaries(good_rows)
    thresholds, checks = audio_e2e._calibration_thresholds(good_summaries)
    assert all(checks.values())

    sparse_calibration = _quality_rows()
    seen: dict[tuple[str, int], int] = {}
    for row in sparse_calibration:
        cell = (row["operation"], row["latent_bucket_frames"])
        seen[cell] = seen.get(cell, 0) + 1
        if seen[cell] > 20:
            row["metrics"]["change_temporal_iou"] = None
    sparse_summary = audio_e2e._all_metric_summaries(sparse_calibration)
    _, sparse_checks = audio_e2e._calibration_thresholds(sparse_summary)
    assert sparse_checks["change_temporal_iou:overall:overall"] is False
    assert sparse_summary["change_temporal_iou"]["overall"]["rows"] == 200
    assert sparse_summary["change_temporal_iou"]["overall"]["group_rows"] == 1_000
    assert sparse_summary["change_temporal_iou"]["overall"]["coverage_fraction"] == 0.2

    sparse_test = []
    for repeat in range(5):
        for row in _quality_rows():
            copied = {
                **row,
                "pair_ordinal": repeat * 1_000 + row["pair_ordinal"],
                "metrics": dict(row["metrics"]),
            }
            sparse_test.append(copied)
    seen.clear()
    for row in sparse_test:
        cell = (row["operation"], row["latent_bucket_frames"])
        seen[cell] = seen.get(cell, 0) + 1
        if seen[cell] > 20:
            row["metrics"]["change_temporal_iou"] = None
    test_checks = audio_e2e._test_threshold_checks(
        audio_e2e._all_metric_summaries(sparse_test), thresholds
    )
    assert test_checks["change_temporal_iou:overall:overall"] is False


def test_speech_addition_and_source_coverage_reject_abstention() -> None:
    good_rows = _quality_rows()
    thresholds, good_checks = audio_e2e._calibration_thresholds(
        audio_e2e._all_metric_summaries(good_rows)
    )
    assert all(good_checks.values())

    sparse = _quality_rows()
    seen: dict[tuple[str, int], int] = {}
    for row in sparse:
        cell = (row["operation"], row["latent_bucket_frames"])
        seen[cell] = seen.get(cell, 0) + 1
        if row["operation"] == "event_addition" and seen[cell] > 4:
            row["metrics"]["speech_addition_wer_progress"] = None
            row["metrics"]["speech_addition_cer_progress"] = None
        if seen[cell] > 4:
            row["metrics"]["unchanged_demix_eligible_sources"] = 0
            row["metrics"]["unchanged_demix_eligibility_fraction"] = 0.0

    summaries = audio_e2e._all_metric_summaries(sparse)
    _, calibration_checks = audio_e2e._calibration_thresholds(summaries)
    assert calibration_checks[
        "speech_addition_wer_progress:by_operation:event_addition"
    ] is False
    assert calibration_checks[
        "speech_addition_cer_progress:by_operation:event_addition"
    ] is False
    assert calibration_checks[
        "unchanged_demix_source_coverage:overall:overall"
    ] is False
    coverage = summaries["unchanged_demix_source_coverage"]["overall"]
    assert coverage["eligible_sources"] == 160
    assert coverage["total_sources"] == 4_000
    assert coverage["coverage_fraction"] == pytest.approx(0.04)

    test_checks = audio_e2e._test_threshold_checks(summaries, thresholds)
    assert test_checks[
        "speech_addition_wer_progress:by_operation:event_addition"
    ] is False
    assert test_checks[
        "unchanged_demix_source_coverage:overall:overall"
    ] is False


def test_unchanged_source_coverage_is_ratio_of_source_counts() -> None:
    rows = [
        {
            "operation": "event_addition",
            "latent_bucket_frames": 432,
            "metrics": {
                "unchanged_demix_eligible_sources": 1,
                "unchanged_demix_total_sources": 1,
            },
        },
        {
            "operation": "event_addition",
            "latent_bucket_frames": 432,
            "metrics": {
                "unchanged_demix_eligible_sources": 0,
                "unchanged_demix_total_sources": 9,
            },
        },
    ]
    overall = audio_e2e._unchanged_source_coverage_summary(rows)["overall"]
    assert overall["eligible_sources"] == 1
    assert overall["total_sources"] == 10
    assert overall["coverage_fraction"] == pytest.approx(0.1)
    assert overall["mean"] == pytest.approx(0.1)


def test_calibration_artifact_binds_threshold_inputs_and_sampling(
    tmp_path: Path,
) -> None:
    model_config = tmp_path / "model.json"
    model_config.write_text("{}\n", encoding="utf-8")
    codec = tmp_path / "codec"
    codec.mkdir()
    validation_index = tmp_path / "validation.sqlite"
    validation_index.touch()
    checkpoint = tmp_path / "step-00005000.pt"
    checkpoint.write_bytes(b"checkpoint")
    selection_path = tmp_path / "SELECTED.json"
    selection_path.write_text("{}\n", encoding="utf-8")
    selection_sha = sha256_file(selection_path)
    source_hashes = {"unit-test-source": "f" * 64}
    independent_content_assets = {
        "contract": "unit-test-independent-content",
        "offline_metric_only": True,
    }
    selection = {
        "selected_checkpoint": str(checkpoint),
        "selected_checkpoint_sha256": sha256_file(checkpoint),
        "codec": {"fingerprint": "codec-fingerprint"},
        "validation_index": {
            "path": str(validation_index),
            "sha256": sha256_file(validation_index),
        },
    }
    rows = _quality_rows()
    connection = sqlite3.connect(validation_index)
    try:
        connection.execute(
            "CREATE TABLE pairs (pair_ordinal INTEGER PRIMARY KEY, "
            "pair_id TEXT NOT NULL, operation TEXT NOT NULL, "
            "latent_bucket_frames INTEGER NOT NULL)"
        )
        connection.executemany(
            "INSERT INTO pairs VALUES (?,?,?,?)",
            [
                (
                    int(row["pair_ordinal"]),
                    str(row["pair_id"]),
                    str(row["operation"]),
                    int(row["latent_bucket_frames"]),
                )
                for row in rows
            ],
        )
        connection.commit()
    finally:
        connection.close()
    selection["validation_index"]["sha256"] = sha256_file(validation_index)
    summaries = audio_e2e._all_metric_summaries(rows)
    thresholds, quality_checks = audio_e2e._calibration_thresholds(summaries)
    assert all(quality_checks.values())
    row_selection = {
        "contract": "sha256_ranked_operation_bucket_5x2x100_v1",
        "rows": 1_000,
        "per_operation_bucket_cell": 100,
        "by_operation_bucket": {
            f"{bucket}:{operation}": 100
            for bucket in (432, 648)
            for operation in audio_e2e.OPERATIONS
        },
        "pair_ordinal_sha256": audio_e2e._ordinal_sha256(
            [int(row["pair_ordinal"]) for row in rows]
        ),
    }
    batch_size_value = audio_e2e._batch_size_sidecar_value(
        phase="calibration", batch_size=2
    )
    batch_size_path = tmp_path / "BATCH_SIZE.json"
    audio_e2e._atomic_json(batch_size_path, batch_size_value)
    batch_size_record = {
        "path": str(batch_size_path.resolve(strict=True)),
        "sha256": sha256_file(batch_size_path),
        "value": batch_size_value,
    }
    run_contract = {
        "phase": "calibration",
        "evaluation_contract": audio_e2e.EVALUATION_CONTRACT,
        "checkpoint_selection_sha256": selection_sha,
        "index": {"sha256": sha256_file(validation_index)},
        "row_selection": row_selection,
        "ode_steps": 20,
        "cfg_scale": 1.0,
        "max_plan_tokens": 512,
        "seed": 42,
        "batch_size_per_rank": 2,
        "batch_size_certification": batch_size_record,
        "unchanged_source_demix": audio_e2e._demix_contract(),
        "independent_content_metric_contract": (
            audio_e2e.INDEPENDENT_CONTENT_METRIC_CONTRACT
        ),
        "independent_content_metric_assets": independent_content_assets,
    }
    run_contract_path = tmp_path / "CONTRACT.json"
    run_contract_path.write_text(json.dumps(run_contract), encoding="utf-8")
    row_records = tmp_path / "ROWS.jsonl"
    audio_e2e._atomic_jsonl(row_records, rows)
    artifact = {
        "schema": audio_e2e.CALIBRATION_SCHEMA,
        "schema_version": 1,
        "status": "PASS",
        "calibration_contract": audio_e2e.CALIBRATION_CONTRACT,
        "evaluation_contract": audio_e2e.EVALUATION_CONTRACT,
        "rows": 1_000,
        "successful_rows": 1_000,
        "checks": quality_checks,
        "quality_checks": quality_checks,
        "thresholds": thresholds,
        "thresholds_sha256": audio_e2e._canonical_sha256(thresholds),
        "metric_summaries": summaries,
        "metric_summaries_sha256": audio_e2e._canonical_sha256(summaries),
        "joint_checkpoint_selection": str(selection_path),
        "joint_checkpoint_selection_sha256": selection_sha,
        "selected_checkpoint": str(checkpoint),
        "selected_checkpoint_sha256": sha256_file(checkpoint),
        "model_config": str(model_config),
        "model_config_sha256": sha256_file(model_config),
        "codec": str(codec),
        "codec_fingerprint": "codec-fingerprint",
        "validation_index": {
            "path": str(validation_index),
            "sha256": sha256_file(validation_index),
            "rows": 20_000,
            "split": "validation",
        },
        "row_selection": row_selection,
        "batch_size_per_rank": 2,
        "batch_size_certification": batch_size_record,
        "ode_steps": 20,
        "cfg_scale": 1.0,
        "max_plan_tokens": 512,
        "seed": 42,
        "per_row_seed_contract": (
            "blake2b_pair_id_namespace_rank_batch_invariant_v1"
        ),
        "frozen_vae_config_sha256": audio_e2e.FROZEN_VAE_CONFIG_SHA256,
        "frozen_vae_checkpoint_sha256": audio_e2e.FROZEN_VAE_CHECKPOINT_SHA256,
        "frozen_qwen_runtime": audio_e2e.verify_frozen_qwen_runtime(),
        "source_sha256": source_hashes,
        "unchanged_source_demix": audio_e2e._demix_contract(),
        "independent_content_metric_contract": (
            audio_e2e.INDEPENDENT_CONTENT_METRIC_CONTRACT
        ),
        "independent_content_metric_assets": independent_content_assets,
        "run_contract": str(run_contract_path),
        "run_contract_sha256": sha256_file(run_contract_path),
        "row_records": str(row_records),
        "row_records_sha256": sha256_file(row_records),
    }
    calibration = tmp_path / "CALIBRATION.json"
    calibration.write_text(json.dumps(artifact), encoding="utf-8")
    assert audio_e2e._validate_calibration(
        calibration,
        expected_sha=sha256_file(calibration),
        selection=selection,
        selection_path=selection_path,
        selection_sha=selection_sha,
        model_config=model_config,
        codec=codec,
        source_hashes=source_hashes,
        independent_content_assets=independent_content_assets,
    )["seed"] == 42

    tampered_rows = [
        {**row, "metrics": dict(row["metrics"])} for row in rows
    ]
    tampered_rows[0]["metrics"]["latent_foa_progress"] = 0.123
    audio_e2e._atomic_jsonl(row_records, tampered_rows)
    artifact["row_records_sha256"] = sha256_file(row_records)
    calibration.write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises(RuntimeError, match="stale"):
        audio_e2e._validate_calibration(
            calibration,
            expected_sha=sha256_file(calibration),
            selection=selection,
            selection_path=selection_path,
            selection_sha=selection_sha,
            model_config=model_config,
            codec=codec,
            source_hashes=source_hashes,
            independent_content_assets=independent_content_assets,
        )
    audio_e2e._atomic_jsonl(row_records, rows)
    artifact["row_records_sha256"] = sha256_file(row_records)

    artifact["seed"] = 41
    calibration.write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises(RuntimeError, match="stale"):
        audio_e2e._validate_calibration(
            calibration,
            expected_sha=sha256_file(calibration),
            selection=selection,
            selection_path=selection_path,
            selection_sha=selection_sha,
            model_config=model_config,
            codec=codec,
            source_hashes=source_hashes,
            independent_content_assets=independent_content_assets,
        )


def _free_records() -> list[dict]:
    rows = []
    ordinal = 0
    for operation in audio_e2e.OPERATIONS:
        for bucket in (432, 648):
            for _ in range(50):
                rows.append(
                    {
                        "status": "ok",
                        "pair_ordinal": ordinal,
                        "operation": operation,
                        "latent_bucket_frames": bucket,
                        "predicted_token_ids": [1, 7, 2],
                        "metrics": {
                            "grammar_legal": 1.0,
                            **{
                                name: 1.0
                                for name in joint_selector.FREE_FLOORS
                            },
                        },
                    }
                )
                ordinal += 1
    return rows


def test_joint_free_gate_requires_exact_bos_and_eos() -> None:
    records = _free_records()
    assert joint_selector._free_gate(records, bos_id=1, eos_id=2)["pass"] is True
    records[0]["predicted_token_ids"][-1] = 3
    gate = joint_selector._free_gate(records, bos_id=1, eos_id=2)
    assert gate["checks"]["all_eos_within_512"] is False
    assert gate["pass"] is False


def test_formal_audio_launcher_has_no_latest_checkpoint_fallback() -> None:
    launcher = (
        REPO_ROOT
        / "scripts/t2a/eval/run_sceneplan_transfusion_editing_end_to_end_5gpu.sh"
    ).read_text(encoding="utf-8")
    assert "LATEST.json" not in launcher
    assert "JOINT_CHECKPOINT_SELECTION" in launcher
    assert "--phase calibration" in launcher
    assert "--phase test" in launcher
    assert "--save-all-audio" in launcher
    assert "validate_published_joint_selection" in launcher
    assert 'FORMAL_BATCH_SIZE="${AUDIO_E2E_BATCH_SIZE:-2}"' in launcher
    assert launcher.count('--batch-size "$FORMAL_BATCH_SIZE"') == 2
    assert launcher.count("validate_audio_e2e_final(") == 2
    assert '"$TEST_DIR/FINAL.json"' in launcher
    assert 'final_path.parent / "SEALED.json"' in launcher
    assert 'exec "$REPO/.venv/bin/torchrun"' not in launcher
    assert launcher.index("validate_published_joint_selection(") < launcher.index(
        "--phase calibration"
    )


def test_formal_audio_test_index_is_opened_only_after_calibration_replay() -> None:
    launcher = (
        REPO_ROOT
        / "scripts/t2a/eval/run_sceneplan_transfusion_editing_end_to_end_5gpu.sh"
    ).read_text(encoding="utf-8")
    assert launcher.index("_validate_calibration(") < launcher.index(
        "test_sha=sha256_file(test)"
    )
    assert launcher.index(
        "validate_audio_e2e_final(\n    calibration_final"
    ) < launcher.index("test_sha=sha256_file(test)")
    assert launcher.index('if [[ "$calibration_status" != "0" ]]') < launcher.index(
        'if [[ ! -r "$TEST_INDEX" ]]'
    )

    main_source = inspect.getsource(audio_e2e.main)
    assert main_source.index("calibration = _validate_calibration(") < (
        main_source.index("index = args.index.expanduser().resolve(strict=True)")
    )


def test_final_seal_replays_all_critical_artifact_classes() -> None:
    source = inspect.getsource(audio_e2e.validate_audio_e2e_final)
    for required in (
        "validate_published_joint_selection",
        "_index_summary",
        "_validated_metric_row_records",
        "_validated_rank_records",
        "_validated_batch_size_record",
        "_source_sha256",
        "verify_independent_content_metric_assets",
        "verify_editing_m2d_clap_assets",
        "verify_frozen_qwen_runtime",
        "_validate_calibration",
        "_structural_report",
        "_evaluation_checks",
        "_require_replayed_final_derivations",
        'final_path.parent / "SEALED.json"',
    ):
        assert required in source
    main_source = inspect.getsource(audio_e2e.main)
    assert "validate_audio_e2e_final(" in main_source
    assert "write_seal=True" in main_source


def test_resumable_audio_batch_rejects_identity_tampering() -> None:
    record = _quality_rows()[0]
    expected = {
        "pair_ordinal": record["pair_ordinal"],
        "pair_id": record["pair_id"],
        "operation": record["operation"],
        "latent_bucket_frames": record["latent_bucket_frames"],
    }
    audio_e2e._validate_batch_records(
        [record], [expected], bucket=int(record["latent_bucket_frames"])
    )
    changed = {**record, "pair_id": "wrong-pair"}
    with pytest.raises(RuntimeError, match="identity/contract"):
        audio_e2e._validate_batch_records(
            [changed], [expected], bucket=int(record["latent_bucket_frames"])
        )


def test_formal_stage_handoffs_forward_selected_artifact_paths() -> None:
    dit_launcher = (
        REPO_ROOT
        / "scripts/t2a/train/run_sceneplan_transfusion_editing_dit_full_5gpu.sh"
    ).read_text(encoding="utf-8")
    joint_launcher = (
        REPO_ROOT
        / "scripts/t2a/train/run_sceneplan_transfusion_editing_ar_joint_full_5gpu.sh"
    ).read_text(encoding="utf-8")
    selector_launcher = (
        REPO_ROOT
        / "scripts/t2a/eval/"
        "run_sceneplan_transfusion_editing_joint_checkpoint_selection_5gpu.sh"
    ).read_text(encoding="utf-8")

    assert 'BASE_DIT_RUN="$RUN_ROOT"' in dit_launcher
    assert 'DIT_CHECKPOINT_SELECTION="$DIT_SELECTION_OUTPUT"' in dit_launcher
    assert 'TRAIN_RUN_CONTRACT="$RUN_ROOT/TRAIN_RUN_CONTRACT.json"' in dit_launcher
    assert 'export SAT_EDITING_RUN_CONTRACT_PATH="$TRAIN_RUN_CONTRACT"' in dit_launcher
    assert "resolve-resume --contract" in dit_launcher
    assert 'RESUME_CKPT="${resume_resolution[1]}"' in dit_launcher
    assert 'recovery-quarantine-' in dit_launcher
    assert 'mv -- "$RUN_ROOT/checkpoints/last.ckpt"' in dit_launcher
    assert 'cp --reflink=auto --preserve=mode,timestamps --' in dit_launcher
    assert '"$RUN_ROOT/checkpoints/last.ckpt"' in dit_launcher
    assert 'JOINT_RUN="$RUN_DIR"' in joint_launcher
    assert 'DIT_CHECKPOINT_SELECTION="$DIT_CHECKPOINT_SELECTION"' in joint_launcher
    assert 'JOINT_RUN="$RUN_DIR"' in selector_launcher
    assert 'JOINT_CHECKPOINT_SELECTION="$OUTPUT"' in selector_launcher


def test_checkpoint_reuse_and_joint_resume_are_fail_closed() -> None:
    dit_selector = (
        REPO_ROOT
        / "scripts/t2a/eval/"
        "select_sceneplan_transfusion_editing_dit_checkpoint.py"
    ).read_text(encoding="utf-8")
    joint_selector_source = (
        REPO_ROOT
        / "scripts/t2a/eval/"
        "select_sceneplan_transfusion_editing_joint_checkpoint.py"
    ).read_text(encoding="utf-8")
    joint_launcher = (
        REPO_ROOT
        / "scripts/t2a/train/"
        "run_sceneplan_transfusion_editing_ar_joint_full_5gpu.sh"
    ).read_text(encoding="utf-8")
    pipeline_source = (
        REPO_ROOT
        / "stable_audio_tools/models/sceneplan_transfusion_editing_pipeline.py"
    ).read_text(encoding="utf-8")
    joint_trainer_source = inspect.getsource(joint_trainer.main)
    assert "current_candidate_sha256" in dit_selector
    assert "current_candidate_sha256" in joint_selector_source
    assert '== sha256_file(candidate_path)' in pipeline_source
    assert 'value.get("selection_ranked_steps")' in joint_selector_source
    assert "recomputed_free_gate" in joint_selector_source
    assert "sceneplan_transfusion_editing_joint_run_contract.py" in joint_launcher
    assert "resolve-resume" in joint_launcher
    assert "--requested-checkpoint" in joint_launcher
    assert 'case "${resume_resolution[0]}"' in joint_launcher
    assert "completed joint Editing run cannot roll back" in joint_trainer_source


def test_real_audio_pipeline_api_cannot_accept_old_plan() -> None:
    parameters = inspect.signature(ScenePlanTransfusionEditingPipeline.edit_audio).parameters
    assert "old_sceneplan" not in parameters
    assert "source_sceneplan" not in parameters
    assert "edit_instructions" in parameters
    assert "source_foa" in parameters

    loader_parameters = inspect.signature(
        load_sceneplan_transfusion_editing_pipeline
    ).parameters
    assert loader_parameters["allow_unselected_diagnostic"].default is False
    loader_source = inspect.getsource(load_sceneplan_transfusion_editing_pipeline)
    assert "formal Editing pipeline loading requires a selected checkpoint" in loader_source
    assert "requires a pinned selection SHA256" in loader_source


def test_independent_target_metrics_run_only_after_model_inference() -> None:
    source = inspect.getsource(audio_e2e._process_batch)
    assert source.index("pipeline.edit_audio") < source.index(
        "content_evaluator.score_batch"
    )
    model_call = source[
        source.index("model_inputs = {") : source.index(
            "_reject_model_key_leak(model_inputs"
        )
    ]
    assert "target_foa" not in model_call
    assert "old_sceneplan" not in model_call
