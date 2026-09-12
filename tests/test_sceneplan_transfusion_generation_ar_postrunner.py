from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace


_POSTRUNNER_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts/t2a/test/run_sceneplan_transfusion_generation_ar_posttrain.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "sceneplan_transfusion_generation_ar_postrunner", _POSTRUNNER_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
_POSTRUNNER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_POSTRUNNER)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_completed_plan_evaluation_validation_is_physically_read_only(
    tmp_path: Path,
) -> None:
    execution_root = tmp_path / "source"
    execution_root.mkdir()
    source = execution_root / "evaluator.py"
    source.write_text("frozen = True\n", encoding="utf-8")
    manifest = tmp_path / "test.sqlite"
    manifest.write_bytes(b"immutable manifest fixture")
    selection = tmp_path / "CHECKPOINT_SELECTION.json"
    selection.write_text('{"status":"COMPLETE"}\n', encoding="utf-8")
    output = tmp_path / "evaluation"
    predictions = output / "predictions"
    predictions.mkdir(parents=True)
    teacher = {"loss": 0.25}
    contract = {
        "schema_version": 2,
        "checkpoint_sha256": "checkpoint-sha",
        "evaluation_split": "test",
        "row_limit": 3,
        "evaluation_manifest": {"path": str(manifest)},
        "evaluation_manifest_sha256": _sha256(manifest),
        "training_completion_proof": {"selection_manifest_sha256": _sha256(selection)},
        "source_sha256": {"evaluator.py": _sha256(source)},
    }
    (output / "RUN_CONTRACT.json").write_text(json.dumps(contract), encoding="utf-8")
    (output / "TEACHER_FORCED.json").write_text(json.dumps(teacher), encoding="utf-8")
    (output / "SUMMARY.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "checkpoint_sha256": "checkpoint-sha",
                "evaluation_split": "test",
                "rows": 3,
                "teacher_forced": teacher,
            }
        ),
        encoding="utf-8",
    )
    contract_sha = _POSTRUNNER._canonical_sha256(contract)
    shard_paths = []
    for rank in range(3):
        path = predictions / f"rank_{rank:03d}.sqlite"
        connection = sqlite3.connect(path)
        connection.execute("CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT)")
        connection.executemany(
            "INSERT INTO metadata(key,value) VALUES (?,?)",
            (("status", "COMPLETE"), ("run_contract_sha256", contract_sha)),
        )
        connection.execute(
            "CREATE TABLE predictions(ordinal INTEGER PRIMARY KEY,status TEXT)"
        )
        connection.execute(
            "INSERT INTO predictions(ordinal,status) VALUES (?,?)", (rank, "ok")
        )
        connection.commit()
        connection.close()
        shard_paths.append(path)
    before = [_sha256(path) for path in shard_paths]

    summary = _POSTRUNNER._validated_plan_evaluation(
        output_dir=output,
        split="test",
        rows=3,
        manifest=manifest.resolve(),
        checkpoint_sha256="checkpoint-sha",
        selection_path=selection,
        execution_root=execution_root,
    )

    assert summary["status"] == "PASS"
    assert [_sha256(path) for path in shard_paths] == before


def test_multiepoch_checkpoint_schedule_is_every_half_epoch() -> None:
    contract = {
        "checkpoint_policy": "interval",
        "epochs_requested": 5,
        "requested_steps": 50,
        "steps_per_epoch": 10,
        "save_every_steps": 5,
    }
    assert _POSTRUNNER._expected_checkpoint_steps(contract) == list(range(5, 51, 5))


def test_continuation_selection_includes_parent_and_new_half_epochs() -> None:
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
    assert _POSTRUNNER._expected_checkpoint_steps(contract) == list(range(55, 101, 5))
    assert _POSTRUNNER._expected_selection_steps(contract) == [
        50,
        *range(55, 101, 5),
    ]


def test_parent_reuse_revalidates_plan_and_audio(
    tmp_path: Path, monkeypatch
) -> None:
    run_dir = tmp_path / "stage2"
    run_dir.mkdir()
    parent_run = tmp_path / "parent"
    (parent_run / "posttrain").mkdir(parents=True)
    execution_root = tmp_path / "stage2_source"
    audio_execution_root = tmp_path / "stage2_audio_source"
    parent_execution_root = tmp_path / "parent_source"
    parent_audio_execution_root = tmp_path / "parent_audio_source"
    venv_root = tmp_path / "venv"
    for root in (
        execution_root,
        audio_execution_root,
        parent_execution_root,
        parent_audio_execution_root,
    ):
        root.mkdir()
    for path in (
        execution_root
        / "scripts/t2a/test/select_sceneplan_transfusion_generation_ar_checkpoint.py",
        execution_root
        / "scripts/t2a/test/evaluate_sceneplan_transfusion_generation_ar_8k.py",
        audio_execution_root
        / "scripts/t2a/test/evaluate_sceneplan_transfusion_generation_ar_p10_audio_8k.py",
        venv_root / "bin/torchrun",
        venv_root / "bin/python",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture\n", encoding="utf-8")

    checkpoint = parent_run / "checkpoints/step_00000010.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"parent checkpoint")
    checkpoint_sha = _sha256(checkpoint)
    (run_dir / "RUN_CONTRACT.json").write_text(
        json.dumps(
            {
                "requested_steps": 20,
                "parent_lineage": {"run_dir": str(parent_run)},
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "FINAL.json").write_text(
        json.dumps({"event": "complete", "mode": "full", "step": 20}),
        encoding="utf-8",
    )
    stage_selection = run_dir / "CHECKPOINT_SELECTION.json"
    stage_selection.write_text("{}", encoding="utf-8")
    parent_selection = parent_run / "CHECKPOINT_SELECTION.json"
    parent_selection.write_text("{}", encoding="utf-8")
    smoke_summary = parent_run / "evaluation/smoke/SUMMARY.json"
    test_summary = parent_run / "evaluation/test8k/SUMMARY.json"
    audio_summary = parent_run / "evaluation/audio/SUMMARY.json"
    for path in (smoke_summary, test_summary, audio_summary):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
    parent_status_path = parent_run / "posttrain/POSTTRAIN_STATUS.json"
    parent_status = {
        "status": "GENERATION_EVALUATION_COMPLETE",
        "execution_root": str(parent_execution_root),
        "audio_execution_root": str(parent_audio_execution_root),
        "selected_checkpoint_sha256": checkpoint_sha,
        "selection_manifest": str(parent_selection),
        "selection_manifest_sha256": _sha256(parent_selection),
        "smoke_summary": str(smoke_summary),
        "smoke_summary_sha256": _sha256(smoke_summary),
        "test8k_summary": str(test_summary),
        "test8k_summary_sha256": _sha256(test_summary),
        "p10_audio8k_summary": str(audio_summary),
        "p10_audio8k_summary_sha256": _sha256(audio_summary),
    }
    parent_status_path.write_text(json.dumps(parent_status), encoding="utf-8")
    validation_manifest = tmp_path / "validation.sqlite"
    test_manifest = tmp_path / "test.sqlite"
    validation_manifest.write_bytes(b"validation")
    test_manifest.write_bytes(b"test")
    evaluation_root = tmp_path / "evaluation"
    args = SimpleNamespace(
        run_dir=run_dir,
        evaluation_root=evaluation_root,
        execution_root=execution_root,
        audio_execution_root=audio_execution_root,
        venv_root=venv_root,
        validation_manifest=validation_manifest,
        test_manifest=test_manifest,
        poll_seconds=1,
    )
    monkeypatch.setattr(_POSTRUNNER, "_parse_args", lambda: args)
    monkeypatch.setattr(_POSTRUNNER, "_training_is_running", lambda _path: False)

    selection_calls = []

    def validated_selection(**kwargs):
        selection_calls.append(kwargs)
        if kwargs["run_dir"] == run_dir.resolve():
            return {
                "selected_checkpoint": str(checkpoint),
                "selected_checkpoint_sha256": checkpoint_sha,
                "selected_lineage_role": "parent",
            }
        return {
            "selected_checkpoint": str(checkpoint),
            "selected_checkpoint_sha256": checkpoint_sha,
        }

    plan_calls = []
    stage_calls = []
    audio_calls = []
    monkeypatch.setattr(_POSTRUNNER, "_validated_selection", validated_selection)
    monkeypatch.setattr(
        _POSTRUNNER,
        "_validated_plan_evaluation",
        lambda **kwargs: plan_calls.append(kwargs) or {"status": "PASS"},
    )
    monkeypatch.setattr(
        _POSTRUNNER,
        "_run_stage",
        lambda **kwargs: stage_calls.append(kwargs),
    )
    monkeypatch.setattr(
        _POSTRUNNER,
        "_validated_audio_evaluation",
        lambda **kwargs: audio_calls.append(kwargs) or {"status": "PASS"},
    )

    assert _POSTRUNNER.main() == 0
    assert [call["run_dir"] for call in selection_calls] == [
        run_dir.resolve(),
        parent_run.resolve(),
    ]
    assert plan_calls[0]["output_dir"] == test_summary.parent
    assert plan_calls[0]["execution_root"] == parent_execution_root.resolve()
    assert stage_calls[0]["name"] == "verify_retained_parent_p10_audio_8k"
    command = stage_calls[0]["command"]
    assert command[-1] == "--verify-only"
    assert command[command.index("--source-root") + 1] == str(
        parent_audio_execution_root.resolve()
    )
    assert audio_calls[0]["output_dir"] == audio_summary.parent
    completed = json.loads(
        (run_dir / "posttrain/POSTTRAIN_STATUS.json").read_text(encoding="utf-8")
    )
    assert completed["status"] == "GENERATION_EVALUATION_COMPLETE"
    assert completed["evaluation_reused_from_parent"] is True
