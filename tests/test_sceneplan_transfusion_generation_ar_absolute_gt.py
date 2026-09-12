from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from scripts.t2a.eval import (
    queue_sceneplan_transfusion_generation_ar_absolute_8k as queue,
)
from scripts.t2a.eval import (
    score_sceneplan_transfusion_generation_ar_absolute_8k as scorer,
)


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, values) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"".join(scorer._canonical_bytes(value) + b"\n" for value in values)
    )


def test_metric_loader_uses_w_for_added_native_foa_system(monkeypatch):
    waveform = torch.tensor(
        [[0.25, -0.5], [0.5, 0.5], [-0.25, 0.25], [1.0, -1.0]],
        dtype=torch.float32,
    )
    monkeypatch.setattr(
        scorer.content.torchaudio,
        "load",
        lambda _path: (waveform.clone(), 48_000),
    )
    inputs = {
        "benchmark_kind": "internal8k",
        "native_foa_system_ids": {scorer.GENERATION_AR_ID},
    }
    actual = scorer.content._load_metric_mono(
        inputs,
        f"{scorer.GENERATION_AR_ID}:panel",
        "unused.wav",
        48_000,
    )
    expected = waveform[:1] / waveform[:1].abs().max() * (10.0 ** (-1.0 / 20.0))
    arithmetic_mean = waveform.mean(dim=0, keepdim=True)
    arithmetic_mean = (
        arithmetic_mean
        / arithmetic_mean.abs().max()
        * (10.0 ** (-1.0 / 20.0))
    )
    assert torch.equal(actual, expected)
    assert not torch.equal(actual, arithmetic_mean)


def test_load_inputs_binds_exact_render_rows_to_frozen_panel(tmp_path, monkeypatch):
    monkeypatch.setattr(scorer, "EXPECTED_ROWS", 2)
    evaluation_root = tmp_path / "absolute"
    baseline_root = tmp_path / "baseline"
    benchmark_root = baseline_root / scorer.BASELINE_BENCHMARK_DIR
    benchmark_root.mkdir(parents=True)
    evaluation_root.mkdir()

    benchmark_contract = benchmark_root / "BENCHMARK_CONTRACT.json"
    panel_path = baseline_root / "panel.jsonl"
    benchmark_contract.write_text("{}\n", encoding="utf-8")
    panel_path.write_text("panel\n", encoding="utf-8")
    reference = tmp_path / "reference.wav"
    prediction = tmp_path / "prediction.wav"
    for path in (reference, prediction):
        path.write_bytes(path.name.encode("utf-8"))

    panel = [
        {
            "ordinal": ordinal,
            "panel_id": f"panel-{ordinal}",
            "sample_id": f"sample-{ordinal}",
            "source_count": 1,
            "source_kinds": ["speech"],
            "noise_seed": 100 + ordinal,
            "reference_foa_path": str(reference.resolve()),
            "reference_foa_sha256": "reference-sha",
            "model_num_samples": 64,
            "latent_frames_valid": 2,
        }
        for ordinal in range(2)
    ]
    base = {
        "benchmark_kind": "internal8k",
        "root": benchmark_root,
        "contract": {},
        "contract_path": benchmark_contract.resolve(),
        "manifest_path": None,
        "panel_path": panel_path.resolve(),
        "panel_meta": {
            row["panel_id"]: {
                "source_count": 1,
                "source_kinds": ("speech",),
                "semantic_prompt": "speech",
                "reference_path": str(reference.resolve()),
                "reference_sha256": "reference-sha",
                "transcript": "hello",
                "speech_seen_speaker": True,
                "length_bucket": 2,
            }
            for row in panel
        },
        "reference_paths": {
            row["panel_id"]: str(reference.resolve()) for row in panel
        },
        "all_domain_ids": {
            "music": set(),
            "sound": set(),
            "speech": {row["panel_id"] for row in panel},
        },
    }
    monkeypatch.setattr(
        scorer,
        "_load_frozen_panel_inputs",
        lambda _baseline: (base, panel),
    )

    output_rows = [
        {
            "ordinal": row["ordinal"],
            "panel_id": row["panel_id"],
            "sample_id": row["sample_id"],
            "source_count": row["source_count"],
            "source_kinds": row["source_kinds"],
            "noise_seed": row["noise_seed"],
            "reference_foa_path": str(reference.resolve()),
            "reference_foa_sha256": "reference-sha",
            "reference_samples": 64,
            "reference_latent_frames": 2,
            "generation_ar_foa_path": str(prediction.resolve()),
        }
        for row in panel
    ]
    manifest_path = evaluation_root / "OUTPUT_MANIFEST.jsonl"
    _write_jsonl(manifest_path, output_rows)
    run = {
        "contract": "p10v11_generation_ar_absolute_gt_8k_render_ar_only_v1",
        "rows": 2,
        "baseline_panel": {
            "root": str(baseline_root.resolve()),
            "benchmark_contract": str(benchmark_contract.resolve()),
            "benchmark_contract_sha256": scorer.sha256_file(benchmark_contract),
            "panel": str(panel_path.resolve()),
            "panel_sha256": scorer.sha256_file(panel_path),
        },
        "plan_evaluation": {
            "generation_ar_checkpoint": "/checkpoint.pt",
            "generation_ar_checkpoint_sha256": "checkpoint-sha",
            "generation_ar_checkpoint_step": 123,
        },
    }
    run_path = evaluation_root / "RUN_CONTRACT.json"
    _write_json(run_path, run)
    summary = {
        "status": "PASS",
        "rows": 2,
        "run_contract_canonical_sha256": scorer._canonical_sha256(run),
        "output_manifest_sha256": scorer.sha256_file(manifest_path),
    }
    _write_json(evaluation_root / "SUMMARY.json", summary)
    (evaluation_root / "GENERATION_COMPLETE").write_text("PASS\n", encoding="utf-8")

    inputs = scorer._load_inputs(evaluation_root, baseline_root)
    assert inputs["domain_system_ids"]["speech"][scorer.GENERATION_AR_ID] == {
        "panel-0",
        "panel-1",
    }
    assert set(inputs["paths"]) == {
        scorer.REFERENCE_ID,
        scorer.GENERATION_AR_ID,
    }
    assert inputs["absolute_identity"]["generation_ar_checkpoint_step"] == 123
    assert inputs["absolute_identity"]["metric_source"]

    output_rows[1]["noise_seed"] += 1
    _write_jsonl(manifest_path, output_rows)
    summary["output_manifest_sha256"] = scorer.sha256_file(manifest_path)
    _write_json(evaluation_root / "SUMMARY.json", summary)
    with pytest.raises(RuntimeError, match="render/panel mismatch"):
        scorer._load_inputs(evaluation_root, baseline_root)


def test_system_gap_reports_signed_deltas_and_safe_ratios():
    def system(scale: float):
        return {
            "music": {
                "clap_text_audio": 0.0,
                "paired_reference_clap": 0.5 * scale,
                "fad_vggish": 1.0 * scale,
                "kl_pann": 0.2 * scale,
                "generated_reference_doa_error_deg": 20.0 * scale,
                "activity_iou": 0.8 * scale,
            },
            "sound": {
                "clap_text_audio": 0.1 * scale,
                "paired_reference_clap": 0.4 * scale,
                "fad_vggish": 1.2 * scale,
                "kl_pann": 0.3 * scale,
                "generated_reference_doa_error_deg": 30.0 * scale,
                "activity_iou": 0.7 * scale,
            },
            "speech": {
                "corpus_wer": 0.2 * scale,
                "corpus_cer": 0.1 * scale,
                "utmos": 2.0 * scale,
                "generated_reference_doa_error_deg": 25.0 * scale,
                "activity_iou": 0.75 * scale,
            },
        }

    gap = scorer._system_gap(system(2.0), system(1.0))
    assert gap["music"]["clap_retention"] is None
    assert gap["music"]["paired_reference_clap_retention"] == pytest.approx(2.0)
    assert gap["sound"]["clap_delta"] == pytest.approx(0.1)
    assert gap["speech"]["wer_excess"] == pytest.approx(0.2)


def test_queue_reuses_only_matching_checkpoint_and_plan_evaluation(tmp_path):
    output_dir = tmp_path / "output"
    comparison = output_dir / "metrics/absolute_gt/COMPARISON.json"
    render_contract = output_dir / "RUN_CONTRACT.json"
    test_summary = tmp_path / "test8k" / "SUMMARY.json"
    _write_json(test_summary, {"status": "PASS", "rows": 8000})
    _write_json(
        render_contract,
        {
            "plan_evaluation": {
                "summary": str(test_summary.resolve()),
                "summary_sha256": queue._sha256(test_summary),
            }
        },
    )
    _write_json(
        comparison,
        {
            "status": "PASS",
            "absolute_gt_identity": {
                "generation_ar_checkpoint_sha256": "checkpoint-sha",
                "render_run_contract": str(render_contract.resolve()),
            },
        },
    )
    (output_dir / "ABSOLUTE_GT_EVALUATION_COMPLETE").write_text(
        "PASS\n", encoding="utf-8"
    )
    assert queue._completed_comparison(
        output_dir,
        checkpoint_sha256="checkpoint-sha",
        test_summary=test_summary.resolve(),
    ) == comparison.resolve()
    assert (
        queue._completed_comparison(
            output_dir,
            checkpoint_sha256="different-checkpoint",
            test_summary=test_summary.resolve(),
        )
        is None
    )
