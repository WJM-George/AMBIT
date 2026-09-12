from copy import deepcopy
import inspect
import json

import pytest
import torch

from scripts.t2a.eval import evaluate_sceneplan_transfusion_editing_gt_audio as gate
from stable_audio_tools.models.sceneplan_transfusion_editing_pipeline import (
    ScenePlanTransfusionEditingDiTPipeline, ScenePlanTransfusionEditingPipeline,
)


def _rows():
    rows = []
    for operation in gate.audio.OPERATIONS:
        for bucket in (432, 648):
            for _ in range(100):
                metrics = {name: max(1.0, spec["floor"] + 1) for name, spec in gate.HIGHER.items()
                           if name not in gate.audio.AGGREGATE_ONLY_METRICS}
                metrics.update({name: spec["ceiling"] - 1 for name, spec in gate.LOWER.items()})
                metrics.update(unchanged_demix_total_sources=2, unchanged_demix_eligible_sources=2)
                rows.append({"status": "ok", "pair_ordinal": len(rows), "operation": operation,
                    "latent_bucket_frames": bucket, "metrics": metrics,
                    "difficulty": {"source_count": 2, "unchanged_source_count": 1, "domain_pair": "speech->sound"}})
    return rows


def test_copying_or_ignoring_reference_cannot_pass():
    rows = _rows()
    assert gate._summarize(rows)["status"] == "PASS"
    for row in rows:
        row["metrics"]["audio_codec_foa_progress"] = 0.0
        row["metrics"]["reference_shuffled_audio_gain"] = 0.0
    result = gate._summarize(rows)
    assert result["status"] == "FAIL"
    assert any(key.startswith("audio_codec_foa_progress:") for key in result["failed_checks"])
    assert any(key.startswith("reference_shuffled_audio_gain:") for key in result["failed_checks"])


def test_sparse_metric_coverage_and_post_joint_regression_fail():
    rows = _rows()
    baseline = {"result": gate._summarize(rows)}
    for row in rows:
        row["metrics"]["audio_codec_foa_progress"] = 0.5  # above fixed .05, below baseline-tolerance
    assert gate._summarize(rows)["status"] == "PASS"
    result = gate._summarize(rows, baseline=baseline)
    assert result["status"] == "FAIL"
    assert any(key.startswith("post_joint_nonregression:") for key in result["failed_checks"])
    for row in rows[:200]:
        row["metrics"]["reference_zero_audio_gain"] = None
    assert gate._summarize(rows)["status"] == "FAIL"


def test_dit_and_ar_use_identical_sampler_and_codec_methods():
    for name in ("encode_source_foa", "decode_foa_latents", "sample_edited_latents", "_conditioning_metadata"):
        assert getattr(ScenePlanTransfusionEditingDiTPipeline, name) is getattr(ScenePlanTransfusionEditingPipeline, name)
    assert not hasattr(ScenePlanTransfusionEditingDiTPipeline, "generate_new_sceneplans")
    assert "old_sceneplan" not in inspect.signature(ScenePlanTransfusionEditingDiTPipeline.sample_edited_latents).parameters


def test_gt_execution_preserves_reference_mask_noise_and_scoring_source(tmp_path, monkeypatch):
    plan = {"sample_id": "target", "sources": []}
    metadata = {"pair_id": "pair-a", "pair_ordinal": 0, "latent_bucket_frames": 432,
                "model_num_samples": 32, "latent_frames_valid": 1, "model_sceneplan": plan}
    truth = {**metadata, "source_sample_id": "source-a", "source_foa": torch.ones(4, 32),
             "offline_new_sceneplan": plan, "offline_old_sceneplan": {"sources": [1, 2]},
             "unchanged_source_ids": ["source_0"], "source_domain": "speech", "target_domain": "sound"}
    donor = {**truth, "pair_id": "pair-b", "pair_ordinal": 1, "source_sample_id": "source-b",
             "model_num_samples": 40, "source_foa": torch.full((4, 40), 3.0)}
    calls = []

    class Pipeline:
        def encode_source_foa(self, wave, *, model_num_samples, vae_seeds):
            # The donor's longer audio is cropped at this same real codec boundary.
            assert model_num_samples == [32]
            mask = torch.zeros(1, 432, dtype=torch.bool)
            mask[:, 0] = True
            latent = torch.zeros(1, 64, 432)
            latent[:, :, 0] = wave[..., :32].mean()
            return latent, mask

        def decode_foa_latents(self, latent, *, model_num_samples):
            return latent[:, :4, :1].expand(-1, -1, 32).clone(), torch.ones(1, 32, dtype=torch.bool)

        def sample_edited_latents(self, reference, mask, plans, **kwargs):
            calls.append((reference.clone(), mask.clone(), deepcopy(plans), kwargs["initial_noise"].clone()))
            return reference + 0.1

    def score(**kwargs):
        result = kwargs["sampled_gt_result"]
        assert kwargs["codec"] is None
        assert torch.all(result["source_codec_foa"] == 1.0)
        assert torch.all(result["edited_foa"] == 1.1)
        return [{"status": "ok", "plan_origin": "ground_truth", "metrics": {}}]

    monkeypatch.setattr(gate.audio, "_process_batch", score)
    record = gate._evaluate_row(pipeline=Pipeline(), scorer=object(), sample=(torch.ones(64, 648), metadata),
        truth=truth, donor_truth=donor, device=torch.device("cpu"), output_dir=tmp_path, contract_sha="f" * 64)
    assert len(calls) == 3
    assert [call[0][0, 0, 0].item() for call in calls] == [1.0, 0.0, 3.0]
    assert all(torch.equal(calls[0][1], call[1]) and torch.equal(calls[0][3], call[3]) for call in calls)
    assert all(call[2] == [plan] for call in calls)
    assert set(record["variant_audio"]) == set(gate.VARIANTS)
    for artifact in record["variant_audio"].values():
        gate._verify_artifact(artifact)


def test_tampered_audio_or_gt_mislabeled_as_ar_is_rejected(tmp_path):
    path = tmp_path / "sample.wav"
    path.write_bytes(b"immutable-generated-audio")
    artifact = gate._artifact(path)
    row = {"pair_ordinal": 1, "pair_id": "pair-a", "operation": "event_removal", "latent_bucket_frames": 432}
    record = {**row, "schema": gate.SCHEMA, "status": "ok", "contract_sha256": "abc",
              "plan_origin": "ground_truth", "metrics": {}, "variant_audio": {k: artifact for k in gate.VARIANTS},
              "model_input_contract": {"editing_ar": None, "old_sceneplan": False},
              "edited_foa_path": artifact["path"], "edited_foa_sha256": artifact["sha256"]}
    gate._check_record(record, row, "abc")
    record["metrics"]["plan_token_accuracy"] = 1.0
    with pytest.raises(RuntimeError, match="free AR evidence"):
        gate._check_record(record, row, "abc")
    record["metrics"].clear()
    path.write_bytes(b"replaced-audio")
    with pytest.raises(RuntimeError, match="artifact changed"):
        gate._check_record(record, row, "abc")


def test_cli_has_no_test_or_threshold_override():
    source = inspect.getsource(gate.main)
    assert '"--test-index"' not in source
    assert '"--threshold"' not in source
    assert '"--index"' not in source


def test_policy_survives_publication_and_degenerate_ablation_is_not_a_gain():
    assert json.loads(json.dumps(gate._policy())) == gate._policy()
    assert gate.audio._progress(0.0, 0.0) is None


def test_equivalent_added_tone_phase_is_not_an_edit_failure():
    t = torch.arange(44100, dtype=torch.float32) / 44100
    source = torch.sin(2 * torch.pi * 220 * t)
    addition = torch.sin(2 * torch.pi * 440 * t)
    target = source + addition
    equivalent_addition = source - addition
    # The same added frequency, envelope and level with opposite phase has
    # four times the target MSE of doing no edit. This metric cannot decide
    # whether event addition happened successfully.
    copy_error = gate.audio._nmse(source, target)
    output_error = gate.audio._nmse(equivalent_addition, target)
    assert output_error == pytest.approx(4 * copy_error, rel=1e-5)
    assert gate.audio._progress(output_error, copy_error) == pytest.approx(-3)
    rows = _rows()
    for row in rows:
        if row["operation"] == "event_addition":
            row["metrics"].update(latent_foa_progress=-3, audio_codec_foa_progress=-3,
                audio_raw_foa_progress=-3, audio_raw_w_progress=-3,
                change_direction_cosine_foa=-1, latent_foa_nmse=4, audio_codec_foa_nmse=4)
    assert gate._summarize(rows)["status"] == "PASS"
    for row in rows:
        if row["operation"] == "event_addition":
            row["metrics"]["independent_clap_edit_text_progress"] = 0
    assert gate._summarize(rows)["status"] == "FAIL"
