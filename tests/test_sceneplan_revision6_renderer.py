from __future__ import annotations

import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import soundfile as sf

from scripts.t2a.data.materialize_sceneplan_v2_shard import (
    REVISION6_MAX_MODEL_SAMPLES,
    calibrate_speech_background,
    load_complete_source,
    max_model_samples_for_row,
)
from scripts.t2a.data.finalize_speech_expansion_donor_registry import canonical_row
from scripts.t2a.data.audit_speech_expansion_sources_noalign_15s import _model_samples
from scripts.t2a.data.build_speech_expansion_speaker_profiles import (
    MAX_SPEAKER_EXCERPT_SAMPLES,
    clean_profile,
    speaker_excerpt,
)
from scripts.t2a.data.sceneplan_v2_common import normalized_transcript
from scripts.t2a.data.sceneplan_v2_common import MAX_MODEL_SAMPLES


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _source(kind: str, onset: int, offset: int, gain_db: float) -> dict:
    return {
        "kind": kind,
        "gain_db": gain_db,
        "activity": [
            {
                "model_onset_sample": onset,
                "model_offset_sample": offset,
            }
        ],
    }


def _qc(gain_db: float) -> dict:
    return {"planned_gain_db": gain_db}


def test_file_backed_speech_requires_audio_and_transcript_lineage(
    tmp_path: Path,
) -> None:
    samples = np.linspace(-0.1, 0.1, 4_410, dtype=np.float32)
    path = tmp_path / "utterance.flac"
    sf.write(path, samples, 44_100, subtype="PCM_24")
    transcript = "A complete metadata transcript."
    source = {
        "kind": "speech",
        "activity": [{"dry_end_sample": len(samples)}],
        "speech": {"transcript": transcript},
        "asset_ref": {
            "dry_audio_path": str(path),
            "identity_hash": _digest(path.read_bytes()),
            "native_sample_rate_hz": 44_100,
            "native_num_samples": len(samples),
            "normalized_transcript_sha256": _digest(
                normalized_transcript(transcript).encode("utf-8")
            ),
        },
    }

    audio, lineage = load_complete_source(source)
    assert audio.shape == (len(samples),)
    assert lineage["source_audio_sha256"] == source["asset_ref"]["identity_hash"]
    assert lineage["random_crop"] is False

    source["asset_ref"]["normalized_transcript_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="transcript lineage mismatch"):
        load_complete_source(source)


def test_revision6_explicitly_extends_complete_source_envelope(
    tmp_path: Path,
) -> None:
    samples = np.linspace(
        -0.1, 0.1, MAX_MODEL_SAMPLES + 1024, dtype=np.float32
    )
    path = tmp_path / "long-utterance.flac"
    sf.write(path, samples, 44_100, subtype="PCM_24")
    transcript = "A complete long metadata transcript."
    source = {
        "kind": "speech",
        "activity": [{"dry_end_sample": len(samples)}],
        "speech": {"transcript": transcript},
        "asset_ref": {
            "dry_audio_path": str(path),
            "identity_hash": _digest(path.read_bytes()),
            "native_sample_rate_hz": 44_100,
            "native_num_samples": len(samples),
            "normalized_transcript_sha256": _digest(
                normalized_transcript(transcript).encode("utf-8")
            ),
        },
    }

    with pytest.raises(ValueError, match="exceeds model limit"):
        load_complete_source(source)
    audio, lineage = load_complete_source(
        source, max_model_samples=REVISION6_MAX_MODEL_SAMPLES
    )
    assert audio.shape == (len(samples),)
    assert lineage["coverage_fraction"] == 1.0
    assert lineage["random_crop"] is False


def test_revision6_parquet_speech_uses_extended_envelope(tmp_path: Path) -> None:
    samples = np.linspace(
        -0.1, 0.1, MAX_MODEL_SAMPLES + 2048, dtype=np.float32
    )
    encoded = io.BytesIO()
    sf.write(encoded, samples, 44_100, format="FLAC", subtype="PCM_24")
    blob = encoded.getvalue()
    transcript = "A complete long Parquet-backed transcript."
    parquet_path = tmp_path / "long-speech.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [{"audio": {"bytes": blob}, "text_normalized": transcript}]
        ),
        parquet_path,
        row_group_size=1,
    )
    source = {
        "kind": "speech",
        "activity": [{"dry_end_sample": len(samples)}],
        "speech": {"transcript": transcript},
        "asset_ref": {
            "parquet_path": str(parquet_path),
            "row_group": 0,
            "row_in_group": 0,
            "identity_hash": _digest(blob),
            "native_sample_rate_hz": 44_100,
            "native_num_samples": len(samples),
        },
    }

    with pytest.raises(ValueError, match="exceeds model limit"):
        load_complete_source(source)
    audio, lineage = load_complete_source(
        source, max_model_samples=REVISION6_MAX_MODEL_SAMPLES
    )
    assert audio.shape == (len(samples),)
    assert lineage["model_num_samples"] == len(samples)
    assert lineage["coverage_fraction"] == 1.0
    assert lineage["random_crop"] is False


def test_render_envelope_is_selected_by_frozen_contract_revision() -> None:
    assert max_model_samples_for_row({}) == MAX_MODEL_SAMPLES
    assert max_model_samples_for_row(
        {"render_recipe_json": json.dumps({"dataset_contract_revision": 5})}
    ) == MAX_MODEL_SAMPLES
    assert max_model_samples_for_row(
        {"render_recipe_json": json.dumps({"dataset_contract_revision": 6})}
    ) == REVISION6_MAX_MODEL_SAMPLES
    with pytest.raises(RuntimeError, match="unsupported dataset contract revision"):
        max_model_samples_for_row(
            {"render_recipe_json": json.dumps({"dataset_contract_revision": 7})}
        )


def test_source_disjoint_parquet_speech_donor_keeps_parquet_locator(
    tmp_path: Path,
) -> None:
    parquet_path = tmp_path / "train.parquet"
    parquet_path.write_bytes(b"immutable-test-placeholder")
    # Legacy strong-QC rounded this rational resample down by one sample;
    # canonical donor geometry must match scipy resample_poly's ceiling.
    model_samples = 450_700
    row = {
        "source_audio_sha256": "1" * 64,
        "source_dataset": "libritts",
        "source_id": "594_127732_000016_000001",
        "speaker_key": "libritts:594",
        "native_sample_rate_hz": 24_000,
        "native_num_samples": 245_279,
        "model_num_samples": model_samples,
        "latent_frames_valid": math.ceil(model_samples / 1024),
        "length_bucket_frames": 648,
        "source_family": "existing_train_long",
        "source_text": "A complete sentence.",
        "renderer_text": "A complete sentence.",
        "normalized_transcript": "a complete sentence",
        "normalized_transcript_sha256": "2" * 64,
        "selection_rank": "0001",
        "locator_json": json.dumps(
            {
                "type": "parquet_row",
                "parquet_path": str(parquet_path),
                "row_group": 3,
                "row_in_group": 7,
            }
        ),
        "source_audio_path": None,
    }

    result = canonical_row(row, ordinal=9, ledger_by_hash={})
    locator = json.loads(result["locator_json"])
    assert result["asset_id"] == "libritts:594_127732_000016_000001"
    assert _model_samples(245_279, 24_000) == 450_701
    assert result["model_num_samples"] == 450_701
    assert result["latent_frames_valid"] == math.ceil(450_701 / 1024)
    assert result["source_audio_path"] is None
    assert locator == {
        "type": "parquet_row",
        "parquet_path": str(parquet_path.resolve()),
        "row_group": 3,
        "row_in_group": 7,
    }


def test_long_speech_profile_uses_bounded_annotation_excerpt_only() -> None:
    sample_rate = 44_100
    seconds = 14.5
    samples = np.linspace(
        -0.1, 0.1, round(sample_rate * seconds), dtype=np.float32
    )
    excerpt = speaker_excerpt(samples, sample_rate)
    assert excerpt.dtype == np.float32
    assert excerpt.shape == (MAX_SPEAKER_EXCERPT_SAMPLES,)
    assert np.isfinite(excerpt).all()


def test_short_complete_speaker_noun_phrase_is_not_overconstrained() -> None:
    assert clean_profile("a warm, clear, low-pitched adult male voice.") == (
        "a warm, clear, low-pitched adult male voice"
    )


def test_sequential_mixing_is_disjoint_and_never_recalibrated() -> None:
    num_samples = 8_820
    present = [
        _source("speech", 0, 4_410, 0.0),
        _source("sound", 4_410, num_samples, -5.0),
    ]
    stems = [np.zeros((4, num_samples), dtype=np.float32) for _ in present]
    stems[0][0, :4_410] = 0.1
    stems[1][0, 4_410:] = 0.1
    source_qc = [_qc(0.0), _qc(-5.0)]

    report = calibrate_speech_background(
        stems,
        present,
        source_qc,
        num_samples=num_samples,
        mixing_mode="sequential_nonoverlap",
    )
    assert report["overlap_samples"] == 0
    assert report["mode"] == "nonoverlap_independent_rms_normalization"
    assert [row["calibrated_gain_correction_db"] for row in source_qc] == [0.0, 0.0]

    present[1]["activity"][0]["model_onset_sample"] = 4_409
    with pytest.raises(RuntimeError, match="unexpectedly overlap"):
        calibrate_speech_background(
            stems,
            present,
            [_qc(0.0), _qc(-5.0)],
            num_samples=num_samples,
            mixing_mode="sequential_nonoverlap",
        )


def test_overlap_mixing_hits_the_frozen_aggregate_ratio() -> None:
    num_samples = 5_000
    present = [
        _source("speech", 0, num_samples, 0.0),
        _source("music", 0, num_samples, -5.0),
    ]
    stems = [np.zeros((4, num_samples), dtype=np.float32) for _ in present]
    stems[0][0] = 0.2
    stems[1][0] = 0.05
    source_qc = [_qc(0.0), _qc(-5.0)]

    report = calibrate_speech_background(
        stems,
        present,
        source_qc,
        num_samples=num_samples,
        mixing_mode="overlap_calibrated",
    )
    assert report["overlap_samples"] == num_samples
    assert report["target_speech_to_aggregate_background_db"] == pytest.approx(5.0)
    assert report["measured_speech_to_aggregate_background_db"] == pytest.approx(
        5.0, abs=1.0e-5
    )
