from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from scripts.t2a.data.build_sceneplan_transfusion_editing_m2d_clap_cache import (
    _batched_audio_embeddings,
    _embedding_blob,
    _resume_rows,
)
from stable_audio_tools.data.sceneplan_transfusion_editing import sha256_json
from stable_audio_tools.data.sceneplan_transfusion_editing_m2d_clap import (
    EDITING_M2D_AUDIO_PREPROCESS,
    EDITING_M2D_TEMPORAL_PILOT_CHECKS,
    EDITING_M2D_VAE_CHECKPOINT_SHA256,
    EDITING_M2D_VAE_CONFIG_SHA256,
    editing_m2d_cache_implementation_sha256,
    validate_editing_m2d_temporal_pilot,
)
from stable_audio_tools.models.sceneplan_transfusion_editing_m2d_runtime import (
    decoded_foa_w_to_m2d_waveform,
    FrozenEditingM2DCLAP,
    M2D_CLAP_BERT_FILES,
    M2D_CLAP_SOURCE_AUDIO_VIEW,
    M2D_CLAP_TEMPORAL_POLICY,
    editing_m2d_software_runtime_fingerprint,
    expected_editing_m2d_clap_asset_report,
)
from stable_audio_tools.models.sceneplan_transfusion_editing_m2d_clap import (
    EditingARSourceSemanticBridge,
    canonicalize_editing_m2d_embedding,
)
from stable_audio_tools.models.sceneplan_transfusion_editing_pipeline import (
    DIAGNOSTIC_EXTERNAL_M2D_EMBEDDING_ORIGIN,
    ScenePlanTransfusionEditingPipeline,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _partial_connection() -> tuple[sqlite3.Connection, bytes, str]:
    connection = sqlite3.connect(":memory:")
    connection.executescript(
        """
        CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE features (
            pair_ordinal INTEGER PRIMARY KEY,
            pair_id TEXT NOT NULL UNIQUE,
            source_sample_id TEXT NOT NULL,
            source_latent_tensor_sha256 TEXT NOT NULL,
            source_caption_sha256 TEXT NOT NULL,
            audio_embedding BLOB NOT NULL,
            text_embedding BLOB NOT NULL,
            audio_embedding_sha256 TEXT NOT NULL,
            text_embedding_sha256 TEXT NOT NULL,
            audio_norm_before_l2 REAL NOT NULL,
            text_norm_before_l2 REAL NOT NULL,
            record_sha256 TEXT NOT NULL
        );
        """
    )
    connection.executemany(
        "INSERT INTO metadata VALUES (?,?)",
        (("schema", "unit-test"), ("state", "building"), ("rows", "1")),
    )
    embedding = np.zeros(768, dtype=np.float16)
    embedding[0] = 1.0
    blob = embedding.tobytes()
    blob_sha = hashlib.sha256(blob).hexdigest()
    record = {
        "pair_ordinal": 0,
        "pair_id": "pair-0",
        "source_sample_id": "source-0",
        "source_latent_tensor_sha256": "a" * 64,
        "source_caption_sha256": "b" * 64,
        "audio_embedding_sha256": blob_sha,
        "text_embedding_sha256": blob_sha,
    }
    connection.execute(
        "INSERT INTO features VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            0,
            "pair-0",
            "source-0",
            "a" * 64,
            "b" * 64,
            blob,
            blob,
            blob_sha,
            blob_sha,
            2.0,
            3.0,
            sha256_json(record),
        ),
    )
    connection.commit()
    return connection, blob, blob_sha


class EditingM2DCacheResumeTests(unittest.TestCase):
    def test_numeric_runtime_fingerprint_binds_precision_backend_flags(self) -> None:
        fingerprint = editing_m2d_software_runtime_fingerprint()
        self.assertEqual(
            fingerprint["schema"], "editing_m2d_numeric_runtime_fingerprint_v2"
        )
        self.assertTrue(
            {
                "cublas_workspace_config",
                "cuda_matmul_allow_fp16_reduced_precision_reduction",
                "cuda_matmul_allow_bf16_reduced_precision_reduction",
                "cuda_matmul_allow_fp16_accumulation",
                "cudnn_enabled",
                "cudnn_benchmark",
                "cudnn_benchmark_limit",
                "cudnn_deterministic",
                "deterministic_algorithms",
                "deterministic_algorithms_warn_only",
                "cuda_flash_sdp_enabled",
                "cuda_mem_efficient_sdp_enabled",
                "cuda_math_sdp_enabled",
                "cuda_cudnn_sdp_enabled",
            }.issubset(fingerprint)
        )

    def test_caption_text_tower_pins_the_actual_bert_weights(self) -> None:
        self.assertEqual(
            M2D_CLAP_BERT_FILES["model.safetensors"],
            "68d45e234eb4a928074dfd868cead0219ab85354cc53d20e772753c6bb9169d3",
        )

    def test_temporal_pilot_is_hash_and_contract_bound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            verifier = REPO_ROOT / (
                "scripts/t2a/test/"
                "validate_sceneplan_transfusion_editing_m2d_temporal_policy.py"
            )
            source_index = root / "validation.sqlite"
            source_index.write_bytes(b"frozen validation index")
            payload = {
                "schema": "sceneplan_transfusion_editing_m2d_temporal_policy_pilot",
                "schema_version": 2,
                "status": "PASS",
                "checks": {
                    key: True for key in EDITING_M2D_TEMPORAL_PILOT_CHECKS
                },
                "physical_gpu": 3,
                "rows": 4,
                "source_index": str(source_index.resolve()),
                "source_index_sha256": hashlib.sha256(
                    source_index.read_bytes()
                ).hexdigest(),
                "source_index_rows": 20_000,
                "old_sceneplan_model_input": False,
                "new_sceneplan_or_target_information_used": False,
                "source_audio_view": M2D_CLAP_SOURCE_AUDIO_VIEW,
                "temporal_policy": M2D_CLAP_TEMPORAL_POLICY,
                "audio_preprocess": EDITING_M2D_AUDIO_PREPROCESS,
                "projector_tokens": [{"first_10_seconds": 310, "full_valid_clip": 470}],
                "vae": {
                    "config_sha256": EDITING_M2D_VAE_CONFIG_SHA256,
                    "checkpoint_sha256": EDITING_M2D_VAE_CHECKPOINT_SHA256,
                },
                "m2d_assets": expected_editing_m2d_clap_asset_report(
                    require_text=False
                ),
                "implementation_sha256": (
                    editing_m2d_cache_implementation_sha256()
                ),
                "numeric_runtime_fingerprint": {
                    **editing_m2d_software_runtime_fingerprint(),
                    "device_name": "unit-test-gpu",
                    "device_capability": [9, 0],
                },
                "verifier": str(verifier.resolve()),
                "verifier_sha256": hashlib.sha256(verifier.read_bytes()).hexdigest(),
            }
            pilot = root / "PASS.json"
            pilot.write_text(json.dumps(payload), encoding="utf-8")
            pilot_sha = hashlib.sha256(pilot.read_bytes()).hexdigest()
            observed = validate_editing_m2d_temporal_pilot(
                pilot, expected_sha256=pilot_sha
            )
            self.assertEqual(observed["sha256"], pilot_sha)
            payload["checks"]["every_real_active_tail_changes_embedding"] = False
            pilot.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "stale or invalid"):
                validate_editing_m2d_temporal_pilot(pilot)

    def test_audio_preprocess_uses_antialiasing_before_16khz_m2d(self) -> None:
        samples = 44_100
        time = torch.arange(samples, dtype=torch.float32) / 44_100

        def convert(frequency: float) -> torch.Tensor:
            foa = torch.zeros(4, samples)
            foa[0] = torch.sin(2 * torch.pi * frequency * time)
            return decoded_foa_w_to_m2d_waveform(
                foa, valid_samples=samples
            )

        in_band = convert(1_000.0)
        above_nyquist = convert(12_000.0)
        self.assertEqual(tuple(in_band.shape), (16_000,))
        self.assertTrue(torch.isfinite(in_band).all())
        self.assertLessEqual(float(in_band.abs().max()), 1.0)
        # A linear 44.1k -> 16k interpolation leaves this 12 kHz tone at about
        # 0.56 RMS through aliasing.  The fixed sinc path must strongly reject
        # it while retaining an in-band tone.
        self.assertGreater(float(in_band.square().mean().sqrt()), 0.5)
        self.assertLess(float(above_nyquist.square().mean().sqrt()), 0.02)

    def test_audio_preprocess_uses_only_decoded_w_channel(self) -> None:
        generator = torch.Generator().manual_seed(42)
        source = torch.randn(4, 44_100, generator=generator)
        changed_directional = source.clone()
        changed_directional[1:] = torch.randn(
            3, 44_100, generator=generator
        ) * 100.0
        expected = decoded_foa_w_to_m2d_waveform(
            source, valid_samples=44_100
        )
        observed = decoded_foa_w_to_m2d_waveform(
            changed_directional, valid_samples=44_100
        )
        self.assertTrue(torch.equal(expected, observed))
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            outer_autocast = decoded_foa_w_to_m2d_waveform(
                source, valid_samples=44_100
            )
        self.assertEqual(outer_autocast.dtype, torch.float32)
        self.assertTrue(torch.equal(expected, outer_autocast))
        self.assertEqual(
            M2D_CLAP_SOURCE_AUDIO_VIEW,
            "frozen_vae_decode_of_source_foa_latent_W",
        )

    def test_aligned_temporal_policy_covers_full_15_second_waveform(self) -> None:
        class FakeAudioProj(torch.nn.Module):
            dont_average = True

            def __init__(self) -> None:
                super().__init__()
                self.seen_tokens: int | None = None
                self.autocast_states: list[bool] = []
                self.input_dtypes: list[torch.dtype] = []

            def forward(self, value: torch.Tensor) -> torch.Tensor:
                self.autocast_states.append(torch.is_autocast_enabled("cpu"))
                self.input_dtypes.append(value.dtype)
                self.seen_tokens = int(value.shape[-2])
                output = torch.zeros(value.shape[0], 768, device=value.device)
                output[:, 0] = 1.0
                return output

        class FakeBackbone(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(()))
                self.audio_proj = FakeAudioProj()
                self.chunk_widths: list[int] = []
                self.autocast_states: list[bool] = []
                self.input_dtypes: list[torch.dtype] = []

            @staticmethod
            def patch_size() -> list[int]:
                return [16, 16]

            def forward_encoder(self, value: torch.Tensor) -> torch.Tensor:
                self.autocast_states.append(torch.is_autocast_enabled("cpu"))
                self.input_dtypes.append(value.dtype)
                self.chunk_widths.append(int(value.shape[-1]))
                tokens = 5 * (int(value.shape[-1]) // 16)
                return torch.zeros(value.shape[0], tokens + 1, 768)

        class FakeRuntime(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.backbone = FakeBackbone()
                self.seen_shape: tuple[int, ...] | None = None
                self.autocast_states: list[bool] = []
                self.input_dtypes: list[torch.dtype] = []
                self.cfg = SimpleNamespace(input_size=[80, 1001], flat_features=True)

            def to_normalized_feature(self, value: torch.Tensor) -> torch.Tensor:
                self.autocast_states.append(torch.is_autocast_enabled("cpu"))
                self.input_dtypes.append(value.dtype)
                self.seen_shape = tuple(value.shape)
                mel_frames = int(value.shape[-1]) // 160 + 1
                return torch.zeros(value.shape[0], 1, 80, mel_frames)

        encoder = FrozenEditingM2DCLAP.__new__(FrozenEditingM2DCLAP)
        torch.nn.Module.__init__(encoder)
        encoder.runtime = FakeRuntime()
        encoder.load_text_encoder = False
        waveform = torch.zeros(2, 15 * 16_000)
        baseline = encoder.encode_audio(waveform)
        encoder.runtime.backbone.chunk_widths.clear()
        encoder.runtime.autocast_states.clear()
        encoder.runtime.input_dtypes.clear()
        encoder.runtime.backbone.autocast_states.clear()
        encoder.runtime.backbone.input_dtypes.clear()
        encoder.runtime.backbone.audio_proj.autocast_states.clear()
        encoder.runtime.backbone.audio_proj.input_dtypes.clear()
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            embedding = encoder.encode_audio(waveform)
        self.assertEqual(encoder.runtime.seen_shape, (2, 240_000))
        self.assertEqual(encoder.runtime.backbone.chunk_widths, [992, 512])
        self.assertEqual(encoder.runtime.backbone.audio_proj.seen_tokens, 470)
        self.assertEqual(tuple(embedding.shape), (2, 768))
        self.assertTrue(torch.equal(baseline, embedding))
        self.assertEqual(encoder.runtime.autocast_states, [False])
        self.assertEqual(encoder.runtime.input_dtypes, [torch.float32])
        self.assertEqual(
            encoder.runtime.backbone.autocast_states, [False, False]
        )
        self.assertEqual(
            encoder.runtime.backbone.input_dtypes,
            [torch.float32, torch.float32],
        )
        self.assertEqual(
            encoder.runtime.backbone.audio_proj.autocast_states, [False]
        )
        self.assertEqual(
            encoder.runtime.backbone.audio_proj.input_dtypes, [torch.float32]
        )
        self.assertIn("full_coverage_chunks_concat", M2D_CLAP_TEMPORAL_POLICY)

    def test_pipeline_m2d_view_decodes_only_the_exact_valid_latent(self) -> None:
        class FakeVAE:
            def __init__(self) -> None:
                self.calls: list[tuple[tuple[int, ...], bool, torch.dtype]] = []

            def decode(self, value: torch.Tensor) -> torch.Tensor:
                self.calls.append(
                    (
                        tuple(value.shape),
                        torch.is_autocast_enabled("cpu"),
                        value.dtype,
                    )
                )
                return torch.zeros(
                    value.shape[0], 4, value.shape[-1] * 1024,
                    device=value.device,
                )

        class FakeM2D:
            def __init__(self) -> None:
                self.calls: list[tuple[tuple[int, ...], bool, torch.dtype]] = []

            def encode_audio(self, value: torch.Tensor) -> torch.Tensor:
                self.calls.append(
                    (
                        tuple(value.shape),
                        torch.is_autocast_enabled("cpu"),
                        value.dtype,
                    )
                )
                result = torch.zeros(value.shape[0], 768, device=value.device)
                result[:, 0 if value.shape[-1] == 240_000 else 1] = 1.0
                return result

        class Harness:
            device = torch.device("cpu")

            def __init__(self) -> None:
                self.vae = FakeVAE()
                self.m2d = FakeM2D()

            def _require_audio_autoencoder(self) -> FakeVAE:
                return self.vae

            def _require_source_semantic_encoder(self) -> FakeM2D:
                return self.m2d

        harness = Harness()
        source = torch.zeros(3, 64, 648)
        source[0, :, 646:] = 123.0
        source[1, :, 431:] = 456.0
        source[2, :, 646:] = 789.0
        mask = torch.zeros(3, 648, dtype=torch.bool)
        mask[0, :646] = True
        mask[1, :431] = True
        mask[2, :646] = True
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            embedding = ScenePlanTransfusionEditingPipeline.encode_source_m2d_audio(
                harness,
                source,
                mask,
                model_num_samples=[15 * 44_100, 10 * 44_100, 15 * 44_100],
            )
        self.assertEqual(
            harness.vae.calls,
            [
                ((2, 64, 646), False, torch.float32),
                ((1, 64, 431), False, torch.float32),
            ],
        )
        self.assertEqual(
            harness.m2d.calls,
            [
                ((2, 240_000), False, torch.float32),
                ((1, 160_000), False, torch.float32),
            ],
        )
        self.assertEqual(tuple(embedding.shape), (3, 768))
        self.assertEqual(embedding.dtype, torch.float16)
        self.assertEqual(embedding[:, :2].tolist(), [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])

    def test_source_vae_encode_and_decode_ignore_outer_autocast(self) -> None:
        class FakeVAE:
            def __init__(self) -> None:
                self.encoder_calls: list[tuple[bool, torch.dtype]] = []
                self.decoder_calls: list[tuple[bool, torch.dtype]] = []

            def encoder(self, value: torch.Tensor) -> torch.Tensor:
                self.encoder_calls.append(
                    (torch.is_autocast_enabled("cpu"), value.dtype)
                )
                return torch.zeros(value.shape[0], 128, 432)

            def decode(self, value: torch.Tensor) -> torch.Tensor:
                self.decoder_calls.append(
                    (torch.is_autocast_enabled("cpu"), value.dtype)
                )
                return value[:, :4].repeat_interleave(1024, dim=-1)

        class Harness:
            device = torch.device("cpu")

            def __init__(self) -> None:
                self.vae = FakeVAE()

            def _require_audio_autoencoder(self) -> FakeVAE:
                return self.vae

        source = torch.zeros(1, 4, 1024)
        baseline_harness = Harness()
        baseline_latent, baseline_mask = (
            ScenePlanTransfusionEditingPipeline.encode_source_foa(
                baseline_harness,
                source,
                model_num_samples=[1024],
                vae_seeds=[73],
            )
        )
        baseline_audio, baseline_sample_mask = (
            ScenePlanTransfusionEditingPipeline.decode_foa_latents(
                baseline_harness,
                baseline_latent,
                model_num_samples=[1024],
            )
        )

        harness = Harness()
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            latent, mask = ScenePlanTransfusionEditingPipeline.encode_source_foa(
                harness,
                source,
                model_num_samples=[1024],
                vae_seeds=[73],
            )
            audio, sample_mask = (
                ScenePlanTransfusionEditingPipeline.decode_foa_latents(
                    harness,
                    latent,
                    model_num_samples=[1024],
                )
            )
        self.assertTrue(torch.equal(baseline_latent, latent))
        self.assertTrue(torch.equal(baseline_mask, mask))
        self.assertTrue(torch.equal(baseline_audio, audio))
        self.assertTrue(torch.equal(baseline_sample_mask, sample_mask))
        self.assertEqual(harness.vae.encoder_calls, [(False, torch.float32)])
        self.assertEqual(harness.vae.decoder_calls, [(False, torch.float32)])

    def test_external_m2d_injection_is_explicit_and_canonicalized(self) -> None:
        class FakeEditingAR:
            source_semantic_mode = "m2d_audio"

        class Harness:
            device = torch.device("cpu")
            editing_ar = FakeEditingAR()

            def __init__(self) -> None:
                self.encode_calls: list[list[int]] = []

            def encode_source_m2d_audio(self, *args, **kwargs) -> torch.Tensor:
                samples = [int(value) for value in kwargs["model_num_samples"]]
                self.encode_calls.append(samples)
                return canonicalize_editing_m2d_embedding(raw)

        harness = Harness()
        source = torch.zeros(2, 64, 432)
        mask = torch.ones(2, 432, dtype=torch.bool)
        raw = torch.randn(2, 768, generator=torch.Generator().manual_seed(19)) * 3.0
        prepare = ScenePlanTransfusionEditingPipeline._prepare_source_m2d_audio_embedding
        with self.assertRaisesRegex(RuntimeError, "diagnostic-only"):
            prepare(
                harness,
                source,
                mask,
                durations=[1.0, 1.0],
                source_m2d_audio_embedding=raw,
                source_m2d_audio_embedding_origin=None,
            )
        observed = prepare(
            harness,
            source,
            mask,
            durations=[1.0, 1.0],
            source_m2d_audio_embedding=raw,
            source_m2d_audio_embedding_origin=(
                DIAGNOSTIC_EXTERNAL_M2D_EMBEDDING_ORIGIN
            ),
        )
        expected = canonicalize_editing_m2d_embedding(raw)
        self.assertEqual(observed.dtype, torch.float16)
        self.assertTrue(torch.equal(observed, expected))
        self.assertEqual(harness.encode_calls, [])
        internal = prepare(
            harness,
            source,
            mask,
            durations=[1.0, 1.0],
            source_m2d_audio_embedding=None,
            source_m2d_audio_embedding_origin=None,
        )
        self.assertTrue(torch.equal(internal, expected))
        self.assertEqual(harness.encode_calls, [[44_100, 44_100]])
        self.assertNotIn(
            "source_m2d_audio_embedding",
            inspect.signature(
                ScenePlanTransfusionEditingPipeline.edit_audio
            ).parameters,
        )

    def test_online_and_cache_m2d_use_identical_fp16_boundary(self) -> None:
        generator = torch.Generator().manual_seed(91)
        raw = torch.randn(768, generator=generator) * 3.7
        online = canonicalize_editing_m2d_embedding(raw)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            outer_autocast = canonicalize_editing_m2d_embedding(raw)
        blob, digest, _ = _embedding_blob(raw)
        self.assertEqual(online.dtype, torch.float16)
        self.assertTrue(torch.equal(online, outer_autocast))
        self.assertEqual(online.cpu().contiguous().numpy().tobytes(), blob)
        self.assertEqual(hashlib.sha256(blob).hexdigest(), digest)

    def test_formal_cache_and_joint_require_numeric_online_replay(self) -> None:
        cache_runner = (
            REPO_ROOT
            / "scripts/t2a/data/"
            "run_sceneplan_transfusion_editing_m2d_clap_cache_5gpu.sh"
        ).read_text(encoding="utf-8")
        joint_source = (
            REPO_ROOT
            / "scripts/t2a/train/"
            "train_sceneplan_transfusion_editing_ar_joint_full.py"
        ).read_text(encoding="utf-8")
        parity_script = REPO_ROOT / (
            "scripts/t2a/test/"
            "validate_sceneplan_transfusion_editing_m2d_cache_online_parity.py"
        )
        self.assertTrue(parity_script.is_file())
        self.assertIn("run_online_parity_gate", cache_runner)
        self.assertIn("--physical-gpu 3", cache_runner)
        self.assertIn("validate_editing_m2d_cache_online_parity", joint_source)
        self.assertIn("cache_online_parity", joint_source)

    def test_whole_example_m2d_dropout_uses_inverted_scaling(self) -> None:
        bridge = EditingARSourceSemanticBridge(
            mode="m2d_audio", hidden_dim=2, semantic_dim=2,
            audio_feature_dropout=0.5,
        )
        bridge.audio_norm = torch.nn.Identity()
        assert isinstance(bridge.audio_to_hidden, torch.nn.Linear)
        with torch.no_grad():
            bridge.audio_to_hidden.weight.copy_(torch.eye(2))
            bridge.audio_to_hidden.bias.zero_()
        bridge.train()
        torch.manual_seed(7)
        rows = 20_000
        source = torch.zeros(rows, 1, 2)
        semantic = torch.ones(rows, 2) / (2.0**0.5)
        observed = bridge.inject(source, semantic).mean(dim=(0, 1))
        self.assertTrue(
            torch.allclose(observed, semantic[0], atol=0.02, rtol=0.0)
        )

    def test_m2d_zero_mask_removes_the_complete_projected_residual(self) -> None:
        bridge = EditingARSourceSemanticBridge(
            mode="m2d_audio", hidden_dim=3, semantic_dim=2
        ).eval()
        bridge.audio_norm = torch.nn.Identity()
        assert isinstance(bridge.audio_to_hidden, torch.nn.Linear)
        with torch.no_grad():
            bridge.audio_to_hidden.weight.fill_(2.0)
            bridge.audio_to_hidden.bias.fill_(7.0)
        source = torch.randn(4, 5, 3)
        semantic = torch.nn.functional.normalize(torch.randn(4, 2), dim=-1)
        observed = bridge.inject(
            source,
            semantic,
            source_m2d_audio_keep_mask=torch.zeros(4, dtype=torch.bool),
        )
        self.assertTrue(torch.equal(observed, source))

    def test_audio_batching_groups_exact_lengths_without_reordering(self) -> None:
        class FakeM2D:
            def __init__(self) -> None:
                self.calls: list[tuple[int, int]] = []

            def encode_audio(self, waveforms: torch.Tensor) -> torch.Tensor:
                self.calls.append(tuple(int(value) for value in waveforms.shape))
                output = torch.zeros(len(waveforms), 768)
                output[:, 0] = waveforms[:, 0]
                return output

        model = FakeM2D()
        waveforms = [
            torch.full((400,), 1.0),
            torch.full((500,), 2.0),
            torch.full((400,), 3.0),
        ]
        embeddings = _batched_audio_embeddings(model, waveforms, batch_size=8)
        self.assertEqual(model.calls, [(2, 400), (1, 500)])
        self.assertEqual([float(value[0]) for value in embeddings], [1.0, 2.0, 3.0])

    def test_valid_partial_row_is_resumed(self) -> None:
        connection, _, _ = _partial_connection()
        self.addCleanup(connection.close)
        expected = {
            0: (
                0, "pair-0", "source-0", 44_100, 44, "latent", "key",
                "a" * 64,
            )
        }
        self.assertEqual(
            _resume_rows(
                connection,
                expected_metadata={"schema": "unit-test"},
                expected_records=expected,
            ),
            {0},
        )

    def test_partial_embedding_corruption_fails_closed(self) -> None:
        connection, blob, _ = _partial_connection()
        self.addCleanup(connection.close)
        changed = bytearray(blob)
        changed[-1] ^= 1
        connection.execute(
            "UPDATE features SET audio_embedding=? WHERE pair_ordinal=0",
            (bytes(changed),),
        )
        connection.commit()
        expected = {
            0: (
                0, "pair-0", "source-0", 44_100, 44, "latent", "key",
                "a" * 64,
            )
        }
        with self.assertRaisesRegex(RuntimeError, "corrupt"):
            _resume_rows(
                connection,
                expected_metadata={"schema": "unit-test"},
                expected_records=expected,
            )


if __name__ == "__main__":
    unittest.main()
