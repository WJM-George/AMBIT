#!/usr/bin/env python3
"""Validate the decoded-W M2D view and seamless 15-second aggregation.

This is a source-only pre-cache pilot.  It opens fixed long rows from the
frozen Editing validation index, decodes their exact source latents with the
pinned FOA VAE, and observes the token sequence presented to the pinned
M2D-CLAP audio semantic projector.  No old/new ScenePlan, instruction, target
latent, or target audio field is queried.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
from typing import Any

import torch
from safetensors import safe_open


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stable_audio_tools.configuration import load_config  # noqa: E402
from stable_audio_tools.data.model_sceneplan import VAE_HOP_SAMPLES  # noqa: E402
from stable_audio_tools.data.sceneplan_transfusion_editing_index import (  # noqa: E402
    sha256_file,
)
from stable_audio_tools.data.sceneplan_transfusion_editing_m2d_clap import (  # noqa: E402
    EDITING_M2D_AUDIO_PREPROCESS,
    EDITING_M2D_VAE_CHECKPOINT_SHA256,
    EDITING_M2D_VAE_CONFIG_SHA256,
    editing_m2d_cache_implementation_sha256,
)
from stable_audio_tools.models.factory import create_model_from_config  # noqa: E402
from stable_audio_tools.models.sceneplan_transfusion_editing_m2d_runtime import (  # noqa: E402
    FrozenEditingM2DCLAP,
    M2D_CLAP_SOURCE_AUDIO_VIEW,
    M2D_CLAP_TEMPORAL_POLICY,
    decoded_foa_w_to_m2d_waveform,
    editing_m2d_numeric_runtime_fingerprint,
)
from stable_audio_tools.models.utils import (  # noqa: E402
    copy_state_dict,
    load_ckpt_state_dict,
)


DEFAULT_INDEX = Path(
    os.environ.get("AMBIT_DATA_ROOT", "data") + "/sceneplan_transfusion_editing_v1/"
    "training_index/validation.sqlite"
)
DEFAULT_OUTPUT = Path(
    os.environ.get("AMBIT_DATA_ROOT", "data") + "/sceneplan_transfusion_editing_v1/"
    "semantic_cache/m2d_clap_v2/temporal_policy_pilot/PASS.json"
)
DEFAULT_VAE_CONFIG = REPO_ROOT / (
    "stable_audio_tools/configs/model_configs/autoencoders/"
    "stable_audio_4ch_vae_ds1024_z64_wdmix_scm.json"
)
DEFAULT_VAE_CHECKPOINT = Path(
    os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/compareVAE_ckpt/unwrapped_wdmix_1350000.ckpt"
)
EXPECTED_INDEX_ROWS = 20_000
MINIMUM_MODEL_SAMPLES = 15 * 44_100
EXPECTED_VALID_FRAMES = 648
MINIMUM_M2D_SAMPLES = 15 * 16_000
REAL_TAIL_ABSOLUTE_RMS_MIN = 1.0e-4
REAL_TAIL_RELATIVE_RMS_MIN = 1.0e-2
SELECT_COLUMNS = (
    "pair_ordinal",
    "pair_id",
    "source_sample_id",
    "model_num_samples",
    "latent_frames_valid",
    "latent_bucket_frames",
    "source_latent_path",
    "source_latent_key",
    "source_latent_tensor_sha256",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--physical-gpu", type=int, default=3)
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--vae-config", type=Path, default=DEFAULT_VAE_CONFIG)
    parser.add_argument("--vae-checkpoint", type=Path, default=DEFAULT_VAE_CHECKPOINT)
    return parser.parse_args()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def _device(physical_gpu: int) -> torch.device:
    physical = int(physical_gpu)
    if physical not in range(3, 8):
        raise ValueError("Editing M2D pilot is restricted to physical GPUs 3--7")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical):
        raise RuntimeError(
            "run the Editing M2D pilot with exactly its requested physical GPU "
            f"visible; expected CUDA_VISIBLE_DEVICES={physical}"
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Editing M2D pilot requires exactly one visible GPU")
    torch.cuda.set_device(0)
    return torch.device("cuda:0")


def _tensor_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()


def _index_rows(index: Path, *, rows: int) -> tuple[str, list[dict[str, Any]]]:
    marker_path = index.with_suffix(index.suffix + ".frozen.json")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    index_sha = sha256_file(index)
    connection = sqlite3.connect(f"file:{index}?mode=ro&immutable=1", uri=True)
    try:
        metadata = dict(connection.execute("SELECT key,value FROM metadata"))
        count = int(connection.execute("SELECT COUNT(*) FROM pairs").fetchone()[0])
        query = (
            f"SELECT {','.join(SELECT_COLUMNS)} FROM pairs "
            "WHERE model_num_samples>=? AND latent_frames_valid=? "
            "AND latent_bucket_frames=648 "
            "ORDER BY pair_ordinal LIMIT ?"
        )
        selected = connection.execute(
            query, (MINIMUM_MODEL_SAMPLES, EXPECTED_VALID_FRAMES, int(rows))
        ).fetchall()
    finally:
        connection.close()
    if not (
        marker.get("schema") == "sceneplan_transfusion_editing_training_index"
        and int(marker.get("schema_version", -1)) == 1
        and marker.get("state") == "materialized_complete_frozen"
        and marker.get("split") == "validation"
        and int(marker.get("rows", -1)) == EXPECTED_INDEX_ROWS
        and marker.get("index_sha256") == index_sha
        and Path(str(marker.get("index_path") or "")).resolve() == index
        and metadata.get("editing_ar_input_contract")
        == "source_foa_latent_plus_raw_edit_request_v2"
        and metadata.get("editing_ar_old_sceneplan_input") == "false"
        and count == EXPECTED_INDEX_ROWS
        and len(selected) == int(rows)
    ):
        raise RuntimeError("frozen Editing validation index cannot support the pilot")
    output = []
    for values in selected:
        record = dict(zip(SELECT_COLUMNS, values))
        record["pair_ordinal"] = int(record["pair_ordinal"])
        record["model_num_samples"] = int(record["model_num_samples"])
        record["latent_frames_valid"] = int(record["latent_frames_valid"])
        record["latent_bucket_frames"] = int(record["latent_bucket_frames"])
        output.append(record)
    return index_sha, output


def _load_source_latents(
    records: list[dict[str, Any]], *, device: torch.device
) -> torch.Tensor:
    rows = []
    for record in records:
        path = Path(str(record["source_latent_path"])).resolve(strict=True)
        key = str(record["source_latent_key"])
        with safe_open(path, framework="pt", device="cpu") as handle:
            if key not in handle.keys():
                raise RuntimeError(f"{record['pair_id']}: source latent key is absent")
            latent = handle.get_tensor(key).clone()
        if (
            latent.dtype != torch.float16
            or tuple(latent.shape) != (64, EXPECTED_VALID_FRAMES)
            or int(record["latent_frames_valid"]) != EXPECTED_VALID_FRAMES
            or int(record["latent_bucket_frames"]) != 648
            or not bool(torch.isfinite(latent).all())
            or _tensor_sha256(latent)
            != str(record["source_latent_tensor_sha256"])
        ):
            raise RuntimeError(f"{record['pair_id']}: source latent identity changed")
        rows.append(latent)
    return torch.stack(rows).to(device=device, dtype=torch.float32)


def _cosine_distance(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return 1.0 - torch.nn.functional.cosine_similarity(left.float(), right.float())


def main() -> int:
    args = _parse_args()
    if os.environ.get("M2D_NONCOMMERCIAL_EVALUATION_ACK") != "1":
        raise RuntimeError(
            "M2D temporal pilot requires authorized internal non-commercial "
            "evaluation acknowledgement"
        )
    if not 1 <= int(args.rows) <= 16:
        raise ValueError("--rows must be within [1,16]")
    device = _device(args.physical_gpu)
    torch.set_float32_matmul_precision("high")
    index = args.index.expanduser().resolve(strict=True)
    index_sha, records = _index_rows(index, rows=int(args.rows))
    vae_config = args.vae_config.expanduser().resolve(strict=True)
    vae_checkpoint = args.vae_checkpoint.expanduser().resolve(strict=True)
    if sha256_file(vae_config) != EDITING_M2D_VAE_CONFIG_SHA256:
        raise RuntimeError("frozen FOA VAE configuration SHA256 changed")
    if sha256_file(vae_checkpoint) != EDITING_M2D_VAE_CHECKPOINT_SHA256:
        raise RuntimeError("frozen FOA VAE checkpoint SHA256 changed")

    vae = create_model_from_config(load_config(vae_config))
    copy_state_dict(vae, load_ckpt_state_dict(str(vae_checkpoint)))
    vae.eval().requires_grad_(False).to(device)
    m2d = FrozenEditingM2DCLAP(
        device=device, load_text_encoder=False
    ).eval().requires_grad_(False)
    latents = _load_source_latents(records, device=device)
    with torch.inference_mode():
        decoded = vae.decode(latents).float()
    if tuple(decoded.shape) != (
        len(records),
        4,
        EXPECTED_VALID_FRAMES * VAE_HOP_SAMPLES,
    ) or not bool(torch.isfinite(decoded).all()):
        raise RuntimeError("frozen VAE pilot decode geometry/value changed")
    waveforms = [
        decoded_foa_w_to_m2d_waveform(
            audio, valid_samples=int(record["model_num_samples"])
        )
        for audio, record in zip(decoded, records)
    ]
    if any(
        waveform.ndim != 1
        or int(waveform.shape[-1]) < MINIMUM_M2D_SAMPLES
        for waveform in waveforms
    ):
        raise RuntimeError("decoded-W pilot did not retain at least 15 seconds")

    projector_token_counts: list[int] = []

    def observe_projector_tokens(_module, inputs) -> None:
        if len(inputs) != 1 or inputs[0].ndim != 3:
            raise RuntimeError("M2D audio projector input contract changed")
        projector_token_counts.append(int(inputs[0].shape[-2]))

    hook = m2d.runtime.backbone.audio_proj.register_forward_pre_hook(
        observe_projector_tokens
    )
    try:
        prefix_rows = []
        complete_rows = []
        zero_tail_rows = []
        row_token_counts = []
        for waveform in waveforms:
            call_start = len(projector_token_counts)
            prefix_rows.append(m2d.encode_audio(waveform[None, : 10 * 16_000])[0])
            complete_rows.append(m2d.encode_audio(waveform[None])[0])
            zero_tail_waveform = waveform.clone()
            zero_tail_waveform[10 * 16_000 :] = 0.0
            zero_tail_rows.append(m2d.encode_audio(zero_tail_waveform[None])[0])
            row_token_counts.append(projector_token_counts[call_start:])
        prefix = torch.stack(prefix_rows)
        complete = torch.stack(complete_rows)
        zero_tail = torch.stack(zero_tail_rows)

        silence = torch.zeros(5, MINIMUM_M2D_SAMPLES, device=device)
        time_axis = torch.arange(5 * 16_000, device=device) / 16_000.0
        tone = 0.5 * torch.sin(2.0 * torch.pi * 440.0 * time_axis)
        probes = silence.clone()
        for segment in range(3):
            start = segment * 5 * 16_000
            probes[segment + 1, start : start + 5 * 16_000] = tone
        # The upstream 1001-mel chunker drops mel frames 992..1000 because
        # 1001 is not divisible by the 16-frame patch stride. A narrow event
        # fully inside that old ~10 s boundary gap must still reach the final
        # semantic vector under the aligned 992-frame policy.
        boundary_start = round(9.94 * 16_000)
        boundary_end = round(9.98 * 16_000)
        boundary_axis = (
            torch.arange(boundary_end - boundary_start, device=device) / 16_000.0
        )
        probes[4, boundary_start:boundary_end] = 0.5 * torch.sin(
            2.0 * torch.pi * 1760.0 * boundary_axis
        )
        probe_embeddings = m2d.encode_audio(probes)
    finally:
        hook.remove()

    if len(projector_token_counts) != 3 * len(records) + 1:
        raise RuntimeError("M2D temporal pilot did not observe every projector call")
    if any(len(value) != 3 for value in row_token_counts):
        raise RuntimeError("M2D temporal pilot lost a real-audio projector trace")
    probe_tokens = projector_token_counts[-1]
    real_tail_distance = _cosine_distance(complete, zero_tail)
    full_rms = torch.stack(
        [waveform.float().square().mean().sqrt() for waveform in waveforms]
    )
    tail_rms = torch.stack(
        [
            waveform[10 * 16_000 :].float().square().mean().sqrt()
            for waveform in waveforms
        ]
    )
    tail_threshold = torch.maximum(
        torch.full_like(tail_rms, REAL_TAIL_ABSOLUTE_RMS_MIN),
        full_rms * REAL_TAIL_RELATIVE_RMS_MIN,
    )
    active_tail = tail_rms >= tail_threshold
    segment_distances = _cosine_distance(
        probe_embeddings[1:4], probe_embeddings[0:1].expand(3, -1)
    )
    boundary_distance = _cosine_distance(
        probe_embeddings[4:5], probe_embeddings[0:1]
    )[0]
    norms = complete.float().norm(dim=-1)
    checks = {
        "source_audio_view_is_decoded_W": M2D_CLAP_SOURCE_AUDIO_VIEW
        == "frozen_vae_decode_of_source_foa_latent_W",
        "all_waveforms_cover_at_least_15_seconds": all(
            int(value.shape[-1]) >= MINIMUM_M2D_SAMPLES for value in waveforms
        ),
        "full_projector_has_more_tokens_than_10s": all(
            complete_tokens > prefix_tokens > 0
            for prefix_tokens, complete_tokens, _ in row_token_counts
        ),
        "tail_intervention_preserves_full_token_geometry": all(
            complete_tokens == zero_tail_tokens
            for _, complete_tokens, zero_tail_tokens in row_token_counts
        ),
        "synthetic_15s_probe_uses_long_sequence": probe_tokens
        > row_token_counts[0][0],
        "all_embeddings_are_finite": bool(
            torch.isfinite(prefix).all()
            and torch.isfinite(complete).all()
            and torch.isfinite(zero_tail).all()
            and torch.isfinite(probe_embeddings).all()
        ),
        "full_embeddings_are_l2_normalized": bool(
            torch.all((norms >= 0.999) & (norms <= 1.001))
        ),
        # A silent post-10-second region should be invariant when zeroed and
        # is not evidence against coverage.  Require every row with measurable
        # real tail content to propagate that tail into the final vector.
        "real_active_tail_rows_present": bool(torch.any(active_tail)),
        "every_real_active_tail_changes_embedding": bool(
            torch.any(active_tail)
            and torch.all(real_tail_distance[active_tail] > 1.0e-8)
        ),
        "early_middle_late_probes_all_reach_embedding": bool(
            torch.all(segment_distances > 1.0e-8)
        ),
        "boundary_narrow_probe_reaches_embedding": bool(
            boundary_distance > 1.0e-8
        ),
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    result = {
        "schema": "sceneplan_transfusion_editing_m2d_temporal_policy_pilot",
        "schema_version": 2,
        "status": status,
        "checks": checks,
        "physical_gpu": int(args.physical_gpu),
        "rows": len(records),
        "source_index": str(index),
        "source_index_sha256": index_sha,
        "source_index_rows": EXPECTED_INDEX_ROWS,
        "selected_query_columns": list(SELECT_COLUMNS),
        "pair_ordinals": [int(record["pair_ordinal"]) for record in records],
        "pair_ids": [str(record["pair_id"]) for record in records],
        "source_latent_tensor_sha256": [
            str(record["source_latent_tensor_sha256"]) for record in records
        ],
        "old_sceneplan_model_input": False,
        "new_sceneplan_or_target_information_used": False,
        "source_audio_view": M2D_CLAP_SOURCE_AUDIO_VIEW,
        "temporal_policy": M2D_CLAP_TEMPORAL_POLICY,
        "audio_preprocess": EDITING_M2D_AUDIO_PREPROCESS,
        "minimum_valid_model_samples": MINIMUM_MODEL_SAMPLES,
        "selected_model_num_samples": [
            int(record["model_num_samples"]) for record in records
        ],
        "selected_m2d_samples": [int(value.shape[-1]) for value in waveforms],
        "projector_tokens": [
            {
                "first_10_seconds": prefix_tokens,
                "full_valid_clip": complete_tokens,
                "full_valid_clip_with_zero_tail": zero_tail_tokens,
            }
            for prefix_tokens, complete_tokens, zero_tail_tokens in row_token_counts
        ],
        "real_tail_zero_cosine_distance": [
            float(value) for value in real_tail_distance.cpu()
        ],
        "real_tail_eligibility": {
            "absolute_rms_min": REAL_TAIL_ABSOLUTE_RMS_MIN,
            "relative_to_full_rms_min": REAL_TAIL_RELATIVE_RMS_MIN,
            "full_rms": [float(value) for value in full_rms.cpu()],
            "post_10_second_rms": [float(value) for value in tail_rms.cpu()],
            "eligible": [bool(value) for value in active_tail.cpu()],
        },
        "synthetic_segment_vs_silence_cosine_distance": {
            name: float(value)
            for name, value in zip(
                ("early", "middle", "late"), segment_distances.cpu()
            )
        },
        "synthetic_boundary_gap_probe": {
            "start_sec": boundary_start / 16_000,
            "end_sec": boundary_end / 16_000,
            "cosine_distance_from_silence": float(boundary_distance.cpu()),
        },
        "vae": {
            "config": str(vae_config),
            "config_sha256": EDITING_M2D_VAE_CONFIG_SHA256,
            "checkpoint": str(vae_checkpoint),
            "checkpoint_sha256": EDITING_M2D_VAE_CHECKPOINT_SHA256,
        },
        "m2d_assets": m2d.asset_report,
        "implementation_sha256": editing_m2d_cache_implementation_sha256(),
        "numeric_runtime_fingerprint": editing_m2d_numeric_runtime_fingerprint(
            device
        ),
        "verifier": str(Path(__file__).resolve()),
        "verifier_sha256": sha256_file(Path(__file__).resolve()),
    }
    output = args.output.expanduser().resolve()
    _atomic_json(output, result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    if status != "PASS":
        raise RuntimeError("Editing M2D 15-second temporal policy pilot failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
