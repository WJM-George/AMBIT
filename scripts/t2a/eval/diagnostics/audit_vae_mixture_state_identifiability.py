#!/usr/bin/env python3
"""Audit whether the frozen FOA VAE exposes identifiable mixture interventions.

This is a target-side diagnostic, not a training route.  For each calibrated
turn-0 family it reconstructs the exact full mixture, every isolated source,
and every leave-one-source-out mixture from the authoritative recipes.  The
VAE posterior parameters are evaluated directly, then eight posterior draws
use one epsilon shared by all states.  This removes unrelated sampling noise
from each ``full - leave-one-out`` intervention without pretending that the
posterior is deterministic.

Two independent questions are answered:

* Does decoding the latent intervention preserve the identity of the removed
  source under target-audio CLAP and target-only AST anchors?
* Is the latent intervention direction stable across posterior draws and
  separable from the other sources in the same mixture?

The linear rectified-flow interpolation cannot provide another direction
test: with shared base noise its paired difference is exactly ``t * delta_z``.
We therefore report intervention-to-state RMS ratios across RF times, but do
not count their necessarily invariant direction as evidence.
"""
from __future__ import annotations
import os

import argparse
import copy
import hashlib
import json
import math
import sys
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from transformers import ASTFeatureExtractor, ASTForAudioClassification

from dataset.synthesis.render_spatial_edit_families import (
    _flac_pcm24_memory_roundtrip,
)
from scripts.t2a.data.build_spatial_cot_source_curriculum import (
    EXPECTED_VAE_SHA256,
    _load_recipe_families,
    _pcm24_track,
)
from scripts.t2a.data.preencode_spatial_cot_family_shard import _load_vae
from scripts.t2a.eval.diagnostics.audit_vae_source_semantics import (
    _active_clap_waveform,
)
from scripts.t2a.eval.diagnostics.score_source_location_semantics import (
    _assignment_metrics,
    _compile_active_source_tracks,
)
from scripts.t2a.eval.diagnostics.score_source_semantics_ast import (
    DEFAULT_CACHE as DEFAULT_AST_CACHE,
    DEFAULT_MODEL as DEFAULT_AST_MODEL,
    _active_ast_waveform,
    _anchor_assignment,
    _labels as _ast_labels,
    _select_target_anchors,
)
from stable_audio_tools.data.spatial_family_dataset import SpatialFamilyDataset
from stable_audio_tools.data.t2a_artifacts import atomic_write_json
from stable_audio_tools.training.metrics.fad_metrics import load_clap_model


DEFAULT_FAMILY_STORE = Path(os.environ.get("AMBIT_DATA_ROOT", "data") + "/spatial_cot_v1/latents/train")
DEFAULT_CAPTION_OVERLAY = Path(
    os.environ.get("AMBIT_DATA_ROOT", "data") + "/spatial_cot_v1/captions/"
    "spatial_source_regions_v3/train"
)
DEFAULT_VAE_CONFIG = REPO_ROOT / (
    "stable_audio_tools/configs/model_configs/autoencoders/"
    "stable_audio_4ch_vae_ds1024_z64_wdmix_scm.json"
)
DEFAULT_VAE_CHECKPOINT = Path(
    os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/compareVAE_ckpt/unwrapped_wdmix_1350000.ckpt"
)

DRAW_COUNT = 8
REQUIRED_DRAW_PASSES = 7
MIN_CLAP_TARGET_GAP = 0.05
MIN_AST_ANCHOR_PROBABILITY = 0.02
MIN_AST_ANCHOR_MARGIN = 0.01
MIN_AST_TARGET_GAP = 0.01
MIN_AST_RETENTION = 0.25
MIN_RMS_RETENTION = 0.25
MAX_RMS_RETENTION = 4.0
MIN_MEDIAN_DIRECTION_COSINE = 0.80
MIN_DIRECTION_COSINE = 0.60
MIN_DIRECTION_MARGIN = 0.05
RF_TIMES = (0.25, 0.5, 0.75, 1.0)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rms(value: torch.Tensor) -> float:
    tensor = torch.as_tensor(value, dtype=torch.float32)
    return float(tensor.square().mean().sqrt())


def _summary(values: Sequence[float]) -> dict[str, float]:
    if not values:
        raise ValueError("cannot summarize an empty sequence")
    tensor = torch.tensor(list(values), dtype=torch.float64)
    return {
        "min": float(tensor.min()),
        "median": float(torch.quantile(tensor, 0.5)),
        "mean": float(tensor.mean()),
        "max": float(tensor.max()),
    }


def _cosine_matrix(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left = torch.as_tensor(left, dtype=torch.float32).flatten(1)
    right = torch.as_tensor(right, dtype=torch.float32).flatten(1)
    left = F.normalize(left, dim=-1)
    right = F.normalize(right, dim=-1)
    return left @ right.transpose(0, 1)


def _group_slice(name: str, n_w: int) -> slice:
    if name == "full":
        return slice(None)
    if name == "w":
        return slice(0, n_w)
    if name == "spatial":
        return slice(n_w, None)
    raise ValueError(f"unknown latent group: {name}")


def _posterior_parameters(
    model,
    audio: Sequence[np.ndarray],
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return exact grouped-VAE posterior mean/stdev without sampling it."""

    if batch_size <= 0:
        raise ValueError("posterior batch size must be positive")
    if model.pretransform is not None:
        raise RuntimeError("the frozen FOA VAE unexpectedly gained a pretransform")
    if not hasattr(model.bottleneck, "n_w"):
        raise RuntimeError("the frozen FOA VAE no longer has a grouped bottleneck")
    means = []
    stdevs = []
    for start in range(0, len(audio), batch_size):
        batch = torch.from_numpy(np.stack(audio[start : start + batch_size])).to(
            device=device, dtype=torch.float32, non_blocking=True
        )
        with torch.inference_mode():
            encoded = model.encoder(batch)
        latent_dim = int(model.latent_dim)
        if encoded.ndim != 3 or encoded.shape[1] != 2 * latent_dim:
            raise RuntimeError(
                f"unexpected pre-bottleneck VAE shape: {tuple(encoded.shape)}"
            )
        mean, scale = encoded.chunk(2, dim=1)
        stdev = F.softplus(scale) + 1.0e-4
        if tuple(mean.shape[1:]) != (latent_dim, 432):
            raise RuntimeError(f"unexpected VAE posterior shape: {tuple(mean.shape)}")
        if not torch.isfinite(mean).all() or not torch.isfinite(stdev).all():
            raise RuntimeError("VAE posterior contains non-finite values")
        means.append(mean)
        stdevs.append(stdev)
    return torch.cat(means, dim=0), torch.cat(stdevs, dim=0)


def _shared_posterior_draws(
    mean: torch.Tensor,
    stdev: torch.Tensor,
    *,
    draw_count: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw one epsilon per draw and broadcast it across every target state."""

    if mean.shape != stdev.shape or mean.ndim != 3:
        raise ValueError("posterior mean/stdev must be matching [state,C,T] tensors")
    generator = torch.Generator(device=mean.device)
    generator.manual_seed(int(seed))
    epsilon = torch.randn(
        (draw_count, 1, *mean.shape[1:]),
        generator=generator,
        device=mean.device,
        dtype=mean.dtype,
    )
    draws = mean.unsqueeze(0) + stdev.unsqueeze(0) * epsilon
    return draws, epsilon[:, 0]


def _direction_group_report(
    draw_delta: torch.Tensor,
    mean_delta: torch.Tensor,
    *,
    source_ids: Sequence[str],
    group: str,
    n_w: int,
) -> dict[str, Any]:
    """Score draw stability and source assignment against mean directions."""

    channel_slice = _group_slice(group, n_w)
    draws = draw_delta[:, :, channel_slice]
    means = mean_delta[:, channel_slice]
    source_count = len(source_ids)
    if tuple(draws.shape[:2]) != (DRAW_COUNT, source_count):
        raise ValueError("draw deltas do not match the frozen audit shape")
    mean_cosine = _cosine_matrix(means, means).cpu()
    source_rows = []
    for source_index, source_id in enumerate(source_ids):
        values = draws[:, source_index]
        own_mean = means[source_index : source_index + 1]
        draw_to_own = _cosine_matrix(values, own_mean)[:, 0].cpu()
        draw_to_all = _cosine_matrix(values, means).cpu()
        assignments = torch.argmax(draw_to_all, dim=1)
        other_indices = [index for index in range(source_count) if index != source_index]
        if other_indices:
            nearest_other = draw_to_all[:, other_indices].amax(dim=1)
            margins = draw_to_all[:, source_index] - nearest_other
            mean_nearest_other = max(
                float(mean_cosine[source_index, index]) for index in other_indices
            )
        else:
            margins = torch.full_like(draw_to_own, 2.0)
            mean_nearest_other = -1.0
        cosine_values = [float(value) for value in draw_to_own]
        margin_values = [float(value) for value in margins]
        correct_count = int((assignments == source_index).sum())
        stable = (
            median(cosine_values) >= MIN_MEDIAN_DIRECTION_COSINE
            and min(cosine_values) >= MIN_DIRECTION_COSINE
            and correct_count >= REQUIRED_DRAW_PASSES
            and median(margin_values) >= MIN_DIRECTION_MARGIN
        )
        source_rows.append(
            {
                "source_id": source_id,
                "mean_delta_rms": _rms(means[source_index]),
                "draw_delta_rms": _summary(
                    [_rms(values[draw]) for draw in range(DRAW_COUNT)]
                ),
                "draw_to_mean_cosine": _summary(cosine_values),
                "draw_assignment_correct_count": correct_count,
                "draw_assignment_margin": _summary(margin_values),
                "mean_nearest_other_cosine": mean_nearest_other,
                "status": "PASS" if stable else "BLOCK",
            }
        )
    return {
        "group": group,
        "mean_direction_cosine": {
            source_id: {
                other_id: float(mean_cosine[row, column])
                for column, other_id in enumerate(source_ids)
            }
            for row, source_id in enumerate(source_ids)
        },
        "sources": source_rows,
        "status": (
            "PASS" if all(row["status"] == "PASS" for row in source_rows) else "BLOCK"
        ),
    }


def _rf_snr_report(
    full_draws: torch.Tensor,
    draw_delta: torch.Tensor,
    *,
    source_ids: Sequence[str],
    seed: int,
) -> list[dict[str, Any]]:
    generator = torch.Generator(device=full_draws.device)
    generator.manual_seed(int(seed) + 97_531)
    base_noise = torch.randn(
        full_draws.shape,
        generator=generator,
        device=full_draws.device,
        dtype=full_draws.dtype,
    )
    rows = []
    for source_index, source_id in enumerate(source_ids):
        times = []
        for time in RF_TIMES:
            state = (1.0 - time) * base_noise + time * full_draws
            ratios = []
            for draw in range(DRAW_COUNT):
                numerator = time * _rms(draw_delta[draw, source_index])
                denominator = max(_rms(state[draw]), 1.0e-8)
                ratios.append(numerator / denominator)
            times.append(
                {
                    "time": time,
                    "intervention_to_full_state_rms": _summary(ratios),
                }
            )
        rows.append(
            {
                "source_id": source_id,
                "times": times,
                "direction_note": (
                    "shared-noise RF pair direction is exactly invariant for t>0"
                ),
            }
        )
    return rows


def _latent_report(
    mean: torch.Tensor,
    stdev: torch.Tensor,
    draws: torch.Tensor,
    epsilon: torch.Tensor,
    *,
    source_ids: Sequence[str],
    state_index: Mapping[str, int],
    n_w: int,
    seed: int,
) -> dict[str, Any]:
    full_index = state_index["full"]
    silence_index = state_index["silence"]
    minus_indices = [state_index[f"minus:{source_id}"] for source_id in source_ids]
    isolated_indices = [
        state_index[f"isolated:{source_id}"] for source_id in source_ids
    ]
    mean_delta = torch.stack(
        [mean[full_index] - mean[index] for index in minus_indices], dim=0
    )
    draw_delta = torch.stack(
        [draws[:, full_index] - draws[:, index] for index in minus_indices], dim=1
    )
    isolated_delta = torch.stack(
        [
            draws[:, index] - draws[:, silence_index]
            for index in isolated_indices
        ],
        dim=1,
    )
    groups = {
        group: _direction_group_report(
            draw_delta,
            mean_delta,
            source_ids=source_ids,
            group=group,
            n_w=n_w,
        )
        for group in ("full", "w", "spatial")
    }
    alignments = []
    posterior_rows = []
    for source_index, source_id in enumerate(source_ids):
        alignment = _cosine_matrix(
            draw_delta[:, source_index], isolated_delta[:, source_index]
        ).diagonal()
        alignments.append(
            {
                "source_id": source_id,
                "mixture_to_isolated_direction_cosine": _summary(
                    [float(value) for value in alignment]
                ),
            }
        )
        minus_index = minus_indices[source_index]
        sigma_delta = stdev[full_index] - stdev[minus_index]
        stochastic_delta = sigma_delta.unsqueeze(0) * epsilon
        mean_rms = _rms(mean_delta[source_index])
        posterior_rows.append(
            {
                "source_id": source_id,
                "posterior_mean_delta_rms": mean_rms,
                "posterior_sigma_delta_rms": _rms(sigma_delta),
                "sampled_stochastic_delta_rms": _summary(
                    [_rms(value) for value in stochastic_delta]
                ),
                "median_stochastic_to_mean_rms": median(
                    [_rms(value) / max(mean_rms, 1.0e-8) for value in stochastic_delta]
                ),
            }
        )
    return {
        "shared_posterior_epsilon": True,
        "draw_count": DRAW_COUNT,
        "posterior": posterior_rows,
        "direction_groups": groups,
        "mixture_to_isolated_alignment": alignments,
        "rf_intervention_snr": _rf_snr_report(
            draws[:, full_index], draw_delta, source_ids=source_ids, seed=seed
        ),
        "status": groups["full"]["status"],
    }


def _decode_w_draws(
    model,
    draws: torch.Tensor,
    *,
    source_ids: Sequence[str],
    state_index: Mapping[str, int],
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode only states needed by semantic gates and retain their W channel."""

    if batch_size <= 0:
        raise ValueError("decode batch size must be positive")
    full_index = state_index["full"]
    minus_indices = [state_index[f"minus:{source_id}"] for source_id in source_ids]
    isolated_indices = [
        state_index[f"isolated:{source_id}"] for source_id in source_ids
    ]
    selected = [full_index, *minus_indices, *isolated_indices]
    delta_rows = []
    isolated_rows = []
    for draw in range(DRAW_COUNT):
        latent = draws[draw, selected]
        decoded_chunks = []
        for start in range(0, len(selected), batch_size):
            with torch.inference_mode():
                decoded = model.decode(latent[start : start + batch_size])
            if decoded.ndim != 3 or decoded.shape[1] != 4:
                raise RuntimeError(f"unexpected decoded FOA shape: {tuple(decoded.shape)}")
            decoded_chunks.append(decoded[:, 0].float().cpu())
        w_audio = torch.cat(decoded_chunks, dim=0) * math.sqrt(2.0)
        full = w_audio[0]
        delta_rows.append(
            torch.stack(
                [full - w_audio[1 + index] for index in range(len(source_ids))]
            )
        )
        isolated_start = 1 + len(source_ids)
        isolated_rows.append(w_audio[isolated_start:])
    return torch.stack(delta_rows), torch.stack(isolated_rows)


def _clap_embeddings(clap_model, waves: Sequence[torch.Tensor]) -> torch.Tensor:
    rows = []
    with torch.inference_mode():
        for wave in waves:
            embedding = clap_model.get_audio_embedding_from_data(
                x=wave[None], use_tensor=True
            ).float()
            if embedding.ndim != 2 or embedding.shape[0] != 1:
                raise RuntimeError(
                    f"CLAP returned unexpected embedding shape: {tuple(embedding.shape)}"
                )
            rows.append(embedding[0])
    return torch.stack(rows)


def _ast_probabilities(
    model: ASTForAudioClassification,
    extractor: ASTFeatureExtractor,
    waves: Sequence[torch.Tensor],
    *,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    if batch_size <= 0:
        raise ValueError("AST batch size must be positive")
    probabilities = []
    for start in range(0, len(waves), batch_size):
        values = [wave.cpu().numpy() for wave in waves[start : start + batch_size]]
        features = extractor(
            values,
            sampling_rate=16_000,
            return_tensors="pt",
            padding=True,
        )
        with torch.inference_mode():
            logits = model(
                **{key: value.to(device) for key, value in features.items()}
            ).logits
        probabilities.append(torch.sigmoid(logits.float()).cpu())
    return torch.cat(probabilities, dim=0)


def _active_rms(
    wave: torch.Tensor,
    active_frames: torch.Tensor,
    *,
    hop: int,
) -> float:
    mask = torch.as_tensor(active_frames, dtype=torch.bool).repeat_interleave(hop)
    value = torch.as_tensor(wave, dtype=torch.float32).cpu().reshape(-1)
    if value.numel() != mask.numel() or not bool(mask.any()):
        raise ValueError("active RMS mask does not match the waveform")
    return _rms(value[mask])


def _semantic_branch_rows(
    *,
    branch: str,
    source_ids: Sequence[str],
    target_clap_matrix: torch.Tensor,
    draw_clap_matrices: Sequence[torch.Tensor],
    ast_anchors: Sequence[Mapping[str, Any]],
    target_ast_matrix: torch.Tensor,
    draw_ast_matrices: Sequence[torch.Tensor],
    target_rms: Sequence[float],
    draw_rms: Sequence[Sequence[float]],
) -> list[dict[str, Any]]:
    target_clap = _assignment_metrics(
        target_clap_matrix,
        target_clap_matrix,
        source_ids=source_ids,
        min_target_gap=MIN_CLAP_TARGET_GAP,
    )
    clap_draws = [
        _assignment_metrics(
            matrix,
            target_clap_matrix,
            source_ids=source_ids,
            min_target_gap=MIN_CLAP_TARGET_GAP,
        )
        for matrix in draw_clap_matrices
    ]
    target_ast = _anchor_assignment(
        target_ast_matrix,
        target_ast_matrix,
        anchors=ast_anchors,
        source_ids=source_ids,
        min_target_gap=MIN_AST_TARGET_GAP,
    )
    ast_draws = [
        _anchor_assignment(
            matrix,
            target_ast_matrix,
            anchors=ast_anchors,
            source_ids=source_ids,
            min_target_gap=MIN_AST_TARGET_GAP,
        )
        for matrix in draw_ast_matrices
    ]
    rows = []
    for source_index, source_id in enumerate(source_ids):
        clap_target_row = target_clap["rows"][source_index]
        ast_target_row = target_ast["rows"][source_index]
        calibrated = bool(clap_target_row["target_valid"]) and bool(
            ast_target_row["target_valid"]
        )
        clap_rows = [report["rows"][source_index] for report in clap_draws]
        ast_rows = [report["rows"][source_index] for report in ast_draws]
        clap_correct = sum(bool(row["generated_correct"]) for row in clap_rows)
        ast_correct = sum(bool(row["generated_correct"]) for row in ast_rows)
        clap_assignments = {
            candidate: sum(row["generated_assignment"] == candidate for row in clap_rows)
            for candidate in source_ids
        }
        ast_assignments = {
            candidate: sum(row["generated_assignment"] == candidate for row in ast_rows)
            for candidate in source_ids
        }
        ast_retention = [
            float(row["generated_to_target_anchor_ratio"]) for row in ast_rows
        ]
        rms_retention = [
            float(draw_rms[draw][source_index])
            / max(float(target_rms[source_index]), 1.0e-8)
            for draw in range(DRAW_COUNT)
        ]
        passed = (
            calibrated
            and clap_correct >= REQUIRED_DRAW_PASSES
            and ast_correct >= REQUIRED_DRAW_PASSES
            and median(ast_retention) >= MIN_AST_RETENTION
            and MIN_RMS_RETENTION
            <= median(rms_retention)
            <= MAX_RMS_RETENTION
        )
        rows.append(
            {
                "source_id": source_id,
                "branch": branch,
                "target_calibrated": calibrated,
                "target_clap_discriminability_gap": float(
                    clap_target_row["target_discriminability_gap"]
                ),
                "target_ast_anchor": str(ast_target_row["anchor_label"]),
                "target_ast_discriminability_gap": float(
                    ast_target_row["target_discriminability_gap"]
                ),
                "clap_correct_draw_count": clap_correct,
                "clap_assignment_counts": clap_assignments,
                "clap_assignment_margin": _summary(
                    [float(row["generated_diagonal_margin"]) for row in clap_rows]
                ),
                "ast_correct_draw_count": ast_correct,
                "ast_assignment_counts": ast_assignments,
                "ast_assignment_margin": _summary(
                    [float(row["generated_diagonal_margin"]) for row in ast_rows]
                ),
                "ast_target_retention": _summary(ast_retention),
                "active_rms_retention": _summary(rms_retention),
                "status": (
                    "ABSTAIN_CALIBRATION"
                    if not calibrated
                    else ("PASS" if passed else "BLOCK")
                ),
            }
        )
    return rows


def _semantic_report(
    *,
    clap_model,
    ast_model: ASTForAudioClassification,
    ast_extractor: ASTFeatureExtractor,
    ast_label_names: Sequence[str],
    device: torch.device,
    plan: Mapping[str, Any],
    source_ids: Sequence[str],
    exact_delta_w: Sequence[torch.Tensor],
    exact_isolated_w: Sequence[torch.Tensor],
    decoded_delta_w: torch.Tensor,
    decoded_isolated_w: torch.Tensor,
    sample_rate: int,
    ast_batch_size: int,
) -> dict[str, Any]:
    source_count = len(source_ids)
    tracks = _compile_active_source_tracks(
        plan, num_frames=432, source_ids=source_ids
    )
    hop = int(exact_delta_w[0].numel() // 432)
    if hop != 1024:
        raise RuntimeError(f"frozen 10-second target hop changed: {hop}")
    plan_sources = {
        str(source["source_id"]): source
        for source in ((plan.get("scene") or {}).get("sources") or [])
    }

    target_delta_clap_waves = []
    target_isolated_clap_waves = []
    draw_delta_clap_waves = []
    draw_isolated_clap_waves = []
    target_delta_ast_waves = []
    target_isolated_ast_waves = []
    draw_delta_ast_waves = []
    draw_isolated_ast_waves = []
    target_delta_rms = []
    target_isolated_rms = []
    draw_delta_rms = [[] for _ in range(DRAW_COUNT)]
    draw_isolated_rms = [[] for _ in range(DRAW_COUNT)]

    for source_index, source_id in enumerate(source_ids):
        source = plan_sources[source_id]
        activity = source.get("activity") or {}
        onset = float(activity["onset_sec"])
        offset = float(activity["offset_sec"])
        active_frames = tracks[source_index, 0] > 0.5
        target_delta_clap_waves.append(
            _active_clap_waveform(
                exact_delta_w[source_index],
                sample_rate=sample_rate,
                onset_sec=onset,
                offset_sec=offset,
                device=device,
            )
        )
        target_isolated_clap_waves.append(
            _active_clap_waveform(
                exact_isolated_w[source_index],
                sample_rate=sample_rate,
                onset_sec=onset,
                offset_sec=offset,
                device=device,
            )
        )
        delta_ast, _ = _active_ast_waveform(
            exact_delta_w[source_index],
            active_frames,
            hop=hop,
            sample_rate=sample_rate,
        )
        isolated_ast, _ = _active_ast_waveform(
            exact_isolated_w[source_index],
            active_frames,
            hop=hop,
            sample_rate=sample_rate,
        )
        target_delta_ast_waves.append(delta_ast)
        target_isolated_ast_waves.append(isolated_ast)
        target_delta_rms.append(
            _active_rms(exact_delta_w[source_index], active_frames, hop=hop)
        )
        target_isolated_rms.append(
            _active_rms(exact_isolated_w[source_index], active_frames, hop=hop)
        )
        for draw in range(DRAW_COUNT):
            draw_delta_clap_waves.append(
                _active_clap_waveform(
                    decoded_delta_w[draw, source_index],
                    sample_rate=sample_rate,
                    onset_sec=onset,
                    offset_sec=offset,
                    device=device,
                )
            )
            draw_isolated_clap_waves.append(
                _active_clap_waveform(
                    decoded_isolated_w[draw, source_index],
                    sample_rate=sample_rate,
                    onset_sec=onset,
                    offset_sec=offset,
                    device=device,
                )
            )
            delta_ast, _ = _active_ast_waveform(
                decoded_delta_w[draw, source_index],
                active_frames,
                hop=hop,
                sample_rate=sample_rate,
            )
            isolated_ast, _ = _active_ast_waveform(
                decoded_isolated_w[draw, source_index],
                active_frames,
                hop=hop,
                sample_rate=sample_rate,
            )
            draw_delta_ast_waves.append(delta_ast)
            draw_isolated_ast_waves.append(isolated_ast)
            draw_delta_rms[draw].append(
                _active_rms(decoded_delta_w[draw, source_index], active_frames, hop=hop)
            )
            draw_isolated_rms[draw].append(
                _active_rms(
                    decoded_isolated_w[draw, source_index], active_frames, hop=hop
                )
            )

    # The loop above is source-major.  Reorder generated rows to draw-major so
    # every square assignment matrix is [decoded source, exact source identity].
    def draw_major(values: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        return [
            values[source * DRAW_COUNT + draw]
            for draw in range(DRAW_COUNT)
            for source in range(source_count)
        ]

    target_isolated_clap = _clap_embeddings(clap_model, target_isolated_clap_waves)
    target_delta_clap = _clap_embeddings(clap_model, target_delta_clap_waves)
    generated_delta_clap = _clap_embeddings(
        clap_model, draw_major(draw_delta_clap_waves)
    ).reshape(DRAW_COUNT, source_count, -1)
    generated_isolated_clap = _clap_embeddings(
        clap_model, draw_major(draw_isolated_clap_waves)
    ).reshape(DRAW_COUNT, source_count, -1)
    target_delta_clap_matrix = _cosine_matrix(
        target_delta_clap, target_isolated_clap
    ).cpu()
    target_isolated_clap_matrix = _cosine_matrix(
        target_isolated_clap, target_isolated_clap
    ).cpu()
    draw_delta_clap_matrices = [
        _cosine_matrix(generated_delta_clap[draw], target_isolated_clap).cpu()
        for draw in range(DRAW_COUNT)
    ]
    draw_isolated_clap_matrices = [
        _cosine_matrix(generated_isolated_clap[draw], target_isolated_clap).cpu()
        for draw in range(DRAW_COUNT)
    ]

    ast_waves = [
        *target_isolated_ast_waves,
        *target_delta_ast_waves,
        *draw_major(draw_isolated_ast_waves),
        *draw_major(draw_delta_ast_waves),
    ]
    ast_probabilities = _ast_probabilities(
        ast_model,
        ast_extractor,
        ast_waves,
        device=device,
        batch_size=ast_batch_size,
    )
    cursor = 0
    target_isolated_ast = ast_probabilities[cursor : cursor + source_count]
    cursor += source_count
    target_delta_ast = ast_probabilities[cursor : cursor + source_count]
    cursor += source_count
    generated_isolated_ast = ast_probabilities[
        cursor : cursor + DRAW_COUNT * source_count
    ].reshape(DRAW_COUNT, source_count, -1)
    cursor += DRAW_COUNT * source_count
    generated_delta_ast = ast_probabilities[
        cursor : cursor + DRAW_COUNT * source_count
    ].reshape(DRAW_COUNT, source_count, -1)
    anchors = _select_target_anchors(
        target_isolated_ast,
        ast_label_names,
        min_probability=MIN_AST_ANCHOR_PROBABILITY,
        min_margin=MIN_AST_ANCHOR_MARGIN,
    )
    anchor_indices = [int(anchor["label_index"]) for anchor in anchors]
    target_isolated_ast_matrix = target_isolated_ast[:, anchor_indices]
    target_delta_ast_matrix = target_delta_ast[:, anchor_indices]
    draw_isolated_ast_matrices = [
        generated_isolated_ast[draw, :, anchor_indices]
        for draw in range(DRAW_COUNT)
    ]
    draw_delta_ast_matrices = [
        generated_delta_ast[draw, :, anchor_indices] for draw in range(DRAW_COUNT)
    ]

    isolated_rows = _semantic_branch_rows(
        branch="isolated",
        source_ids=source_ids,
        target_clap_matrix=target_isolated_clap_matrix,
        draw_clap_matrices=draw_isolated_clap_matrices,
        ast_anchors=anchors,
        target_ast_matrix=target_isolated_ast_matrix,
        draw_ast_matrices=draw_isolated_ast_matrices,
        target_rms=target_isolated_rms,
        draw_rms=draw_isolated_rms,
    )
    delta_rows = _semantic_branch_rows(
        branch="full_minus_leave_one_out",
        source_ids=source_ids,
        target_clap_matrix=target_delta_clap_matrix,
        draw_clap_matrices=draw_delta_clap_matrices,
        ast_anchors=anchors,
        target_ast_matrix=target_delta_ast_matrix,
        draw_ast_matrices=draw_delta_ast_matrices,
        target_rms=target_delta_rms,
        draw_rms=draw_delta_rms,
    )
    source_rows = []
    for source_id, isolated, delta in zip(source_ids, isolated_rows, delta_rows):
        calibrated = (
            isolated["status"] != "ABSTAIN_CALIBRATION"
            and delta["status"] != "ABSTAIN_CALIBRATION"
        )
        passed = isolated["status"] == "PASS" and delta["status"] == "PASS"
        source_rows.append(
            {
                "source_id": source_id,
                "isolated": isolated,
                "intervention": delta,
                "status": (
                    "ABSTAIN_CALIBRATION"
                    if not calibrated
                    else ("PASS" if passed else "BLOCK")
                ),
            }
        )
    if any(row["status"] == "ABSTAIN_CALIBRATION" for row in source_rows):
        status = "ABSTAIN_CALIBRATION"
    else:
        status = "PASS" if all(row["status"] == "PASS" for row in source_rows) else "BLOCK"
    return {
        "anchor_source": "exact_isolated_pre_vae_audio",
        "sources": source_rows,
        "status": status,
    }


def _materialize_family(
    family: Mapping[str, Any], recipe_family: Mapping[str, Any]
) -> dict[str, Any]:
    rank = int(family["family_rank"])
    family_id = str(family["family_id"])
    recipe = recipe_family["recipes"][0]
    turn = family["turns"][0]
    plan = copy.deepcopy(turn["after"]["scene_plan"])
    plan_sources = ((plan.get("scene") or {}).get("sources") or [])
    source_ids = [str(source["source_id"]) for source in plan_sources]
    recipe_by_id = {str(source["source_id"]): source for source in recipe["sources"]}
    if set(source_ids) != set(recipe_by_id) or len(source_ids) < 2:
        raise RuntimeError(f"family {rank} source identities changed")
    sample_rate = int(recipe["audio"]["sample_rate"])
    if sample_rate != 44_100:
        raise RuntimeError(f"family {rank} sample rate changed: {sample_rate}")
    master_gain = float(
        (family.get("render_provenance") or {}).get(
            "family_master_gain_linear", 1.0
        )
    )
    if not math.isfinite(master_gain) or master_gain <= 0.0:
        raise RuntimeError(f"family {rank} has invalid master gain")
    components = []
    for source_id in source_ids:
        source = recipe_by_id[source_id]
        gain = 10.0 ** (float(source.get("gain_db", 0.0)) / 20.0)
        component = (_pcm24_track(recipe, source) * gain * master_gain).astype(
            np.float32, copy=False
        )
        components.append(component)
    shape = {tuple(component.shape) for component in components}
    if shape != {(4, 442_368)}:
        raise RuntimeError(f"family {rank} source audio shape changed: {shape}")

    def pcm24(value: np.ndarray) -> np.ndarray:
        return _flac_pcm24_memory_roundtrip(
            np.clip(value, -1.0, 1.0).astype(np.float32, copy=False), sample_rate
        )

    full = pcm24(np.sum(np.stack(components), axis=0))
    isolated = [pcm24(component) for component in components]
    leave_one_out = [
        pcm24(
            np.sum(
                np.stack(
                    [component for index, component in enumerate(components) if index != removed]
                ),
                axis=0,
            )
        )
        for removed in range(len(components))
    ]
    target_stats = turn["after"]["signal_stats"]
    peak_error = abs(float(np.max(np.abs(full))) - float(target_stats["peak"]))
    rms_error = abs(
        float(np.sqrt(np.mean(np.square(full, dtype=np.float64))))
        - float(target_stats["rms"])
    )
    if peak_error > 2.0e-5 or rms_error > 2.0e-5:
        raise RuntimeError(
            f"family {rank} exact mixture reconstruction changed: "
            f"peak={peak_error} rms={rms_error}"
        )
    state_names = [
        "full",
        "silence",
        *[f"minus:{source_id}" for source_id in source_ids],
        *[f"isolated:{source_id}" for source_id in source_ids],
    ]
    state_audio = [
        full,
        np.zeros_like(full),
        *leave_one_out,
        *isolated,
    ]
    return {
        "family_rank": rank,
        "family_id": family_id,
        "plan": plan,
        "sample_rate": sample_rate,
        "source_ids": source_ids,
        "state_names": state_names,
        "state_audio": state_audio,
        "full": full,
        "leave_one_out": leave_one_out,
        "isolated": isolated,
        "exact_mixture_peak_error": peak_error,
        "exact_mixture_rms_error": rms_error,
    }


def _audit_family(
    materialized: Mapping[str, Any],
    *,
    vae,
    clap_model,
    ast_model: ASTForAudioClassification,
    ast_extractor: ASTFeatureExtractor,
    ast_label_names: Sequence[str],
    device: torch.device,
    seed: int,
    encode_batch_size: int,
    decode_batch_size: int,
    ast_batch_size: int,
) -> dict[str, Any]:
    source_ids = list(materialized["source_ids"])
    state_names = list(materialized["state_names"])
    state_index = {name: index for index, name in enumerate(state_names)}
    mean, stdev = _posterior_parameters(
        vae,
        materialized["state_audio"],
        device=device,
        batch_size=encode_batch_size,
    )
    family_seed = int(seed) + int(materialized["family_rank"]) * 10_003
    draws, epsilon = _shared_posterior_draws(
        mean, stdev, draw_count=DRAW_COUNT, seed=family_seed
    )
    latent = _latent_report(
        mean,
        stdev,
        draws,
        epsilon,
        source_ids=source_ids,
        state_index=state_index,
        n_w=int(vae.bottleneck.n_w),
        seed=family_seed,
    )
    decoded_delta_w, decoded_isolated_w = _decode_w_draws(
        vae,
        draws,
        source_ids=source_ids,
        state_index=state_index,
        batch_size=decode_batch_size,
    )
    exact_delta_w = [
        torch.from_numpy(materialized["full"][0] - audio[0]) * math.sqrt(2.0)
        for audio in materialized["leave_one_out"]
    ]
    exact_isolated_w = [
        torch.from_numpy(audio[0]) * math.sqrt(2.0)
        for audio in materialized["isolated"]
    ]
    semantic = _semantic_report(
        clap_model=clap_model,
        ast_model=ast_model,
        ast_extractor=ast_extractor,
        ast_label_names=ast_label_names,
        device=device,
        plan=materialized["plan"],
        source_ids=source_ids,
        exact_delta_w=exact_delta_w,
        exact_isolated_w=exact_isolated_w,
        decoded_delta_w=decoded_delta_w,
        decoded_isolated_w=decoded_isolated_w,
        sample_rate=int(materialized["sample_rate"]),
        ast_batch_size=ast_batch_size,
    )
    latent_by_id = {
        row["source_id"]: row
        for row in latent["direction_groups"]["full"]["sources"]
    }
    source_rows = []
    for semantic_row in semantic["sources"]:
        source_id = semantic_row["source_id"]
        latent_row = latent_by_id[source_id]
        if semantic_row["status"] == "ABSTAIN_CALIBRATION":
            outcome = "ABSTAIN_CALIBRATION"
        elif semantic_row["status"] != "PASS":
            outcome = "VAE_SOURCE_IDENTITY_LOSS"
        elif latent_row["status"] != "PASS":
            outcome = "RECOVERABLE_BUT_ENTANGLED"
        else:
            outcome = "STABLE_RECOVERABLE"
        source_rows.append(
            {
                "source_id": source_id,
                "semantic_status": semantic_row["status"],
                "latent_status": latent_row["status"],
                "outcome": outcome,
            }
        )
    outcomes = {row["outcome"] for row in source_rows}
    if "ABSTAIN_CALIBRATION" in outcomes:
        family_outcome = "ABSTAIN_CALIBRATION"
    elif "VAE_SOURCE_IDENTITY_LOSS" in outcomes:
        family_outcome = "VAE_SOURCE_IDENTITY_LOSS"
    elif "RECOVERABLE_BUT_ENTANGLED" in outcomes:
        family_outcome = "RECOVERABLE_BUT_ENTANGLED"
    else:
        family_outcome = "STABLE_RECOVERABLE"
    return {
        "family_rank": int(materialized["family_rank"]),
        "family_id": str(materialized["family_id"]),
        "source_ids": source_ids,
        "source_count": len(source_ids),
        "exact_mixture_peak_error": float(materialized["exact_mixture_peak_error"]),
        "exact_mixture_rms_error": float(materialized["exact_mixture_rms_error"]),
        "latent": latent,
        "semantic": semantic,
        "source_outcomes": source_rows,
        "outcome": family_outcome,
    }


def audit(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    ranks = sorted(set(int(rank) for rank in args.family_ranks))
    if not ranks or min(ranks) < 0:
        raise ValueError("--family-ranks must contain non-negative ranks")
    if args.output.exists():
        raise FileExistsError(args.output)
    vae_checkpoint = args.vae_checkpoint.expanduser().resolve()
    if _sha256(vae_checkpoint) != EXPECTED_VAE_SHA256:
        raise RuntimeError(f"frozen VAE checksum mismatch: {vae_checkpoint}")

    dataset = SpatialFamilyDataset(
        [
            {
                "path": args.family_store.expanduser().resolve(),
                "caption_overlay_path": args.caption_overlay.expanduser().resolve(),
            }
        ],
        require_ready=True,
    )
    if max(ranks) >= len(dataset):
        raise ValueError("a family rank is outside the canonical store")
    families = []
    for rank in ranks:
        _, info = dataset[rank]
        family = copy.deepcopy(info["spatial_family"])
        if int(family["family_rank"]) != rank:
            raise RuntimeError(f"family rank changed at {rank}")
        families.append(family)
    recipes = _load_recipe_families(families)

    vae, vae_config = _load_vae(
        args.vae_config.expanduser().resolve(), vae_checkpoint, device
    )
    clap_model = load_clap_model(args.clap_model, device=str(device))
    ast_cache = args.ast_cache.expanduser().resolve()
    ast_extractor = ASTFeatureExtractor.from_pretrained(
        args.ast_model,
        cache_dir=str(ast_cache),
        local_files_only=not args.allow_ast_download,
    )
    ast_model = ASTForAudioClassification.from_pretrained(
        args.ast_model,
        cache_dir=str(ast_cache),
        local_files_only=not args.allow_ast_download,
    ).to(device)
    ast_model.eval()
    ast_label_names = _ast_labels(ast_model)

    results = []
    for family in families:
        materialized = _materialize_family(
            family, recipes[str(family["family_id"])]
        )
        result = _audit_family(
            materialized,
            vae=vae,
            clap_model=clap_model,
            ast_model=ast_model,
            ast_extractor=ast_extractor,
            ast_label_names=ast_label_names,
            device=device,
            seed=args.seed,
            encode_batch_size=args.encode_batch_size,
            decode_batch_size=args.decode_batch_size,
            ast_batch_size=args.ast_batch_size,
        )
        results.append(result)
        print(
            json.dumps(
                {
                    "family_rank": result["family_rank"],
                    "source_count": result["source_count"],
                    "outcome": result["outcome"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    report = {
        "schema": "stable_audio_tools.vae_mixture_state_identifiability_audit",
        "schema_version": 1,
        "family_ranks": ranks,
        "family_count": len(results),
        "draw_count": DRAW_COUNT,
        "shared_posterior_epsilon": True,
        "seed": int(args.seed),
        "thresholds": {
            "required_draw_passes": REQUIRED_DRAW_PASSES,
            "min_clap_target_gap": MIN_CLAP_TARGET_GAP,
            "min_ast_anchor_probability": MIN_AST_ANCHOR_PROBABILITY,
            "min_ast_anchor_margin": MIN_AST_ANCHOR_MARGIN,
            "min_ast_target_gap": MIN_AST_TARGET_GAP,
            "min_ast_retention": MIN_AST_RETENTION,
            "min_rms_retention": MIN_RMS_RETENTION,
            "max_rms_retention": MAX_RMS_RETENTION,
            "min_median_direction_cosine": MIN_MEDIAN_DIRECTION_COSINE,
            "min_direction_cosine": MIN_DIRECTION_COSINE,
            "min_direction_margin": MIN_DIRECTION_MARGIN,
        },
        "rf_times": list(RF_TIMES),
        "rf_direction_interpretation": (
            "shared-noise linear RF direction is algebraically invariant; only SNR is diagnostic"
        ),
        "family_store": str(args.family_store.expanduser().resolve()),
        "caption_overlay": str(args.caption_overlay.expanduser().resolve()),
        "vae_config": str(args.vae_config.expanduser().resolve()),
        "vae_checkpoint": str(vae_checkpoint),
        "vae_checkpoint_sha256": EXPECTED_VAE_SHA256,
        "vae_sample_rate": int(vae_config["sample_rate"]),
        "clap_model": args.clap_model,
        "ast_model": args.ast_model,
        "families": results,
    }
    atomic_write_json(args.output, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family-ranks", nargs="+", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20_260_813)
    parser.add_argument("--family-store", type=Path, default=DEFAULT_FAMILY_STORE)
    parser.add_argument("--caption-overlay", type=Path, default=DEFAULT_CAPTION_OVERLAY)
    parser.add_argument("--vae-config", type=Path, default=DEFAULT_VAE_CONFIG)
    parser.add_argument("--vae-checkpoint", type=Path, default=DEFAULT_VAE_CHECKPOINT)
    parser.add_argument("--clap-model", default="630k-audioset-fusion-best.pt")
    parser.add_argument("--ast-model", default=DEFAULT_AST_MODEL)
    parser.add_argument("--ast-cache", type=Path, default=DEFAULT_AST_CACHE)
    parser.add_argument("--allow-ast-download", action="store_true")
    parser.add_argument("--encode-batch-size", type=int, default=2)
    parser.add_argument("--decode-batch-size", type=int, default=4)
    parser.add_argument("--ast-batch-size", type=int, default=8)
    args = parser.parse_args()
    report = audit(args)
    print(
        json.dumps(
            {
                "output": str(args.output.expanduser().resolve()),
                "family_count": report["family_count"],
                "outcomes": {
                    outcome: sum(
                        family["outcome"] == outcome for family in report["families"]
                    )
                    for outcome in (
                        "STABLE_RECOVERABLE",
                        "RECOVERABLE_BUT_ENTANGLED",
                        "VAE_SOURCE_IDENTITY_LOSS",
                        "ABSTAIN_CALIBRATION",
                    )
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
