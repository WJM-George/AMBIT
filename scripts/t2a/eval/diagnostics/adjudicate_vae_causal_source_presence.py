#!/usr/bin/env python3
"""Adjudicate causal source presence in decoded full and leave-one-out states.

The frozen mixture-intervention audit intentionally classified
``decode(full) - decode(minus)`` against exact isolated sources.  That is a
strict and useful local-linearity test, but a nonlinear decoder is not required
to make waveform subtraction itself a source separator.  This independent
target-side adjudication therefore scores the two decoded states directly.

For every removal intervention, the authoritative ScenePlan demixes full and
leave-one-out FOA into persistent source slots.  Exact isolated source audio
freezes CLAP identities and target-only AST labels.  A valid causal effect must
remove the intended anchor while retaining every non-removed slot.  No model
training, generated-source waveform route, or post-hoc mixture is involved.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from transformers import ASTFeatureExtractor, ASTForAudioClassification

from scripts.t2a.data.build_spatial_cot_source_curriculum import (
    EXPECTED_VAE_SHA256,
    _load_recipe_families,
)
from scripts.t2a.data.preencode_spatial_cot_family_shard import _load_vae
from scripts.t2a.eval.diagnostics.audit_vae_mixture_state_identifiability import (
    DEFAULT_CAPTION_OVERLAY,
    DEFAULT_FAMILY_STORE,
    DEFAULT_VAE_CHECKPOINT,
    DEFAULT_VAE_CONFIG,
    DRAW_COUNT,
    REQUIRED_DRAW_PASSES,
    _active_rms,
    _ast_probabilities,
    _clap_embeddings,
    _cosine_matrix,
    _materialize_family,
    _posterior_parameters,
    _sha256,
    _shared_posterior_draws,
    _summary,
)
from scripts.t2a.eval.diagnostics.audit_vae_source_semantics import (
    _active_clap_waveform,
)
from scripts.t2a.eval.diagnostics.score_source_location_semantics import (
    _compile_active_source_tracks,
    _demix_foa_sources,
)
from scripts.t2a.eval.diagnostics.score_source_semantics_ast import (
    DEFAULT_CACHE as DEFAULT_AST_CACHE,
    DEFAULT_MODEL as DEFAULT_AST_MODEL,
    _active_ast_waveform,
    _labels as _ast_labels,
    _select_target_anchors,
)
from stable_audio_tools.data.spatial_family_dataset import SpatialFamilyDataset
from stable_audio_tools.data.t2a_artifacts import atomic_write_json
from stable_audio_tools.training.metrics.fad_metrics import load_clap_model


MIN_CLAP_ASSIGNMENT_GAP = 0.05
MIN_CLAP_CAUSAL_GAP = 0.05
MIN_AST_CAUSAL_GAP = 0.01
MIN_TARGET_INSTANCE_COSINE = 0.75
MIN_AST_ANCHOR_PROBABILITY = 0.02
MIN_RETENTION = 0.25
MAX_RMS_RETENTION = 4.0
DEMIX_RIDGE = 0.05


def _assignment_row(
    matrix: torch.Tensor,
    row: int,
    *,
    source_ids: Sequence[str],
    min_gap: float,
) -> dict[str, Any]:
    values = torch.as_tensor(matrix, dtype=torch.float32).cpu()
    count = len(source_ids)
    if tuple(values.shape) != (count, count):
        raise ValueError("assignment matrix is not source-square")
    other = [index for index in range(count) if index != row]
    own = float(values[row, row])
    nearest = max((float(values[row, index]) for index in other), default=-1.0)
    predicted = int(torch.argmax(values[row]))
    gap = own - nearest
    return {
        "source_id": source_ids[row],
        "own": own,
        "nearest_other": nearest,
        "gap": gap,
        "assignment": source_ids[predicted],
        "correct": predicted == row,
        "valid": predicted == row and gap >= min_gap,
    }


def _decode_demixed_states(
    vae,
    draws: torch.Tensor,
    *,
    state_index: Mapping[str, int],
    source_ids: Sequence[str],
    tracks: torch.Tensor,
    batch_size: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if batch_size <= 0:
        raise ValueError("decode batch size must be positive")
    selected = [
        state_index["full"],
        *[state_index[f"minus:{source_id}"] for source_id in source_ids],
    ]
    draw_rows = []
    separator = None
    for draw in range(DRAW_COUNT):
        latent = draws[draw, selected]
        decoded_chunks = []
        for start in range(0, len(selected), batch_size):
            with torch.inference_mode():
                decoded = vae.decode(latent[start : start + batch_size])
            if decoded.ndim != 3 or tuple(decoded.shape[1:]) != (4, 442_368):
                raise RuntimeError(f"decoded FOA shape changed: {tuple(decoded.shape)}")
            decoded_chunks.append(decoded.float().cpu())
        decoded_states = torch.cat(decoded_chunks, dim=0)
        stems = []
        for audio in decoded_states:
            demixed, metadata = _demix_foa_sources(
                audio, tracks, ridge=DEMIX_RIDGE
            )
            if separator is None:
                separator = metadata
            elif metadata != separator:
                raise RuntimeError("Plan demixer metadata changed across states")
            stems.append(demixed)
        draw_rows.append(torch.stack(stems))
    if separator is None:
        raise RuntimeError("no decoded states were demixed")
    return torch.stack(draw_rows), separator


def _target_demixed_states(
    materialized: Mapping[str, Any], tracks: torch.Tensor
) -> tuple[torch.Tensor, dict[str, Any]]:
    audios = [materialized["full"], *materialized["leave_one_out"]]
    stems = []
    separator = None
    for audio in audios:
        demixed, metadata = _demix_foa_sources(
            torch.from_numpy(audio), tracks, ridge=DEMIX_RIDGE
        )
        if separator is None:
            separator = metadata
        elif metadata != separator:
            raise RuntimeError("target Plan demixer metadata changed across states")
        stems.append(demixed)
    if separator is None:
        raise RuntimeError("no exact target states were demixed")
    return torch.stack(stems), separator


def _semantic_state_tensors(
    *,
    clap_model,
    ast_model: ASTForAudioClassification,
    ast_extractor: ASTFeatureExtractor,
    ast_label_names: Sequence[str],
    device: torch.device,
    plan: Mapping[str, Any],
    source_ids: Sequence[str],
    exact_isolated_w: Sequence[torch.Tensor],
    target_stems: torch.Tensor,
    decoded_stems: torch.Tensor,
    tracks: torch.Tensor,
    sample_rate: int,
    ast_batch_size: int,
) -> dict[str, Any]:
    source_count = len(source_ids)
    state_count = source_count + 1
    if tuple(target_stems.shape[:2]) != (state_count, source_count):
        raise ValueError("target stem states do not match source count")
    if tuple(decoded_stems.shape[:3]) != (
        DRAW_COUNT,
        state_count,
        source_count,
    ):
        raise ValueError("decoded stem states do not match frozen audit dimensions")
    plan_sources = {
        str(source["source_id"]): source
        for source in ((plan.get("scene") or {}).get("sources") or [])
    }
    hop = int(target_stems.shape[-1] // 432)
    if hop != 1024:
        raise RuntimeError(f"frozen source stem hop changed: {hop}")

    reference_clap_waves = []
    target_clap_waves = []
    decoded_clap_waves = []
    reference_ast_waves = []
    target_ast_waves = []
    decoded_ast_waves = []
    target_rms = torch.zeros(state_count, source_count)
    decoded_rms = torch.zeros(DRAW_COUNT, state_count, source_count)
    for source_index, source_id in enumerate(source_ids):
        source = plan_sources[source_id]
        activity = source.get("activity") or {}
        onset = float(activity["onset_sec"])
        offset = float(activity["offset_sec"])
        active_frames = tracks[source_index, 0] > 0.5
        reference_clap_waves.append(
            _active_clap_waveform(
                exact_isolated_w[source_index],
                sample_rate=sample_rate,
                onset_sec=onset,
                offset_sec=offset,
                device=device,
            )
        )
        ast_wave, _ = _active_ast_waveform(
            exact_isolated_w[source_index],
            active_frames,
            hop=hop,
            sample_rate=sample_rate,
        )
        reference_ast_waves.append(ast_wave)

    # State-major, then source-major.  The same deterministic ordering is used
    # for CLAP, AST, and raw signal retention.
    for state in range(state_count):
        for source_index, source_id in enumerate(source_ids):
            source = plan_sources[source_id]
            activity = source.get("activity") or {}
            onset = float(activity["onset_sec"])
            offset = float(activity["offset_sec"])
            active_frames = tracks[source_index, 0] > 0.5
            target_clap_waves.append(
                _active_clap_waveform(
                    target_stems[state, source_index],
                    sample_rate=sample_rate,
                    onset_sec=onset,
                    offset_sec=offset,
                    device=device,
                )
            )
            ast_wave, _ = _active_ast_waveform(
                target_stems[state, source_index],
                active_frames,
                hop=hop,
                sample_rate=sample_rate,
            )
            target_ast_waves.append(ast_wave)
            target_rms[state, source_index] = _active_rms(
                target_stems[state, source_index], active_frames, hop=hop
            )
    for draw in range(DRAW_COUNT):
        for state in range(state_count):
            for source_index, source_id in enumerate(source_ids):
                source = plan_sources[source_id]
                activity = source.get("activity") or {}
                onset = float(activity["onset_sec"])
                offset = float(activity["offset_sec"])
                active_frames = tracks[source_index, 0] > 0.5
                decoded_clap_waves.append(
                    _active_clap_waveform(
                        decoded_stems[draw, state, source_index],
                        sample_rate=sample_rate,
                        onset_sec=onset,
                        offset_sec=offset,
                        device=device,
                    )
                )
                ast_wave, _ = _active_ast_waveform(
                    decoded_stems[draw, state, source_index],
                    active_frames,
                    hop=hop,
                    sample_rate=sample_rate,
                )
                decoded_ast_waves.append(ast_wave)
                decoded_rms[draw, state, source_index] = _active_rms(
                    decoded_stems[draw, state, source_index],
                    active_frames,
                    hop=hop,
                )

    reference_clap = _clap_embeddings(clap_model, reference_clap_waves)
    target_clap = _clap_embeddings(clap_model, target_clap_waves).reshape(
        state_count, source_count, -1
    )
    decoded_clap = _clap_embeddings(clap_model, decoded_clap_waves).reshape(
        DRAW_COUNT, state_count, source_count, -1
    )
    target_clap_similarity = _cosine_matrix(
        target_clap.flatten(0, 1), reference_clap
    ).reshape(state_count, source_count, source_count).cpu()
    decoded_clap_similarity = torch.stack(
        [
            _cosine_matrix(decoded_clap[draw].flatten(0, 1), reference_clap)
            .reshape(state_count, source_count, source_count)
            .cpu()
            for draw in range(DRAW_COUNT)
        ]
    )
    target_clap_instances = torch.stack(
        [
            _cosine_matrix(target_clap[state], target_clap[state]).cpu()
            for state in range(state_count)
        ]
    )
    decoded_to_target_clap = torch.stack(
        [
            torch.stack(
                [
                    _cosine_matrix(
                        decoded_clap[draw, state], target_clap[state]
                    ).cpu()
                    for state in range(state_count)
                ]
            )
            for draw in range(DRAW_COUNT)
        ]
    )

    ast_waves = [
        *reference_ast_waves,
        *target_ast_waves,
        *decoded_ast_waves,
    ]
    ast_probabilities = _ast_probabilities(
        ast_model,
        ast_extractor,
        ast_waves,
        device=device,
        batch_size=ast_batch_size,
    )
    cursor = 0
    reference_ast = ast_probabilities[cursor : cursor + source_count]
    cursor += source_count
    target_ast = ast_probabilities[
        cursor : cursor + state_count * source_count
    ].reshape(state_count, source_count, -1)
    cursor += state_count * source_count
    decoded_ast = ast_probabilities[cursor:].reshape(
        DRAW_COUNT, state_count, source_count, -1
    )
    anchors = _select_target_anchors(
        reference_ast,
        ast_label_names,
        min_probability=0.02,
        min_margin=0.01,
    )
    anchor_indices = [int(anchor["label_index"]) for anchor in anchors]
    return {
        "target_clap": target_clap_similarity,
        "decoded_clap": decoded_clap_similarity,
        "target_clap_instances": target_clap_instances,
        "decoded_to_target_clap": decoded_to_target_clap,
        "target_ast": target_ast[:, :, anchor_indices].cpu(),
        "decoded_ast": decoded_ast[:, :, :, anchor_indices].cpu(),
        "target_rms": target_rms,
        "decoded_rms": decoded_rms,
        "anchors": anchors,
    }


def _source_causal_report(
    tensors: Mapping[str, Any],
    *,
    removed_index: int,
    source_ids: Sequence[str],
) -> dict[str, Any]:
    source_count = len(source_ids)
    minus_state = 1 + removed_index
    target_clap = tensors["target_clap"]
    target_clap_instances = tensors["target_clap_instances"]
    target_ast = tensors["target_ast"]
    target_rms = tensors["target_rms"]
    decoded_clap = tensors["decoded_clap"]
    decoded_to_target_clap = tensors["decoded_to_target_clap"]
    decoded_ast = tensors["decoded_ast"]
    decoded_rms = tensors["decoded_rms"]
    anchors = tensors["anchors"]
    nonremoved = [index for index in range(source_count) if index != removed_index]

    target_full_clap = [
        _assignment_row(
            target_clap_instances[0],
            index,
            source_ids=source_ids,
            min_gap=MIN_CLAP_ASSIGNMENT_GAP,
        )
        for index in range(source_count)
    ]
    target_minus_clap = [
        _assignment_row(
            target_clap_instances[minus_state],
            index,
            source_ids=source_ids,
            min_gap=MIN_CLAP_ASSIGNMENT_GAP,
        )
        for index in nonremoved
    ]
    target_full_ast_probabilities = [
        float(target_ast[0, index, index]) for index in range(source_count)
    ]
    target_minus_ast_probabilities = [
        float(target_ast[minus_state, index, index]) for index in nonremoved
    ]
    target_clap_gap = float(
        target_clap[0, removed_index, removed_index]
        - target_clap[minus_state, removed_index, removed_index]
    )
    target_ast_gap = float(
        target_ast[0, removed_index, removed_index]
        - target_ast[minus_state, removed_index, removed_index]
    )
    calibrated = (
        all(bool(anchor["target_valid"]) for anchor in anchors)
        and all(row["valid"] for row in target_full_clap)
        and all(row["valid"] for row in target_minus_clap)
        and min(target_full_ast_probabilities) >= MIN_AST_ANCHOR_PROBABILITY
        and min(
            target_minus_ast_probabilities, default=float("inf")
        )
        >= MIN_AST_ANCHOR_PROBABILITY
        and target_clap_gap >= MIN_CLAP_CAUSAL_GAP
        and target_ast_gap >= MIN_AST_CAUSAL_GAP
    )

    draw_rows = []
    failure_counts = {
        "full_target_instance": 0,
        "full_ast_retention": 0,
        "full_rms_retention": 0,
        "removed_clap_causal_gap": 0,
        "removed_ast_causal_gap": 0,
        "nonremoved_target_instance": 0,
        "nonremoved_ast_retention": 0,
        "nonremoved_rms_retention": 0,
    }
    for draw in range(DRAW_COUNT):
        full_instance_rows = [
            _assignment_row(
                decoded_to_target_clap[draw, 0],
                index,
                source_ids=source_ids,
                min_gap=0.0,
            )
            for index in range(source_count)
        ]
        minus_instance_rows = [
            _assignment_row(
                decoded_to_target_clap[draw, minus_state],
                index,
                source_ids=source_ids,
                min_gap=0.0,
            )
            for index in nonremoved
        ]
        full_ast_retention = [
            float(decoded_ast[draw, 0, index, index])
            / max(float(target_ast[0, index, index]), 1.0e-8)
            for index in range(source_count)
        ]
        full_rms_retention = [
            float(decoded_rms[draw, 0, index])
            / max(float(target_rms[0, index]), 1.0e-8)
            for index in range(source_count)
        ]
        post_clap_gap = float(
            decoded_clap[draw, 0, removed_index, removed_index]
            - decoded_clap[draw, minus_state, removed_index, removed_index]
        )
        post_ast_gap = float(
            decoded_ast[draw, 0, removed_index, removed_index]
            - decoded_ast[draw, minus_state, removed_index, removed_index]
        )
        clap_gap_retention = post_clap_gap / max(target_clap_gap, 1.0e-8)
        ast_gap_retention = post_ast_gap / max(target_ast_gap, 1.0e-8)
        nonremoved_ast_retention = [
            float(decoded_ast[draw, minus_state, index, index])
            / max(float(target_ast[minus_state, index, index]), 1.0e-8)
            for index in nonremoved
        ]
        nonremoved_rms_retention = [
            float(decoded_rms[draw, minus_state, index])
            / max(float(target_rms[minus_state, index]), 1.0e-8)
            for index in nonremoved
        ]
        checks = {
            "full_target_instance": all(
                row["correct"] and row["own"] >= MIN_TARGET_INSTANCE_COSINE
                for row in full_instance_rows
            ),
            "full_ast_retention": min(full_ast_retention) >= MIN_RETENTION,
            "full_rms_retention": min(full_rms_retention) >= MIN_RETENTION
            and max(full_rms_retention) <= MAX_RMS_RETENTION,
            "removed_clap_causal_gap": clap_gap_retention >= MIN_RETENTION,
            "removed_ast_causal_gap": ast_gap_retention >= MIN_RETENTION,
            "nonremoved_target_instance": all(
                row["correct"] and row["own"] >= MIN_TARGET_INSTANCE_COSINE
                for row in minus_instance_rows
            ),
            "nonremoved_ast_retention": min(
                nonremoved_ast_retention, default=float("inf")
            )
            >= MIN_RETENTION,
            "nonremoved_rms_retention": min(
                nonremoved_rms_retention, default=float("inf")
            )
            >= MIN_RETENTION
            and max(nonremoved_rms_retention, default=float("-inf"))
            <= MAX_RMS_RETENTION,
        }
        for name, passed in checks.items():
            failure_counts[name] += not passed
        draw_rows.append(
            {
                "draw": draw,
                "pass": all(checks.values()),
                "checks": checks,
                "clap_causal_gap_retention": clap_gap_retention,
                "ast_causal_gap_retention": ast_gap_retention,
                "min_full_ast_retention": min(full_ast_retention),
                "min_full_rms_retention": min(full_rms_retention),
                "min_nonremoved_ast_retention": min(
                    nonremoved_ast_retention, default=None
                ),
                "min_nonremoved_rms_retention": min(
                    nonremoved_rms_retention, default=None
                ),
            }
        )
    pass_count = sum(bool(row["pass"]) for row in draw_rows)
    if not calibrated:
        status = "ABSTAIN_CALIBRATION"
    else:
        status = "PASS" if pass_count >= REQUIRED_DRAW_PASSES else "BLOCK"
    return {
        "source_id": source_ids[removed_index],
        "target_calibrated": calibrated,
        "target_clap_causal_gap": target_clap_gap,
        "target_ast_causal_gap": target_ast_gap,
        "target_full_clap_instance": target_full_clap,
        "target_nonremoved_minus_clap_instance": target_minus_clap,
        "target_full_ast_anchor_probabilities": target_full_ast_probabilities,
        "target_nonremoved_minus_ast_anchor_probabilities": (
            target_minus_ast_probabilities
        ),
        "draw_pass_count": pass_count,
        "failure_draw_counts": failure_counts,
        "clap_causal_gap_retention": _summary(
            [float(row["clap_causal_gap_retention"]) for row in draw_rows]
        ),
        "ast_causal_gap_retention": _summary(
            [float(row["ast_causal_gap_retention"]) for row in draw_rows]
        ),
        "draws": draw_rows,
        "status": status,
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
    include_posterior_mean: bool,
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
    draws, _ = _shared_posterior_draws(
        mean, stdev, draw_count=DRAW_COUNT, seed=family_seed
    )
    tracks = _compile_active_source_tracks(
        materialized["plan"], num_frames=432, source_ids=source_ids
    )
    target_stems, target_separator = _target_demixed_states(materialized, tracks)
    exact_isolated_w = [
        torch.from_numpy(audio[0]) * math.sqrt(2.0)
        for audio in materialized["isolated"]
    ]

    def adjudicate(draw_values: torch.Tensor) -> tuple[list[dict[str, Any]], str]:
        decoded_stems, decoded_separator = _decode_demixed_states(
            vae,
            draw_values,
            state_index=state_index,
            source_ids=source_ids,
            tracks=tracks,
            batch_size=decode_batch_size,
        )
        if target_separator != decoded_separator:
            raise RuntimeError("target and decoded Plan demixers differ")
        tensors = _semantic_state_tensors(
            clap_model=clap_model,
            ast_model=ast_model,
            ast_extractor=ast_extractor,
            ast_label_names=ast_label_names,
            device=device,
            plan=materialized["plan"],
            source_ids=source_ids,
            exact_isolated_w=exact_isolated_w,
            target_stems=target_stems,
            decoded_stems=decoded_stems,
            tracks=tracks,
            sample_rate=int(materialized["sample_rate"]),
            ast_batch_size=ast_batch_size,
        )
        rows = [
            _source_causal_report(
                tensors, removed_index=index, source_ids=source_ids
            )
            for index in range(len(source_ids))
        ]
        if any(row["status"] == "ABSTAIN_CALIBRATION" for row in rows):
            result = "ABSTAIN_CALIBRATION"
        elif all(row["status"] == "PASS" for row in rows):
            result = "CAUSAL_PRESENCE_PRESERVED"
        else:
            result = "CAUSAL_SOURCE_PRESENCE_LOSS"
        return rows, result

    source_rows, outcome = adjudicate(draws)
    posterior_mean = None
    if include_posterior_mean:
        mean_draws = mean.unsqueeze(0).expand(DRAW_COUNT, -1, -1, -1)
        mean_rows, mean_outcome = adjudicate(mean_draws)
        posterior_mean = {
            "repeated_identical_draws": DRAW_COUNT,
            "sources": mean_rows,
            "outcome": mean_outcome,
        }
    return {
        "family_rank": int(materialized["family_rank"]),
        "family_id": str(materialized["family_id"]),
        "source_ids": source_ids,
        "source_count": len(source_ids),
        "separator": target_separator,
        "sources": source_rows,
        "outcome": outcome,
        "posterior_mean": posterior_mean,
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
    expected_vae_sha256 = str(args.expected_vae_sha256).strip().lower()
    if len(expected_vae_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in expected_vae_sha256
    ):
        raise ValueError("--expected-vae-sha256 must be a lowercase SHA-256 digest")
    actual_vae_sha256 = _sha256(vae_checkpoint)
    if actual_vae_sha256 != expected_vae_sha256:
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
    families = []
    for rank in ranks:
        _, info = dataset[rank]
        family = copy.deepcopy(info["spatial_family"])
        if int(family["family_rank"]) != rank:
            raise RuntimeError(f"family rank changed at {rank}")
        families.append(family)
    recipes = _load_recipe_families(families)
    vae, _ = _load_vae(
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
            include_posterior_mean=args.include_posterior_mean,
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
        "schema": "stable_audio_tools.vae_causal_source_presence_adjudication",
        "schema_version": 2,
        "adjudication": "differential_causal_gap_plus_exact_target_instance_fidelity",
        "family_ranks": ranks,
        "family_count": len(results),
        "draw_count": DRAW_COUNT,
        "shared_posterior_epsilon": True,
        "posterior_mean_included": bool(args.include_posterior_mean),
        "seed": int(args.seed),
        "thresholds": {
            "required_draw_passes": REQUIRED_DRAW_PASSES,
            "min_clap_assignment_gap": MIN_CLAP_ASSIGNMENT_GAP,
            "min_clap_causal_gap": MIN_CLAP_CAUSAL_GAP,
            "min_ast_causal_gap": MIN_AST_CAUSAL_GAP,
            "min_target_instance_cosine": MIN_TARGET_INSTANCE_COSINE,
            "min_ast_anchor_probability": MIN_AST_ANCHOR_PROBABILITY,
            "min_retention": MIN_RETENTION,
            "max_rms_retention": MAX_RMS_RETENTION,
            "demix_ridge": DEMIX_RIDGE,
        },
        "family_store": str(args.family_store.expanduser().resolve()),
        "caption_overlay": str(args.caption_overlay.expanduser().resolve()),
        "vae_config": str(args.vae_config.expanduser().resolve()),
        "vae_checkpoint": str(vae_checkpoint),
        "vae_checkpoint_sha256": actual_vae_sha256,
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
    parser.add_argument(
        "--expected-vae-sha256", default=EXPECTED_VAE_SHA256
    )
    parser.add_argument("--clap-model", default="630k-audioset-fusion-best.pt")
    parser.add_argument("--ast-model", default=DEFAULT_AST_MODEL)
    parser.add_argument("--ast-cache", type=Path, default=DEFAULT_AST_CACHE)
    parser.add_argument("--allow-ast-download", action="store_true")
    parser.add_argument("--encode-batch-size", type=int, default=2)
    parser.add_argument("--decode-batch-size", type=int, default=4)
    parser.add_argument("--ast-batch-size", type=int, default=8)
    parser.add_argument("--include-posterior-mean", action="store_true")
    args = parser.parse_args()
    report = audit(args)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "family_count": report["family_count"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
