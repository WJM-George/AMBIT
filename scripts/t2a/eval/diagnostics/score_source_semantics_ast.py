#!/usr/bin/env python3
"""Score target-calibrated source semantics with an independent AudioSet model.

The checkpoint evaluator emits one complete generated FOA mixture and its
target.  This diagnostic demixes both with the authoritative target ScenePlan,
keeps only each source's planned active samples, and classifies the resulting
signals with AST.  Target stems choose their own discriminative AudioSet anchor
labels; generated stems are never allowed to choose easier labels after seeing
the output.

AST is an evaluation-only second opinion.  It is not a differentiable training
loss, and a source is eligible for a semantic decision only when the target
stem itself has a sufficiently strong, source-specific AST anchor.  Raw active
RMS is retained beside peak-normalized semantic scores so a silent source
cannot pass merely because normalization hides its missing energy.
"""
from __future__ import annotations
import os

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torchaudio
from transformers import ASTFeatureExtractor, ASTForAudioClassification

from scripts.t2a.eval.diagnostics.diagnose_spatial_conditions import _atomic_json
from scripts.t2a.eval.diagnostics.score_source_location_semantics import (
    _compile_active_source_tracks,
    _demix_foa_sources,
    _resolve_scoring_inputs,
    _scene_sources,
)


DEFAULT_MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"
DEFAULT_CACHE = Path(os.environ.get("AMBIT_DATA_ROOT", "data") + "/.cache/hf_ast_source_gate/hub")


def _active_ast_waveform(
    stem: torch.Tensor,
    active_frames: torch.Tensor,
    *,
    hop: int,
    sample_rate: int,
    target_seconds: int = 10,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Concatenate planned activity and return a deterministic 16-kHz AST input."""

    stem = torch.as_tensor(stem, dtype=torch.float32).cpu().reshape(-1)
    active_frames = torch.as_tensor(active_frames, dtype=torch.bool).cpu().reshape(-1)
    if hop <= 0 or sample_rate <= 0 or target_seconds <= 0:
        raise ValueError("hop, sample_rate, and target_seconds must be positive")
    if stem.numel() != active_frames.numel() * hop:
        raise ValueError(
            f"stem/activity size mismatch: {stem.numel()} != "
            f"{active_frames.numel()} * {hop}"
        )
    sample_mask = active_frames.repeat_interleave(hop)
    active = stem[sample_mask]
    if active.numel() == 0:
        raise ValueError("source has no planned active samples")
    finite = bool(torch.isfinite(active).all())
    if not finite:
        raise ValueError("source active samples are non-finite")
    peak = float(active.abs().amax())
    rms = float(active.square().mean().sqrt())
    silent = peak <= 1.0e-8
    if silent:
        normalized = torch.zeros_like(active)
    else:
        normalized = active / peak * (10.0 ** (-1.0 / 20.0))
    target_samples = target_seconds * sample_rate
    normalized = normalized.repeat(math.ceil(target_samples / normalized.numel()))[
        :target_samples
    ]
    if sample_rate != 16_000:
        normalized = torchaudio.functional.resample(
            normalized[None], sample_rate, 16_000
        )[0]
    expected = target_seconds * 16_000
    if normalized.numel() != expected:
        raise RuntimeError(
            f"AST waveform length changed: {normalized.numel()} != {expected}"
        )
    return normalized.clamp(-1.0, 1.0), {
        "active_frame_count": int(active_frames.sum()),
        "active_sample_count": int(active.numel()),
        "active_rms": rms,
        "active_peak": peak,
        "silent": silent,
        "semantic_normalization": "active peak to -1 dBFS, repeat to exactly 10 s",
    }


def _cosine_matrix(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left = torch.nn.functional.normalize(left.float(), dim=-1)
    right = torch.nn.functional.normalize(right.float(), dim=-1)
    return left @ right.transpose(0, 1)


def _labels(model: ASTForAudioClassification) -> list[str]:
    count = int(model.config.num_labels)
    mapping = model.config.id2label or {}
    labels = [str(mapping.get(index, mapping.get(str(index), index))) for index in range(count)]
    if len(set(labels)) != count:
        raise RuntimeError("AST label names are not unique")
    return labels


def _select_target_anchors(
    target_probabilities: torch.Tensor,
    labels: Sequence[str],
    *,
    min_probability: float,
    min_margin: float,
) -> list[dict[str, Any]]:
    """Freeze one target-only discriminative AudioSet label per source."""

    values = torch.as_tensor(target_probabilities, dtype=torch.float32).cpu()
    if values.ndim != 2 or values.shape[1] != len(labels):
        raise ValueError("target probabilities and AST labels do not match")
    if not 0.0 <= min_probability <= 1.0 or not 0.0 <= min_margin <= 1.0:
        raise ValueError("AST anchor thresholds must lie in [0, 1]")
    source_count = values.shape[0]
    anchors = []
    for source_index in range(source_count):
        own = values[source_index]
        if source_count == 1:
            other = torch.zeros_like(own)
        else:
            other = torch.cat(
                (values[:source_index], values[source_index + 1 :]), dim=0
            ).amax(dim=0)
        margins = own - other
        eligible = torch.nonzero(
            (own >= min_probability) & (margins >= min_margin), as_tuple=False
        ).flatten()
        candidates = eligible if eligible.numel() else torch.arange(values.shape[1])
        # Margin is the primary calibration criterion; probability breaks ties.
        candidate_scores = margins[candidates] * 2.0 + own[candidates]
        label_index = int(candidates[int(torch.argmax(candidate_scores))])
        anchors.append(
            {
                "source_index": source_index,
                "label_index": label_index,
                "label": str(labels[label_index]),
                "target_probability": float(own[label_index]),
                "max_other_target_probability": float(other[label_index]),
                "target_margin": float(margins[label_index]),
                "target_valid": bool(eligible.numel()),
                "eligible_label_count": int(eligible.numel()),
            }
        )
    return anchors


def _anchors_from_spec(
    spec: Mapping[str, Any],
    *,
    family_rank: int,
    family_id: str,
    source_ids: Sequence[str],
    labels: Sequence[str],
) -> list[dict[str, Any]]:
    """Resolve source-only anchors that were frozen before rollout scoring."""

    if (
        spec.get("schema") != "stable_audio_tools.independent_ast_source_anchors"
        or int(spec.get("schema_version", -1)) != 1
        or int(spec.get("family_rank", -1)) != family_rank
        or str(spec.get("family_id")) != family_id
    ):
        raise ValueError("independent AST anchor spec does not match the result family")
    rows = spec.get("source_anchors")
    if not isinstance(rows, list):
        raise ValueError("independent AST anchor spec has no source_anchors")
    by_id = {str(row.get("source_id")): row for row in rows if isinstance(row, dict)}
    if set(by_id) != set(source_ids):
        raise ValueError("independent AST anchor source IDs do not match the ScenePlan")
    label_to_index = {str(label): index for index, label in enumerate(labels)}
    anchors = []
    for source_index, source_id in enumerate(source_ids):
        row = by_id[source_id]
        label = str(row.get("ast_label") or "")
        if label not in label_to_index:
            raise ValueError(f"AST anchor label is unavailable: {label!r}")
        reference_probability = float(row.get("post_vae_probability"))
        pre_probability = float(row.get("pre_vae_probability"))
        valid = (
            str(row.get("status")) == "PASS"
            and math.isfinite(reference_probability)
            and math.isfinite(pre_probability)
            and 0.0 < reference_probability <= 1.0
            and 0.0 < pre_probability <= 1.0
        )
        anchors.append(
            {
                "source_index": source_index,
                "label_index": label_to_index[label],
                "label": label,
                "target_valid": valid,
                "anchor_origin": "independent_isolated_vae_target",
                "reference_probability": reference_probability,
                "pre_vae_probability": pre_probability,
                "event_label": str(row.get("event_label") or source_id),
                "calibration_status": str(row.get("status")),
            }
        )
    return anchors


def _anchor_assignment(
    generated: torch.Tensor,
    target: torch.Tensor,
    *,
    anchors: Sequence[Mapping[str, Any]],
    source_ids: Sequence[str],
    min_target_gap: float,
    require_mixture_target_gap: bool = True,
) -> dict[str, Any]:
    """Assign spatial rows to target-selected AST anchors and fail closed."""

    generated = torch.as_tensor(generated, dtype=torch.float32).cpu()
    target = torch.as_tensor(target, dtype=torch.float32).cpu()
    count = len(source_ids)
    if tuple(generated.shape) != (count, count) or tuple(target.shape) != (
        count,
        count,
    ):
        raise ValueError("AST anchor matrices must be square source matrices")
    rows = []
    valid_correct = []
    valid_margins = []
    for index, source_id in enumerate(source_ids):
        others = [column for column in range(count) if column != index]
        target_other = max(
            (float(target[index, column]) for column in others), default=0.0
        )
        generated_other = max(
            (float(generated[index, column]) for column in others), default=0.0
        )
        target_gap = float(target[index, index]) - target_other
        generated_gap = float(generated[index, index]) - generated_other
        predicted = int(torch.argmax(generated[index]))
        valid = bool(anchors[index]["target_valid"]) and (
            target_gap >= min_target_gap or not require_mixture_target_gap
        )
        correct = predicted == index
        if valid:
            valid_correct.append(correct)
            valid_margins.append(generated_gap)
        reference_probability = float(
            anchors[index].get("reference_probability", target[index, index])
        )
        rows.append(
            {
                "source_id": source_id,
                "anchor_label": str(anchors[index]["label"]),
                "target_discriminability_gap": target_gap,
                "mixture_target_gap_required": bool(require_mixture_target_gap),
                "target_valid": valid,
                "generated_anchor_probability": float(generated[index, index]),
                "target_anchor_probability": float(target[index, index]),
                "reference_anchor_probability": reference_probability,
                "generated_to_target_anchor_ratio": float(
                    generated[index, index] / target[index, index].clamp_min(1.0e-8)
                ),
                "generated_to_reference_anchor_ratio": float(
                    generated[index, index] / max(reference_probability, 1.0e-8)
                ),
                "generated_diagonal_margin": generated_gap,
                "generated_assignment": source_ids[predicted],
                "generated_correct": correct,
            }
        )
    return {
        "min_target_gap": float(min_target_gap),
        "mixture_target_gap_required": bool(require_mixture_target_gap),
        "valid_source_count": len(valid_correct),
        "source_count": count,
        "accuracy_on_target_discriminable_sources": (
            sum(valid_correct) / len(valid_correct) if valid_correct else None
        ),
        "mean_generated_diagonal_margin_on_valid_sources": (
            sum(valid_margins) / len(valid_margins) if valid_margins else None
        ),
        "rows": rows,
    }


def _matrix_rows(
    matrix: torch.Tensor,
    *,
    row_ids: Sequence[str],
    column_ids: Sequence[str],
) -> dict[str, dict[str, float]]:
    matrix = torch.as_tensor(matrix, dtype=torch.float32).cpu()
    return {
        row_id: {
            column_id: float(matrix[row, column])
            for column, column_id in enumerate(column_ids)
        }
        for row, row_id in enumerate(row_ids)
    }


def _top_labels(
    probabilities: torch.Tensor, labels: Sequence[str], *, top_k: int
) -> list[dict[str, Any]]:
    values, indices = torch.topk(
        torch.as_tensor(probabilities, dtype=torch.float32).cpu(),
        k=min(top_k, len(labels)),
    )
    return [
        {"label": str(labels[int(index)]), "probability": float(value)}
        for value, index in zip(values, indices)
    ]


@torch.inference_mode()
def _score_report(
    model: ASTForAudioClassification,
    extractor: ASTFeatureExtractor,
    result_path: Path,
    *,
    device: torch.device,
    ridge: float,
    min_anchor_probability: float,
    min_anchor_margin: float,
    min_target_gap: float,
    top_k: int,
    anchor_spec: Mapping[str, Any] | None,
) -> dict[str, Any]:
    report = json.loads(result_path.read_text(encoding="utf-8"))
    resolved = _resolve_scoring_inputs(report, result_path)
    plan = resolved["plan"]
    sources = _scene_sources(plan)
    source_ids = [str(source["source_id"]) for source in sources]
    generated, generated_rate = torchaudio.load(str(resolved["generated_path"]))
    target, target_rate = torchaudio.load(str(resolved["target_path"]))
    if generated_rate != target_rate or generated.shape != target.shape:
        raise ValueError("generated and target FOA files must share rate and shape")
    hop = int(resolved["hop"])
    frame_count_hint = int(resolved["frame_count"])
    if hop > 0:
        if generated.shape[-1] % hop:
            raise ValueError("FOA sample count is not divisible by the reported hop")
        frame_count = generated.shape[-1] // hop
    elif frame_count_hint > 0 and generated.shape[-1] % frame_count_hint == 0:
        frame_count = frame_count_hint
        hop = generated.shape[-1] // frame_count
    else:
        raise ValueError("source semantic report lacks a valid hop/frame count")
    if frame_count_hint > 0 and frame_count != frame_count_hint:
        raise ValueError("source semantic frame count changed")

    tracks = _compile_active_source_tracks(
        plan, num_frames=frame_count, source_ids=source_ids
    )
    generated_stems, separator = _demix_foa_sources(generated, tracks, ridge=ridge)
    target_stems, target_separator = _demix_foa_sources(target, tracks, ridge=ridge)
    if separator != target_separator:
        raise RuntimeError("generated and target AST separators differ")

    waveforms = []
    signal_rows = []
    for index, source_id in enumerate(source_ids):
        active_frames = tracks[index, 0] > 0.5
        generated_wave, generated_signal = _active_ast_waveform(
            generated_stems[index],
            active_frames,
            hop=hop,
            sample_rate=generated_rate,
        )
        target_wave, target_signal = _active_ast_waveform(
            target_stems[index],
            active_frames,
            hop=hop,
            sample_rate=target_rate,
        )
        waveforms.extend((generated_wave.numpy(), target_wave.numpy()))
        target_rms = float(target_signal["active_rms"])
        signal_rows.append(
            {
                "source_id": source_id,
                "generated": generated_signal,
                "target": target_signal,
                "active_rms_ratio": float(
                    generated_signal["active_rms"] / max(target_rms, 1.0e-8)
                ),
            }
        )

    features = extractor(
        waveforms,
        sampling_rate=16_000,
        return_tensors="pt",
        padding=True,
    )
    logits = model(
        **{key: value.to(device) for key, value in features.items()}
    ).logits.float().cpu()
    probabilities = torch.sigmoid(logits)
    generated_probabilities = probabilities[0::2]
    target_probabilities = probabilities[1::2]
    label_names = _labels(model)
    if anchor_spec is None:
        anchors = _select_target_anchors(
            target_probabilities,
            label_names,
            min_probability=min_anchor_probability,
            min_margin=min_anchor_margin,
        )
        anchor_source = "demixed_mixture_target"
    else:
        anchors = _anchors_from_spec(
            anchor_spec,
            family_rank=int(report["family_rank"]),
            family_id=str(report["family_id"]),
            source_ids=source_ids,
            labels=label_names,
        )
        anchor_source = "independent_isolated_vae_target"
    anchor_indices = [int(anchor["label_index"]) for anchor in anchors]
    generated_anchor_matrix = generated_probabilities[:, anchor_indices]
    target_anchor_matrix = target_probabilities[:, anchor_indices]
    anchor_assignment = _anchor_assignment(
        generated_anchor_matrix,
        target_anchor_matrix,
        anchors=anchors,
        source_ids=source_ids,
        min_target_gap=min_target_gap,
        require_mixture_target_gap=anchor_spec is None,
    )
    generated_to_target = _cosine_matrix(
        generated_probabilities, target_probabilities
    )
    target_to_target = _cosine_matrix(target_probabilities, target_probabilities)

    source_rows = []
    for index, source in enumerate(sources):
        source_rows.append(
            {
                **signal_rows[index],
                "caption": str((source.get("event") or {}).get("label") or source_ids[index]),
                "anchor": anchors[index],
                "generated_top_labels": _top_labels(
                    generated_probabilities[index], label_names, top_k=top_k
                ),
                "target_top_labels": _top_labels(
                    target_probabilities[index], label_names, top_k=top_k
                ),
                "generated_to_own_target_ast_cosine": float(
                    generated_to_target[index, index]
                ),
            }
        )

    return {
        "schema": "stable_audio_tools.source_semantics_ast",
        "schema_version": 1,
        "source_result": str(result_path.resolve()),
        "source_report_kind": resolved["kind"],
        "turn_index": resolved["turn_index"],
        "family_rank": int(report["family_rank"]),
        "family_id": str(report["family_id"]),
        "source_ids": source_ids,
        "model": str(model.config._name_or_path),
        "anchor_source": anchor_source,
        "separator": separator,
        "thresholds": {
            "min_anchor_probability": float(min_anchor_probability),
            "min_anchor_margin": float(min_anchor_margin),
            "min_target_gap": float(min_target_gap),
        },
        "sources": source_rows,
        "anchor_assignment": anchor_assignment,
        "matrices": {
            "generated_location_to_target_ast_anchor": _matrix_rows(
                generated_anchor_matrix,
                row_ids=source_ids,
                column_ids=source_ids,
            ),
            "target_location_to_target_ast_anchor": _matrix_rows(
                target_anchor_matrix,
                row_ids=source_ids,
                column_ids=source_ids,
            ),
            "generated_location_to_target_ast_posterior_cosine": _matrix_rows(
                generated_to_target,
                row_ids=source_ids,
                column_ids=source_ids,
            ),
            "target_location_to_target_ast_posterior_cosine": _matrix_rows(
                target_to_target,
                row_ids=source_ids,
                column_ids=source_ids,
            ),
        },
        "interpretation": (
            "AST anchors are selected from target stems only. Semantic scores "
            "are peak-normalized; active RMS ratios remain a separate required "
            "signal-retention diagnostic."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_json", nargs="+", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--ridge", type=float, default=0.05)
    parser.add_argument("--min-anchor-probability", type=float, default=0.02)
    parser.add_argument("--min-anchor-margin", type=float, default=0.01)
    parser.add_argument("--min-target-gap", type=float, default=0.01)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument(
        "--anchor-spec",
        type=Path,
        help="optional independent source-only AST anchors frozen before rollout",
    )
    args = parser.parse_args()
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")
    for path in args.result_json:
        if not path.is_file():
            raise FileNotFoundError(path)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA AST scoring requested but unavailable")
    cache_dir = args.cache_dir.expanduser().resolve()
    extractor = ASTFeatureExtractor.from_pretrained(
        args.model,
        cache_dir=str(cache_dir),
        local_files_only=not args.allow_download,
    )
    model = ASTForAudioClassification.from_pretrained(
        args.model,
        cache_dir=str(cache_dir),
        local_files_only=not args.allow_download,
    ).to(device)
    model.eval()
    anchor_spec = None
    if args.anchor_spec is not None:
        anchor_path = args.anchor_spec.expanduser().resolve()
        anchor_spec = json.loads(anchor_path.read_text(encoding="utf-8"))
        if not isinstance(anchor_spec, dict):
            raise ValueError("--anchor-spec must contain a JSON object")
    summaries = []
    for result_path in args.result_json:
        score = _score_report(
            model,
            extractor,
            result_path,
            device=device,
            ridge=args.ridge,
            min_anchor_probability=args.min_anchor_probability,
            min_anchor_margin=args.min_anchor_margin,
            min_target_gap=args.min_target_gap,
            top_k=args.top_k,
            anchor_spec=anchor_spec,
        )
        output_path = result_path.parent / "SOURCE_SEMANTICS_AST.json"
        _atomic_json(output_path, score)
        summaries.append(
            {
                "family_id": score["family_id"],
                "output": str(output_path),
                "anchor_assignment": score["anchor_assignment"],
            }
        )
    print(json.dumps(summaries, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
