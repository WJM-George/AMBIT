#!/usr/bin/env python3
"""Score whether generated source content occurs at the planned FOA locations.

The global W channel can say whether a mixture contains a dog, speech, or
music, but it cannot say which event occupies which source trajectory.  This
diagnostic uses the authoritative ScenePlan to build a regularized first-order
demixer for each active frame.  It then compares each generated directional
stem with both the corresponding target stem and every source caption in CLAP
space.

FOA demixing is imperfect in reverberant or nearly co-located scenes.  Scores
therefore include target-only discriminability gaps and abstain from aggregate
assignment accuracy when the retained target itself cannot distinguish a
source.  The metric is evidence for content-location binding, not a substitute
for full-mixture content, waveform, spatial-field, silence, or listening tests.
"""
from __future__ import annotations

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

from stable_audio_tools.data.spatial_story import compile_source_tracks
from stable_audio_tools.training.metrics.fad_metrics import load_clap_model

from scripts.t2a.eval.diagnostics.diagnose_spatial_conditions import (
    _atomic_json,
)


def _cosine_matrix(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left = torch.nn.functional.normalize(left.float(), dim=-1)
    right = torch.nn.functional.normalize(right.float(), dim=-1)
    return left @ right.transpose(0, 1)


def _clap_text_embeddings(clap_model, captions: Sequence[str]) -> torch.Tensor:
    """Embed captions without triggering LAION-CLAP's singleton squeeze."""

    queries = list(captions)
    if not queries:
        raise ValueError("CLAP source scoring requires at least one caption")
    requested = queries if len(queries) > 1 else queries * 2
    embeddings = clap_model.get_text_embedding(
        requested, use_tensor=True
    ).float()
    if embeddings.ndim != 2 or embeddings.shape[0] != len(requested):
        raise RuntimeError(
            "CLAP returned unexpected text embeddings: "
            f"{tuple(embeddings.shape)} for {len(requested)} captions"
        )
    return embeddings[: len(queries)]


def _source_caption(source: Mapping[str, Any]) -> str:
    event = source.get("event") or {}
    content = source.get("content") or {}
    label = str(event.get("label") or event.get("category") or "sound").strip()
    transcript = str(content.get("transcript") or "").strip()
    if transcript:
        return f'{label} saying "{transcript}"'
    return label


def _scene_sources(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    sources = ((plan.get("scene") or {}).get("sources") or [])
    if not isinstance(sources, list) or not sources:
        raise ValueError("ScenePlan must contain at least one source")
    if not all(isinstance(source, dict) for source in sources):
        raise ValueError("ScenePlan sources must be dictionaries")
    source_ids = [str(source.get("source_id") or "") for source in sources]
    if "" in source_ids or len(source_ids) != len(set(source_ids)):
        raise ValueError("ScenePlan source IDs must be present and unique")
    return sources


def _compile_active_source_tracks(
    plan: Mapping[str, Any],
    *,
    num_frames: int,
    source_ids: Sequence[str],
) -> torch.Tensor:
    """Compile persistent slots, then return active tracks in scene order.

    Isolated curriculum rows intentionally retain IDs such as ``source_1``
    even when ``source_0`` is absent.  ``compile_source_tracks`` therefore
    needs enough persistent-slot capacity, while the demixer needs only the
    active tracks in the same order as the captions and source IDs.
    """

    persistent_slots = []
    for source_index, source_id in enumerate(source_ids):
        if source_id.startswith("source_") and source_id[7:].isdigit():
            persistent_slots.append(int(source_id[7:]))
        elif source_id.startswith("s") and source_id[1:].isdigit():
            persistent_slots.append(int(source_id[1:]))
        else:
            persistent_slots.append(source_index)
    compiled = compile_source_tracks(
        plan,
        num_frames=num_frames,
        max_sources=max(len(source_ids), max(persistent_slots) + 1),
    )
    compiled_slot_by_id = {
        str(source_id): slot
        for slot, source_id in enumerate(compiled["source_ids"])
        if source_id is not None
    }
    if set(compiled_slot_by_id) != set(source_ids):
        raise RuntimeError(
            "compiled source identities changed: "
            f"{sorted(compiled_slot_by_id)} != {sorted(source_ids)}"
        )
    return torch.stack(
        [compiled["tracks"][compiled_slot_by_id[source_id]] for source_id in source_ids]
    )


def _report_path(result_path: Path, value: Any, *, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty path")
    path = Path(value)
    if not path.is_absolute():
        path = result_path.parent / path
    return path.resolve()


def _resolve_scoring_inputs(
    report: Mapping[str, Any], result_path: Path
) -> dict[str, Any]:
    """Normalize intervention and standard one-turn evaluation reports.

    The intervention diagnostic pins its plan and audio paths at the report
    root.  The normal checkpoint evaluator stores the same authoritative
    information in its sole turn.  Supporting both keeps source-binding
    scoring attached to the canonical evaluator instead of requiring a second
    generation pass.
    """

    if report.get("schema") == "stable_audio_tools.semantic_condition_diagnostic":
        plan = report.get("correct_scene_plan")
        if not isinstance(plan, dict):
            raise ValueError(
                f"{result_path} has no pinned correct_scene_plan; regenerate "
                "with schema v2"
            )
        settings = report.get("settings") or {}
        baseline_name = "plan_semantics_correct__caption_correct"
        conditions = report.get("conditions") or {}
        baseline = conditions.get(baseline_name) or {}
        return {
            "kind": "semantic_condition_diagnostic",
            "turn_index": None,
            "plan": plan,
            "generated_path": _report_path(
                result_path,
                baseline.get("audio_path"),
                field="baseline audio_path",
            ),
            "target_path": _report_path(
                result_path, report.get("target_audio_path"), field="target_audio_path"
            ),
            "sample_rate": int(settings.get("sample_rate") or 0),
            "hop": int(settings.get("downsampling_ratio") or 0),
            "frame_count": 0,
        }

    if int(report.get("evaluator_version") or 0) >= 2:
        turns = report.get("turn_results")
        if not isinstance(turns, list) or len(turns) != 1:
            raise ValueError(
                "standard checkpoint source-location scoring requires exactly "
                "one evaluated turn"
            )
        turn = turns[0]
        if not isinstance(turn, dict):
            raise ValueError("checkpoint evaluator turn result must be an object")
        plan_path = _report_path(
            result_path, turn.get("target_plan_path"), field="target_plan_path"
        )
        if not plan_path.is_file():
            raise FileNotFoundError(plan_path)
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        if not isinstance(plan, dict):
            raise ValueError(f"target ScenePlan must be an object: {plan_path}")
        spatial = turn.get("spatial_alignment") or {}
        silence = turn.get("silence_alignment") or {}
        return {
            "kind": "checkpoint_evaluation",
            "turn_index": int(turn.get("turn") or 0),
            "plan": plan,
            "generated_path": _report_path(
                result_path, turn.get("audio_path"), field="turn audio_path"
            ),
            "target_path": _report_path(
                result_path,
                turn.get("target_audio_path"),
                field="turn target_audio_path",
            ),
            # content_alignment.sample_rate is the CLAP resampling rate, not
            # necessarily the retained FOA file rate.  For normal evaluator
            # results the generated and target file headers are authoritative.
            "sample_rate": 0,
            "hop": int(
                spatial.get("hop") or silence.get("frame_samples") or 0
            ),
            "frame_count": int(
                spatial.get("frame_count") or silence.get("frame_count") or 0
            ),
        }

    raise ValueError(f"unsupported source-location result schema: {result_path}")


def _demix_foa_sources(
    audio: torch.Tensor,
    tracks: torch.Tensor,
    *,
    ridge: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Regularized frame-wise FOA pseudoinverse for active source tracks.

    ``audio`` uses ACN/SN3D ``[W,Y,Z,X]``.  Each point-source steering column
    is ``[1/sqrt(2), y, z, x]``.  Inactive columns are zero, so their solved
    stems remain exactly zero.  The output keeps the original full duration;
    activity masks prevent inactive tails or unrelated sources from becoming
    semantic evidence for a slot.
    """

    audio = torch.as_tensor(audio, dtype=torch.float32).cpu()
    tracks = torch.as_tensor(tracks, dtype=torch.float32).cpu()
    if audio.ndim != 2 or audio.shape[0] != 4:
        raise ValueError(f"audio must be FOA [4,N], got {tuple(audio.shape)}")
    if tracks.ndim != 3 or tracks.shape[1] != 8:
        raise ValueError(
            "source tracks must be [source,8,frame], got "
            f"{tuple(tracks.shape)}"
        )
    if not math.isfinite(ridge) or ridge <= 0.0:
        raise ValueError("ridge must be finite and positive")
    source_count, _, frame_count = tracks.shape
    if source_count < 1 or audio.shape[-1] % frame_count:
        raise ValueError(
            f"audio length {audio.shape[-1]} is not divisible by "
            f"track frames {frame_count}"
        )
    hop = audio.shape[-1] // frame_count
    active = tracks[:, 0].transpose(0, 1) > 0.5  # [frame, source]
    directions = tracks[:, 1:4].permute(2, 0, 1).contiguous()  # [frame, source, xyz]
    directions = directions / directions.norm(dim=-1, keepdim=True).clamp_min(
        1.0e-8
    )

    steering = torch.zeros(frame_count, 4, source_count)
    steering[:, 0, :] = active.float() / math.sqrt(2.0)
    steering[:, 1, :] = directions[:, :, 1]  # Y
    steering[:, 2, :] = directions[:, :, 2]  # Z
    steering[:, 3, :] = directions[:, :, 0]  # X
    identity = torch.eye(source_count).unsqueeze(0)
    decoder = torch.linalg.solve(
        steering.transpose(1, 2) @ steering + ridge * identity,
        steering.transpose(1, 2),
    )
    framed_audio = audio.reshape(4, frame_count, hop).permute(1, 0, 2)
    stems = (decoder @ framed_audio).permute(1, 0, 2).reshape(source_count, -1)
    sample_mask = active.transpose(0, 1).repeat_interleave(hop, dim=1)
    stems = stems * sample_mask

    condition_numbers = []
    active_counts = active.sum(dim=1)
    for frame_index in torch.nonzero(active_counts > 1, as_tuple=False).flatten():
        columns = active[frame_index]
        condition_numbers.append(
            float(torch.linalg.cond(steering[frame_index, :, columns]))
        )
    condition_tensor = torch.tensor(condition_numbers, dtype=torch.float32)
    condition_summary = {
        "multi_source_frame_count": len(condition_numbers),
        "median": (
            float(condition_tensor.median()) if condition_numbers else None
        ),
        "p90": (
            float(torch.quantile(condition_tensor, 0.9))
            if condition_numbers
            else None
        ),
        "max": float(condition_tensor.max()) if condition_numbers else None,
    }
    metadata = {
        "hop": hop,
        "frame_count": frame_count,
        "source_count": source_count,
        "active_frame_counts": [int(value) for value in active.sum(dim=0)],
        "condition_number": condition_summary,
        "ridge": float(ridge),
        "steering": "ACN_SN3D_WYZX_point_source_regularized_pseudoinverse",
    }
    return stems, metadata


def _prepare_clap_stems(
    stems: torch.Tensor,
    *,
    sample_rate: int,
    device: torch.device,
) -> torch.Tensor:
    values = []
    for stem in stems:
        mono = stem[None].float().to(device)
        peak = mono.abs().amax().clamp_min(1.0e-8)
        mono = mono / peak * (10.0 ** (-1.0 / 20.0))
        if sample_rate != 48_000:
            mono = torchaudio.functional.resample(mono, sample_rate, 48_000)
        values.append(mono.clamp(-1.0, 1.0))
    sample_counts = {int(value.shape[-1]) for value in values}
    if len(sample_counts) != 1:
        raise RuntimeError(f"resampled stems differ in length: {sample_counts}")
    return torch.cat(values, dim=0)


def _save_listening_stem(
    path: Path, stem: torch.Tensor, *, sample_rate: int
) -> float:
    """Save a non-clipping mono preview and return its applied gain."""

    mono = torch.as_tensor(stem, dtype=torch.float32).cpu().reshape(1, -1)
    peak = float(mono.abs().amax())
    gain = min(1.0, (10.0 ** (-1.0 / 20.0)) / max(peak, 1.0e-8))
    torchaudio.save(str(path), (mono * gain).clamp(-1.0, 1.0), sample_rate)
    return gain


def _matrix_rows(
    matrix: torch.Tensor,
    *,
    row_ids: Sequence[str],
    column_ids: Sequence[str],
) -> dict[str, dict[str, float]]:
    if tuple(matrix.shape) != (len(row_ids), len(column_ids)):
        raise ValueError("matrix labels do not match its shape")
    return {
        row_id: {
            column_id: float(matrix[row, column])
            for column, column_id in enumerate(column_ids)
        }
        for row, row_id in enumerate(row_ids)
    }


def _assignment_metrics(
    generated_matrix: torch.Tensor,
    target_reference_matrix: torch.Tensor,
    *,
    source_ids: Sequence[str],
    min_target_gap: float,
) -> dict[str, Any]:
    """Target-calibrated diagonal assignment statistics.

    Rows are location-aligned stems and columns are source identities.  A row
    is valid only when the corresponding target row prefers its own identity
    by ``min_target_gap``.  This prevents generic music labels, co-located
    sources, or reverberant leakage from becoming a false model failure.
    """

    generated_matrix = torch.as_tensor(generated_matrix, dtype=torch.float32)
    target_reference_matrix = torch.as_tensor(
        target_reference_matrix, dtype=torch.float32
    )
    count = len(source_ids)
    expected = (count, count)
    if tuple(generated_matrix.shape) != expected or tuple(
        target_reference_matrix.shape
    ) != expected:
        raise ValueError("assignment matrices must be square source matrices")
    if not math.isfinite(min_target_gap) or min_target_gap < 0.0:
        raise ValueError("min_target_gap must be finite and non-negative")

    rows = []
    valid_correct = []
    for index, source_id in enumerate(source_ids):
        other = [column for column in range(count) if column != index]
        target_diagonal = float(target_reference_matrix[index, index])
        target_other = (
            max(float(target_reference_matrix[index, column]) for column in other)
            if other
            else -1.0
        )
        target_gap = target_diagonal - target_other
        generated_diagonal = float(generated_matrix[index, index])
        generated_other = (
            max(float(generated_matrix[index, column]) for column in other)
            if other
            else -1.0
        )
        generated_gap = generated_diagonal - generated_other
        predicted = int(torch.argmax(generated_matrix[index]))
        valid = target_gap >= min_target_gap
        correct = predicted == index
        if valid:
            valid_correct.append(correct)
        rows.append(
            {
                "source_id": source_id,
                "target_discriminability_gap": target_gap,
                "target_valid": valid,
                "generated_diagonal": generated_diagonal,
                "generated_diagonal_margin": generated_gap,
                "generated_assignment": source_ids[predicted],
                "generated_correct": correct,
            }
        )
    valid_gaps = [row["generated_diagonal_margin"] for row in rows if row["target_valid"]]
    return {
        "min_target_gap": float(min_target_gap),
        "valid_source_count": len(valid_correct),
        "source_count": count,
        "accuracy_on_target_discriminable_sources": (
            sum(valid_correct) / len(valid_correct) if valid_correct else None
        ),
        "mean_generated_diagonal_margin_on_valid_sources": (
            sum(valid_gaps) / len(valid_gaps) if valid_gaps else None
        ),
        "rows": rows,
    }


@torch.inference_mode()
def _score_report(
    clap_model,
    result_path: Path,
    *,
    ridge: float,
    min_target_audio_gap: float,
    min_target_caption_gap: float,
    save_stems: bool,
) -> dict[str, Any]:
    report = json.loads(result_path.read_text(encoding="utf-8"))
    resolved = _resolve_scoring_inputs(report, result_path)
    plan = resolved["plan"]
    sources = _scene_sources(plan)
    source_ids = [str(source["source_id"]) for source in sources]
    captions = [_source_caption(source) for source in sources]
    expected_sample_rate = int(resolved["sample_rate"])
    hop = int(resolved["hop"])
    frame_count_hint = int(resolved["frame_count"])
    generated_path = resolved["generated_path"]
    target_path = resolved["target_path"]
    generated, generated_rate = torchaudio.load(str(generated_path))
    target, target_rate = torchaudio.load(str(target_path))
    if generated_rate != target_rate or (
        expected_sample_rate > 0 and generated_rate != expected_sample_rate
    ):
        raise ValueError(
            f"audio rates disagree: generated={generated_rate}, target={target_rate}, "
            f"expected={expected_sample_rate}"
        )
    if generated.shape != target.shape:
        raise ValueError(
            f"generated/target shape mismatch: {generated.shape} != {target.shape}"
        )
    if hop > 0:
        if generated.shape[-1] % hop:
            raise ValueError(
                f"audio length {generated.shape[-1]} is not divisible by hop {hop}"
            )
        frame_count = generated.shape[-1] // hop
    elif frame_count_hint > 0:
        if generated.shape[-1] % frame_count_hint:
            raise ValueError(
                f"audio length {generated.shape[-1]} is not divisible by "
                f"frame_count {frame_count_hint}"
            )
        frame_count = frame_count_hint
        hop = generated.shape[-1] // frame_count
    else:
        raise ValueError("source-location report lacks hop/frame-count metadata")
    if frame_count_hint > 0 and frame_count != frame_count_hint:
        raise ValueError(
            f"frame-count metadata mismatch: {frame_count} != {frame_count_hint}"
        )
    source_tracks = _compile_active_source_tracks(
        plan,
        num_frames=frame_count,
        source_ids=source_ids,
    )
    generated_stems, separator = _demix_foa_sources(
        generated, source_tracks, ridge=ridge
    )
    target_stems, target_separator = _demix_foa_sources(
        target, source_tracks, ridge=ridge
    )
    if separator != target_separator:
        raise RuntimeError("generated and target separators are not identical")

    stem_paths: dict[str, dict[str, str]] = {}
    stem_preview_gains: dict[str, dict[str, float]] = {}
    if save_stems:
        stem_dir = result_path.parent / "source_location_stems"
        stem_dir.mkdir(parents=True, exist_ok=True)
        for index, source_id in enumerate(source_ids):
            generated_stem_path = stem_dir / f"{source_id}.generated.wav"
            target_stem_path = stem_dir / f"{source_id}.target.wav"
            generated_preview_gain = _save_listening_stem(
                generated_stem_path,
                generated_stems[index],
                sample_rate=generated_rate,
            )
            target_preview_gain = _save_listening_stem(
                target_stem_path,
                target_stems[index],
                sample_rate=target_rate,
            )
            stem_paths[source_id] = {
                "generated": str(generated_stem_path),
                "target": str(target_stem_path),
            }
            stem_preview_gains[source_id] = {
                "generated": generated_preview_gain,
                "target": target_preview_gain,
            }

    device = next(clap_model.model.parameters()).device
    clap_audio = torch.cat(
        (
            _prepare_clap_stems(
                generated_stems, sample_rate=generated_rate, device=device
            ),
            _prepare_clap_stems(target_stems, sample_rate=target_rate, device=device),
        ),
        dim=0,
    )
    audio_embeddings = clap_model.get_audio_embedding_from_data(
        x=clap_audio, use_tensor=True
    ).float()
    text_embeddings = _clap_text_embeddings(clap_model, captions)
    source_count = len(source_ids)
    generated_embeddings = audio_embeddings[:source_count]
    target_embeddings = audio_embeddings[source_count:]
    generated_to_target = _cosine_matrix(
        generated_embeddings, target_embeddings
    ).cpu()
    target_to_target = _cosine_matrix(target_embeddings, target_embeddings).cpu()
    generated_to_caption = _cosine_matrix(
        generated_embeddings, text_embeddings
    ).cpu()
    target_to_caption = _cosine_matrix(target_embeddings, text_embeddings).cpu()

    return {
        "schema": "stable_audio_tools.source_location_semantic_scores",
        "schema_version": 1,
        "source_result": str(result_path.resolve()),
        "source_report_kind": resolved["kind"],
        "turn_index": resolved["turn_index"],
        "family_rank": report["family_rank"],
        "family_id": report["family_id"],
        "source_ids": source_ids,
        "source_captions": dict(zip(source_ids, captions)),
        "separator": separator,
        "stem_rms": {
            source_id: {
                "generated": float(generated_stems[index].square().mean().sqrt()),
                "target": float(target_stems[index].square().mean().sqrt()),
            }
            for index, source_id in enumerate(source_ids)
        },
        "stem_paths": stem_paths,
        "stem_preview_gains": stem_preview_gains,
        "stems_saved": bool(save_stems),
        "matrices": {
            "generated_location_to_target_source_audio": _matrix_rows(
                generated_to_target, row_ids=source_ids, column_ids=source_ids
            ),
            "target_location_to_target_source_audio": _matrix_rows(
                target_to_target, row_ids=source_ids, column_ids=source_ids
            ),
            "generated_location_to_source_caption": _matrix_rows(
                generated_to_caption, row_ids=source_ids, column_ids=source_ids
            ),
            "target_location_to_source_caption": _matrix_rows(
                target_to_caption, row_ids=source_ids, column_ids=source_ids
            ),
        },
        "target_audio_assignment": _assignment_metrics(
            generated_to_target,
            target_to_target,
            source_ids=source_ids,
            min_target_gap=min_target_audio_gap,
        ),
        "caption_assignment": _assignment_metrics(
            generated_to_caption,
            target_to_caption,
            source_ids=source_ids,
            min_target_gap=min_target_caption_gap,
        ),
        "interpretation": (
            "Assignment accuracy is reported only for sources that the target "
            "directional stems distinguish by the configured target-only gap. "
            "A causal source-slot response alone does not establish correct binding."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_json", nargs="+", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--clap-model", default="630k-audioset-fusion-best.pt")
    parser.add_argument("--ridge", type=float, default=0.05)
    parser.add_argument("--min-target-audio-gap", type=float, default=0.02)
    parser.add_argument("--min-target-caption-gap", type=float, default=0.01)
    parser.add_argument(
        "--save-stems",
        action="store_true",
        help="persist normalized listening stems; disabled by default to limit disk use",
    )
    args = parser.parse_args()
    for path in args.result_json:
        if not path.is_file():
            raise FileNotFoundError(path)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA scoring requested but unavailable")
    clap_model = load_clap_model(args.clap_model, device=str(device))
    summaries = []
    for result_path in args.result_json:
        score = _score_report(
            clap_model,
            result_path,
            ridge=args.ridge,
            min_target_audio_gap=args.min_target_audio_gap,
            min_target_caption_gap=args.min_target_caption_gap,
            save_stems=args.save_stems,
        )
        score["clap_model"] = args.clap_model
        output_path = result_path.parent / "SOURCE_LOCATION_METRICS.json"
        _atomic_json(output_path, score)
        summaries.append(
            {
                "family_id": score["family_id"],
                "output": str(output_path),
                "target_audio_assignment": score["target_audio_assignment"],
                "caption_assignment": score["caption_assignment"],
            }
        )
    print(json.dumps(summaries, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
