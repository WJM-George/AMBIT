#!/usr/bin/env python3
"""Score frame-level source presence with the research-only OpenFLAM model.

Global CLAP can reward a mixture dominated by one event.  This diagnostic asks
whether every ScenePlan source is present during its own activity interval and
absent outside it.  Generated probabilities are always reported beside the
retained target, so labels that OpenFLAM cannot recognize do not become silent
model failures.

OpenFLAM code and weights use the non-commercial Adobe Research License.  This
script is an optional research evaluator; it is never part of training or a
redistributable model artifact.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Every required model asset is explicit and checksum-verified.  Prevent an
# evaluator invocation from silently fetching or changing dependencies.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
import torchaudio

from stable_audio_tools.data.spatial_story import compile_source_tracks

from scripts.t2a.eval.diagnostics.diagnose_spatial_conditions import (
    _atomic_json,
)
from scripts.t2a.eval.diagnostics.score_source_location_semantics import (
    _resolve_scoring_inputs,
    _scene_sources,
    _source_caption,
)


OPENFLAM_SAMPLE_RATE = 48_000
OPENFLAM_DURATION_SEC = 10.0
OPENFLAM_SAMPLE_COUNT = int(OPENFLAM_SAMPLE_RATE * OPENFLAM_DURATION_SEC)
OPENFLAM_EXPECTED_SHA256 = (
    "0f7329b67ad6c3b3a31d5bbb30e91c74dcf7b9007f78d9d7af0bcc643bed1054"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _prepare_foa_w(path: Path) -> tuple[torch.Tensor, dict[str, Any]]:
    audio, sample_rate = torchaudio.load(str(path))
    if audio.ndim != 2 or audio.shape[0] != 4:
        raise ValueError(f"OpenFLAM input must be FOA [4,N], got {audio.shape}")
    mono = audio[0].float().reshape(1, -1)
    original_peak = float(mono.abs().amax())
    original_rms = float(mono.square().mean().sqrt())
    if sample_rate != OPENFLAM_SAMPLE_RATE:
        mono = torchaudio.functional.resample(
            mono, sample_rate, OPENFLAM_SAMPLE_RATE
        )
    if mono.shape[-1] < OPENFLAM_SAMPLE_COUNT:
        mono = torch.nn.functional.pad(
            mono, (0, OPENFLAM_SAMPLE_COUNT - mono.shape[-1])
        )
    else:
        mono = mono[:, :OPENFLAM_SAMPLE_COUNT]
    clipped_fraction = float((mono.abs() > 1.0).float().mean())
    return mono.clamp(-1.0, 1.0).squeeze(0), {
        "path": str(path),
        "source_sample_rate": int(sample_rate),
        "source_sample_count": int(audio.shape[-1]),
        "source_duration_sec": float(audio.shape[-1] / sample_rate),
        "w_peak": original_peak,
        "w_rms": original_rms,
        "clipped_fraction_before_clamp": clipped_fraction,
        "analysis_sample_rate": OPENFLAM_SAMPLE_RATE,
        "analysis_sample_count": OPENFLAM_SAMPLE_COUNT,
        "analysis_duration_sec": OPENFLAM_DURATION_SEC,
    }


def _masked_summary(values: torch.Tensor, mask: torch.Tensor) -> dict[str, Any]:
    values = torch.as_tensor(values, dtype=torch.float32).flatten()
    mask = torch.as_tensor(mask, dtype=torch.bool).flatten()
    if values.shape != mask.shape:
        raise ValueError("probabilities and activity mask must have equal shape")
    selected = values[mask]
    if selected.numel() == 0:
        return {"frame_count": 0, "mean": None, "median": None, "p90": None}
    return {
        "frame_count": int(selected.numel()),
        "mean": float(selected.mean()),
        "median": float(selected.median()),
        "p90": float(torch.quantile(selected, 0.9)),
    }


def _presence_summary(
    values: torch.Tensor, active_mask: torch.Tensor
) -> dict[str, Any]:
    active = _masked_summary(values, active_mask)
    inactive = _masked_summary(values, ~active_mask)
    margin = (
        active["mean"] - inactive["mean"]
        if active["mean"] is not None and inactive["mean"] is not None
        else None
    )
    return {"active": active, "inactive": inactive, "activity_margin": margin}


def _safe_ratio(numerator: Any, denominator: Any) -> float | None:
    if numerator is None or denominator is None:
        return None
    numerator = float(numerator)
    denominator = float(denominator)
    if not math.isfinite(numerator) or not math.isfinite(denominator):
        return None
    return numerator / max(denominator, 1.0e-6)


def _load_openflam(
    *, package_root: Path, checkpoint: Path, text_cache: Path, device: torch.device
):
    if not package_root.is_dir():
        raise FileNotFoundError(package_root)
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))
    import openflam
    import openflam.hook as hook

    hook.hf_hub_download = lambda *args, **kwargs: str(checkpoint)
    model = openflam.OpenFLAM(
        model_name="v1-base", default_ckpt_path=str(text_cache)
    )
    return model.eval().to(device)


@torch.inference_mode()
def _score_result(model, result_path: Path) -> dict[str, Any]:
    report = json.loads(result_path.read_text(encoding="utf-8"))
    resolved = _resolve_scoring_inputs(report, result_path)
    plan = resolved["plan"]
    sources = _scene_sources(plan)
    source_ids = [str(source["source_id"]) for source in sources]
    captions = [_source_caption(source) for source in sources]
    generated, generated_metadata = _prepare_foa_w(resolved["generated_path"])
    target, target_metadata = _prepare_foa_w(resolved["target_path"])
    device = next(model.parameters()).device
    audio = torch.stack((generated, target)).to(device)
    # OpenFLAM v1 squeezes its text-feature axis when exactly one caption is
    # supplied, which breaks the package's own cross-product bias broadcast.
    # Duplicating the sole query is mathematically neutral; score the first
    # copy and discard the compatibility copy below.
    query_captions = captions if len(captions) > 1 else captions * 2
    probabilities = model.get_local_similarity(
        audio, query_captions, method="unbiased", cross_product=True
    ).float().cpu()
    if probabilities.ndim != 3 or tuple(probabilities.shape[:2]) != (
        2,
        len(query_captions),
    ):
        raise ValueError(
            "OpenFLAM returned an unexpected probability tensor: "
            f"{tuple(probabilities.shape)}"
        )
    probabilities = probabilities[:, : len(sources)]
    frame_count = int(probabilities.shape[-1])

    analysis_plan = copy.deepcopy(plan)
    analysis_plan.setdefault("audio", {})["duration_sec"] = OPENFLAM_DURATION_SEC
    persistent_slots = []
    for source_index, source_id in enumerate(source_ids):
        if source_id.startswith("source_") and source_id[7:].isdigit():
            persistent_slots.append(int(source_id[7:]))
        elif source_id.startswith("s") and source_id[1:].isdigit():
            persistent_slots.append(int(source_id[1:]))
        else:
            persistent_slots.append(source_index)
    compiled = compile_source_tracks(
        analysis_plan,
        num_frames=frame_count,
        max_sources=max(len(sources), max(persistent_slots) + 1),
    )
    compiled_slot_by_id = {
        str(source_id): slot
        for slot, source_id in enumerate(compiled["source_ids"])
        if source_id is not None
    }
    if set(compiled_slot_by_id) != set(source_ids):
        raise RuntimeError("compiled OpenFLAM source identities changed")
    active_masks = torch.stack(
        [
            compiled["tracks"][compiled_slot_by_id[source_id], 0] > 0.5
            for source_id in source_ids
        ]
    )

    rows = []
    generated_active_means = []
    target_active_means = []
    for index, (source_id, caption) in enumerate(zip(source_ids, captions)):
        generated_summary = _presence_summary(
            probabilities[0, index], active_masks[index]
        )
        target_summary = _presence_summary(
            probabilities[1, index], active_masks[index]
        )
        generated_active = generated_summary["active"]["mean"]
        target_active = target_summary["active"]["mean"]
        if generated_active is not None:
            generated_active_means.append(generated_active)
        if target_active is not None:
            target_active_means.append(target_active)
        rows.append(
            {
                "source_id": source_id,
                "caption": caption,
                "generated": generated_summary,
                "target": target_summary,
                "active_presence_retention": _safe_ratio(
                    generated_active, target_active
                ),
                "activity_margin_delta": (
                    generated_summary["activity_margin"]
                    - target_summary["activity_margin"]
                    if generated_summary["activity_margin"] is not None
                    and target_summary["activity_margin"] is not None
                    else None
                ),
            }
        )

    return {
        "schema": "stable_audio_tools.source_presence_openflam",
        "schema_version": 1,
        "source_result": str(result_path.resolve()),
        "source_report_kind": resolved["kind"],
        "turn_index": resolved["turn_index"],
        "family_rank": report["family_rank"],
        "family_id": report["family_id"],
        "method": "OpenFLAM unbiased framewise activation, W channel",
        "frame_count": frame_count,
        "audio": {"generated": generated_metadata, "target": target_metadata},
        "multi_flam": {
            "generated_active_mean": (
                sum(generated_active_means) / len(generated_active_means)
                if generated_active_means
                else None
            ),
            "target_active_mean": (
                sum(target_active_means) / len(target_active_means)
                if target_active_means
                else None
            ),
        },
        "sources": rows,
        "interpretation": (
            "Compare generated probabilities to the retained target for each "
            "source. A low target score means the evaluator does not recognize "
            "that label and should not be used as a model-failure gate."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_json", nargs="+", type=Path)
    parser.add_argument(
        "--package-root", type=Path, default=Path(os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/sat_tools/openflam-py")
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/evaluators/openflam/open_flam_oct17.pth"
        ),
    )
    parser.add_argument(
        "--license",
        type=Path,
        default=Path(os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/evaluators/openflam/LICENSE"),
    )
    parser.add_argument(
        "--text-cache",
        type=Path,
        default=Path(
            "." + "/codex-home/.cache/huggingface/hub"
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    for path in [*args.result_json, args.checkpoint, args.license]:
        if not path.is_file():
            raise FileNotFoundError(path)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA OpenFLAM scoring requested but unavailable")
    checkpoint_sha256 = _sha256(args.checkpoint)
    if checkpoint_sha256 != OPENFLAM_EXPECTED_SHA256:
        raise RuntimeError(
            "OpenFLAM checkpoint checksum mismatch: "
            f"{checkpoint_sha256} != {OPENFLAM_EXPECTED_SHA256}"
        )
    model = _load_openflam(
        package_root=args.package_root,
        checkpoint=args.checkpoint,
        text_cache=args.text_cache,
        device=device,
    )
    summaries = []
    for result_path in args.result_json:
        score = _score_result(model, result_path)
        score["model"] = {
            "name": "kechenadobe/OpenFLAM v1-base",
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": checkpoint_sha256,
            "package_root": str(args.package_root.resolve()),
            "license": {
                "name": "Adobe Research License",
                "restriction": "non-commercial research evaluation only",
                "path": str(args.license.resolve()),
            },
        }
        output_path = result_path.parent / "SOURCE_PRESENCE_OPENFLAM.json"
        _atomic_json(output_path, score)
        summaries.append(
            {
                "family_id": score["family_id"],
                "output": str(output_path),
                "multi_flam": score["multi_flam"],
                "source_retention": {
                    row["source_id"]: row["active_presence_retention"]
                    for row in score["sources"]
                },
            }
        )
    print(json.dumps(summaries, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
