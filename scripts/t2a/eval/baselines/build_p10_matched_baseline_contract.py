#!/usr/bin/env python3
"""Freeze public-baseline requests for an existing matched P10 eval panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from build_p10_60k_baseline_contract import BASELINES


DEFAULT_SOURCE_EVAL = Path(
    os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/archives/"
    "sceneplan_dit_v7_sao_300m_from_scratch_160k/evaluation/"
    "p10_40k_60k_80k_100k_120k_140k_50x3_v1"
)
DEFAULT_OUTPUT_ROOT = Path(
    os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/baselines/p10_v7_40k_140k_50x3_v1"
)
REPO_LOCK = Path(
    "." + "/evaluation_benchmark/contracts/REPOS.lock.json"
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_panel(
    panel: list[dict[str, Any]], expected_rows_per_domain: int
) -> dict[str, int]:
    domains = ("music", "sound", "speech")
    counts = {
        domain: sum(row.get("domain") == domain for row in panel)
        for domain in domains
    }
    expected = {domain: expected_rows_per_domain for domain in domains}
    if counts != expected or len(panel) != expected_rows_per_domain * len(domains):
        raise RuntimeError(f"panel count changed: rows={len(panel)} counts={counts}")
    panel_ids = [str(row["panel_id"]) for row in panel]
    sample_ids = [str(row["sample_id"]) for row in panel]
    if len(panel_ids) != len(set(panel_ids)):
        raise RuntimeError("panel_id is not unique")
    if len(sample_ids) != len(set(sample_ids)):
        raise RuntimeError("sample_id is not unique")
    for row in panel:
        sources = row["scene_plan"]["sources"]
        if len(sources) != 1 or sources[0]["kind"] != row["domain"]:
            raise RuntimeError(
                f"quality baseline panel must be single-source and domain-pure: "
                f"{row['panel_id']}"
            )
        for key in ("semantic_text", "reference_foa_path", "noise_seed"):
            if key not in row:
                raise RuntimeError(f"missing {key!r} in {row['panel_id']}")
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-eval", type=Path, default=DEFAULT_SOURCE_EVAL)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--rows-per-domain", type=int, default=50)
    parser.add_argument(
        "--checkpoint-steps",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Optional subset of source-evaluation checkpoints to include in the "
            "cross-system table. By default every frozen checkpoint is included."
        ),
    )
    args = parser.parse_args()

    source = args.source_eval.expanduser().resolve(strict=True)
    output_root = args.output_root.expanduser().resolve()
    eval_contract_path = source / "EVAL_CONTRACT.json"
    eval_contract = json.loads(eval_contract_path.read_text(encoding="utf-8"))
    panel_filename = eval_contract["test_set"]["panel_filename"]
    panel_path = source / panel_filename
    panel = _read_jsonl(panel_path)
    counts = _validate_panel(panel, int(args.rows_per_domain))
    panel_sha256 = _sha256(panel_path)
    if panel_sha256 != eval_contract["test_set"]["panel_sha256"]:
        raise RuntimeError("panel SHA256 disagrees with source evaluation contract")

    checkpoints = [
        {"step": int(row["step"]), "path": str(Path(row["path"]).resolve(strict=True))}
        for row in eval_contract["checkpoints"]
    ]
    steps = [row["step"] for row in checkpoints]
    if len(steps) != len(set(steps)) or steps != sorted(steps):
        raise RuntimeError(f"invalid checkpoint steps: {steps}")
    if args.checkpoint_steps is not None:
        requested_steps = [int(step) for step in args.checkpoint_steps]
        if (
            not requested_steps
            or len(requested_steps) != len(set(requested_steps))
            or requested_steps != sorted(requested_steps)
            or any(step <= 0 for step in requested_steps)
        ):
            raise ValueError(
                "--checkpoint-steps must be unique positive integers in ascending order"
            )
        checkpoints_by_step = {row["step"]: row for row in checkpoints}
        missing_steps = [step for step in requested_steps if step not in checkpoints_by_step]
        if missing_steps:
            raise RuntimeError(
                f"requested checkpoints are absent from source contract: {missing_steps}"
            )
        checkpoints = [checkpoints_by_step[step] for step in requested_steps]
        steps = requested_steps

    repo_lock = json.loads(REPO_LOCK.read_text(encoding="utf-8"))
    repo_commits = {row["name"]: row["commit"] for row in repo_lock["repositories"]}
    requests: list[dict[str, Any]] = []
    for baseline in BASELINES:
        if baseline["repo"] not in repo_commits:
            raise RuntimeError(f"unlocked upstream repo: {baseline['repo']}")
        for row in panel:
            domain = str(row["domain"])
            if domain not in baseline["domains"]:
                continue
            request: dict[str, Any] = {
                "baseline_id": baseline["id"],
                "baseline_display_name": baseline["display_name"],
                "domain": domain,
                "panel_id": row["panel_id"],
                "sample_id": row["sample_id"],
                "seed": int(row["noise_seed"] % (2**31 - 1)),
                "duration_sec": float(row["duration_sec"]),
                "semantic_prompt": row["semantic_text"],
                "reference_foa_path": row["reference_foa_path"],
                "native_output_path": str(
                    output_root
                    / "outputs"
                    / baseline["id"]
                    / domain
                    / row["panel_id"]
                    / "native.wav"
                ),
                "quality_w_path": str(
                    output_root
                    / "outputs"
                    / baseline["id"]
                    / domain
                    / row["panel_id"]
                    / "quality_w.wav"
                ),
            }
            if domain == "speech":
                source_spec = row["scene_plan"]["sources"][0]
                request["transcript"] = source_spec["transcript"]
                request["speaker_description"] = source_spec["speaker_description"]
                request["speech_seen_speaker"] = bool(
                    row.get("speech_seen_speaker", False)
                )
                request["speech_speaker_key"] = row.get("speech_speaker_key")
            for key in (
                "scene_composition",
                "length_bucket",
                "source_kinds",
                "source_kind_counts",
            ):
                if key in row:
                    request[key] = row[key]
            requests.append(request)

    expected_requests = sum(
        len(baseline["domains"]) * int(args.rows_per_domain)
        for baseline in BASELINES
    )
    if len(requests) != expected_requests:
        raise RuntimeError(
            f"generation request count changed: {len(requests)} != {expected_requests}"
        )

    contract = {
        "schema": "sceneplan_foa.p10_matched_cross_system_baseline_contract",
        "schema_version": 3,
        "status": "FROZEN",
        "source_eval_root": str(source),
        "source_eval_contract": str(eval_contract_path),
        "source_eval_contract_sha256": _sha256(eval_contract_path),
        "source_panel_path": str(panel_path),
        "source_panel_sha256": panel_sha256,
        "ours_checkpoints": checkpoints,
        "domain_counts": counts,
        "single_source_domain_pure": True,
        "music_profile": (
            "representative single-source formal-test music; singing or vocal "
            "timbre is allowed when present in the frozen test distribution, but "
            "no exact TTS transcript is supplied"
        ),
        "generation_request_count": len(requests),
        "quality_view": {
            "ours": "native FOA W channel only",
            "mono_baseline": "unchanged",
            "stereo_baseline": "fixed arithmetic mean downmix",
            "level_handling": "per-clip peak normalization to -1 dBFS in metric preprocessing",
            "forbidden": "treating W,W,W,W as spatial audio or duplicating channels in metric weights",
        },
        "spatial_view": {
            "ours": "native WYZX/ACN/SN3D FOA",
            "non_spatial_baselines": "N/A",
            "binaural_baselines": "separate common-render lane only",
        },
        "metrics": {
            "music_sound": [
                "CLAP",
                "paired CLAP",
                "FD-CLAP diagnostic",
                "FAD-VGGish diagnostic",
                "FD-PANN diagnostic",
                "paired KL-PANN",
            ],
            "speech": ["corpus WER", "corpus CER", "UTMOS"],
            "spatial": [
                "plan spherical DoA error",
                "azimuth circular MAE",
                "elevation MAE",
                "trajectory extent error",
                "activity IoU",
            ],
            "sample_warning": (
                f"{args.rows_per_domain} matched clips per domain; FAD/FD are "
                "checkpoint/baseline diagnostics, not the 1000-row publication estimates"
            ),
        },
        "baselines": [
            {
                **row,
                "domains": list(row["domains"]),
                "repo_commit": repo_commits[row["repo"]],
            }
            for row in BASELINES
        ],
        "withheld": [
            {
                "id": "stable_audio_3",
                "reason": "official weights remain gated in the frozen local setup",
            },
            {
                "id": "native_foa_literature_baselines",
                "systems": [
                    "Diff-SAGe",
                    "ImmerseDiffusion",
                    "SonicMotion",
                    "SwanSphere",
                ],
                "reason": "no verified runnable official checkpoint on this panel",
            },
        ],
    }
    _atomic_json(output_root / "BENCHMARK_CONTRACT.json", contract)
    _atomic_jsonl(output_root / "generation_requests.jsonl", requests)
    print(
        json.dumps(
            {
                "status": "PASS",
                "panel_rows": len(panel),
                "requests": len(requests),
                "checkpoints": steps,
                "output_root": str(output_root),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
