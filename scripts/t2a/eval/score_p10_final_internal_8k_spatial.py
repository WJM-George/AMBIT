#!/usr/bin/env python3
"""Score native-FOA spatial and temporal execution on the final internal 8k.

These metrics apply only to P10 and the native FOA reference.  Public
mono/stereo baselines remain N/A.  Scene-level values are reported for every
Music/Sound/Speech-presence lane and one-through-four-source stratum.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.t2a.eval.sceneplan_dit_p10_panel_common import (
    load_foa,
    read_jsonl,
    summarize,
)
from scripts.t2a.eval.score_sceneplan_dit_p10_core import (
    _activity_metrics,
    _doa_metrics,
    _paired_doa_metrics,
)


DEFAULT_ROOT = Path(
    os.environ.get("AMBIT_DATA_ROOT", "data") + "/sceneplan_v2_1p124m/revisions/"
    "speech_expansion_noalign_15s_v1/evaluation/"
    "p10_v11_150k_full_test_8000_semantic_v2"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    os.replace(temporary, path)


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "generated_reference_doa_error_deg": summarize(
            row["generated_reference_doa"]["spherical_error_mean_deg"] for row in rows
        ),
        "generated_plan_doa_error_deg": summarize(
            row["generated_doa"]["spherical_error_mean_deg"] for row in rows
        ),
        "reference_plan_doa_error_deg": summarize(
            row["reference_doa"]["spherical_error_mean_deg"] for row in rows
        ),
        "generated_activity_iou": summarize(
            row["generated_activity"]["temporal_iou"] for row in rows
        ),
        "reference_activity_iou": summarize(
            row["reference_activity"]["temporal_iou"] for row in rows
        ),
        "generated_activity_onset_error_sec": summarize(
            row["generated_activity"]["onset_abs_error_sec"] for row in rows
        ),
        "generated_activity_offset_error_sec": summarize(
            row["generated_activity"]["offset_abs_error_sec"] for row in rows
        ),
        "generated_valid_direction_fraction": summarize(
            row["generated_doa"]["valid_direction_fraction"] for row in rows
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    root = args.eval_root.expanduser().resolve(strict=True)
    (root / "INFERENCE_COMPLETE").resolve(strict=True)
    contract_path = (root / "EVAL_CONTRACT.json").resolve(strict=True)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    panel_path = (root / contract["test_set"]["panel_filename"]).resolve(strict=True)
    if _sha256(panel_path) != contract["test_set"]["panel_sha256"]:
        raise RuntimeError("internal 8k panel SHA256 changed")
    panel = read_jsonl(panel_path)
    if len(panel) != 8000:
        raise RuntimeError("internal spatial panel changed")

    scored: list[dict[str, Any]] = []
    for index, panel_row in enumerate(panel, start=1):
        metadata_path = (
            root
            / "outputs"
            / "step_150000"
            / panel_row["domain"]
            / panel_row["panel_id"]
            / "metadata.json"
        ).resolve(strict=True)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not (
            metadata.get("status") == "PASS"
            and int(metadata.get("checkpoint_step", -1)) == 150000
            and metadata.get("panel_id") == panel_row["panel_id"]
            and metadata.get("sample_id") == panel_row["sample_id"]
            and metadata.get("reference_foa_sha256")
            == panel_row["reference_foa_sha256"]
        ):
            raise RuntimeError(f"P10 output metadata mismatch: {metadata_path}")
        samples = int(metadata["model_num_samples"])
        frames = int(metadata["latent_frames"])
        generated, generated_rate = load_foa(
            metadata["generated_foa_path"], expected_samples=samples
        )
        reference, reference_rate = load_foa(
            metadata["reference_foa_path"], expected_samples=samples
        )
        if generated_rate != 44_100 or reference_rate != 44_100:
            raise RuntimeError("internal spatial sample rate changed")
        scene_plan = panel_row["scene_plan"]
        scored.append(
            {
                "panel_id": panel_row["panel_id"],
                "sample_id": panel_row["sample_id"],
                "source_count": int(panel_row["source_count"]),
                "source_kinds": panel_row["source_kinds"],
                "generated_doa": _doa_metrics(
                    generated,
                    scene_plan,
                    model_num_samples=samples,
                    latent_frames=frames,
                ),
                "reference_doa": _doa_metrics(
                    reference,
                    scene_plan,
                    model_num_samples=samples,
                    latent_frames=frames,
                ),
                "generated_reference_doa": _paired_doa_metrics(
                    generated,
                    reference,
                    scene_plan,
                    model_num_samples=samples,
                    latent_frames=frames,
                ),
                "generated_activity": _activity_metrics(
                    generated,
                    scene_plan,
                    model_num_samples=samples,
                    latent_frames=frames,
                ),
                "reference_activity": _activity_metrics(
                    reference,
                    scene_plan,
                    model_num_samples=samples,
                    latent_frames=frames,
                ),
            }
        )
        if index % 50 == 0 or index == len(panel):
            print(
                json.dumps(
                    {"event": "internal_spatial", "completed": index, "total": len(panel)}
                ),
                flush=True,
            )

    domains: dict[str, Any] = {}
    for domain in ("music", "sound", "speech"):
        domain_rows = [row for row in scored if domain in row["source_kinds"]]
        strata = {"all": _aggregate(domain_rows)}
        for source_count in range(1, 5):
            chosen = [row for row in domain_rows if row["source_count"] == source_count]
            strata[f"source_{source_count}"] = _aggregate(chosen)
        domains[domain] = {"strata": strata}

    report = {
        "schema": "sceneplan_foa.p10_final_internal_8k_spatial_metrics",
        "schema_version": 1,
        "status": "PASS",
        "system_id": "ours_p10_150k",
        "rows": len(scored),
        "contract": str(contract_path),
        "contract_sha256": _sha256(contract_path),
        "panel": str(panel_path),
        "panel_sha256": _sha256(panel_path),
        "domains": domains,
        "protocol": {
            "foa": "native WYZX/ACN/SN3D only",
            "paired_doa": "energy-weighted spherical error between generated and rendered-reference FOA intensity trajectories",
            "plan_doa": "energy-weighted plan direction error on exactly-one-active-source frames",
            "activity": "scene-level energy activity IoU against the union of planned source intervals",
            "public_baselines": "N/A because ordinary public systems are mono/stereo",
        },
    }
    metric_root = root / "cross_system_baselines_final_8k/metrics/final_spatial"
    _atomic_jsonl(metric_root / "PER_OUTPUT.jsonl", scored)
    _atomic_json(metric_root / "SPATIAL_METRICS.json", report)
    (metric_root / "SPATIAL_METRICS_COMPLETE").write_text("PASS\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "metrics": str(metric_root / "SPATIAL_METRICS.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
