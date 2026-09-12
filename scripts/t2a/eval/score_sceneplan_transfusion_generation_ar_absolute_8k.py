#!/usr/bin/env python3
"""Score Generation-AR P10 renders against real GT FOA on the frozen 8K.

This is an adapter around the already frozen cross-system content protocol. It
adds one native-FOA system without modifying the historical benchmark:

* ``generation_ar_p10``: free Generation-AR ScenePlan -> P10-v11.

The historical ``ours_p10_150k`` lane (original high-precision GT ScenePlan)
and public baseline results are joined only after the new lanes are scored.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
BASELINE_SCRIPT_DIR = REPO_ROOT / "scripts/t2a/eval/baselines"
if str(BASELINE_SCRIPT_DIR) not in sys.path:
    # One historical baseline dependency still uses a script-local import.
    # Preserve its direct-script import environment when reused as a module.
    sys.path.insert(1, str(BASELINE_SCRIPT_DIR))

from scripts.t2a.eval.baselines import score_p10_final_content_benchmark as content
from scripts.t2a.eval.sceneplan_dit_p10_panel_common import (
    load_foa,
    read_jsonl,
    sha256_file,
    summarize,
)
from scripts.t2a.eval.score_sceneplan_dit_p10_core import (
    _activity_metrics,
    _doa_metrics,
    _paired_doa_metrics,
)


SCHEMA = "stable_audio_tools.sceneplan_transfusion_generation_ar_absolute_gt_metrics"
CONTRACT = "p10v11_generation_ar_absolute_gt_8k_metrics_ar_only_v1"
EXPECTED_ROWS = 8_000
REFERENCE_ID = content.REFERENCE_ID
ORIGINAL_ORACLE_ID = content.OURS_ID
GENERATION_AR_ID = "generation_ar_p10"
NEW_SYSTEM_IDS = (GENERATION_AR_ID,)
DEFAULT_BASELINE_ROOT = Path(
    "/mnt/sdb/audio_dataset/sceneplan_v2_1p124m/revisions/"
    "speech_expansion_noalign_15s_v1/evaluation/"
    "p10_v11_150k_full_test_8000_semantic_v2"
)
BASELINE_BENCHMARK_DIR = "cross_system_baselines_final_8k"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument(
        "--baseline-root", type=Path, default=DEFAULT_BASELINE_ROOT
    )
    parser.add_argument(
        "--arm",
        choices=("clap", "vggish", "panns", "speech", "spatial", "merge", "compare"),
        required=True,
    )
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-speech-shards", type=int, default=3)
    return parser.parse_args()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_jsonl(path: Path, values: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(_canonical_bytes(value).decode("utf-8") + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _metric_source_identity() -> dict[str, dict[str, Any]]:
    paths = {
        "absolute_scorer": Path(__file__).resolve(strict=True),
        "content_scorer": Path(content.__file__).resolve(strict=True),
        "panel_common": (
            REPO_ROOT / "scripts/t2a/eval/sceneplan_dit_p10_panel_common.py"
        ).resolve(strict=True),
        "spatial_core": (
            REPO_ROOT / "scripts/t2a/eval/score_sceneplan_dit_p10_core.py"
        ).resolve(strict=True),
        "content_metric_helpers": (
            REPO_ROOT / "scripts/t2a/eval/baselines/score_p10_60k_cross_system.py"
        ).resolve(strict=True),
        "frechet_helper": (
            REPO_ROOT
            / "scripts/t2a/eval/baselines/score_p10_v11_stratified_3000_public_baselines.py"
        ).resolve(strict=True),
    }
    return {
        key: {
            "path": str(path),
            "bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
        }
        for key, path in sorted(paths.items())
    }


def _load_frozen_panel_inputs(
    baseline: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load only the frozen panel needed by the new native-FOA lane.

    The historical loader also opens 8K per-output P10 metadata files and all
    public generation requests.  Those artifacts are not inputs to rescoring
    the new lane; historical systems are joined from their already-frozen
    metric reports in ``_compare``.  Avoiding that scan makes every metric arm
    cheap to resume without weakening panel lineage.
    """

    benchmark_root = (baseline / BASELINE_BENCHMARK_DIR).resolve(strict=True)
    contract_path = (benchmark_root / "BENCHMARK_CONTRACT.json").resolve(
        strict=True
    )
    complete_path = (benchmark_root / "BASELINE_GENERATION_COMPLETE").resolve(
        strict=True
    )
    if not complete_path.read_text(encoding="utf-8").strip():
        raise RuntimeError("frozen baseline completion marker is empty")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if (
        contract.get("schema")
        != "sceneplan_foa.p10_final_8000_public_baseline_contract"
        or contract.get("status") != "FROZEN"
        or int(contract.get("source_panel_rows", -1)) != EXPECTED_ROWS
    ):
        raise RuntimeError("baseline root is not the frozen internal 8K contract")
    panel_path = Path(str(contract["source_panel_path"])).resolve(strict=True)
    if sha256_file(panel_path) != contract["source_panel_sha256"]:
        raise RuntimeError("frozen baseline panel SHA256 changed")
    panel = read_jsonl(panel_path)
    if len(panel) != EXPECTED_ROWS:
        raise RuntimeError("frozen baseline panel is not exactly 8K")

    panel_meta: dict[str, dict[str, Any]] = {}
    reference_paths: dict[str, str] = {}
    all_domain_ids = {domain: set() for domain in ("music", "sound", "speech")}
    for expected_ordinal, row in enumerate(panel):
        panel_id = str(row["panel_id"])
        if panel_id in panel_meta or int(row["ordinal"]) != expected_ordinal:
            raise RuntimeError("frozen baseline panel IDs/ordinals are not canonical")
        source_kinds = tuple(str(value) for value in row["source_kinds"])
        if not source_kinds or any(value not in all_domain_ids for value in source_kinds):
            raise RuntimeError(f"invalid source kinds in frozen panel: {panel_id}")
        speech = content._speech_source(row["scene_plan"])
        reference_path = str(Path(row["reference_foa_path"]).resolve(strict=True))
        panel_meta[panel_id] = {
            "panel_id": panel_id,
            "source_count": int(row["source_count"]),
            "source_kinds": source_kinds,
            "semantic_prompt": row["semantic_text"],
            "reference_path": reference_path,
            "reference_sha256": row["reference_foa_sha256"],
            "transcript": None if speech is None else speech["transcript"],
            "speech_seen_speaker": row.get("speech_seen_speaker"),
            "length_bucket": int(row["length_bucket"]),
        }
        reference_paths[panel_id] = reference_path
        for domain in set(source_kinds):
            all_domain_ids[domain].add(panel_id)

    expected_domains = {"music": 4013, "sound": 4013, "speech": 5000}
    actual_domains = {key: len(value) for key, value in all_domain_ids.items()}
    if actual_domains != expected_domains:
        raise RuntimeError(f"frozen baseline domain counts changed: {actual_domains}")
    base = {
        "benchmark_kind": "internal8k",
        "root": benchmark_root,
        "contract": contract,
        "contract_path": contract_path,
        # New metric partials must bind the rendered audio inventory, not the
        # unrelated public-baseline generation request manifest.
        "manifest_path": None,
        "panel_path": panel_path,
        "panel_meta": panel_meta,
        "reference_paths": reference_paths,
        "all_domain_ids": all_domain_ids,
    }
    return base, panel


def _load_inputs(evaluation_root: Path, baseline_root: Path) -> dict[str, Any]:
    root = evaluation_root.expanduser().resolve(strict=True)
    baseline = baseline_root.expanduser().resolve(strict=True)
    complete_path = (root / "GENERATION_COMPLETE").resolve(strict=True)
    if complete_path.read_text(encoding="utf-8") != "PASS\n":
        raise RuntimeError("absolute render completion marker changed")
    run_path = (root / "RUN_CONTRACT.json").resolve(strict=True)
    summary_path = (root / "SUMMARY.json").resolve(strict=True)
    manifest_path = (root / "OUTPUT_MANIFEST.jsonl").resolve(strict=True)
    run = json.loads(run_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        run.get("contract")
        != "p10v11_generation_ar_absolute_gt_8k_render_ar_only_v1"
        or int(run.get("rows", -1)) != EXPECTED_ROWS
        or summary.get("status") != "PASS"
        or int(summary.get("rows", -1)) != EXPECTED_ROWS
        or summary.get("run_contract_canonical_sha256") != _canonical_sha256(run)
        or summary.get("output_manifest_sha256") != sha256_file(manifest_path)
    ):
        raise RuntimeError("absolute render contract/summary is invalid")
    output_rows = read_jsonl(manifest_path)
    if len(output_rows) != EXPECTED_ROWS:
        raise RuntimeError("absolute render manifest is not exactly 8K")

    base, panel = _load_frozen_panel_inputs(baseline)
    base["manifest_path"] = manifest_path
    baseline_identity = run.get("baseline_panel", {})
    if not (
        Path(str(baseline_identity.get("root", ""))).resolve(strict=True) == baseline
        and Path(str(baseline_identity.get("benchmark_contract", ""))).resolve(
            strict=True
        )
        == base["contract_path"]
        and baseline_identity.get("benchmark_contract_sha256")
        == sha256_file(base["contract_path"])
        and Path(str(baseline_identity.get("panel", ""))).resolve(strict=True)
        == base["panel_path"]
        and baseline_identity.get("panel_sha256") == sha256_file(base["panel_path"])
    ):
        raise RuntimeError("rendered audio is not bound to this frozen baseline panel")
    output_by_id = {str(row["panel_id"]): row for row in output_rows}
    if len(output_by_id) != EXPECTED_ROWS:
        raise RuntimeError("absolute render panel IDs are not unique")
    for panel_row in panel:
        panel_id = str(panel_row["panel_id"])
        row = output_by_id.get(panel_id)
        if row is None or not (
            row["sample_id"] == panel_row["sample_id"]
            and int(row["ordinal"]) == int(panel_row["ordinal"])
            and row["reference_foa_path"]
            == str(Path(panel_row["reference_foa_path"]).resolve(strict=True))
            and row["reference_foa_sha256"] == panel_row["reference_foa_sha256"]
            and int(row["source_count"]) == int(panel_row["source_count"])
            and tuple(row["source_kinds"])
            == tuple(str(value) for value in panel_row["source_kinds"])
            and int(row["noise_seed"]) == int(panel_row["noise_seed"])
            and int(row["reference_samples"]) == int(panel_row["model_num_samples"])
            and int(row["reference_latent_frames"])
            == int(panel_row["latent_frames_valid"])
        ):
            raise RuntimeError(f"absolute render/panel mismatch: {panel_id}")
        Path(str(row["generation_ar_foa_path"])).resolve(strict=True)

    paths = {
        REFERENCE_ID: dict(base["reference_paths"]),
        GENERATION_AR_ID: {
            panel_id: str(row["generation_ar_foa_path"])
            for panel_id, row in output_by_id.items()
        },
    }
    all_domain_ids = base["all_domain_ids"]
    domain_system_ids = {
        "music": {
            system_id: set(all_domain_ids["music"])
            for system_id in (REFERENCE_ID, *NEW_SYSTEM_IDS)
        },
        "sound": {
            system_id: set(all_domain_ids["sound"])
            for system_id in (REFERENCE_ID, *NEW_SYSTEM_IDS)
        },
        # Ground-truth speech was already scored in the frozen benchmark.  The
        # new P10 lane is rescored here with the identical ASR/MOS stack.
        "speech": {
            system_id: set(all_domain_ids["speech"])
            for system_id in NEW_SYSTEM_IDS
        },
    }
    inputs = {
        **base,
        "paths": paths,
        "domain_system_ids": domain_system_ids,
        "display_names": {
            REFERENCE_ID: "Ground truth",
            GENERATION_AR_ID: "Generation AR ScenePlan + P10",
        },
        "conditioning": {
            REFERENCE_ID: "reference",
            GENERATION_AR_ID: "raw request -> Generation AR ScenePlan",
        },
        "native_foa_system_ids": {
            REFERENCE_ID,
            GENERATION_AR_ID,
        },
        "absolute_evaluation_root": str(root),
        "absolute_output_rows": output_by_id,
        "absolute_identity": {
            "render_run_contract": str(run_path),
            "render_run_contract_sha256": sha256_file(run_path),
            "render_run_contract_canonical_sha256": _canonical_sha256(run),
            "render_summary": str(summary_path),
            "render_summary_sha256": sha256_file(summary_path),
            "output_manifest": str(manifest_path),
            "output_manifest_sha256": sha256_file(manifest_path),
            "generation_ar_checkpoint": run["plan_evaluation"][
                "generation_ar_checkpoint"
            ],
            "generation_ar_checkpoint_sha256": run["plan_evaluation"][
                "generation_ar_checkpoint_sha256"
            ],
            "generation_ar_checkpoint_step": run["plan_evaluation"][
                "generation_ar_checkpoint_step"
            ],
            "metric_source": _metric_source_identity(),
        },
    }
    return inputs


def _add_identity(report: dict[str, Any], inputs: Mapping[str, Any]) -> dict[str, Any]:
    return {
        **report,
        "absolute_gt_contract": CONTRACT,
        "absolute_gt_identity": inputs["absolute_identity"],
    }


def _validate_partial_identity(report: Mapping[str, Any], inputs: Mapping[str, Any]) -> None:
    if (
        report.get("absolute_gt_contract") != CONTRACT
        or report.get("absolute_gt_identity") != inputs["absolute_identity"]
    ):
        raise RuntimeError("absolute metric partial identity mismatch")


def _spatial_aggregate(rows: list[Mapping[str, Any]], system_id: str) -> dict[str, Any]:
    def values(*keys: str):
        for row in rows:
            value: Any = row["systems"][system_id]
            for key in keys:
                value = value[key]
            yield value

    return {
        "rows": len(rows),
        "generated_reference_doa_error_deg": summarize(
            values("generated_reference_doa", "spherical_error_mean_deg")
        ),
        "generated_gt_plan_doa_error_deg": summarize(
            values("generated_gt_plan_doa", "spherical_error_mean_deg")
        ),
        "generated_activity_iou": summarize(
            values("generated_activity", "temporal_iou")
        ),
        "generated_activity_onset_error_sec": summarize(
            values("generated_activity", "onset_abs_error_sec")
        ),
        "generated_activity_offset_error_sec": summarize(
            values("generated_activity", "offset_abs_error_sec")
        ),
        "generated_valid_direction_fraction": summarize(
            values("generated_gt_plan_doa", "valid_direction_fraction")
        ),
    }


def _score_spatial(inputs: Mapping[str, Any], root: Path) -> dict[str, Any]:
    panel = read_jsonl(inputs["panel_path"])
    output_rows = inputs["absolute_output_rows"]
    scored: list[dict[str, Any]] = []
    for index, panel_row in enumerate(panel, start=1):
        panel_id = str(panel_row["panel_id"])
        rendered = output_rows[panel_id]
        samples = int(panel_row["model_num_samples"])
        frames = int(panel_row["latent_frames_valid"])
        reference, reference_rate = load_foa(
            panel_row["reference_foa_path"], expected_samples=samples
        )
        if reference_rate != 44_100:
            raise RuntimeError("real GT FOA sample rate changed")
        systems = {}
        for system_id, key in (
            (GENERATION_AR_ID, "generation_ar_foa_path"),
        ):
            generated, generated_rate = load_foa(
                rendered[key], expected_samples=samples
            )
            if generated_rate != 44_100:
                raise RuntimeError("generated FOA sample rate changed")
            systems[system_id] = {
                "generated_reference_doa": _paired_doa_metrics(
                    generated,
                    reference,
                    panel_row["scene_plan"],
                    model_num_samples=samples,
                    latent_frames=frames,
                ),
                "generated_gt_plan_doa": _doa_metrics(
                    generated,
                    panel_row["scene_plan"],
                    model_num_samples=samples,
                    latent_frames=frames,
                ),
                "generated_activity": _activity_metrics(
                    generated,
                    panel_row["scene_plan"],
                    model_num_samples=samples,
                    latent_frames=frames,
                ),
            }
        scored.append(
            {
                "ordinal": int(panel_row["ordinal"]),
                "panel_id": panel_id,
                "sample_id": str(panel_row["sample_id"]),
                "source_count": int(panel_row["source_count"]),
                "source_kinds": panel_row["source_kinds"],
                "systems": systems,
            }
        )
        if index % 50 == 0 or index == len(panel):
            print(
                json.dumps(
                    {"event": "absolute_spatial", "completed": index, "total": len(panel)}
                ),
                flush=True,
            )

    systems: dict[str, Any] = {}
    for system_id in NEW_SYSTEM_IDS:
        systems[system_id] = {}
        for domain in ("music", "sound", "speech"):
            domain_rows = [row for row in scored if domain in row["source_kinds"]]
            strata = {"all": _spatial_aggregate(domain_rows, system_id)}
            for source_count in range(1, 5):
                chosen = [
                    row
                    for row in domain_rows
                    if int(row["source_count"]) == source_count
                ]
                strata[f"source_{source_count}"] = _spatial_aggregate(
                    chosen, system_id
                )
            systems[system_id][domain] = {"strata": strata}
    report = _add_identity(
        {
            "schema": SCHEMA + "_spatial",
            "schema_version": 1,
            "status": "PASS",
            "rows": len(scored),
            "systems": systems,
            "protocol": {
                "foa": "native WYZX/ACN/SN3D only",
                "target": "the original high-precision GT ScenePlan and real GT FOA",
                "paired_doa": (
                    "energy-weighted spherical error between generated and real GT "
                    "FOA active-intensity trajectories"
                ),
                "duration": (
                    "native P10 output is tail-cropped/right-zero-padded to the real "
                    "GT sample count; predicted duration is never repaired"
                ),
            },
        },
        inputs,
    )
    metric_root = root / "metrics/absolute_gt"
    _atomic_jsonl(metric_root / "SPATIAL_PER_OUTPUT.jsonl", scored)
    _atomic_json(metric_root / "SPATIAL_METRICS.json", report)
    return report


def _mean(row: Mapping[str, Any], key: str) -> float | None:
    value = row.get(key)
    if isinstance(value, Mapping):
        value = value.get("mean")
    return None if value is None else float(value)


def _content_row(value: Mapping[str, Any]) -> dict[str, Any]:
    row = value["strata"]["all"]
    return {
        "rows": row["rows"],
        "clap_text_audio": _mean(row, "clap"),
        "paired_reference_clap": _mean(row, "paired_reference_clap"),
        "fad_vggish": row["fad_vggish"],
        "kl_pann": _mean(row, "kl_pann"),
    }


def _speech_row(value: Mapping[str, Any]) -> dict[str, Any]:
    row = value["strata"]["all"]
    return {
        "rows": row["rows"],
        "corpus_wer": row["corpus_wer"],
        "corpus_cer": row["corpus_cer"],
        "utmos": _mean(row, "utmos"),
    }


def _spatial_row(value: Mapping[str, Any]) -> dict[str, Any]:
    row = value["strata"]["all"]
    return {
        "rows": row["rows"],
        "generated_reference_doa_error_deg": _mean(
            row, "generated_reference_doa_error_deg"
        ),
        "generated_gt_plan_doa_error_deg": _mean(
            row, "generated_gt_plan_doa_error_deg"
        ),
        "activity_iou": _mean(row, "generated_activity_iou"),
        "activity_onset_error_sec": _mean(
            row, "generated_activity_onset_error_sec"
        ),
        "activity_offset_error_sec": _mean(
            row, "generated_activity_offset_error_sec"
        ),
    }


def _fmt(value: Any, digits: int = 3) -> str:
    return "N/A" if value is None else f"{float(value):.{digits}f}"


def _safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or abs(float(denominator)) < 1.0e-12:
        return None
    return float(numerator) / float(denominator)


def _system_gap(
    candidate: Mapping[str, Any], oracle: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "music": {
            "clap_delta": candidate["music"]["clap_text_audio"]
            - oracle["music"]["clap_text_audio"],
            "clap_retention": _safe_ratio(
                candidate["music"]["clap_text_audio"],
                oracle["music"]["clap_text_audio"],
            ),
            "paired_reference_clap_delta": candidate["music"][
                "paired_reference_clap"
            ]
            - oracle["music"]["paired_reference_clap"],
            "paired_reference_clap_retention": _safe_ratio(
                candidate["music"]["paired_reference_clap"],
                oracle["music"]["paired_reference_clap"],
            ),
            "fad_excess": candidate["music"]["fad_vggish"]
            - oracle["music"]["fad_vggish"],
            "kl_pann_excess": candidate["music"]["kl_pann"]
            - oracle["music"]["kl_pann"],
            "doa_error_excess_deg": candidate["music"][
                "generated_reference_doa_error_deg"
            ]
            - oracle["music"]["generated_reference_doa_error_deg"],
            "activity_iou_delta": candidate["music"]["activity_iou"]
            - oracle["music"]["activity_iou"],
        },
        "sound": {
            "clap_delta": candidate["sound"]["clap_text_audio"]
            - oracle["sound"]["clap_text_audio"],
            "clap_retention": _safe_ratio(
                candidate["sound"]["clap_text_audio"],
                oracle["sound"]["clap_text_audio"],
            ),
            "paired_reference_clap_delta": candidate["sound"][
                "paired_reference_clap"
            ]
            - oracle["sound"]["paired_reference_clap"],
            "paired_reference_clap_retention": _safe_ratio(
                candidate["sound"]["paired_reference_clap"],
                oracle["sound"]["paired_reference_clap"],
            ),
            "fad_excess": candidate["sound"]["fad_vggish"]
            - oracle["sound"]["fad_vggish"],
            "kl_pann_excess": candidate["sound"]["kl_pann"]
            - oracle["sound"]["kl_pann"],
            "doa_error_excess_deg": candidate["sound"][
                "generated_reference_doa_error_deg"
            ]
            - oracle["sound"]["generated_reference_doa_error_deg"],
            "activity_iou_delta": candidate["sound"]["activity_iou"]
            - oracle["sound"]["activity_iou"],
        },
        "speech": {
            "wer_excess": candidate["speech"]["corpus_wer"]
            - oracle["speech"]["corpus_wer"],
            "cer_excess": candidate["speech"]["corpus_cer"]
            - oracle["speech"]["corpus_cer"],
            "utmos_delta": candidate["speech"]["utmos"]
            - oracle["speech"]["utmos"],
            "doa_error_excess_deg": candidate["speech"][
                "generated_reference_doa_error_deg"
            ]
            - oracle["speech"]["generated_reference_doa_error_deg"],
            "activity_iou_delta": candidate["speech"]["activity_iou"]
            - oracle["speech"]["activity_iou"],
        },
    }


def _compare(inputs: Mapping[str, Any], root: Path, baseline_root: Path) -> dict[str, Any]:
    new_content_path = (root / "metrics/final_content/CONTENT_METRICS.json").resolve(
        strict=True
    )
    new_spatial_path = (root / "metrics/absolute_gt/SPATIAL_METRICS.json").resolve(
        strict=True
    )
    old_content_path = (
        baseline_root
        / "cross_system_baselines_final_8k/metrics/final_content/CONTENT_METRICS.json"
    ).resolve(strict=True)
    old_spatial_path = (
        baseline_root
        / "cross_system_baselines_final_8k/metrics/final_spatial/SPATIAL_METRICS.json"
    ).resolve(strict=True)
    new_content = json.loads(new_content_path.read_text(encoding="utf-8"))
    new_spatial = json.loads(new_spatial_path.read_text(encoding="utf-8"))
    old_content = json.loads(old_content_path.read_text(encoding="utf-8"))
    old_spatial = json.loads(old_spatial_path.read_text(encoding="utf-8"))
    _validate_partial_identity(new_spatial, inputs)
    _validate_partial_identity(new_content, inputs)
    if (
        new_content.get("status") != "PASS"
        or new_spatial.get("status") != "PASS"
        or int(new_spatial.get("rows", -1)) != EXPECTED_ROWS
    ):
        raise RuntimeError("new absolute metric reports are incomplete")
    expected_panel_sha = sha256_file(inputs["panel_path"])
    if (
        old_content.get("status") != "PASS"
        or old_spatial.get("status") != "PASS"
        or old_content.get("panel_sha256") != expected_panel_sha
        or old_spatial.get("panel_sha256") != expected_panel_sha
        or int(old_spatial.get("rows", -1)) != EXPECTED_ROWS
    ):
        raise RuntimeError("historical oracle/baseline metric lineage is invalid")

    systems: dict[str, Any] = {}
    for system_id in NEW_SYSTEM_IDS:
        systems[system_id] = {
            "display_name": inputs["display_names"][system_id],
            "music": {
                **_content_row(new_content["audio_domains"]["music"][system_id]),
                **_spatial_row(new_spatial["systems"][system_id]["music"]),
            },
            "sound": {
                **_content_row(new_content["audio_domains"]["sound"][system_id]),
                **_spatial_row(new_spatial["systems"][system_id]["sound"]),
            },
            "speech": {
                **_speech_row(new_content["speech"][system_id]),
                **_spatial_row(new_spatial["systems"][system_id]["speech"]),
            },
        }
    systems[ORIGINAL_ORACLE_ID] = {
        "display_name": old_content["audio_domains"]["music"][ORIGINAL_ORACLE_ID][
            "display_name"
        ],
        "music": {
            **_content_row(old_content["audio_domains"]["music"][ORIGINAL_ORACLE_ID]),
            **{
                "generated_reference_doa_error_deg": _mean(
                    old_spatial["domains"]["music"]["strata"]["all"],
                    "generated_reference_doa_error_deg",
                ),
                "activity_iou": _mean(
                    old_spatial["domains"]["music"]["strata"]["all"],
                    "generated_activity_iou",
                ),
            },
        },
        "sound": {
            **_content_row(old_content["audio_domains"]["sound"][ORIGINAL_ORACLE_ID]),
            **{
                "generated_reference_doa_error_deg": _mean(
                    old_spatial["domains"]["sound"]["strata"]["all"],
                    "generated_reference_doa_error_deg",
                ),
                "activity_iou": _mean(
                    old_spatial["domains"]["sound"]["strata"]["all"],
                    "generated_activity_iou",
                ),
            },
        },
        "speech": {
            **_speech_row(old_content["speech"][ORIGINAL_ORACLE_ID]),
            **{
                "generated_reference_doa_error_deg": _mean(
                    old_spatial["domains"]["speech"]["strata"]["all"],
                    "generated_reference_doa_error_deg",
                ),
                "activity_iou": _mean(
                    old_spatial["domains"]["speech"]["strata"]["all"],
                    "generated_activity_iou",
                ),
            },
        },
    }
    public = {
        domain: {
            system_id: {
                "display_name": value["display_name"],
                **(
                    _content_row(value)
                    if domain in ("music", "sound")
                    else _speech_row(value)
                ),
            }
            for system_id, value in (
                old_content["audio_domains"][domain].items()
                if domain in ("music", "sound")
                else old_content["speech"].items()
            )
            if system_id not in {REFERENCE_ID, ORIGINAL_ORACLE_ID}
        }
        for domain in ("music", "sound", "speech")
    }
    ar = systems[GENERATION_AR_ID]
    original_oracle = systems[ORIGINAL_ORACLE_ID]
    report = {
        "schema": SCHEMA + "_comparison",
        "schema_version": 1,
        "status": "PASS",
        "contract": CONTRACT,
        "absolute_gt_identity": inputs["absolute_identity"],
        "systems": systems,
        "public_baseline_content_metrics": public,
        "generation_ar_vs_original_gt_oracle": _system_gap(ar, original_oracle),
        "source_reports": {
            str(path): sha256_file(path)
            for path in (
                new_content_path,
                new_spatial_path,
                old_content_path,
                old_spatial_path,
            )
        },
        "interpretation": (
            "The new Generation-AR P10 lane and the historical native-P10 oracle "
            "use the same frozen 8K real-GT FOA panel and the same frozen P10-v11. "
            "The historical oracle uses the original high-precision GT ScenePlan "
            "and is reused from its frozen metrics. Public systems have content "
            "but no native-FOA spatial score."
        ),
    }
    report["report_sha256_without_self"] = _canonical_sha256(report)
    output = root / "metrics/absolute_gt/COMPARISON.json"
    _atomic_json(output, report)

    lines = [
        "# Generation AR -> P10 absolute GT benchmark",
        "",
        "All values compare generated audio with the same real GT FOA/audio panel.",
        "",
        "| System | Music CLAP ↑ | Music FAD ↓ | Music DoA ↓ | Sound CLAP ↑ | Sound FAD ↓ | Sound DoA ↓ | Speech WER ↓ | Speech UTMOS ↑ | Speech DoA ↓ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for system_id in (GENERATION_AR_ID, ORIGINAL_ORACLE_ID):
        value = systems[system_id]
        lines.append(
            "| {name} | {mc} | {mf} | {md} | {sc} | {sf} | {sd} | {wer} | {mos} | {spd} |".format(
                name=value["display_name"],
                mc=_fmt(value["music"]["clap_text_audio"]),
                mf=_fmt(value["music"]["fad_vggish"]),
                md=_fmt(value["music"]["generated_reference_doa_error_deg"], 2),
                sc=_fmt(value["sound"]["clap_text_audio"]),
                sf=_fmt(value["sound"]["fad_vggish"]),
                sd=_fmt(value["sound"]["generated_reference_doa_error_deg"], 2),
                wer=_fmt(value["speech"]["corpus_wer"]),
                mos=_fmt(value["speech"]["utmos"]),
                spd=_fmt(value["speech"]["generated_reference_doa_error_deg"], 2),
            )
        )
    for domain, title in (("music", "Music"), ("sound", "Sound")):
        lines.extend(
            [
                "",
                f"## {title}: public raw-text baselines",
                "",
                "| System | CLAP text-audio ↑ | Paired-reference CLAP ↑ | FAD-VGGish ↓ | KL-PANN ↓ |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for value in public[domain].values():
            lines.append(
                "| {name} | {clap} | {paired} | {fad} | {kl} |".format(
                    name=value["display_name"],
                    clap=_fmt(value["clap_text_audio"]),
                    paired=_fmt(value["paired_reference_clap"]),
                    fad=_fmt(value["fad_vggish"]),
                    kl=_fmt(value["kl_pann"]),
                )
            )
    lines.extend(
        [
            "",
            "## Speech: public raw-text baselines",
            "",
            "| System | WER ↓ | CER ↓ | UTMOS ↑ |",
            "|---|---:|---:|---:|",
        ]
    )
    for value in public["speech"].values():
        lines.append(
            "| {name} | {wer} | {cer} | {utmos} |".format(
                name=value["display_name"],
                wer=_fmt(value["corpus_wer"]),
                cer=_fmt(value["corpus_cer"]),
                utmos=_fmt(value["utmos"]),
            )
        )
    _atomic_text(root / "metrics/absolute_gt/COMPARISON.md", "\n".join(lines) + "\n")
    _atomic_text(root / "ABSOLUTE_GT_EVALUATION_COMPLETE", "PASS\n")
    return report


def main() -> int:
    args = _parse_args()
    root = args.evaluation_root.expanduser().resolve(strict=True)
    baseline_root = args.baseline_root.expanduser().resolve(strict=True)
    inputs = _load_inputs(root, baseline_root)
    partial = root / "metrics/final_content/partials"
    partial.mkdir(parents=True, exist_ok=True)
    if args.arm == "spatial":
        report = _score_spatial(inputs, root)
        print(json.dumps({"status": report["status"], "arm": "spatial"}))
        return 0
    if args.arm == "merge":
        for name in ("CLAP", "VGGISH", "PANNS"):
            _validate_partial_identity(
                json.loads((partial / f"{name}.json").read_text(encoding="utf-8")),
                inputs,
            )
        for shard in range(args.num_speech_shards):
            _validate_partial_identity(
                json.loads(
                    (
                        partial
                        / f"SPEECH_SHARD_{shard:02d}_OF_{args.num_speech_shards:02d}.json"
                    ).read_text(encoding="utf-8")
                ),
                inputs,
            )
        report = content._merge(inputs, root, args.num_speech_shards)
        report = _add_identity(report, inputs)
        _atomic_json(root / "metrics/final_content/CONTENT_METRICS.json", report)
        print(json.dumps({"status": report["status"], "arm": "merge"}))
        return 0
    if args.arm == "compare":
        report = _compare(inputs, root, baseline_root)
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "comparison": str(root / "metrics/absolute_gt/COMPARISON.json"),
                }
            )
        )
        return 0
    if not torch.cuda.is_available():
        raise RuntimeError("absolute content metrics require CUDA")
    device = torch.device(f"cuda:{args.device_index}")
    if args.arm == "clap":
        report = content._score_clap(inputs, device)
        output = partial / "CLAP.json"
    elif args.arm == "vggish":
        report = content._score_vggish(inputs, device)
        output = partial / "VGGISH.json"
    elif args.arm == "panns":
        report = content._score_panns(inputs, device)
        output = partial / "PANNS.json"
    else:
        report = content._score_speech_shard(
            inputs,
            device_index=args.device_index,
            shard_index=args.shard_index,
            num_shards=args.num_speech_shards,
        )
        output = (
            partial
            / f"SPEECH_SHARD_{args.shard_index:02d}_OF_{args.num_speech_shards:02d}.json"
        )
    report = _add_identity(report, inputs)
    _atomic_json(output, report)
    print(json.dumps({"status": "PASS", "arm": args.arm, "output": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
