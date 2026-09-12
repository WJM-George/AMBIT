#!/usr/bin/env python3
"""Build and checksum the six final P10 paper-facing benchmark tables."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any


DEFAULT_INTERNAL = Path(
    os.environ.get("AMBIT_DATA_ROOT", "data") + "/sceneplan_v2_1p124m/revisions/"
    "speech_expansion_noalign_15s_v1/evaluation/"
    "p10_v11_150k_full_test_8000_semantic_v2/cross_system_baselines_final_8k"
)
DEFAULT_OOD = Path(
    os.environ.get("AMBIT_DATA_ROOT", "data") + "/evaluation_benchmark/p10_ood_3000_v1/"
    "cross_system_benchmark"
)
DEFAULT_OUTPUT = Path(
    os.environ.get("AMBIT_DATA_ROOT", "data") + "/evaluation_benchmark/p10_final_paper_six_tables_v1"
)

ORDER = (
    "ground_truth",
    "ours_p10_150k",
    "audiox_maf_mmdit",
    "audiox_maf",
    "audiox_turbo",
    "stable_audio_open_1_0",
    "tangoflux",
    "mmaudio_large_44k_v2_text_only",
    "woosh_flow",
    "qwen3_tts_1p7b_voice_design",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(text.rstrip() + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def _fmt(value: float | None, digits: int = 3) -> str:
    if value is None or not math.isfinite(float(value)):
        return "N/A"
    return f"{float(value):.{digits}f}"


def _content_cell(domain: str, value: dict[str, Any] | None) -> str:
    if value is None:
        return "N/A"
    if domain in {"music", "sound"}:
        return "N={}; {}/{}/{}".format(
            int(value["rows"]),
            _fmt(value["clap"]["mean"]),
            _fmt(value["fad_vggish"]),
            _fmt(value["kl_pann"]["mean"]),
        )
    return "N={}; {}/{}/{}".format(
        int(value["rows"]),
        _fmt(value["corpus_wer"]),
        _fmt(value["corpus_cer"]),
        _fmt(value["utmos"]["mean"]),
    )


def _spatial_cell(
    spatial: dict[str, Any], domain: str, system_id: str, stratum: str
) -> str:
    row = spatial["domains"][domain]["strata"][stratum]
    if not row["rows"]:
        return "N/A"
    if system_id == "ground_truth":
        doa = 0.0
        iou = row["reference_activity_iou"]["mean"]
    elif system_id == "ours_p10_150k":
        doa = row["generated_reference_doa_error_deg"]["mean"]
        iou = row["generated_activity_iou"]["mean"]
    else:
        return "N/A"
    return f"{_fmt(doa, 2)}/{_fmt(iou)}"


def _domain_map(metrics: dict[str, Any], domain: str) -> dict[str, Any]:
    return metrics["audio_domains"][domain] if domain in {"music", "sound"} else metrics["speech"]


def _expected_systems(domain: str) -> set[str]:
    general = {
        "ground_truth",
        "ours_p10_150k",
        "audiox_maf_mmdit",
        "audiox_maf",
        "audiox_turbo",
        "stable_audio_open_1_0",
        "tangoflux",
    }
    if domain == "sound":
        return general | {"mmaudio_large_44k_v2_text_only", "woosh_flow"}
    if domain == "speech":
        return general | {"qwen3_tts_1p7b_voice_design"}
    return general


def _validate_metrics(metrics: dict[str, Any], kind: str) -> None:
    if metrics.get("status") != "PASS" or metrics.get("benchmark_kind") != kind:
        raise RuntimeError(f"invalid {kind} content metric marker")
    for domain in ("music", "sound", "speech"):
        systems = set(_domain_map(metrics, domain))
        if systems != _expected_systems(domain):
            raise RuntimeError(
                f"{kind}/{domain} system set changed: {sorted(systems)}"
            )
        for system_id, value in _domain_map(metrics, domain).items():
            row = value["strata"]["all"]
            if row is None or int(row["rows"]) <= 0:
                raise RuntimeError(f"missing all-stratum result: {kind}/{domain}/{system_id}")


def _internal_table(
    domain: str,
    metrics: dict[str, Any],
    spatial: dict[str, Any],
) -> str:
    content_legend = (
        "CLAP ↑ / FAD-VGGish ↓ / KL-PANN ↓"
        if domain in {"music", "sound"}
        else "WER ↓ / CER ↓ / UTMOS ↑"
    )
    systems = _domain_map(metrics, domain)
    lines = [
        f"# Internal 8k — {domain.title()}-present scenes",
        "",
        f"Content cells: `N; {content_legend}`. Spatial cells: `generated↔GT DoA error (degrees) ↓ / activity IoU ↑`.",
        "",
        "| System | Input | All content | 1-src content | 2-src content | 3-src content | 4-src content | All spatial | 1-src spatial | 2-src spatial | 3-src spatial | 4-src spatial |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    strata = ("all", "source_1", "source_2", "source_3", "source_4")
    for system_id in ORDER:
        if system_id not in systems:
            continue
        value = systems[system_id]
        content = [_content_cell(domain, value["strata"].get(stratum)) for stratum in strata]
        spatial_cells = [
            _spatial_cell(spatial, domain, system_id, stratum) for stratum in strata
        ]
        lines.append(
            f"| {value['display_name']} | {value['conditioning']} | "
            + " | ".join(content + spatial_cells)
            + " |"
        )
    lines.extend(
        [
            "",
            "Ours uses the complete native ScenePlan; public baselines receive only raw source descriptions joined in ScenePlan order. This is a native-interface capability comparison, not equal conditioning bandwidth.",
            "",
            "Domain lanes are presence-based: a mixed Music+Sound+Speech scene appears in each applicable table, while each table is independently stratified by total scene source count.",
            "",
            "Public mono/stereo models have N/A spatial entries. Ground truth has zero self-DoA error; its activity IoU is the renderer/reference ceiling under the identical detector.",
            "",
            "The frozen panel contains 2,000 clips longer than the approximately 10.03 s native window of the released AudioX-family adapters (and other fixed-window systems). Their native output is right-padded to the requested reference duration, so the all-row score intentionally includes unsupported-duration behavior; no separate 10/15 s headline stratum is introduced.",
            "",
        ]
    )
    return "\n".join(lines)


def _ood_table(domain: str, metrics: dict[str, Any]) -> str:
    systems = _domain_map(metrics, domain)
    if domain in {"music", "sound"}:
        columns = "| System | Input | N | CLAP ↑ | FAD-VGGish ↓ | KL-PANN ↓ |"
        separator = "|---|---|---:|---:|---:|---:|"
    else:
        columns = "| System | Input | N | WER ↓ | CER ↓ | UTMOS ↑ |"
        separator = "|---|---|---:|---:|---:|---:|"
    lines = [f"# OOD 3k — {domain.title()}", "", columns, separator]
    for system_id in ORDER:
        if system_id not in systems:
            continue
        value = systems[system_id]
        row = value["strata"]["all"]
        if domain in {"music", "sound"}:
            fields = (
                row["rows"],
                _fmt(row["clap"]["mean"]),
                _fmt(row["fad_vggish"]),
                _fmt(row["kl_pann"]["mean"]),
            )
        else:
            fields = (
                row["rows"],
                _fmt(row["corpus_wer"]),
                _fmt(row["corpus_cer"]),
                _fmt(row["utmos"]["mean"]),
            )
        lines.append(
            f"| {value['display_name']} | {value['conditioning']} | "
            + " | ".join(str(field) for field in fields)
            + " |"
        )
    lines.extend(
        [
            "",
            "OOD v1 contains 1,000 acoustically deduplicated single-source examples per domain. Natural references are content recordings without ground-truth FOA trajectories, so no OOD spatial metric is claimed; multichannel references use an arithmetic-mean content downmix.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--internal-root", type=Path, default=DEFAULT_INTERNAL)
    parser.add_argument("--ood-root", type=Path, default=DEFAULT_OOD)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    internal_root = args.internal_root.expanduser().resolve(strict=True)
    ood_root = args.ood_root.expanduser().resolve(strict=True)
    output_root = args.output_root.expanduser().resolve()
    (internal_root / "FINAL_CONTENT_METRICS_COMPLETE").resolve(strict=True)
    (ood_root / "FINAL_CONTENT_METRICS_COMPLETE").resolve(strict=True)
    internal_metrics_path = (
        internal_root / "metrics/final_content/CONTENT_METRICS.json"
    ).resolve(strict=True)
    ood_metrics_path = (
        ood_root / "metrics/final_content/CONTENT_METRICS.json"
    ).resolve(strict=True)
    spatial_path = (
        internal_root / "metrics/final_spatial/SPATIAL_METRICS.json"
    ).resolve(strict=True)
    (spatial_path.parent / "SPATIAL_METRICS_COMPLETE").resolve(strict=True)
    internal = json.loads(internal_metrics_path.read_text(encoding="utf-8"))
    ood = json.loads(ood_metrics_path.read_text(encoding="utf-8"))
    spatial = json.loads(spatial_path.read_text(encoding="utf-8"))
    internal_contract_path = (internal_root / "BENCHMARK_CONTRACT.json").resolve(
        strict=True
    )
    ood_contract_path = (ood_root / "BENCHMARK_CONTRACT.json").resolve(strict=True)
    internal_contract = json.loads(
        internal_contract_path.read_text(encoding="utf-8")
    )
    ood_contract = json.loads(ood_contract_path.read_text(encoding="utf-8"))
    _validate_metrics(internal, "internal8k")
    _validate_metrics(ood, "ood3k")
    if spatial.get("status") != "PASS" or int(spatial.get("rows", -1)) != 8000:
        raise RuntimeError("internal spatial metrics are incomplete")

    output_root.mkdir(parents=True, exist_ok=True)
    table_paths = []
    for domain in ("music", "sound", "speech"):
        path = output_root / f"INTERNAL_{domain.upper()}_TABLE.md"
        _atomic_text(path, _internal_table(domain, internal, spatial))
        table_paths.append(path)
    for domain in ("music", "sound", "speech"):
        path = output_root / f"OOD_{domain.upper()}_TABLE.md"
        _atomic_text(path, _ood_table(domain, ood))
        table_paths.append(path)

    readme_path = output_root / "README.md"
    _atomic_text(
        readme_path,
        "\n".join(
            [
                "# Final P10 evaluation — internal 8k and OOD 3k",
                "",
                "This directory is generated only after every contracted inference output and every metric arm passes its audit.",
                "",
                "- Internal tables are presence-based Music, Sound, and Speech lanes, each stratified by total scene source count 1–4.",
                "- OOD tables contain 1,000 acoustically deduplicated natural references per domain.",
                "- Ours uses its native full ScenePlan. Public systems use raw source descriptions only; the tables therefore report native-interface capability, not equal conditioning bandwidth.",
                "- Content uses FOA W for internal GT/ours; public and OOD content recordings use an arithmetic-mean downmix across all available channels.",
                "- Ordinary mono/stereo systems never receive FOA spatial scores.",
                "- Internal fixed-window systems are deterministically right-padded on the 2,000 clips longer than approximately 10.03 s; this duration limitation remains part of their all-row capability result.",
                "",
                "Metric cells are LAION-CLAP / FAD-VGGish / KL-PANN for Music and Sound, and WER / CER / UTMOS for Speech. Internal spatial cells are generated-to-reference spherical DoA error / activity IoU.",
                "",
                "The exact manifests, panel hashes, metric reports, and all table checksums are recorded in `FINAL_RESULTS.json` and `ARTIFACT_MANIFEST.json`.",
                "",
            ]
        ),
    )

    combined = {
        "schema": "sceneplan_foa.p10_final_six_tables",
        "schema_version": 1,
        "status": "PASS",
        "internal_contract": str(internal_contract_path),
        "internal_contract_sha256": _sha256(internal_contract_path),
        "internal_panel": internal_contract["source_panel_path"],
        "internal_panel_sha256": internal_contract["source_panel_sha256"],
        "ood_contract": str(ood_contract_path),
        "ood_contract_sha256": _sha256(ood_contract_path),
        "ood_panel": ood_contract["source_panel_path"],
        "ood_panel_sha256": ood_contract["source_panel_sha256"],
        "internal_metrics": str(internal_metrics_path),
        "internal_metrics_sha256": _sha256(internal_metrics_path),
        "internal_spatial_metrics": str(spatial_path),
        "internal_spatial_metrics_sha256": _sha256(spatial_path),
        "ood_metrics": str(ood_metrics_path),
        "ood_metrics_sha256": _sha256(ood_metrics_path),
        "tables": [str(path) for path in table_paths],
    }
    combined_path = output_root / "FINAL_RESULTS.json"
    _atomic_json(combined_path, combined)
    artifacts = table_paths + [readme_path, combined_path]
    manifest = {
        "schema": "sceneplan_foa.p10_final_six_table_artifacts",
        "schema_version": 1,
        "status": "PASS",
        "artifacts": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in artifacts
        ],
    }
    _atomic_json(output_root / "ARTIFACT_MANIFEST.json", manifest)
    (output_root / "FINAL_SIX_TABLES_COMPLETE").write_text("PASS\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
