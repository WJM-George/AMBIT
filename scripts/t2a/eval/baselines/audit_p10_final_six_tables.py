#!/usr/bin/env python3
"""Independently audit the final P10 six-table evaluation artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path(
    "/mnt/sdb/audio_dataset/evaluation_benchmark/p10_final_paper_six_tables_v1"
)
EXPECTED_TABLES = {
    "INTERNAL_MUSIC_TABLE.md",
    "INTERNAL_SOUND_TABLE.md",
    "INTERNAL_SPEECH_TABLE.md",
    "OOD_MUSIC_TABLE.md",
    "OOD_SOUND_TABLE.md",
    "OOD_SPEECH_TABLE.md",
}
GENERAL = {
    "ground_truth",
    "ours_p10_150k",
    "audiox_maf_mmdit",
    "audiox_maf",
    "audiox_turbo",
    "stable_audio_open_1_0",
    "tangoflux",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _assert_finite(value: Any, path: str = "root") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise RuntimeError(f"non-finite number at {path}: {value}")
    if isinstance(value, dict):
        for key, child in value.items():
            _assert_finite(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_finite(child, f"{path}[{index}]")


def _domain_map(metrics: dict[str, Any], domain: str) -> dict[str, Any]:
    return metrics["audio_domains"][domain] if domain != "speech" else metrics["speech"]


def _expected_systems(domain: str) -> set[str]:
    if domain == "sound":
        return GENERAL | {"mmaudio_large_44k_v2_text_only", "woosh_flow"}
    if domain == "speech":
        return GENERAL | {"qwen3_tts_1p7b_voice_design"}
    return GENERAL


def _check_metrics(path: Path, kind: str) -> dict[str, Any]:
    metrics = json.loads(path.read_text(encoding="utf-8"))
    if metrics.get("status") != "PASS" or metrics.get("benchmark_kind") != kind:
        raise RuntimeError(f"invalid {kind} content metrics: {path}")
    _assert_finite(metrics, kind)
    for domain in ("music", "sound", "speech"):
        systems = _domain_map(metrics, domain)
        if set(systems) != _expected_systems(domain):
            raise RuntimeError(f"{kind}/{domain} system set changed")
        for system_id, result in systems.items():
            all_rows = result["strata"]["all"]
            if all_rows is None or int(all_rows["rows"]) <= 0:
                raise RuntimeError(f"empty metric lane: {kind}/{domain}/{system_id}")
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    root = args.output_root.expanduser().resolve(strict=True)
    (root / "FINAL_SIX_TABLES_COMPLETE").resolve(strict=True)
    manifest_path = (root / "ARTIFACT_MANIFEST.json").resolve(strict=True)
    results_path = (root / "FINAL_RESULTS.json").resolve(strict=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    results = json.loads(results_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "PASS" or results.get("status") != "PASS":
        raise RuntimeError("final artifact status is not PASS")
    _assert_finite(manifest, "manifest")
    _assert_finite(results, "results")

    artifact_paths = []
    for row in manifest["artifacts"]:
        path = Path(row["path"]).resolve(strict=True)
        if path.parent != root:
            raise RuntimeError(f"artifact escaped final root: {path}")
        if path.stat().st_size != int(row["bytes"]):
            raise RuntimeError(f"artifact byte count changed: {path}")
        if _sha256(path) != row["sha256"]:
            raise RuntimeError(f"artifact SHA256 changed: {path}")
        artifact_paths.append(path)
    if len(artifact_paths) != len(set(artifact_paths)):
        raise RuntimeError("duplicate final artifact path")
    observed_tables = {path.name for path in artifact_paths if path.suffix == ".md"}
    if observed_tables != EXPECTED_TABLES | {"README.md"}:
        raise RuntimeError(f"final markdown set changed: {sorted(observed_tables)}")

    for key in (
        "internal_contract",
        "ood_contract",
        "internal_metrics",
        "internal_spatial_metrics",
        "ood_metrics",
    ):
        path = Path(results[key]).resolve(strict=True)
        if _sha256(path) != results[f"{key}_sha256"]:
            raise RuntimeError(f"recorded lineage SHA256 changed: {key}")
    for prefix in ("internal", "ood"):
        panel = Path(results[f"{prefix}_panel"]).resolve(strict=True)
        if _sha256(panel) != results[f"{prefix}_panel_sha256"]:
            raise RuntimeError(f"{prefix} panel SHA256 changed")

    internal = _check_metrics(Path(results["internal_metrics"]), "internal8k")
    ood = _check_metrics(Path(results["ood_metrics"]), "ood3k")
    expected_internal = {"music": 4013, "sound": 4013, "speech": 5000}
    for domain, rows in expected_internal.items():
        for system_id in GENERAL:
            observed = int(_domain_map(internal, domain)[system_id]["strata"]["all"]["rows"])
            if observed != rows:
                raise RuntimeError(
                    f"internal row count changed: {domain}/{system_id}={observed}"
                )
    for system_id in ("mmaudio_large_44k_v2_text_only", "woosh_flow"):
        observed = int(
            internal["audio_domains"]["sound"][system_id]["strata"]["all"][
                "rows"
            ]
        )
        if observed != 4013:
            raise RuntimeError(f"internal Sound-specialist count changed: {system_id}")
    qwen_rows = int(
        internal["speech"]["qwen3_tts_1p7b_voice_design"]["strata"]["all"][
            "rows"
        ]
    )
    if qwen_rows != 1250:
        raise RuntimeError("internal Qwen3-TTS eligibility count changed")
    for domain in ("music", "sound", "speech"):
        for system_id, result in _domain_map(ood, domain).items():
            if int(result["strata"]["all"]["rows"]) != 1000:
                raise RuntimeError(f"OOD row count changed: {domain}/{system_id}")

    spatial = json.loads(
        Path(results["internal_spatial_metrics"]).read_text(encoding="utf-8")
    )
    _assert_finite(spatial, "spatial")
    if spatial.get("status") != "PASS" or int(spatial.get("rows", -1)) != 8000:
        raise RuntimeError("internal spatial result is incomplete")

    for table_name in EXPECTED_TABLES:
        text = (root / table_name).read_text(encoding="utf-8")
        lowered = text.lower()
        if "| system |" not in lowered or "sceneplan dit (ours, 150k)" not in lowered:
            raise RuntimeError(f"malformed table: {table_name}")
        if " nan" in lowered or " none" in lowered:
            raise RuntimeError(f"invalid literal in table: {table_name}")

    report = {
        "schema": "sceneplan_foa.p10_final_evaluation_audit",
        "schema_version": 1,
        "status": "PASS",
        "output_root": str(root),
        "artifact_manifest": str(manifest_path),
        "artifact_manifest_sha256": _sha256(manifest_path),
        "final_results": str(results_path),
        "final_results_sha256": _sha256(results_path),
        "tables": sorted(EXPECTED_TABLES),
        "internal_rows": 8000,
        "ood_rows": 3000,
    }
    audit_path = root / "FINAL_AUDIT.json"
    _atomic_json(audit_path, report)
    (root / "FINAL_EVALUATION_COMPLETE").write_text("PASS\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
