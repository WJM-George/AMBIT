#!/usr/bin/env python3
"""Select a representative 400/400/400 single-source P10 test panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REVISION_ROOT = Path(
    os.environ.get("AMBIT_DATA_ROOT", "data") + "/sceneplan_v2_1p124m/"
    "revisions/speech_expansion_noalign_15s_v1"
)
DEFAULT_SOURCE = REVISION_ROOT / (
    "evaluation/p10_v9_full_test_8000_ckpt20k_100k_v1"
)
DEFAULT_OUTPUT = REVISION_ROOT / (
    "evaluation/p10_v9_balanced_1200_ckpt20k_100k_v1"
)
SELECTION_NAMESPACE = "sceneplan-p10-revision6-balanced-400x3-v1-20260830"
DOMAINS = ("music", "sound", "speech")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
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


def stable_rank(domain: str, sample_id: str) -> str:
    return hashlib.sha256(
        f"{SELECTION_NAMESPACE}\0{domain}\0{sample_id}".encode("utf-8")
    ).hexdigest()


def stratum(row: dict[str, Any]) -> tuple[Any, ...]:
    source = row["scene_plan"]["sources"][0]
    base: tuple[Any, ...] = (
        str(row["room_type"]),
        str(source["trajectory"]["type"]),
        int(row["length_bucket"]),
    )
    if source["kind"] == "speech":
        return (*base, bool(row["speech_seen_speaker"]))
    return base


def proportional_sample(
    candidates: list[dict[str, Any]], domain: str, target: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        groups[stratum(row)].append(row)
    for values in groups.values():
        values.sort(key=lambda row: stable_rank(domain, row["sample_id"]))

    quotas: dict[tuple[Any, ...], int] = {}
    remainders: list[tuple[float, str, tuple[Any, ...]]] = []
    for key, values in groups.items():
        exact = target * len(values) / len(candidates)
        quotas[key] = int(math.floor(exact))
        remainders.append((exact - quotas[key], repr(key), key))
    remaining = target - sum(quotas.values())
    for _fraction, _label, key in sorted(remainders, reverse=True)[:remaining]:
        quotas[key] += 1
    if sum(quotas.values()) != target:
        raise RuntimeError(f"{domain}: proportional quotas do not sum to {target}")
    if any(quotas[key] > len(groups[key]) for key in groups):
        raise RuntimeError(f"{domain}: proportional quota exceeds a stratum")

    selected = [row for key in sorted(groups, key=repr) for row in groups[key][: quotas[key]]]
    selected.sort(key=lambda row: stable_rank(domain, row["sample_id"]))
    if len(selected) != target or len({row["sample_id"] for row in selected}) != target:
        raise RuntimeError(f"{domain}: invalid selected panel")
    audit = {
        "candidate_rows": len(candidates),
        "selected_rows": len(selected),
        "candidate_strata": {repr(key): len(groups[key]) for key in sorted(groups, key=repr)},
        "selected_strata": {repr(key): quotas[key] for key in sorted(groups, key=repr)},
    }
    return selected, audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--rows-per-domain", type=int, default=400)
    args = parser.parse_args()
    source_root = args.source_root.expanduser().resolve(strict=True)
    output_root = args.output_root.expanduser().resolve()
    rows_per_domain = int(args.rows_per_domain)
    if rows_per_domain != 400:
        raise ValueError("the frozen balanced contract requires 400 rows per domain")

    source_contract_path = source_root / "EVAL_CONTRACT.json"
    source_contract = json.loads(source_contract_path.read_text(encoding="utf-8"))
    panel_path = source_root / source_contract["test_set"]["panel_filename"]
    if sha256_file(panel_path) != source_contract["test_set"]["panel_sha256"]:
        raise RuntimeError("complete test panel SHA256 changed")
    rows = [
        json.loads(line)
        for line in panel_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if len(rows) != 8_000:
        raise RuntimeError(f"complete test panel changed: {len(rows)}")

    panel: list[dict[str, Any]] = []
    selection_audit: dict[str, Any] = {}
    candidate_counts: dict[str, int] = {}
    for domain in DOMAINS:
        candidates = [
            row
            for row in rows
            if int(row["source_count"]) == 1
            and row["source_kind_counts"] == {domain: 1}
        ]
        candidate_counts[domain] = len(candidates)
        if len(candidates) < rows_per_domain:
            raise RuntimeError(
                f"not enough single-source {domain}: {len(candidates)} < {rows_per_domain}"
            )
        selected, audit = proportional_sample(candidates, domain, rows_per_domain)
        selection_audit[domain] = audit
        width = len(str(rows_per_domain))
        for index, source_row in enumerate(selected, start=1):
            row = dict(source_row)
            row["full_test_panel_id"] = source_row["panel_id"]
            row["panel_id"] = f"{domain}_{index:0{width}d}"
            row["domain"] = domain
            panel.append(row)

    counts = Counter(row["domain"] for row in panel)
    expected = {domain: rows_per_domain for domain in DOMAINS}
    if dict(counts) != expected or len(panel) != 1_200:
        raise RuntimeError(f"balanced panel count changed: {dict(counts)}")
    if len({row["sample_id"] for row in panel}) != len(panel):
        raise RuntimeError("balanced panel sample IDs are not unique")

    output_root.mkdir(parents=True, exist_ok=True)
    output_panel = output_root / "balanced_test_1200.jsonl"
    atomic_jsonl(output_panel, panel)
    contract = {
        "schema": "stable_audio_tools.sceneplan_dit_p10_balanced_test_contract",
        "schema_version": 1,
        "status": "FROZEN_BALANCED_SINGLE_SOURCE_400X3_V1",
        "purpose": "checkpoint and runnable-baseline comparison on 400 music, 400 sound, and 400 speech scenes",
        "source_full_test_contract": str(source_contract_path),
        "source_full_test_contract_sha256": sha256_file(source_contract_path),
        "selection_namespace": SELECTION_NAMESPACE,
        "selection": {
            "single_source_domain_pure": True,
            "representative_proportional_strata": [
                "room_type",
                "motion_type",
                "length_bucket",
                "speech_seen_speaker_when_applicable",
            ],
            "candidate_counts": candidate_counts,
            "audit": selection_audit,
        },
        "test_set": {
            **source_contract["test_set"],
            "all_rows": 8_000,
            "evaluation_rows": 1_200,
            "evaluation_subset": "representative single-source 400 music + 400 sound + 400 speech",
            "panel_filename": output_panel.name,
            "panel_sha256": sha256_file(output_panel),
            "domain_counts": expected,
            "single_source_domain_pure": True,
        },
        "leakage_audit": source_contract["leakage_audit"],
        "checkpoints": source_contract["checkpoints"],
        "sampling": source_contract["sampling"],
        "metric_protocol": {
            "music_sound_quality_channel": "native FOA W; mono baseline unchanged; stereo baseline arithmetic mean",
            "speech_quality_channel": "native FOA W cropped to planned speech activity for ASR/MOS",
            "spatial": "native FOA only; mono/stereo baselines are N/A",
            "checkpoint_noise": "same per-sample seed at every checkpoint",
            "baseline_prompts": "same semantic prompt and target duration for every compatible system",
        },
    }
    atomic_json(output_root / "EVAL_CONTRACT.json", contract)
    summary = {
        "status": "PASS",
        "output_root": str(output_root),
        "rows": len(panel),
        "domain_counts": dict(counts),
        "candidate_counts": candidate_counts,
        "selected_room_motion_length_seen_counts": {
            domain: dict(
                Counter(
                    repr(stratum(row))
                    for row in panel
                    if row["domain"] == domain
                )
            )
            for domain in DOMAINS
        },
        "speech_seen_speaker_rows": sum(
            row["domain"] == "speech" and row["speech_seen_speaker"] is True
            for row in panel
        ),
        "speech_unseen_speaker_rows": sum(
            row["domain"] == "speech" and row["speech_seen_speaker"] is False
            for row in panel
        ),
    }
    atomic_json(output_root / "BUILD_SUMMARY.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
