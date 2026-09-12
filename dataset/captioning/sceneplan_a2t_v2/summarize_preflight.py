#!/usr/bin/env python3
"""Freeze the 1k throughput/QC and revised-100 ScenePlan preflight report."""

from __future__ import annotations

from collections import Counter
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

from description_contract import validate_source_description


DATASET_ROOT = Path(os.environ.get("AMBIT_DATA_ROOT", "data") + "/sceneplan_v2_1p124m")
SOURCE_ROOT = DATASET_ROOT / "source_annotations/nonspeech_instruct_v2"
STRESS_ROOT = DATASET_ROOT / "audit/a2t_throughput_1k_20260816"
REVISED_ROOT = DATASET_ROOT / "pilots/revised_sceneplan_100_registry_v1"
EXPECTED_FULL_ROWS = 873_502
ONE_DAY_SECONDS = 86_400


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile(values: list[int], value: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), value))


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> int:
    source_summary_path = SOURCE_ROOT / "summary.json"
    source_summary = json.loads(source_summary_path.read_text(encoding="utf-8"))
    finalizer_path = STRESS_ROOT / "finalizer/finalizer_audit.json"
    finalizer = json.loads(finalizer_path.read_text(encoding="utf-8"))
    revised_summary_path = REVISED_ROOT / "summary.json"
    revised = json.loads(revised_summary_path.read_text(encoding="utf-8"))
    summary_paths = sorted(
        STRESS_ROOT.glob("source_descriptions_instruct.shard*-of-*.summary.json")
    )
    output_paths = sorted(
        STRESS_ROOT.glob("source_descriptions_instruct.shard*-of-*.jsonl")
    )
    if len(summary_paths) != 4 or len(output_paths) != 4:
        raise RuntimeError("1k stress test requires four summaries and four outputs")
    worker_summaries = [json.loads(path.read_text()) for path in summary_paths]
    rows = [
        json.loads(line)
        for path in output_paths
        for line in path.open("r", encoding="utf-8")
        if line.strip()
    ]
    if len(rows) != 1_000 or len({row["id"] for row in rows}) != 1_000:
        raise RuntimeError("stress outputs are not exactly 1,000 unique rows")
    hard_failures = 0
    terminal_punctuation = 0
    words: list[int] = []
    generated_tokens: list[int] = []
    for row in rows:
        qc = validate_source_description(row["source_description"])
        hard_failures += int(
            bool(qc.hard_flags)
            or bool(row["description_hard_qc_flags"])
            or row["finish_reason"] != "stop"
            or bool(row["generation_capped"])
        )
        terminal_punctuation += int(
            bool(re.search(r"""[.!?]["'”’)]?$""", row["source_description"]))
        )
        words.append(qc.word_count)
        generated_tokens.append(int(row["generated_tokens"]))

    log_paths = sorted((STRESS_ROOT / "logs").glob("*.log"))
    timestamps: list[datetime] = []
    for path in log_paths:
        for line in path.open("r", encoding="utf-8"):
            if re.match(r"^\d{4}-\d{2}-\d{2} ", line):
                timestamps.append(
                    datetime.strptime(line[:23], "%Y-%m-%d %H:%M:%S,%f")
                )
    wall_seconds = (max(timestamps) - min(timestamps)).total_seconds()
    steady_aggregate = sum(
        float(row["end_to_end_clips_per_second_excluding_model_load"])
        for row in worker_summaries
    )
    generation_aggregate = sum(
        float(row["clips_per_second"]) for row in worker_summaries
    )
    projected_seconds = EXPECTED_FULL_ROWS / steady_aggregate
    one_day_target = EXPECTED_FULL_ROWS / ONE_DAY_SECONDS

    universe = pq.read_table(
        SOURCE_ROOT / "source_universe.parquet",
        columns=["kind", "duration_sec"],
    )
    first_1k_duration = universe.slice(0, 1000)["duration_sec"]
    duration_by_kind = {}
    for kind in ("music", "sound"):
        durations = pc.filter(
            universe["duration_sec"],
            pc.equal(universe["kind"], kind),
        )
        duration_by_kind[kind] = {
            "rows": len(durations),
            "mean_sec": pc.mean(durations).as_py(),
            "p50_sec": pc.quantile(durations, q=0.5)[0].as_py(),
            "p90_sec": pc.quantile(durations, q=0.9)[0].as_py(),
        }

    full_output_base = SOURCE_ROOT / "annotations/source_descriptions_instruct.jsonl"
    production_command = (
        "python dataset/captioning/sceneplan_a2t_v2/launch_transformers_scaleout.py "
        f"--log-dir {SOURCE_ROOT / 'annotations/logs'} -- "
        f"--input-jsonl {SOURCE_ROOT / 'instruct_input.jsonl'} "
        f"--out {full_output_base} "
        "--batch-size 256 --device-map balanced "
        "--attn-implementation sdpa --safety-max-generation-tokens 256"
    )
    ready = (
        source_summary["source_hash_rows"] == EXPECTED_FULL_ROWS
        and hard_failures == 0
        and terminal_punctuation == 1_000
        and all(row["completed_this_run"] == 250 for row in worker_summaries)
        and all(row["generation_capped"] == 0 for row in worker_summaries)
        and finalizer["annotation_rows"] == 1_000
        and finalizer["spoken_label_rows"] == 1_000
        and finalizer["full_annotation_started"] is False
        and revised["ok"] is True
        and revised["source_description_registry"][
            "formal_tts_scenes_with_spoken_language_background"
        ]
        == 0
    )
    report = {
        "schema": "stable_audio_tools.sceneplan_a2t_full_scale_preflight",
        "schema_version": 1,
        "status": (
            "ready_waiting_for_user_confirmation"
            if ready
            else "preflight_failed"
        ),
        "ready_for_full_scale_confirmation": ready,
        "full_annotation_started": False,
        "p8_started": False,
        "p9_started": False,
        "frozen_input": {
            "rows": source_summary["source_hash_rows"],
            "by_kind": source_summary["by_kind"],
            "by_split_kind": source_summary["by_split_kind"],
            "jsonl": source_summary["instruct_input_jsonl"],
            "jsonl_sha256": source_summary["instruct_input_sha256"],
            "universe_parquet": source_summary["universe_parquet"],
            "universe_parquet_sha256": source_summary[
                "universe_parquet_sha256"
            ],
            "input_schema_sha256": source_summary[
                "instruct_input_schema_sha256"
            ],
            "registry_schema_sha256": source_summary["registry_schema_sha256"],
        },
        "throughput_1k": {
            "rows": len(rows),
            "kind_counts": dict(sorted(Counter(row["kind"] for row in rows).items())),
            "duration_sec": {
                "stress_mean": pc.mean(first_1k_duration).as_py(),
                "full_by_kind": duration_by_kind,
            },
            "gpu_topology": "4 independent workers x 2 RTX 4090",
            "batch_size": 256,
            "attention": "sdpa",
            "last_token_logits_only": True,
            "worker_end_to_end_clips_per_second": [
                row["end_to_end_clips_per_second_excluding_model_load"]
                for row in worker_summaries
            ],
            "steady_aggregate_clips_per_second": steady_aggregate,
            "generation_only_aggregate_clips_per_second": generation_aggregate,
            "observed_wall_seconds_including_model_load": wall_seconds,
            "observed_wall_clips_per_second_including_model_load": 1000
            / wall_seconds,
            "one_day_required_clips_per_second": one_day_target,
            "steady_margin_over_one_day_target": steady_aggregate
            / one_day_target,
            "projected_full_seconds": projected_seconds,
            "projected_full_hours": projected_seconds / 3600,
            "projected_full_days": projected_seconds / ONE_DAY_SECONDS,
            "peak_cuda_memory_allocated_bytes_by_worker": [
                row["cuda_peak_memory_allocated_bytes"]
                for row in worker_summaries
            ],
        },
        "description_qc_1k": {
            "hard_failures": hard_failures,
            "generation_capped": sum(
                int(row["generation_capped"]) for row in rows
            ),
            "terminal_punctuation": terminal_punctuation,
            "word_count": {
                "min": min(words),
                "p50": percentile(words, 50),
                "p90": percentile(words, 90),
                "p99": percentile(words, 99),
                "max": max(words),
                "within_soft_20_30": sum(20 <= value <= 30 for value in words),
            },
            "generated_tokens_observability_only": {
                "min": min(generated_tokens),
                "p50": percentile(generated_tokens, 50),
                "p99": percentile(generated_tokens, 99),
                "max": max(generated_tokens),
            },
            "spoken_language_background_true": finalizer[
                "spoken_language_background_true"
            ],
        },
        "resume_and_finalizer": {
            "partial_rows_accepted": finalizer["annotation_rows"],
            "rows_by_shard": finalizer["rows_by_shard"],
            "incomplete_tail_files_ignored": finalizer[
                "incomplete_tail_files_ignored"
            ],
            "missing_annotation_rows": finalizer["missing_annotation_rows"],
            "missing_spoken_label_rows": finalizer[
                "missing_spoken_label_rows"
            ],
            "incomplete_finalize_refusal_test": True,
            "registry_finalized": False,
        },
        "revised_sceneplan_100": {
            "summary": str(revised_summary_path),
            "summary_sha256": file_sha256(revised_summary_path),
            "rows": revised["rows"],
            "family_counts": revised["family_counts"],
            "source_count_counts": revised["source_count_counts"],
            "registry_rows": revised["source_description_registry"][
                "approved_unique"
            ],
            "registry_rows_referenced": revised[
                "source_description_registry"
            ]["referenced_unique"],
            "spoken_language_registry_rows": revised[
                "source_description_registry"
            ]["spoken_language_background_registry_rows"],
            "formal_tts_spoken_background_violations": revised[
                "source_description_registry"
            ]["formal_tts_scenes_with_spoken_language_background"],
            "caption_tokens": revised["caption_tokens"],
            "conditioning": revised["conditioning"],
            "sceneplan_jsonl": revised["outputs"]["sceneplan_jsonl"],
            "sceneplan_jsonl_sha256": revised["outputs"][
                "sceneplan_jsonl_sha256"
            ],
        },
        "selected_production_runner": {
            "engine": "transformers",
            "attention": "sdpa",
            "batch_size": 256,
            "gpu_groups": "0,1;2,3;4,5;6,7",
            "failure_policy": "fail_fast_all_workers_then_resume_same_shards",
            "flash_attention": "not_installed_source_build_cancelled_before_install",
            "launch_command_not_executed": production_command,
        },
    }
    output = STRESS_ROOT / "preflight_report.json"
    atomic_json(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
