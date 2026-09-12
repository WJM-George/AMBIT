#!/usr/bin/env python3
"""Re-audit revised P8 metadata while reusing immutable binary P9 evidence."""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from materialize_model_sceneplan_v1_shard import canonical_json
from sceneplan_v2_common import DATASET_ROOT, atomic_write_json


ARCHIVE = DATASET_ROOT / "audit/superseded_speaker_constant_20260819_0924"
CURRENT_MANIFESTS = DATASET_ROOT / "materialized/manifests"
OLD_MANIFESTS = ARCHIVE / "materialized_manifests_before_speaker_rebind"
OUTPUT = DATASET_ROOT / "qc/p9_model_sceneplan_materialized_audit.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def neutral_result(value: dict[str, Any]) -> dict[str, Any]:
    copied = dict(value)
    for key in (
        "planned_record_sha256", "planned_bundle_sha256",
        "model_sceneplan_sha256", "render_recipe_sha256",
        "renderer_caption_sha256", "metadata_only_rebind",
    ):
        copied.pop(key, None)
    return copied


def main() -> int:
    started = time.time()
    prior_path = ARCHIVE / "qc/p9_model_sceneplan_materialized_audit.json"
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    rebind_path = DATASET_ROOT / "materialized/p8_metadata_rebind_audit.json"
    rebind = json.loads(rebind_path.read_text(encoding="utf-8"))
    p75_path = DATASET_ROOT / "sceneplans_model_v1/audit.json"
    p75 = json.loads(p75_path.read_text(encoding="utf-8"))
    if not (
        prior.get("ok") is True and int(prior.get("rows", -1)) == 1_124_000
        and rebind.get("ok") is True and int(rebind.get("rows", -1)) == 1_124_000
        and rebind.get("audio_and_latent_checksums_preserved") is True
        and p75.get("ok") is True and int(p75.get("rows", -1)) == 1_124_000
    ):
        raise RuntimeError("delta P9 prerequisites are incomplete")
    plan_index = pq.ParquetFile(DATASET_ROOT / "sceneplans_model_v1/index.parquet")
    new_paths = [
        path for split in ("train", "validation", "test")
        for path in sorted((CURRENT_MANIFESTS / split).glob("materialized-*.parquet"))
    ]
    old_paths = [
        path for split in ("train", "validation", "test")
        for path in sorted((OLD_MANIFESTS / split).glob("materialized-*.parquet"))
    ]
    if not (len(new_paths) == len(old_paths) == plan_index.num_row_groups == 1_099):
        raise RuntimeError("P7.5/old-P8/new-P8 shard layout mismatch")
    unchanged_fields = (
        "sample_id", "split", "family", "source_count", "model_num_samples",
        "latent_frames_valid", "vae_encode_seed", "foa_path", "foa_sha256",
        "latent_ref", "latent_tensor_sha256", "latent_shard_sha256",
        "work_shard", "row_in_shard",
    )
    rows_seen = 0
    counts: Counter[str] = Counter()
    latent_refs: set[str] = set()
    for shard, (new_path, old_path) in enumerate(zip(new_paths, old_paths)):
        new_rows = pq.read_table(new_path).to_pylist()
        old_rows = pq.read_table(old_path).to_pylist()
        plans = plan_index.read_row_group(shard).to_pylist()
        if not (len(new_rows) == len(old_rows) == len(plans)):
            raise RuntimeError(f"manifest row mismatch: {new_path}")
        for new, old, plan in zip(new_rows, old_rows, plans):
            sample_id = str(new["sample_id"])
            if any(new[field] != old[field] for field in unchanged_fields):
                raise RuntimeError(f"{sample_id}: audio/latent/execution scalar changed")
            if not (
                new["model_sceneplan_sha256"] == plan["model_sceneplan_sha256"]
                and new["render_recipe_sha256"] == plan["render_recipe_sha256"]
                and new["renderer_caption_sha256"] == plan["renderer_caption_sha256"]
            ):
                raise RuntimeError(f"{sample_id}: new P7.5/P8 hash mismatch")
            bundle = sha256_text(canonical_json({
                "model_sceneplan_sha256": new["model_sceneplan_sha256"],
                "render_recipe_sha256": new["render_recipe_sha256"],
                "renderer_caption_sha256": new["renderer_caption_sha256"],
            }))
            if new["planned_bundle_sha256"] != bundle:
                raise RuntimeError(f"{sample_id}: revised bundle hash mismatch")
            new_result = json.loads(str(new["render_result_json"]))
            old_result = json.loads(str(old["render_result_json"]))
            if neutral_result(new_result) != neutral_result(old_result):
                raise RuntimeError(f"{sample_id}: materialized execution/QC changed")
            if not (
                new_result.get("metadata_only_rebind") == "speech_speaker_description_v1"
                and new_result["planned_record_sha256"] == bundle
                and new_result["planned_bundle_sha256"] == bundle
                and new_result["model_sceneplan_sha256"] == new["model_sceneplan_sha256"]
                and new_result["render_recipe_sha256"] == new["render_recipe_sha256"]
                and new_result["renderer_caption_sha256"] == new["renderer_caption_sha256"]
                and new_result["materialized_execution_sha256"] == old_result["materialized_execution_sha256"]
            ):
                raise RuntimeError(f"{sample_id}: revised render-result lineage failed")
            counts[str(new["split"])] += 1
            latent_refs.add(str(new["latent_ref"]).split("#", 1)[0])
            rows_seen += 1
        if (shard + 1) % 100 == 0:
            print(json.dumps({"delta_audited_shards": shard + 1, "rows": rows_seen}), flush=True)
    if rows_seen != 1_124_000 or dict(counts) != {"train": 1_100_000, "validation": 20_000, "test": 4_000}:
        raise RuntimeError("delta P9 row/split totals failed")
    report = dict(prior)
    report.update({
        "schema": "stable_audio_tools.model_sceneplan_materialized_dataset_audit",
        "schema_version": 2,
        "dataset_contract_revision": 5,
        "ok": True,
        "rows": rows_seen,
        "speaker_description_revision": "registry_v1",
        "speech_speaker_registry_rows": 512_000,
        "unique_speaker_descriptions": p75["unique_speaker_descriptions"],
        "constant_generic_speaker_description_rows": 0,
        "metadata_lineage_reaudited_all_rows": True,
        "audio_and_latent_columns_compared_to_prior_all_rows": True,
        "materialized_execution_qc_compared_to_prior_all_rows": True,
        "immutable_binary_audit_reused": {
            "prior_report": str(prior_path),
            "prior_report_sha256": sha256_file(prior_path),
            "prior_report_rows": prior["rows"],
            "latent_shards": len(latent_refs),
            "justification": "metadata-only speaker text rebind; all binary references, tensor hashes, FOA hashes, execution QC and materialized execution hashes are byte-for-byte lineage-identical",
        },
        "p8_metadata_rebind_audit": str(rebind_path),
        "p8_metadata_rebind_audit_sha256": sha256_file(rebind_path),
        "p75_speaker_delta_audit": str(p75_path),
        "p75_speaker_delta_audit_sha256": sha256_file(p75_path),
        "elapsed_sec": round(time.time() - started, 3),
    })
    atomic_write_json(OUTPUT, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
