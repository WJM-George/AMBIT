#!/usr/bin/env python3
"""Exhaustively audit a materialized Transfusion Editing pilot shard."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sqlite3
import sys
import zlib
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch
from safetensors import safe_open


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stable_audio_tools.data.sceneplan_transfusion_editing import (  # noqa: E402
    EDIT_OPERATIONS,
    EditingPairMutation,
    canonical_json,
    sha256_json,
    validate_editing_pair_mutation,
)
from stable_audio_tools.data.sceneplan_transfusion_editing_index import (  # noqa: E402
    scene_domain,
    sha256_file,
)


SCHEMA = "sceneplan_transfusion_editing_pilot_audit_v1"
PAIR_GAIN_POLICY = (
    "fixed_member_corrections_nonboosting_peak_safe_target_master_v1"
)


def _unpack(value: bytes) -> Any:
    return json.loads(zlib.decompress(value))


def _tensor_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()


def _read_latent(reference: str, expected_sha: str, frames: int) -> torch.Tensor:
    path_text, key = reference.rsplit("#", 1)
    path = Path(path_text).resolve(strict=True)
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        if key not in handle.keys():
            raise RuntimeError(f"latent key is absent: {reference}")
        value = handle.get_tensor(key).clone()
    if value.dtype != torch.float16 or tuple(value.shape) != (64, int(frames)):
        raise RuntimeError(f"latent geometry changed: {reference} {value.shape}")
    if not torch.isfinite(value).all():
        raise RuntimeError(f"latent contains non-finite values: {reference}")
    if _tensor_sha256(value) != expected_sha:
        raise RuntimeError(f"latent tensor SHA256 changed: {reference}")
    return value


def _stem_by_id(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(item["source_id"]): item for item in result.get("stem_refs") or ()
    }


def _read_foa(path: str) -> np.ndarray:
    value, rate = sf.read(path, dtype="float32", always_2d=True)
    if rate != 44_100 or value.shape[1] != 4 or not np.isfinite(value).all():
        raise RuntimeError(f"invalid retained FOA stem: {path}")
    return value.T


def _audit_unchanged_stems(
    pair_row: sqlite3.Row, target_result: dict[str, Any]
) -> dict[str, int]:
    source_result = _unpack(pair_row["source_render_result_zlib"])
    retained_source_parity = {
        "stem_refs": target_result.get("source_parity_stem_refs") or ()
    }
    old_stems = _stem_by_id(retained_source_parity)
    if not old_stems:
        old_stems = _stem_by_id(source_result)
    new_stems = _stem_by_id(target_result)
    unchanged = json.loads(pair_row["unchanged_source_ids_json"])
    exact = 0
    scaled = 0
    unavailable = 0
    source_master = float(target_result["pair_gain_qc"]["source_master_gain"])
    target_master = float(target_result["pair_gain_qc"]["target_master_gain"])
    ratio = target_master / source_master
    for source_id in unchanged:
        old_ref = old_stems.get(source_id)
        new_ref = new_stems.get(source_id)
        if (
            old_ref is None
            or new_ref is None
            or not Path(str(old_ref.get("path") or "")).is_file()
            or not Path(str(new_ref.get("path") or "")).is_file()
        ):
            unavailable += 1
            continue
        if not target_result["pair_gain_qc"]["target_master_clamped"]:
            if str(old_ref["sha256"]) != str(new_ref["sha256"]):
                raise RuntimeError(
                    f"{pair_row['pair_id']}/{source_id}: unchanged stem SHA drift"
                )
            exact += 1
            continue
        old_audio = _read_foa(str(old_ref["path"]))
        new_audio = _read_foa(str(new_ref["path"]))
        if old_audio.shape != new_audio.shape:
            raise RuntimeError("clamped unchanged stem geometry changed")
        # Both inputs are PCM24.  Scaling an already-quantized old stem incurs
        # at most a few output LSBs versus scaling before target quantization.
        error = np.max(np.abs(new_audio - old_audio * ratio))
        if float(error) > 8.0 / (2**23):
            raise RuntimeError(
                f"{pair_row['pair_id']}/{source_id}: global-only stem scaling "
                f"error is too large: {error}"
            )
        scaled += 1
    return {"exact": exact, "global_scaled": scaled, "unavailable": unavailable}


def validate(pair_index: Path, manifest_path: Path, *, require_source_parity: bool) -> dict[str, Any]:
    pair_index = pair_index.resolve(strict=True)
    manifest_path = manifest_path.resolve(strict=True)
    uri = f"file:{pair_index}?mode=ro&immutable=1"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    try:
        metadata = dict(connection.execute("SELECT key,value FROM metadata"))
        pairs = list(connection.execute("SELECT * FROM pairs ORDER BY pair_ordinal"))
    finally:
        connection.close()
    if metadata.get("pair_gain_policy") != PAIR_GAIN_POLICY:
        raise RuntimeError("pilot pair-gain policy changed")
    materialized = pq.read_table(manifest_path).to_pylist()
    by_id = {str(row["pair_id"]): row for row in materialized}
    if len(by_id) != len(materialized) or len(pairs) != len(materialized):
        raise RuntimeError("pair/materialized row cardinality differs")

    operation_counts: Counter[str] = Counter()
    bucket_counts: Counter[int] = Counter()
    source_domains: Counter[str] = Counter()
    target_domains: Counter[str] = Counter()
    source_target_equal = 0
    source_target_l1 = []
    clamped = 0
    stem_counts: Counter[str] = Counter()
    for pair in pairs:
        pair_id = str(pair["pair_id"])
        row = by_id.get(pair_id)
        if row is None or int(row["pair_ordinal"]) != int(pair["pair_ordinal"]):
            raise RuntimeError(f"{pair_id}: materialized join failed")
        old_plan = _unpack(pair["old_sceneplan_zlib"])
        new_plan = _unpack(pair["new_sceneplan_zlib"])
        source_recipe = _unpack(pair["source_render_recipe_zlib"])
        target_recipe = _unpack(pair["target_render_recipe_zlib"])
        source_members = _unpack(pair["source_members_zlib"])
        target_members = _unpack(pair["target_members_zlib"])
        mutation = EditingPairMutation(
            pair_id=pair_id,
            split=str(pair["split"]),
            operation_family=str(pair["operation_family"]),
            operation=str(pair["operation"]),
            instruction=str(pair["raw_edit_request"]),
            instruction_template_id=str(pair["instruction_template_id"]),
            old_sceneplan=old_plan,
            new_sceneplan=new_plan,
            source_render_recipe=source_recipe,
            target_render_recipe=target_recipe,
            edited_source_ids=tuple(json.loads(pair["edited_source_ids_json"])),
            unchanged_source_ids=tuple(
                json.loads(pair["unchanged_source_ids_json"])
            ),
            source_members=tuple(source_members),
            target_members=tuple(target_members),
        )
        validate_editing_pair_mutation(mutation)
        for value, expected, label in (
            (sha256_json(old_plan), pair["old_sceneplan_sha256"], "old plan"),
            (sha256_json(new_plan), pair["new_sceneplan_sha256"], "new plan"),
            (
                sha256_json(source_recipe),
                pair["source_render_recipe_sha256"],
                "source recipe",
            ),
            (
                sha256_json(target_recipe),
                pair["target_render_recipe_sha256"],
                "target recipe",
            ),
            (
                sha256_json(source_members),
                pair["source_members_sha256"],
                "source members",
            ),
            (
                sha256_json(target_members),
                pair["target_members_sha256"],
                "target members",
            ),
        ):
            if str(value) != str(expected):
                raise RuntimeError(f"{pair_id}: {label} hash changed")
        result = json.loads(row["target_render_result_json"])
        result_text = canonical_json(result)
        if hashlib.sha256(result_text.encode()).hexdigest() != row[
            "target_render_result_sha256"
        ]:
            raise RuntimeError(f"{pair_id}: render-result hash changed")
        if (
            result.get("status") != "ok"
            or result.get("pair_record_sha256") != pair["pair_record_sha256"]
            or result.get("pair_gain_qc", {}).get("policy") != PAIR_GAIN_POLICY
        ):
            raise RuntimeError(f"{pair_id}: invalid target render result")
        if require_source_parity and not bool(row["source_parity_verified"]):
            raise RuntimeError(f"{pair_id}: source parity was not proven")
        gain = result["pair_gain_qc"]
        if float(gain["target_master_gain"]) > float(gain["source_master_gain"]) + 1e-12:
            raise RuntimeError(f"{pair_id}: target master gain was boosted")
        if float(gain["target_true_peak"]) > float(gain["true_peak_ceiling"]) + 1e-5:
            raise RuntimeError(f"{pair_id}: target true-peak ceiling failed")
        clamped += int(bool(gain["target_master_clamped"]))
        target_foa_path = Path(str(row["target_foa_path"])).resolve(strict=True)
        if sha256_file(target_foa_path) != str(row["target_foa_sha256"]):
            raise RuntimeError(f"{pair_id}: target FOA SHA256 changed")

        frames = int(pair["latent_frames_valid"])
        source_latent = _read_latent(
            str(pair["source_latent_ref"]),
            str(pair["source_latent_tensor_sha256"]),
            frames,
        )
        target_latent = _read_latent(
            str(row["target_latent_ref"]),
            str(row["target_latent_tensor_sha256"]),
            frames,
        )
        if tuple(source_latent.shape) != tuple(target_latent.shape):
            raise RuntimeError(f"{pair_id}: source/target latent time axes differ")
        equal = torch.equal(source_latent, target_latent)
        source_target_equal += int(equal)
        source_target_l1.append(
            float((source_latent.float() - target_latent.float()).abs().mean())
        )
        for key, count in _audit_unchanged_stems(pair, result).items():
            stem_counts[key] += count
        operation_counts[str(pair["operation"])] += 1
        bucket_counts[int(pair["latent_bucket_frames"])] += 1
        source_domains[scene_domain(old_plan)] += 1
        target_domains[scene_domain(new_plan)] += 1
    if set(operation_counts) != set(EDIT_OPERATIONS):
        raise RuntimeError("pilot does not cover all Editing operations")
    if source_target_equal:
        raise RuntimeError("an edited target latent is exactly its source latent")
    return {
        "schema": SCHEMA,
        "status": "pass",
        "pair_index": str(pair_index),
        "pair_index_sha256": sha256_file(pair_index),
        "materialized_manifest": str(manifest_path),
        "materialized_manifest_sha256": sha256_file(manifest_path),
        "rows": len(pairs),
        "operation_counts": dict(sorted(operation_counts.items())),
        "latent_bucket_counts": dict(sorted(bucket_counts.items())),
        "source_domain_counts": dict(sorted(source_domains.items())),
        "target_domain_counts": dict(sorted(target_domains.items())),
        "source_parity_verified_rows": sum(
            bool(row["source_parity_verified"]) for row in materialized
        ),
        "source_target_exact_equal_rows": source_target_equal,
        "source_target_latent_mean_l1_min": min(source_target_l1),
        "source_target_latent_mean_l1_max": max(source_target_l1),
        "target_master_clamped_rows": clamped,
        "unchanged_stem_checks": dict(sorted(stem_counts.items())),
        "frame_alignment_contract": "source_and_target_are_64x_same_T",
        "pair_gain_policy": PAIR_GAIN_POLICY,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-index", type=Path, required=True)
    parser.add_argument("--materialized-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-source-parity", action="store_true")
    args = parser.parse_args()
    report = validate(
        args.pair_index,
        args.materialized_manifest,
        require_source_parity=bool(args.require_source_parity),
    )
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(output.name + f".tmp.{os.getpid()}")
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, output)
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
