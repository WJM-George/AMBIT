#!/usr/bin/env python3
"""Fail-closed validator for active audio-aware P11 manifest-v8."""

from __future__ import annotations

import argparse
import json
import math
import re
import sqlite3
import sys
import zlib
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stable_audio_tools.configuration import load_config  # noqa: E402
from stable_audio_tools.data.model_sceneplan_codec import load_model_sceneplan_codec  # noqa: E402
from stable_audio_tools.data.sceneplan_edit_patch import (  # noqa: E402
    ACTIVE_OPERATION_TOKENS,
    PATCH_MAX_TOKENS,
    RETIME_FRAME_LEVELS,
    RETIME_MODES,
    RETIME_POLICY,
    ScenePlanEditPatchCodec,
)
from stable_audio_tools.data.sceneplan_p11_single_turn import (  # noqa: E402
    P11_EDITING_CONTRACT,
    P11_EDITING_INPUT_CONTRACT,
    P11_EDITING_OUTPUT_CONTRACT,
    P11_EDIT_EVIDENCE_MODES,
    P11_MODEL_CONTRACT,
    SCENEPLAN_PLAN_MAX_TOKENS,
    SEMANTIC_CAPTION_MAX_TOKENS,
    canonicalize_sceneplan_source_ids,
)
from stable_audio_tools.data.sceneplan_p11_dataset import (  # noqa: E402
    P11_DATA_CONTRACT,
    P11_MANIFEST_VERSION,
    ScenePlanP11Dataset,
)


DEFAULT_MODEL = REPO_ROOT / (
    "stable_audio_tools/configs/model_configs/txt2audio/t2a/sceneplan_p11/"
    "qwen35_0p8b_sceneplan_p11_audio_aware_v1.json"
)
DEFAULT_DATASET = REPO_ROOT / (
    "stable_audio_tools/configs/dataset_configs/"
    "sceneplan_p11_audio_aware_v1_pilot90.json"
)
GENERATION_DURATION = re.compile(
    r"^Create a ([0-9]+(?:\.[0-9]+)?)-second FOA spatial-audio scene\."
)
EXPECTED_ROW_COLUMNS = (
    "ordinal",
    "task",
    "source_ordinal",
    "input_audio_ordinal",
    "prompt",
    "observed_sceneplan_zlib",
    "input_sceneplan_zlib",
    "target_sceneplan_zlib",
    "editing_evidence_mode",
    "edit_kind",
    "edit_spec_json",
    "prior_corruption_json",
)


def _readonly(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True)
    connection.execute("PRAGMA query_only=ON")
    return connection


def _plan(payload: bytes | None) -> dict[str, Any] | None:
    if payload is None:
        return None
    value = json.loads(zlib.decompress(payload))
    if not isinstance(value, dict):
        raise RuntimeError("P11 ScenePlan payload is not an object")
    return value


def _percentile(values: Sequence[int], percentile: float) -> float:
    return float(np.percentile(np.asarray(values), percentile))


def _finite(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, Mapping):
        return all(_finite(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return all(_finite(item) for item in value)
    return True


def _slot(source_id: Any) -> int:
    return int(str(source_id)[7:])


def _assert_source_identity(
    observed: Mapping[str, Any],
    revised: Mapping[str, Any],
    spec: Mapping[str, Any],
) -> None:
    before = [str(source["source_id"]) for source in observed["sources"]]
    after = [str(source["source_id"]) for source in revised["sources"]]
    operation = str(spec["operation"])
    if operation == "add_source":
        occupied = {_slot(source_id) for source_id in before}
        first_free = next(slot for slot in range(4) if slot not in occupied)
        added = str(spec["source"]["source_id"])
        if added != f"source_{first_free}" or sorted(set(after) - set(before)) != [added]:
            raise RuntimeError("ADD_SOURCE did not consume exactly the first free slot")
        if not set(before).issubset(after):
            raise RuntimeError("ADD_SOURCE renumbered an observed source")
    elif operation == "remove_source":
        owner = str(spec["source_id"])
        if set(after) != set(before) - {owner}:
            raise RuntimeError("REMOVE_SOURCE changed the wrong source inventory")
    elif set(after) != set(before):
        raise RuntimeError(f"{operation} changed persistent source ids")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--dataset-config", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--base-scenes",
        type=int,
        default=10_000,
        help="Triplets to inspect; 0 validates every triplet.",
    )
    parser.add_argument("--grammar-scenes", type=int, default=100)
    parser.add_argument(
        "--scope",
        choices=("manifest", "full"),
        default="full",
        help="full additionally loads FOA latents and the exact semantic cache",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.base_scenes < 0 or args.grammar_scenes < 0:
        raise ValueError("validator scene counts must be non-negative")

    model_config = load_config(args.model_config)
    dataset_config = load_config(args.dataset_config)
    if model_config.get("model_type") != "sceneplan_p11_audio_aware_v1":
        raise ValueError("validator requires the active audio-aware P11 model config")
    text_config = model_config["model"]["text"]
    if int(text_config.get("patch_max_tokens", -1)) != PATCH_MAX_TOKENS:
        raise RuntimeError(f"P11 patch ceiling must be {PATCH_MAX_TOKENS}")
    if dataset_config.get("p11_contract") != P11_DATA_CONTRACT:
        raise ValueError("validator requires the active audio-aware dataset contract")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        text_config["model_path"], local_files_only=True, use_fast=True
    )
    manifest_path = Path(dataset_config["manifest_path"]).resolve(strict=True)
    index_path = Path(dataset_config["datasets"][0]["path"]).resolve(strict=True)
    codec = load_model_sceneplan_codec(dataset_config["codec_path"])
    patch_codec = ScenePlanEditPatchCodec(codec)
    connection = _readonly(manifest_path)
    metadata = dict(connection.execute("SELECT key, value FROM metadata"))
    if int(metadata.get("schema_version", -1)) != P11_MANIFEST_VERSION:
        raise RuntimeError(
            f"validator accepts only canonical P11 manifest v{P11_MANIFEST_VERSION}"
        )
    expected_metadata = {
        "model_contract": P11_MODEL_CONTRACT,
        "data_contract": P11_DATA_CONTRACT,
        "editing_contract": P11_EDITING_CONTRACT,
        "editing_input_contract": P11_EDITING_INPUT_CONTRACT,
        "editing_output_contract": P11_EDITING_OUTPUT_CONTRACT,
        "editing_input_audio_required": "true",
        "editing_old_sceneplan_role": "optional_fallible_prior",
        "editing_revised_authority": "deterministic_patch_applied_to_observed",
        "target_audio_supervision": "forbidden",
        "seed": "42",
        "source_index": str(index_path),
        "retime_policy": RETIME_POLICY,
        "retime_frame_levels": ",".join(map(str, RETIME_FRAME_LEVELS)),
        "retime_modes": ",".join(RETIME_MODES),
    }
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            raise RuntimeError(
                f"manifest metadata {key}={metadata.get(key)!r}, expected {expected!r}"
            )
    columns = tuple(
        str(row[1]) for row in connection.execute("PRAGMA table_info(rows)")
    )
    if columns != EXPECTED_ROW_COLUMNS:
        raise RuntimeError(f"manifest row schema changed: {columns}")
    if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
        raise RuntimeError("manifest SQLite integrity_check failed")

    total_base = int(metadata["base_samples"])
    selected_base = total_base if args.base_scenes == 0 else min(args.base_scenes, total_base)
    if selected_base <= 0:
        raise RuntimeError("P11 validator selected no scenes")
    grammar_base = min(args.grammar_scenes, selected_base)

    tasks: Counter[str] = Counter()
    edit_kinds: Counter[str] = Counter()
    evidence_modes: Counter[str] = Counter()
    corruptions: Counter[str] = Counter()
    source_counts: Counter[int] = Counter()
    kinds: Counter[str] = Counter()
    motions: Counter[str] = Counter()
    duration_buckets: Counter[str] = Counter()
    retime_modes: Counter[str] = Counter()
    retime_magnitudes: Counter[int] = Counter()
    plan_lengths: list[int] = []
    patch_lengths: list[int] = []
    prompts: list[str] = []
    grammar_transitions = 0
    no_op_hash_preserved = 0
    for position in range(selected_base):
        rows = connection.execute(
            """
            SELECT ordinal,task,source_ordinal,input_audio_ordinal,prompt,
                   observed_sceneplan_zlib,input_sceneplan_zlib,
                   target_sceneplan_zlib,editing_evidence_mode,edit_kind,
                   edit_spec_json,prior_corruption_json
            FROM rows WHERE ordinal BETWEEN ? AND ? ORDER BY ordinal
            """,
            (position * 3, position * 3 + 2),
        ).fetchall()
        if len(rows) != 3 or [row[1] for row in rows] != [
            "generation",
            "understanding",
            "editing",
        ]:
            raise RuntimeError(f"P11 triplet {position} is incomplete")
        if len({int(row[2]) for row in rows}) != 1:
            raise RuntimeError(f"P11 triplet {position} crosses base scenes")
        if rows[0][3] is not None or int(rows[1][3]) != int(rows[1][2]) or int(rows[2][3]) != int(rows[2][2]):
            raise RuntimeError("G/U/E input-audio truth table failed")

        generation_target = _plan(rows[0][7])
        understanding_observed = _plan(rows[1][5])
        understanding_target = _plan(rows[1][7])
        editing_observed = _plan(rows[2][5])
        if not (
            generation_target
            == understanding_observed
            == understanding_target
            == editing_observed
        ):
            raise RuntimeError(f"P11 triplet {position} observed truth diverged")
        observed = editing_observed
        if observed is None:
            raise RuntimeError("Editing observed ScenePlan is absent")
        if canonicalize_sceneplan_source_ids(observed) != observed:
            raise RuntimeError(f"P11 triplet {position} observed ids are not canonical")
        if not _finite(observed):
            raise RuntimeError("observed ScenePlan contains non-finite values")
        observed_tokens = codec.encode(
            observed, max_tokens=SCENEPLAN_PLAN_MAX_TOKENS
        )["input_ids"]
        if codec.decode(observed_tokens, sample_id=observed["sample_id"]) != observed:
            raise RuntimeError(f"P11 triplet {position} plan round-trip failed")
        source_counts[len(observed["sources"])] += 1
        for source in observed["sources"]:
            kinds[str(source["kind"])] += 1
            motions[str(source["trajectory"]["type"])] += 1
        duration_frames = codec._duration_frame(observed["duration_sec"])
        duration_buckets["433_648" if duration_frames > 432 else "1_432"] += 1

        for row in rows:
            tasks[str(row[1])] += 1
            prompts.append(str(row[4]))
        duration_match = GENERATION_DURATION.match(" ".join(str(rows[0][4]).split()))
        if duration_match is None:
            raise RuntimeError("P11 generation prompt omits requested duration")
        if not math.isclose(
            codec.snap_numeric_to_grid("seconds", float(duration_match.group(1))),
            float(observed["duration_sec"]),
            abs_tol=1.0e-9,
        ):
            raise RuntimeError("P11 generation duration does not identify its target")
        plan_lengths.extend([int(observed_tokens.numel())] * 3)

        edit_row = rows[2]
        mode = str(edit_row[8])
        if mode not in P11_EDIT_EVIDENCE_MODES:
            raise RuntimeError("Editing evidence mode is invalid")
        prior = _plan(edit_row[6])
        corruption = None if edit_row[11] is None else json.loads(str(edit_row[11]))
        if mode == "no_plan":
            if prior is not None or corruption is not None:
                raise RuntimeError("no-plan row leaked an old ScenePlan")
        elif mode == "correct_plan":
            if prior != observed or corruption is not None:
                raise RuntimeError("correct-plan row does not carry observed truth")
        else:
            if prior is None or prior == observed or corruption is None:
                raise RuntimeError("corrupt-plan row lacks a real fallible prior")
            if corruption.get("supervision_targets_unchanged") is not True:
                raise RuntimeError("prior corruption can alter supervision targets")
            corruptions[str(corruption.get("type"))] += 1
        evidence_modes[mode] += 1

        revised = _plan(edit_row[7])
        if revised is None or not _finite(revised):
            raise RuntimeError("revised ScenePlan is absent or non-finite")
        edit_kind = str(edit_row[9])
        spec = json.loads(str(edit_row[10]))
        if spec.get("contract") != "audio_aware_atomic_patch_v1":
            raise RuntimeError("P11 Editing patch has a stale contract")
        if str(spec.get("operation")) != edit_kind or edit_kind not in ACTIVE_OPERATION_TOKENS:
            raise RuntimeError("P11 active edit kind/spec mismatch")
        patch_codec.assert_target(observed, spec, revised)
        _assert_source_identity(observed, revised, spec)
        patch_tokens = patch_codec.encode(spec)["input_ids"]
        if not torch.equal(
            patch_tokens, patch_codec.canonicalize(patch_tokens)["input_ids"]
        ):
            raise RuntimeError("P11 patch round-trip is not exact")
        applied = patch_codec.apply(observed, patch_tokens)
        if not torch.equal(
            codec.encode(applied)["input_ids"], codec.encode(revised)["input_ids"]
        ):
            raise RuntimeError("apply(patch, observed) != revised")
        if edit_kind == "no_op":
            if not torch.equal(codec.encode(observed)["input_ids"], codec.encode(revised)["input_ids"]):
                raise RuntimeError("no-op changed the canonical ScenePlan hash")
            no_op_hash_preserved += 1
        elif edit_kind == "retime_source":
            if spec.get("retime_policy") != RETIME_POLICY:
                raise RuntimeError("retime row uses a stale scale policy")
            retime_mode = str(spec.get("retime_mode") or "")
            retime_magnitude = int(spec.get("retime_magnitude_frames", -1))
            if retime_mode not in RETIME_MODES:
                raise RuntimeError("retime row has an invalid mode")
            if retime_magnitude not in RETIME_FRAME_LEVELS:
                raise RuntimeError("retime row has an invalid frame magnitude")
            source_id = str(spec["source_id"])
            before = next(
                source for source in observed["sources"]
                if str(source["source_id"]) == source_id
            )
            after = next(
                source for source in revised["sources"]
                if str(source["source_id"]) == source_id
            )
            before_frames = tuple(
                codec._frame_from_seconds(before["activity"][key], mode="nearest")
                for key in ("onset_sec", "offset_sec")
            )
            after_frames = tuple(
                codec._frame_from_seconds(after["activity"][key], mode="nearest")
                for key in ("onset_sec", "offset_sec")
            )
            actual_magnitude = max(
                abs(left - right)
                for left, right in zip(before_frames, after_frames)
            )
            if actual_magnitude != retime_magnitude or actual_magnitude < 4:
                raise RuntimeError("retime row is not a meaningful P10-frame edit")
            retime_modes[retime_mode] += 1
            retime_magnitudes[retime_magnitude] += 1
        if position < grammar_base:
            for prefix_length, expected in enumerate(patch_tokens.tolist()):
                allowed = patch_codec.allowed_next_ids(
                    patch_tokens[:prefix_length], input_sceneplan=observed
                )
                if int(expected) not in allowed:
                    raise RuntimeError(
                        f"patch grammar rejected {edit_kind} token {prefix_length}"
                    )
                grammar_transitions += 1
            if patch_codec.allowed_next_ids(patch_tokens, input_sceneplan=observed):
                raise RuntimeError("P11 patch grammar did not terminate")
        edit_kinds[edit_kind] += 1
        patch_lengths.append(int(patch_tokens.numel()))

    connection.close()
    if selected_base >= len(ACTIVE_OPERATION_TOKENS) and set(edit_kinds) != set(ACTIVE_OPERATION_TOKENS):
        raise RuntimeError(f"pilot does not cover every active edit: {edit_kinds}")
    if selected_base >= 10 and set(evidence_modes) != set(P11_EDIT_EVIDENCE_MODES):
        raise RuntimeError("pilot does not cover every old-plan evidence mode")
    if evidence_modes["corrupt_plan"] >= len(PRIOR_CORRUPTION_TYPES := ("room", "source_count", "description", "activity", "trajectory")) and set(corruptions) != set(PRIOR_CORRUPTION_TYPES):
        raise RuntimeError(f"pilot does not cover every prior corruption: {corruptions}")
    if json.loads(metadata.get("retime_mode_counts", "{}")) != dict(retime_modes):
        raise RuntimeError("manifest retime mode counts are stale")
    if {
        int(key): int(value)
        for key, value in json.loads(
            metadata.get("retime_magnitude_counts", "{}")
        ).items()
    } != dict(retime_magnitudes):
        raise RuntimeError("manifest retime magnitude counts are stale")

    prompt_lengths: list[int] = []
    for start in range(0, len(prompts), 256):
        encoded = tokenizer(
            prompts[start : start + 256], add_special_tokens=True, truncation=False
        )["input_ids"]
        prompt_lengths.extend(len(value) for value in encoded)
    if max(prompt_lengths) > SEMANTIC_CAPTION_MAX_TOKENS:
        raise RuntimeError("P11 manifest contains a truncated prompt")

    runtime_truth: list[dict[str, Any]] = []
    if args.scope == "full":
        dataset = ScenePlanP11Dataset(
            manifest_path,
            index_path=index_path,
            codec_path=dataset_config["codec_path"],
            tokenizer_spec=(tokenizer, SEMANTIC_CAPTION_MAX_TOKENS, None),
            expected_num_samples=int(dataset_config["expected_num_samples"]),
            index_num_samples=int(dataset_config["index_num_samples"]),
            require_frozen=True,
            semantic_cache_path=dataset_config["semantic_cache_path"],
            semantic_dim=int(dataset_config.get("semantic_dim", 512)),
            semantic_encoder_revision=dataset_config["semantic_encoder_revision"],
        )
        runtime_ordinals = [0, 1]
        for mode in P11_EDIT_EVIDENCE_MODES:
            row = _readonly(manifest_path).execute(
                "SELECT ordinal FROM rows WHERE task='editing' AND editing_evidence_mode=? LIMIT 1",
                (mode,),
            ).fetchone()
            runtime_ordinals.append(int(row[0]))
        for ordinal in runtime_ordinals:
            _, row = dataset[ordinal]
            runtime_truth.append(
                {
                    "task": row["p11_task"],
                    "evidence_mode": row["p11_editing_evidence_mode"],
                    "has_audio": row["p11_input_foa"] is not None,
                    "has_semantic": row["p11_input_semantic"] is not None,
                    "has_old_plan": row["p11_prior_sceneplan_tokens"] is not None,
                    "has_observed_target": row["p11_observed_sceneplan_target"] is not None,
                }
            )

    report = {
        "status": "PASS",
        "validator": "sceneplan_p11_audio_aware_manifest_v8",
        "scope": args.scope,
        "runtime_dataset_gate": "PASS" if args.scope == "full" else "PENDING_SEMANTIC_CACHE",
        "manifest": str(manifest_path),
        "source_index": str(index_path),
        "base_scenes_audited": selected_base,
        "rows_audited": selected_base * 3,
        "tasks": dict(sorted(tasks.items())),
        "edit_kinds": dict(sorted(edit_kinds.items())),
        "editing_evidence_modes": dict(sorted(evidence_modes.items())),
        "prior_corruptions": dict(sorted(corruptions.items())),
        "source_counts": {str(key): value for key, value in sorted(source_counts.items())},
        "source_kinds": dict(sorted(kinds.items())),
        "motion_types": dict(sorted(motions.items())),
        "duration_frame_buckets": dict(sorted(duration_buckets.items())),
        "retime_policy": RETIME_POLICY,
        "retime_modes": dict(sorted(retime_modes.items())),
        "retime_magnitudes": {
            str(key): value for key, value in sorted(retime_magnitudes.items())
        },
        "plan_tokens": {
            "median": _percentile(plan_lengths, 50),
            "p99": _percentile(plan_lengths, 99),
            "max": max(plan_lengths),
        },
        "patch_tokens": {
            "median": _percentile(patch_lengths, 50),
            "p99": _percentile(patch_lengths, 99),
            "max": max(patch_lengths),
            "hard_gate": PATCH_MAX_TOKENS,
        },
        "prompt_tokens": {
            "p99": _percentile(prompt_lengths, 99),
            "max": max(prompt_lengths),
            "hard_gate": SEMANTIC_CAPTION_MAX_TOKENS,
        },
        "patch_grammar_transitions": grammar_transitions,
        "no_op_hash_preserved": no_op_hash_preserved,
        "runtime_truth_table": runtime_truth,
        "contracts": {
            "model": P11_MODEL_CONTRACT,
            "data": P11_DATA_CONTRACT,
            "editing": P11_EDITING_CONTRACT,
            "editing_input": P11_EDITING_INPUT_CONTRACT,
            "editing_output": P11_EDITING_OUTPUT_CONTRACT,
            "patch_codec": patch_codec.fingerprint,
        },
        "truncated": 0,
        "non_finite": 0,
    }
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        destination = args.output.expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
