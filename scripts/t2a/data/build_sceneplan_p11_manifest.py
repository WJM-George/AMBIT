#!/usr/bin/env python3
"""Build the active manifest-v8 for audio-aware P11 G/U/E.

The immutable source index owns each observed FOA latent and its ScenePlan.
Editing derives a revised ScenePlan symbolically but never stores target audio.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sqlite3
import sys
import tempfile
import time
import zlib
from collections import Counter
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stable_audio_tools.data.model_sceneplan import compile_model_renderer_caption  # noqa: E402
from stable_audio_tools.data.model_sceneplan_codec import load_model_sceneplan_codec  # noqa: E402
from stable_audio_tools.data.model_sceneplan_codec_v3 import ModelScenePlanCodecV3  # noqa: E402
from stable_audio_tools.data.model_sceneplan_codec_v4 import ModelScenePlanCodecV4  # noqa: E402
from stable_audio_tools.data.sceneplan_edit_patch import (  # noqa: E402
    ACTIVE_OPERATION_TOKENS,
    PATCH_OUTPUT_CONTRACT,
    RETIME_FRAME_LEVELS,
    RETIME_MODES,
    RETIME_POLICY,
    ScenePlanEditPatchCodec,
    make_deterministic_edit,
)
from stable_audio_tools.data.sceneplan_p11_single_turn import (  # noqa: E402
    P11_EDITING_CONTRACT,
    P11_EDITING_INPUT_CONTRACT,
    P11_EDITING_OUTPUT_CONTRACT,
    P11_EDIT_EVIDENCE_MODES,
    P11_GENERATION_NATURAL_CONTRACT,
    P11_MODEL_CONTRACT,
    P11_SOURCE_IDENTITY_CONTRACT,
    canonicalize_sceneplan_source_ids,
)
from stable_audio_tools.data.sceneplan_p11_dataset import (  # noqa: E402
    P11_DATA_CONTRACT,
    P11_MANIFEST_SCHEMA,
    P11_MANIFEST_VERSION,
)


EDIT_MODE_CYCLE = (
    "no_plan",
    "no_plan",
    "no_plan",
    "no_plan",
    "correct_plan",
    "correct_plan",
    "correct_plan",
    "corrupt_plan",
    "corrupt_plan",
    "corrupt_plan",
)
PRIOR_CORRUPTION_TYPES = (
    "room",
    "source_count",
    "description",
    "activity",
    "trajectory",
)


def _compress(plan: Mapping[str, Any]) -> bytes:
    payload = json.dumps(
        plan, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return zlib.compress(payload, level=9)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _coprime_multiplier(count: int, seed: int) -> int:
    # A tiny multiplier walks only the beginning of a composition-sorted
    # source index when constructing a pilot.  Derive a large deterministic
    # multiplier so even the first few selected rows span the full split.
    payload = hashlib.sha256(
        f"p11-audio-aware-v1-seed-{int(seed)}".encode("utf-8")
    ).digest()
    candidate = max(1, int.from_bytes(payload[:8], "big") % count)
    while math.gcd(candidate, count) != 1:
        candidate = (candidate + 1) % count
        if candidate == 0:
            candidate = 1
    return candidate


def _slot(source_id: Any) -> int:
    text = str(source_id)
    if text not in {f"source_{index}" for index in range(4)}:
        raise ValueError(f"invalid source id {text!r}")
    return int(text[7:])


def _first_free(plan: Mapping[str, Any]) -> str:
    occupied = {_slot(source["source_id"]) for source in plan["sources"]}
    return next(f"source_{slot}" for slot in range(4) if slot not in occupied)


def generation_prompt(plan: Mapping[str, Any]) -> str:
    caption = str(compile_model_renderer_caption(plan)["text"])
    return (
        f"Create a {float(plan['duration_sec']):.2f}-second FOA spatial-audio "
        f"scene. {caption} Return one complete executable ScenePlan within the "
        "frozen P10 limit of four static/linear sources."
    )


def understanding_prompt() -> str:
    return (
        "Infer the complete observed ScenePlan from the input FOA. Include every "
        "audible source, exact speech, frame-aligned activity, room, and "
        "static/linear trajectory."
    )


def _edit_prompt(instruction: str, *, has_prior: bool) -> str:
    evidence = (
        "A fallible old ScenePlan prior is supplied; verify it against the FOA "
        "and do not copy contradictions. "
        if has_prior
        else "No old ScenePlan is supplied; infer the observed scene from FOA first. "
    )
    return (
        evidence
        + instruction
        + " Return the observed ScenePlan and exactly one atomic patch. The revised "
        "ScenePlan must be the deterministic result of applying that patch to the "
        "observed ScenePlan."
    )


def _position_shift(position: Mapping[str, Any]) -> dict[str, float]:
    return {
        "azimuth_deg": float(((int(round(float(position["azimuth_deg"]))) + 90 + 180) % 360) - 180),
        "elevation_deg": float(max(-90, min(90, int(round(float(position["elevation_deg"]))) + 15))),
        "distance_m": float(position["distance_m"]) * 1.25,
    }


def _corrupt_prior(
    codec: ModelScenePlanCodecV3,
    observed: Mapping[str, Any],
    *,
    position: int,
    corruption_index: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Change only an input copy; observed/revised truth is never touched."""

    output = copy.deepcopy(dict(observed))
    corruption = PRIOR_CORRUPTION_TYPES[int(corruption_index) % len(PRIOR_CORRUPTION_TYPES)]
    sources = output["sources"]
    source = sources[(int(position) * 7 + int(seed)) % len(sources)]
    changed_paths: list[str]

    if corruption == "room":
        rooms = ("dry", "moderate", "reverberant", "outdoor")
        current = str(output["room"]["type"])
        output["room"] = {"type": rooms[(rooms.index(current) + 1) % len(rooms)]}
        changed_paths = ["room.type"]
    elif corruption == "source_count":
        if len(sources) > 1:
            removed = sources.pop(-1)
            changed_paths = [f"sources.{removed['source_id']}"]
        else:
            duration = codec._duration_frame(output["duration_sec"])
            added = {
                "source_id": _first_free(output),
                "kind": "sound",
                "description": "an incorrectly hypothesized quiet click",
                "activity": {"onset_sec": 0.0, "offset_sec": duration * 1024 / 44_100},
                "trajectory": {
                    "type": "static",
                    "position": {"azimuth_deg": 135.0, "elevation_deg": 0.0, "distance_m": 2.0},
                },
                "gain_db": 0.0,
            }
            sources.append(added)
            sources.sort(key=lambda item: _slot(item["source_id"]))
            changed_paths = [f"sources.{added['source_id']}"]
    elif corruption == "description":
        if source["kind"] == "speech":
            source["speaker_description"] = "an incorrect distant speaker"
            changed_paths = [f"sources.{source['source_id']}.speaker_description"]
        else:
            source["description"] = "an incorrect unrelated sound"
            changed_paths = [f"sources.{source['source_id']}.description"]
    elif corruption == "activity":
        duration = codec._duration_frame(output["duration_sec"])
        onset = codec._frame_from_seconds(source["activity"]["onset_sec"], mode="nearest")
        offset = codec._frame_from_seconds(source["activity"]["offset_sec"], mode="nearest")
        if offset < duration:
            onset, offset = onset + 1, offset + 1
        elif onset > 0:
            onset, offset = onset - 1, offset - 1
        elif offset - onset > 1:
            offset -= 1
        source["activity"] = {
            "onset_sec": onset * 1024 / 44_100,
            "offset_sec": offset * 1024 / 44_100,
        }
        changed_paths = [f"sources.{source['source_id']}.activity"]
    else:
        trajectory = source["trajectory"]
        if trajectory["type"] == "static":
            trajectory["position"] = _position_shift(trajectory["position"])
        elif trajectory["type"] == "linear":
            trajectory["start"] = _position_shift(trajectory["start"])
            trajectory["end"] = _position_shift(trajectory["end"])
        else:
            raise ValueError("audio-aware prior corruption forbids keyframed motion")
        changed_paths = [f"sources.{source['source_id']}.trajectory"]

    output = codec.project_plan(output)
    if codec.encode(output)["input_ids"].equal(codec.encode(observed)["input_ids"]):
        raise RuntimeError("prior corruption was a no-op after codec projection")
    return output, {
        "contract": "fallible_old_sceneplan_prior_corruption_v1",
        "type": corruption,
        "changed_paths": changed_paths,
        "supervision_targets_unchanged": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--codec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-samples", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if int(args.seed) != 42:
        raise ValueError("active P11 experiments use the frozen seed 42")
    index = args.index.expanduser().resolve(strict=True)
    codec_path = args.codec.expanduser().resolve(strict=True)
    codec = load_model_sceneplan_codec(codec_path)
    if not isinstance(codec, ModelScenePlanCodecV4):
        raise ValueError("canonical P11 requires the frozen 648-frame codec-v4")
    patch_codec = ScenePlanEditPatchCodec(codec)
    source = sqlite3.connect(f"file:{index}?mode=ro&immutable=1", uri=True)
    total = int(source.execute("SELECT COUNT(*) FROM samples").fetchone()[0])
    selected = total if args.base_samples is None else int(args.base_samples)
    if not 2 <= selected <= total:
        raise ValueError(f"--base-samples must be within [2,{total}]")
    multiplier = _coprime_multiplier(total, args.seed)
    offset = int(args.seed) % total
    ordinals = [(multiplier * value + offset) % total for value in range(selected)]

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not args.overwrite:
        raise FileExistsError(output)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".sqlite", dir=output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    destination = sqlite3.connect(temporary)
    edit_counts: Counter[str] = Counter()
    evidence_counts: Counter[str] = Counter()
    corruption_counts: Counter[str] = Counter()
    retime_mode_counts: Counter[str] = Counter()
    retime_magnitude_counts: Counter[int] = Counter()
    started = time.perf_counter()
    completed = False
    try:
        destination.executescript(
            """
            PRAGMA journal_mode=OFF;
            PRAGMA synchronous=OFF;
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE rows (
                ordinal INTEGER PRIMARY KEY,
                task TEXT NOT NULL,
                source_ordinal INTEGER NOT NULL,
                input_audio_ordinal INTEGER,
                prompt TEXT NOT NULL,
                observed_sceneplan_zlib BLOB,
                input_sceneplan_zlib BLOB,
                target_sceneplan_zlib BLOB NOT NULL,
                editing_evidence_mode TEXT,
                edit_kind TEXT,
                edit_spec_json TEXT,
                prior_corruption_json TEXT
            );
            CREATE INDEX rows_task ON rows(task);
            CREATE INDEX rows_source ON rows(source_ordinal);
            CREATE INDEX rows_edit_kind ON rows(edit_kind);
            CREATE INDEX rows_evidence_mode ON rows(editing_evidence_mode);
            """
        )
        pending: list[tuple[Any, ...]] = []
        corrupt_index = 0
        for position, source_ordinal in enumerate(ordinals):
            record = source.execute(
                "SELECT scene_plan_zlib FROM samples WHERE ordinal = ?",
                (int(source_ordinal),),
            ).fetchone()
            if record is None:
                raise RuntimeError(f"frozen source index lacks ordinal {source_ordinal}")
            base = json.loads(zlib.decompress(record[0]))
            observed = canonicalize_sceneplan_source_ids(codec.project_plan(base))
            eligible = [
                "no_op",
                "replace_source",
                "move_source",
                "retime_source",
                "room_change",
            ]
            if len(observed["sources"]) < 4:
                eligible.append("add_source")
            if len(observed["sources"]) > 1:
                eligible.append("remove_source")
            if any(source["kind"] == "speech" for source in observed["sources"]):
                eligible.extend(["change_speech_description", "change_transcript"])
            operation_order = list(ACTIVE_OPERATION_TOKENS)
            requested_operation = min(
                eligible,
                key=lambda operation: (
                    edit_counts[operation],
                    (operation_order.index(operation) - position) % len(operation_order),
                ),
            )
            revised, instruction, edit_kind, edit_spec = make_deterministic_edit(
                codec,
                observed,
                ordinal=source_ordinal,
                seed=args.seed,
                requested_operation=requested_operation,
            )
            if edit_kind not in ACTIVE_OPERATION_TOKENS:
                raise RuntimeError(f"builder emitted retired operation {edit_kind!r}")
            if edit_kind == "retime_source":
                if edit_spec.get("retime_policy") != RETIME_POLICY:
                    raise RuntimeError("builder emitted an unversioned retime target")
                retime_mode = str(edit_spec.get("retime_mode") or "")
                retime_magnitude = int(edit_spec.get("retime_magnitude_frames", -1))
                if retime_mode not in RETIME_MODES:
                    raise RuntimeError("builder emitted an invalid retime mode")
                if retime_magnitude not in RETIME_FRAME_LEVELS:
                    raise RuntimeError("builder emitted an invalid retime magnitude")
                retime_mode_counts[retime_mode] += 1
                retime_magnitude_counts[retime_magnitude] += 1
            patch_codec.assert_target(observed, edit_spec, revised)
            mode = EDIT_MODE_CYCLE[(position + args.seed) % len(EDIT_MODE_CYCLE)]
            prior = None
            corruption = None
            if mode == "correct_plan":
                prior = observed
            elif mode == "corrupt_plan":
                prior, corruption = _corrupt_prior(
                    codec,
                    observed,
                    position=position,
                    corruption_index=corrupt_index,
                    seed=args.seed,
                )
                corrupt_index += 1
                corruption_counts[str(corruption["type"])] += 1
            edit_counts[edit_kind] += 1
            evidence_counts[mode] += 1
            observed_payload = _compress(observed)
            target_payload = _compress(revised)
            output_ordinal = position * 3
            pending.extend(
                [
                    (
                        output_ordinal,
                        "generation",
                        source_ordinal,
                        None,
                        generation_prompt(observed),
                        None,
                        None,
                        observed_payload,
                        None,
                        None,
                        None,
                        None,
                    ),
                    (
                        output_ordinal + 1,
                        "understanding",
                        source_ordinal,
                        source_ordinal,
                        understanding_prompt(),
                        observed_payload,
                        None,
                        observed_payload,
                        None,
                        None,
                        None,
                        None,
                    ),
                    (
                        output_ordinal + 2,
                        "editing",
                        source_ordinal,
                        source_ordinal,
                        _edit_prompt(instruction, has_prior=prior is not None),
                        observed_payload,
                        None if prior is None else _compress(prior),
                        target_payload,
                        mode,
                        edit_kind,
                        json.dumps(edit_spec, ensure_ascii=False, sort_keys=True),
                        None if corruption is None else json.dumps(corruption, sort_keys=True),
                    ),
                ]
            )
            if len(pending) >= 3000:
                destination.executemany("INSERT INTO rows VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", pending)
                pending.clear()
            if (position + 1) % 50_000 == 0:
                destination.commit()
                print(
                    json.dumps(
                        {
                            "event": "progress",
                            "base_samples": position + 1,
                            "base_samples_total": selected,
                            "elapsed_sec": time.perf_counter() - started,
                        }
                    ),
                    flush=True,
                )
        if pending:
            destination.executemany("INSERT INTO rows VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", pending)

        codec_base_source = REPO_ROOT / "stable_audio_tools/data/model_sceneplan_codec_v3.py"
        codec_source = REPO_ROOT / "stable_audio_tools/data/model_sceneplan_codec_v4.py"
        patch_source = REPO_ROOT / "stable_audio_tools/data/sceneplan_edit_patch.py"
        builder_source = Path(__file__).resolve(strict=True)
        metadata = {
            "schema": P11_MANIFEST_SCHEMA,
            "schema_version": str(P11_MANIFEST_VERSION),
            "p10_sceneplan_contract_revision": "5",
            "p10_conditioning_contract_revision": "2",
            "p10_dataset_contract_revision": "6",
            "planner_max_latent_frames": "648",
            "single_turn_only": "true",
            "max_input_audio_spans": "1",
            "p11_target_audio_spans": "0",
            "target_audio_supervision": "forbidden",
            "execution_contract": "external_p10_sceneplan_executor_v1",
            "model_contract": P11_MODEL_CONTRACT,
            "data_contract": P11_DATA_CONTRACT,
            "codec_fingerprint": codec.fingerprint,
            "codec_implementation_sha256": _sha256_file(codec_source),
            "codec_base_implementation_sha256": _sha256_file(codec_base_source),
            "patch_codec_fingerprint": patch_codec.fingerprint,
            "patch_codec_implementation_sha256": _sha256_file(patch_source),
            "builder_implementation_sha256": _sha256_file(builder_source),
            "source_index": str(index),
            "source_index_rows": str(total),
            "base_samples": str(selected),
            "rows": str(selected * 3),
            "task_generation_rows": str(selected),
            "task_understanding_rows": str(selected),
            "task_editing_rows": str(selected),
            "task_order": "generation,understanding,editing",
            "output_contract": PATCH_OUTPUT_CONTRACT,
            "editing_contract": P11_EDITING_CONTRACT,
            "editing_input_contract": P11_EDITING_INPUT_CONTRACT,
            "editing_output_contract": P11_EDITING_OUTPUT_CONTRACT,
            "editing_input_audio_required": "true",
            "editing_old_sceneplan_role": "optional_fallible_prior",
            "editing_revised_authority": "deterministic_patch_applied_to_observed",
            "edit_evidence_modes": ",".join(P11_EDIT_EVIDENCE_MODES),
            "edit_evidence_distribution": "0.4,0.3,0.3",
            "retime_policy": RETIME_POLICY,
            "retime_frame_levels": ",".join(map(str, RETIME_FRAME_LEVELS)),
            "retime_modes": ",".join(RETIME_MODES),
            "retime_mode_counts": json.dumps(retime_mode_counts, sort_keys=True),
            "retime_magnitude_counts": json.dumps(
                retime_magnitude_counts, sort_keys=True
            ),
            "source_identity_contract": P11_SOURCE_IDENTITY_CONTRACT,
            "generation_prompt_contract": P11_GENERATION_NATURAL_CONTRACT,
            "understanding_audio_bridge": "hybrid_temporal_semantic_v1",
            "editing_audio_bridge": "hybrid_temporal_semantic_v1",
            "target_projection_contract": "p10_latent_grid_v1",
            "base_split_before_edit_augmentation": "true",
            "edit_kind_counts": json.dumps(edit_counts, sort_keys=True),
            "editing_evidence_mode_counts": json.dumps(evidence_counts, sort_keys=True),
            "prior_corruption_counts": json.dumps(corruption_counts, sort_keys=True),
            "seed": str(args.seed),
        }
        destination.executemany(
            "INSERT INTO metadata(key,value) VALUES (?,?)", metadata.items()
        )
        destination.commit()
        destination.execute("VACUUM")
        destination.commit()
        completed = True
    finally:
        destination.close()
        source.close()
        if not completed:
            temporary.unlink(missing_ok=True)
    os.replace(temporary, output)
    print(
        json.dumps(
            {
                "output": str(output),
                "base_samples": selected,
                "rows": selected * 3,
                "edit_kind_counts": edit_counts,
                "editing_evidence_mode_counts": evidence_counts,
                "prior_corruption_counts": corruption_counts,
                "retime_mode_counts": retime_mode_counts,
                "retime_magnitude_counts": retime_magnitude_counts,
                "codec_fingerprint": codec.fingerprint,
                "patch_codec_fingerprint": patch_codec.fingerprint,
                "elapsed_sec": time.perf_counter() - started,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
