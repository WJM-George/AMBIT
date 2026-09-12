#!/usr/bin/env python3
"""Build the canonical matched 10k-scene/30k-row P11-v4 screening overlay.

The frozen owner-balanced manifest supplies one matched G/U/E triplet per
scene.  This overlay keeps that 1:1:1 budget while replacing pilot-specific
27-way oversampling with population-scale, train-only supervision:

* G: exact, missing-layout, and coarse-control prompts are balanced;
* U: identity plus eight deterministic FOA/CLAP degradations are balanced;
* E: all six owner-balanced manifest operations receive train-only clean
  instruction surfaces without leaking resulting target bins.

No core40 sidecar is read.  The runtime compiles canonical SceneSketch and
[5,15] ExecutionState targets directly from the frozen ScenePlan rows.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import time
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.t2a.data.build_sceneplan_p11_v4_challenge import (  # noqa: E402
    P10_CHECKPOINT,
    P10_CHECKPOINT_SHA256,
    _reference_completions,
)
from scripts.t2a.data.build_sceneplan_p11_v4_curriculum import (  # noqa: E402
    _decompress,
    _identity_evidence,
    _row,
    _stable_int,
    _train_evidence_specs,
    _train_generation_prompt,
)
from stable_audio_tools.data.model_sceneplan_codec import (  # noqa: E402
    load_model_sceneplan_codec,
)
from stable_audio_tools.data.sceneplan_edit_patch import (  # noqa: E402
    ScenePlanEditPatchCodec,
)
from stable_audio_tools.data.sceneplan_p11_single_turn import (  # noqa: E402
    canonicalize_sceneplan_source_ids,
    validate_p11_executor_profile,
)
from stable_audio_tools.data.sceneplan_p11_v4_curriculum import (  # noqa: E402
    P11_V4_CURRICULUM_PARTITION,
    P11_V4_CURRICULUM_SCHEMA,
    P11_V4_CURRICULUM_VERSION,
    P11_V4_SCREENING_CONTRACT,
)
from stable_audio_tools.data.sceneplan_p11_v4_dataset import (  # noqa: E402
    P11_V4_DATA_CONTRACT,
    P11_V4_SEQUENCE_CONTRACT,
)


P11_V4_SCREENING_ORDERING = "p11_v4_matched_triplet_interleaved_batch8_v1"
DEFAULT_ROOT = Path(
    os.environ.get("AMBIT_DATA_ROOT", "data") + "/sceneplan_v2_1p124m/p11_single_turn_15s_v2"
)
DEFAULT_MANIFEST = DEFAULT_ROOT / "manifests/p11_train_trial30k_owner_balanced_v1.sqlite"
DEFAULT_INDEX = Path(
    os.environ.get("AMBIT_DATA_ROOT", "data") + "/sceneplan_v2_1p124m/revisions/"
    "speech_expansion_noalign_15s_v1/training_index/train.sqlite"
)
DEFAULT_CODEC = DEFAULT_ROOT / "model_sceneplan_codec_v4"
DEFAULT_OUTPUT = DEFAULT_ROOT / (
    "p11_v4_curriculum/p11_train_trial30k_matched_screening_v1.sqlite"
)
DEFAULT_HELDOUT = REPO_ROOT / (
    "artifacts/sceneplan_p11/challenges/"
    "p11_v4_heldout_challenge_v1_20260901.sqlite"
)


_U_PROMPT_SURFACES = (
    "Infer one complete executable ScenePlan from the supplied FOA evidence. "
    "Use observable content for stable source IDs and recover room, activity, "
    "and P10-supported static or linear trajectories.",
    "Analyze this FOA recording and return its canonical executable ScenePlan. "
    "Identify every audible source, preserve reliable speech wording, and infer "
    "the supported temporal and spatial controls.",
    "Recover the latent scene behind the input FOA as one P10-compatible "
    "ScenePlan. Assign contiguous source IDs by observed content and include all "
    "audible sources, timing, room, and trajectories.",
    "Describe the supplied spatial audio with a complete canonical ScenePlan. "
    "Fuse acoustic, spatial, semantic, and reliable lexical evidence without "
    "inventing unsupported motion types.",
    "Solve the inverse ScenePlan problem for this FOA clip. Return one valid "
    "1-to-4-source plan with the observed semantics and precise executable "
    "activity and geometry.",
    "Map the input FOA evidence back to the structured scene that P10 could "
    "execute. Keep source identity content-based and emit a complete ScenePlan.",
    "Reconstruct a P10-v11-compatible ScenePlan from the provided FOA. Include "
    "room, source kind and wording, activity intervals, and static/linear 3D paths.",
    "Interpret the spatial recording and output exactly one canonical ScenePlan. "
    "Use CLAP/FOA evidence for all sources and trustworthy ASR only as lexical "
    "evidence for speech.",
)


_E_PROMPT_SURFACES = (
    "Use the current ScenePlan as immutable authority. {instruction} Preserve "
    "every unspecified field and emit exactly one atomic patch.",
    "Apply one local edit to the supplied plan: {instruction} Keep all other "
    "source semantics and controls identical; return only the atomic operation.",
    "Execute this single P10-supported change: {instruction} The current source "
    "IDs remain authoritative and no full ScenePlan rewrite is allowed.",
    "Modify only what the instruction names: {instruction} Leave the rest of the "
    "scene untouched and answer with one canonical atomic patch.",
    "Starting from the current ScenePlan, {instruction} Preserve every unrelated "
    "field exactly and output the single executable edit.",
    "Perform one atomic ScenePlan operation: {instruction} Do not change any "
    "unmentioned room, content, timing, or geometry value.",
    "Treat the supplied plan as persistent state and {instruction} Respond with "
    "only one P10-compatible patch while retaining all other state.",
    "Make the requested local change and nothing else: {instruction} Emit one "
    "atomic operation using the plan's persistent source IDs.",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized(value: str) -> str:
    return " ".join(str(value).split())


def _edit_instruction(spec: Mapping[str, Any]) -> str:
    operation = str(spec["operation"])
    source_id = str(spec.get("source_id") or "")
    if operation == "rotate_source":
        delta = int(spec["delta_azimuth_deg"])
        return f"rotate {source_id} by the signed azimuth delta {delta:+d} degrees."
    if operation == "distance_source":
        factor = float(spec["distance_factor"])
        relation = "nearer" if factor < 1.0 else "farther"
        return f"move {source_id} {relation} using the exact distance factor {factor:.2f}."
    if operation == "retime_source":
        onset, offset = [int(value) for value in spec["new_interval_frames"]]
        return f"set {source_id} activity to frame interval [{onset},{offset})."
    if operation == "remove_source":
        return f"remove {source_id} completely from the scene."
    if operation == "room_change":
        return f"change only the acoustic room to {spec['new_room']}."
    if operation == "no_op":
        return "make no change; preserve the complete input plan exactly."
    raise ValueError(f"unsupported screening edit operation {operation!r}")


def _edit_prompt(
    spec: Mapping[str, Any], *, sample_id: str, seed: int
) -> tuple[str, str]:
    operation = str(spec["operation"])
    surface = _stable_int(seed, sample_id, operation, "screen-e-surface") % len(
        _E_PROMPT_SURFACES
    )
    return (
        _E_PROMPT_SURFACES[surface].format(instruction=_edit_instruction(spec)),
        f"train_reserved/e_{operation}/surface_{surface:02d}",
    )


def _readonly(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(
        f"file:{path}?mode=ro&immutable=1", uri=True
    )
    connection.execute("PRAGMA query_only=ON")
    return connection


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--codec", type=Path, default=DEFAULT_CODEC)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--heldout-challenge", type=Path, default=DEFAULT_HELDOUT)
    parser.add_argument("--base-scenes", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.base_scenes <= 0:
        raise ValueError("--base-scenes must be positive")

    manifest_path = args.manifest.expanduser().resolve(strict=True)
    index_path = args.index.expanduser().resolve(strict=True)
    codec_path = args.codec.expanduser().resolve(strict=True)
    challenge_path = args.heldout_challenge.expanduser().resolve(strict=True)
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(output_path)

    codec = load_model_sceneplan_codec(codec_path)
    patch_codec = ScenePlanEditPatchCodec(codec)
    manifest = _readonly(manifest_path)
    source = _readonly(index_path)
    heldout = _readonly(challenge_path)
    manifest_metadata = dict(manifest.execute("SELECT key,value FROM metadata"))
    if manifest_metadata.get("source_index") != str(index_path):
        raise RuntimeError("screening manifest/index provenance mismatch")
    if manifest_metadata.get("p10_checkpoint_sha256") != P10_CHECKPOINT_SHA256:
        raise RuntimeError("screening manifest is not bound to canonical P10-v11")
    total_base = int(manifest_metadata["base_samples"])
    if args.base_scenes > total_base:
        raise ValueError(f"screening requests {args.base_scenes}>{total_base} scenes")

    heldout_ids = {
        str(row[0]) for row in heldout.execute("SELECT DISTINCT sample_id FROM rows")
    }
    heldout_prompts = {
        _normalized(row[0]) for row in heldout.execute("SELECT prompt FROM rows")
    }
    heldout_eval_templates = {
        str(row[0])
        for row in heldout.execute(
            "SELECT DISTINCT template_id FROM rows WHERE template_id LIKE 'eval_reserved/%'"
        )
    }

    rows: list[tuple[Any, ...]] = []
    sample_ids: set[str] = set()
    target_ordinals: set[int] = set()
    counts: Counter[str] = Counter()
    edit_operations: Counter[str] = Counter()
    edit_owners: Counter[str] = Counter()
    edit_directions: Counter[str] = Counter()
    template_ids: set[str] = set()
    prompt_overlap = 0
    started = time.perf_counter()
    try:
        for position in range(args.base_scenes):
            base_rows = manifest.execute(
                """
                SELECT ordinal,task,target_ordinal,prompt,target_sceneplan_zlib,
                       edit_kind,edit_spec_json
                FROM rows WHERE ordinal BETWEEN ? AND ? ORDER BY ordinal
                """,
                (position * 3, position * 3 + 2),
            ).fetchall()
            if len(base_rows) != 3 or [str(row[1]) for row in base_rows] != [
                "generation",
                "understanding",
                "editing",
            ]:
                raise RuntimeError(f"invalid matched base triplet at {position}")
            target_ordinal = int(base_rows[0][2])
            if any(int(row[2]) != target_ordinal for row in base_rows):
                raise RuntimeError("G/U/E target ordinal diverged")
            index_row = source.execute(
                "SELECT sample_id,latent_frames_valid FROM samples WHERE ordinal=?",
                (target_ordinal,),
            ).fetchone()
            if index_row is None:
                raise RuntimeError(f"source index lacks ordinal {target_ordinal}")
            sample_id, valid_frames = str(index_row[0]), int(index_row[1])
            if sample_id in sample_ids or target_ordinal in target_ordinals:
                raise RuntimeError("screening base scene was duplicated")
            if sample_id in heldout_ids:
                raise RuntimeError("held-out sample leaked into screening train")
            sample_ids.add(sample_id)
            target_ordinals.add(target_ordinal)
            current = canonicalize_sceneplan_source_ids(
                codec.project_plan(
                    _decompress(base_rows[0][4]), sample_id=sample_id
                )
            )
            validate_p11_executor_profile(current)
            identity = _identity_evidence()

            # Exact counts, while the frozen source manifest already provides
            # a pseudorandom scene order across content/source-count strata.
            g_mode = ("exact", "numeric_layout", "coarse_numeric")[position % 3]
            if g_mode == "exact":
                g_prompt = str(base_rows[0][3])
                g_template = "manifest_renderer_exact_v0"
                g_known = [
                    "duration",
                    "room",
                    "source_semantics",
                    "temporal",
                    "geometry",
                    "gain",
                ]
                g_target = current
                g_variant = 0
                g_family = "exact_compatibility"
            else:
                surface = _stable_int(
                    args.seed, sample_id, g_mode, "screen-g-surface"
                ) % 3
                g_prompt, g_template, g_known = _train_generation_prompt(
                    current, mode=g_mode, variant=surface
                )
                references = _reference_completions(
                    current,
                    codec=codec,
                    known_groups=g_known,
                    mode=g_mode,
                    seed=_stable_int(
                        args.seed, sample_id, g_mode, "screen-g-targets"
                    ),
                    count=4,
                )
                g_variant = _stable_int(
                    args.seed, sample_id, g_mode, "screen-g-choice"
                ) % len(references)
                g_target = references[g_variant]
                g_family = "generation_population_posterior"
            records = [
                {
                    "curriculum_id": f"{sample_id}/screen/g/{g_mode}",
                    "task": "generation",
                    "family": g_family,
                    "view_id": f"{g_mode}_screen_v1",
                    "template_id": g_template,
                    "base_manifest_ordinal": int(base_rows[0][0]),
                    "base_target_ordinal": target_ordinal,
                    "sample_id": sample_id,
                    "prompt": g_prompt,
                    "known_groups": g_known,
                    "target_plan": g_target,
                    "edit_spec": None,
                    "evidence": identity,
                    "target_variant": int(g_variant),
                    "pair_id": None,
                    "pair_label": None,
                }
            ]

            evidence_options = [identity] + _train_evidence_specs(
                valid_frames,
                seed=_stable_int(args.seed, sample_id, "screen-u-evidence"),
            )
            evidence = evidence_options[position % len(evidence_options)]
            u_surface = _stable_int(
                args.seed, sample_id, "screen-u-surface"
            ) % len(_U_PROMPT_SURFACES)
            records.append(
                {
                    "curriculum_id": (
                        f"{sample_id}/screen/u/{evidence['transform_id']}"
                    ),
                    "task": "understanding",
                    "family": (
                        "understanding_identity"
                        if evidence["transform_id"] == "identity_v1"
                        else "understanding_train_evidence_stress"
                    ),
                    "view_id": str(evidence["transform_id"]),
                    "template_id": (
                        f"train_reserved/u_reconstruction/surface_{u_surface:02d}"
                    ),
                    "base_manifest_ordinal": int(base_rows[1][0]),
                    "base_target_ordinal": target_ordinal,
                    "sample_id": sample_id,
                    "prompt": _U_PROMPT_SURFACES[u_surface],
                    "known_groups": ["foa_evidence"],
                    "target_plan": current,
                    "edit_spec": None,
                    "evidence": evidence,
                    "target_variant": position % len(evidence_options),
                    "pair_id": None,
                    "pair_label": None,
                }
            )

            e_target = codec.project_plan(
                _decompress(base_rows[2][4]), sample_id=sample_id
            )
            e_spec = json.loads(str(base_rows[2][6]))
            patch_codec.assert_target(current, e_spec, e_target)
            e_prompt, e_template = _edit_prompt(
                e_spec, sample_id=sample_id, seed=args.seed
            )
            records.append(
                {
                    "curriculum_id": (
                        f"{sample_id}/screen/e/{e_spec['operation']}"
                    ),
                    "task": "editing",
                    "family": "editing_owner_balanced_atomic",
                    "view_id": f"{e_spec['operation']}_screen_v1",
                    "template_id": e_template,
                    "base_manifest_ordinal": int(base_rows[2][0]),
                    "base_target_ordinal": target_ordinal,
                    "sample_id": sample_id,
                    "prompt": e_prompt,
                    "known_groups": ["input_sceneplan_and_instruction"],
                    "target_plan": e_target,
                    "edit_spec": e_spec,
                    "evidence": identity,
                    "target_variant": 0,
                    "pair_id": None,
                    "pair_label": None,
                }
            )

            for record in records:
                normalized_prompt = _normalized(record["prompt"])
                prompt_overlap += int(normalized_prompt in heldout_prompts)
                template_ids.add(str(record["template_id"]))
                counts[f"task:{record['task']}"] += 1
                counts[f"family:{record['family']}"] += 1
                if record["task"] == "generation":
                    counts[f"g_view:{g_mode}"] += 1
                elif record["task"] == "understanding":
                    counts[f"u_transform:{evidence['transform_id']}"] += 1
                else:
                    operation = str(e_spec["operation"])
                    owner = str(e_spec.get("source_id") or "global")
                    edit_operations[operation] += 1
                    edit_owners[f"{operation}:{owner}"] += 1
                    if operation == "rotate_source":
                        direction = -1 if int(e_spec["delta_azimuth_deg"]) < 0 else 1
                        edit_directions[f"{operation}:{direction:+d}"] += 1
                    elif operation == "distance_source":
                        direction = -1 if float(e_spec["distance_factor"]) < 1.0 else 1
                        edit_directions[f"{operation}:{direction:+d}"] += 1
                rows.append(_row(ordinal=len(rows), **record))
            if (position + 1) % 1000 == 0:
                print(
                    json.dumps(
                        {
                            "event": "progress",
                            "base_scenes": position + 1,
                            "rows": len(rows),
                            "elapsed_seconds": time.perf_counter() - started,
                        }
                    ),
                    flush=True,
                )

        if len(rows) != args.base_scenes * 3:
            raise RuntimeError("screening row count changed")
        if any(counts[f"task:{task}"] != args.base_scenes for task in (
            "generation", "understanding", "editing"
        )):
            raise RuntimeError("screening G/U/E task balance changed")
        if template_ids & heldout_eval_templates:
            raise RuntimeError("eval-reserved template leaked into train overlay")
        if prompt_overlap:
            raise RuntimeError(
                f"{prompt_overlap} exact held-out prompts leaked into train overlay"
            )

        output_metadata = {
            "schema": P11_V4_CURRICULUM_SCHEMA,
            "schema_version": str(P11_V4_CURRICULUM_VERSION),
            "contract": P11_V4_SCREENING_CONTRACT,
            "partition": P11_V4_CURRICULUM_PARTITION,
            "source_manifest": str(manifest_path),
            "source_manifest_sha256": _sha256_file(manifest_path),
            "source_index": str(index_path),
            "source_index_sha256": _sha256_file(index_path),
            "codec_path": str(codec_path),
            "codec_fingerprint": codec.fingerprint,
            "p11_v4_data_contract": P11_V4_DATA_CONTRACT,
            "p11_v4_sequence_contract": P11_V4_SEQUENCE_CONTRACT,
            "p10_release": "p10-sceneplan-dit-v11-step150000",
            "p10_checkpoint": str(P10_CHECKPOINT),
            "p10_checkpoint_sha256": P10_CHECKPOINT_SHA256,
            "p10_max_latent_frames": "648",
            "p10_duration_limit_sec": "15.0465",
            "p10_source_count": "1-4",
            "p10_motion_profile": "static,linear",
            "base_scenes": str(args.base_scenes),
            "rows": str(len(rows)),
            "rows_per_base_scene": "3",
            "rows_per_task_per_base_scene": "1",
            "ordering_contract": P11_V4_SCREENING_ORDERING,
            "ordering_batch_size": "8",
            "drop_last_rows": "0",
            "seed": str(args.seed),
            "g_prompt_mix": json.dumps(
                {
                    key.split(":", 1)[1]: value
                    for key, value in counts.items()
                    if key.startswith("g_view:")
                },
                sort_keys=True,
            ),
            "g_underspecified_target_policy": (
                "one_deterministic_legal_draw_from_four_nonexhaustive_anchors_v1"
            ),
            "u_evidence_mix": json.dumps(
                {
                    key.split(":", 1)[1]: value
                    for key, value in counts.items()
                    if key.startswith("u_transform:")
                },
                sort_keys=True,
            ),
            "e_operation_counts": json.dumps(edit_operations, sort_keys=True),
            "e_operation_owner_counts": json.dumps(edit_owners, sort_keys=True),
            "e_direction_counts": json.dumps(edit_directions, sort_keys=True),
            "e_target_bin_leakage": "forbidden",
            "core40_sidecar_used": "false",
            "runtime_execution_state": "p10_execution_state_core15_v1_[5,15]",
            "template_partition": P11_V4_CURRICULUM_PARTITION,
            "train_template_ids_json": json.dumps(sorted(template_ids)),
            "eval_reserved_templates_present": "false",
            "heldout_challenge": str(challenge_path),
            "heldout_challenge_sha256": _sha256_file(challenge_path),
            "heldout_sample_overlap": "0",
            "heldout_prompt_exact_overlap": "0",
            "builder": str(Path(__file__).resolve()),
            "builder_sha256": _sha256_file(Path(__file__).resolve()),
        }

        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{output_path.name}.", suffix=".sqlite", dir=output_path.parent
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        destination = sqlite3.connect(temporary)
        completed = False
        try:
            destination.executescript(
                """
                PRAGMA journal_mode=OFF;
                PRAGMA synchronous=OFF;
                CREATE TABLE metadata (
                    key TEXT PRIMARY KEY, value TEXT NOT NULL
                ) WITHOUT ROWID;
                CREATE TABLE rows (
                    ordinal INTEGER PRIMARY KEY,
                    curriculum_id TEXT NOT NULL UNIQUE,
                    task TEXT NOT NULL,
                    family TEXT NOT NULL,
                    view_id TEXT NOT NULL,
                    template_id TEXT NOT NULL,
                    base_manifest_ordinal INTEGER NOT NULL,
                    base_target_ordinal INTEGER NOT NULL,
                    sample_id TEXT NOT NULL,
                    prompt TEXT NOT NULL,
                    known_field_groups_json TEXT NOT NULL,
                    target_sceneplan_zlib BLOB NOT NULL,
                    edit_spec_json TEXT,
                    evidence_transform_json TEXT NOT NULL,
                    target_variant INTEGER NOT NULL,
                    pair_id TEXT,
                    pair_label TEXT
                );
                CREATE INDEX rows_task_family ON rows(task,family);
                CREATE INDEX rows_base_scene ON rows(base_target_ordinal);
                CREATE INDEX rows_template ON rows(template_id);
                """
            )
            destination.executemany(
                "INSERT INTO rows VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                rows,
            )
            destination.executemany(
                "INSERT INTO metadata(key,value) VALUES (?,?)",
                sorted(output_metadata.items()),
            )
            destination.commit()
            destination.execute("VACUUM")
            destination.commit()
            if destination.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise RuntimeError("screening SQLite integrity check failed")
            completed = True
        finally:
            destination.close()
            if not completed:
                temporary.unlink(missing_ok=True)
        os.replace(temporary, output_path)
    finally:
        manifest.close()
        source.close()
        heldout.close()

    report = {
        "schema": "stable_audio_tools.p11_v4_screening_build",
        "schema_version": 1,
        "status": "BUILT",
        "contract": P11_V4_SCREENING_CONTRACT,
        "output": str(output_path),
        "output_sha256": _sha256_file(output_path),
        "base_scenes": args.base_scenes,
        "rows": len(rows),
        "counts": dict(sorted(counts.items())),
        "edit_operations": dict(sorted(edit_operations.items())),
        "edit_owners": dict(sorted(edit_owners.items())),
        "edit_directions": dict(sorted(edit_directions.items())),
        "heldout_sample_overlap": 0,
        "heldout_prompt_exact_overlap": 0,
        "eval_reserved_template_overlap": 0,
        "core40_sidecar_used": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    report_path = output_path.with_suffix(".build.json")
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
