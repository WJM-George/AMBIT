#!/usr/bin/env python3
"""Build the immutable P10-v11-aligned P11-v4 falsification challenge."""

from __future__ import annotations

import argparse
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
from typing import Any, Mapping, Sequence

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stable_audio_tools.data.model_sceneplan_codec import (  # noqa: E402
    load_model_sceneplan_codec,
)
from stable_audio_tools.data.scene_sketch_v1 import (  # noqa: E402
    DeltaSceneSketchCodec,
    assemble_sceneplan,
    compile_execution_state,
    compile_scene_sketch,
    execution_state_core,
    execution_state_from_core,
)
from stable_audio_tools.data.sceneplan_edit_patch import (  # noqa: E402
    ScenePlanEditPatchCodec,
)
from stable_audio_tools.data.sceneplan_p11_metrics import (  # noqa: E402
    score_generation_constraints,
)
from stable_audio_tools.data.sceneplan_p11_single_turn import (  # noqa: E402
    canonicalize_sceneplan_source_ids,
    validate_p11_executor_profile,
)
from stable_audio_tools.data.sceneplan_p11_v4_challenge import (  # noqa: E402
    P11_V4_CHALLENGE_CONTRACT,
    P11_V4_CHALLENGE_SCHEMA,
    P11_V4_CHALLENGE_VERSION,
    P11_V4_EVIDENCE_TRANSFORM_CONTRACT,
    P11_V4_REFERENCE_SET_CONTRACT,
)
from stable_audio_tools.data.sceneplan_p11_v4_dataset import (  # noqa: E402
    P11_V4_DATA_CONTRACT,
    P11_V4_SEQUENCE_CONTRACT,
)


DEFAULT_MANIFEST = Path(
    "/mnt/sdb/audio_dataset/sceneplan_v2_1p124m/p11_single_turn_15s_v2/"
    "manifests/p11_validation_heldout900_v6.sqlite"
)
DEFAULT_INDEX = Path(
    "/mnt/sdb/audio_dataset/sceneplan_v2_1p124m/revisions/"
    "speech_expansion_noalign_15s_v1/training_index/validation.sqlite"
)
DEFAULT_CODEC = Path(
    "/mnt/sdb/audio_dataset/sceneplan_v2_1p124m/p11_single_turn_15s_v2/"
    "model_sceneplan_codec_v4"
)
DEFAULT_OUTPUT = REPO_ROOT / (
    "artifacts/sceneplan_p11/challenges/"
    "p11_v4_heldout_challenge_v1_20260901.sqlite"
)
P10_CHECKPOINT = Path(
    "/mnt/sdc/ckpts/dit/sceneplan_dit_v11_semantic_v2_protected_resume_150k/"
    "checkpoints/epoch=48-step=150000.ckpt"
)
P10_CHECKPOINT_SHA256 = (
    "be8c90cd1434bd71f73951531175c3674ff0f3173d5db591e2e1c476152ff59e"
)

_FEATURES = {
    name: index
    for index, name in enumerate(
        (
            "duration_frames_norm",
            "onset_frame_norm",
            "offset_frame_norm",
            "motion_static",
            "motion_linear",
            "start_sin_azimuth",
            "start_cos_azimuth",
            "start_sin_elevation",
            "start_cos_elevation",
            "start_log_distance_norm",
            "end_sin_azimuth",
            "end_cos_azimuth",
            "end_sin_elevation",
            "end_cos_elevation",
            "end_log_distance_norm",
        )
    )
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_int(*values: Any) -> int:
    payload = json.dumps(
        values, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _compress(value: Any) -> bytes:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return zlib.compress(payload, level=9)


def _decompress(payload: bytes) -> Any:
    return json.loads(zlib.decompress(payload))


def _json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _source_words(source: Mapping[str, Any]) -> str:
    if str(source["kind"]) == "speech":
        return (
            f"speech by {source['speaker_description']} saying "
            f"\"{source['transcript']}\""
        )
    return f"{source['kind']}: {source['description']}"


def _octant(azimuth_deg: float) -> str:
    labels = (
        "front",
        "front-left",
        "left",
        "rear-left",
        "rear",
        "rear-right",
        "right",
        "front-right",
    )
    normalized = (float(azimuth_deg) + 360.0) % 360.0
    return labels[int(((normalized + 22.5) % 360.0) // 45.0)]


def _distance_band(distance_m: float) -> str:
    value = float(distance_m)
    return "near" if value < 1.2 else "mid-distance" if value < 2.0 else "far"


def _trajectory_words(source: Mapping[str, Any]) -> str:
    trajectory = source["trajectory"]
    if str(trajectory["type"]) == "static":
        position = trajectory["position"]
        return (
            f"stationary {_octant(position['azimuth_deg'])}, "
            f"{_distance_band(position['distance_m'])}"
        )
    start, end = trajectory["start"], trajectory["end"]
    return (
        f"moving {_octant(start['azimuth_deg'])} to {_octant(end['azimuth_deg'])}, "
        f"{_distance_band(start['distance_m'])} to "
        f"{_distance_band(end['distance_m'])}"
    )


def _coarse_activity(source: Mapping[str, Any], duration: float) -> str:
    activity = source["activity"]
    onset = float(activity["onset_sec"]) / duration
    offset = float(activity["offset_sec"]) / duration
    if onset < 0.05 and offset > 0.95:
        return "throughout"
    if offset <= 0.5:
        return "early"
    if onset >= 0.5:
        return "late"
    return "middle"


def _generation_prompt(
    plan: Mapping[str, Any], *, view: str, template_variant: int
) -> tuple[str, str, list[str]]:
    duration = float(plan["duration_sec"])
    room = str(plan["room"]["type"])
    if view == "numeric_layout":
        clauses = []
        for source in plan["sources"]:
            activity = source["activity"]
            clauses.append(
                f"{_source_words(source)}, active {float(activity['onset_sec']):.2f}-"
                f"{float(activity['offset_sec']):.2f} seconds"
            )
        bodies = (
            "Choose any plausible static or linear 3D layout supported by the "
            "renderer; exact positions and distances are intentionally unspecified.",
            "Complete the missing P10-supported static/linear positions and "
            "distances plausibly; there is deliberately more than one valid layout.",
        )
        prompt = (
            f"Create a {duration:.2f}-second FOA scene in a {room} environment. "
            + "; ".join(clauses)
            + ". "
            + bodies[template_variant % len(bodies)]
            + " Return one complete executable ScenePlan with canonical source IDs."
        )
        return (
            prompt,
            f"eval_reserved/g_numeric_layout/{template_variant % len(bodies):02d}",
            ["duration", "room", "source_semantics", "temporal"],
        )
    if view != "coarse_numeric":
        raise ValueError(view)
    clauses = [
        f"{_source_words(source)}, {_coarse_activity(source, duration)}, "
        f"{_trajectory_words(source)}"
        for source in plan["sources"]
    ]
    bodies = (
        "Resolve only the exact frame boundaries and within-bin geometry.",
        "Choose precise values consistent with those coarse timing and spatial bins.",
    )
    prompt = (
        f"Create a {duration:.2f}-second FOA scene in a {room} environment. "
        + "; ".join(clauses)
        + ". "
        + bodies[template_variant % len(bodies)]
        + " Return one complete executable ScenePlan with canonical source IDs."
    )
    return (
        prompt,
        f"eval_reserved/g_coarse_numeric/{template_variant % len(bodies):02d}",
        [
            "duration",
            "room",
            "source_semantics",
            "coarse_temporal",
            "coarse_geometry",
        ],
    )


def _write_angle(vector: np.ndarray, prefix: str, axis: str, degrees: float) -> None:
    radians = math.radians(float(degrees))
    vector[_FEATURES[f"{prefix}_sin_{axis}"]] = math.sin(radians)
    vector[_FEATURES[f"{prefix}_cos_{axis}"]] = math.cos(radians)


def _reference_completions(
    plan: Mapping[str, Any],
    *,
    codec: Any,
    known_groups: Sequence[str],
    mode: str,
    seed: int,
    count: int = 8,
) -> list[dict[str, Any]]:
    """Create deterministic, non-exhaustive legal anchors for numeric ambiguity."""

    target = validate_p11_executor_profile(codec.project_plan(plan))
    sketch = compile_scene_sketch(target, codec)
    target_core = execution_state_core(compile_execution_state(target, codec))
    duration_frames = codec._duration_frame(float(target["duration_sec"]))
    references = [target]
    hashes = {_json_sha256(target)}
    rng = np.random.default_rng(int(seed) % (2**63))
    for _ in range(4096):
        core = target_core.copy()
        for source_index, source in enumerate(sketch["sources"]):
            slot = int(str(source["source_id"]).rsplit("_", 1)[1]) + 1
            vector = core[slot]
            if mode == "numeric_layout":
                linear = bool(rng.integers(0, 2))
                vector[_FEATURES["motion_static"]] = float(not linear)
                vector[_FEATURES["motion_linear"]] = float(linear)
                for prefix in ("start", "end"):
                    _write_angle(
                        vector, prefix, "azimuth", float(rng.uniform(-180.0, 180.0))
                    )
                    _write_angle(
                        vector, prefix, "elevation", float(rng.uniform(-30.0, 30.0))
                    )
                    vector[_FEATURES[f"{prefix}_log_distance_norm"]] = float(
                        rng.uniform(0.25, 0.65)
                    )
                if not linear:
                    for suffix in (
                        "sin_azimuth",
                        "cos_azimuth",
                        "sin_elevation",
                        "cos_elevation",
                        "log_distance_norm",
                    ):
                        vector[_FEATURES[f"end_{suffix}"]] = vector[
                            _FEATURES[f"start_{suffix}"]
                        ]
            elif mode == "coarse_numeric":
                # Small continuous changes stay inside most coarse bins; the
                # frozen constraint scorer below rejects boundary crossings.
                for prefix in ("start", "end"):
                    azimuth = math.degrees(
                        math.atan2(
                            float(vector[_FEATURES[f"{prefix}_sin_azimuth"]]),
                            float(vector[_FEATURES[f"{prefix}_cos_azimuth"]]),
                        )
                    )
                    elevation = math.degrees(
                        math.atan2(
                            float(vector[_FEATURES[f"{prefix}_sin_elevation"]]),
                            float(vector[_FEATURES[f"{prefix}_cos_elevation"]]),
                        )
                    )
                    _write_angle(
                        vector,
                        prefix,
                        "azimuth",
                        azimuth + float(rng.uniform(-8.0, 8.0)),
                    )
                    _write_angle(
                        vector,
                        prefix,
                        "elevation",
                        elevation + float(rng.uniform(-5.0, 5.0)),
                    )
                    distance_index = _FEATURES[f"{prefix}_log_distance_norm"]
                    vector[distance_index] = float(
                        np.clip(
                            vector[distance_index] + rng.uniform(-0.015, 0.015),
                            0.0,
                            1.0,
                        )
                    )
            else:
                raise ValueError(mode)
        state = execution_state_from_core(
            core,
            sketch,
            codec,
            sample_id=str(target["sample_id"]),
            duration_frames=duration_frames,
        )
        candidate = assemble_sceneplan(sketch, state, codec)
        digest = _json_sha256(candidate)
        if digest in hashes:
            continue
        score = score_generation_constraints(
            target,
            candidate,
            known_field_groups=known_groups,
            source_matching="permutation_invariant",
        )
        if not math.isclose(float(score["task_score"]), 1.0, abs_tol=1.0e-12):
            continue
        references.append(candidate)
        hashes.add(digest)
        if len(references) == count:
            return references
    raise RuntimeError(
        f"could not construct {count} legal {mode} references for {target['sample_id']}"
    )


def _identity_evidence() -> dict[str, Any]:
    return {
        "contract": P11_V4_EVIDENCE_TRANSFORM_CONTRACT,
        "transform_id": "identity_v1",
        "foa_time_mask": None,
        "semantic_channel_mask": None,
    }


def _u_evidence_specs(valid_frames: int, *, seed: int) -> list[dict[str, Any]]:
    def time_mask(ratio: float, salt: str) -> list[int]:
        width = max(1, min(valid_frames, int(round(valid_frames * ratio))))
        start = _stable_int(seed, salt) % (valid_frames - width + 1)
        return [int(start), int(start + width)]

    def channel_mask(width: int, salt: str) -> list[int]:
        start = _stable_int(seed, salt) % (512 - width + 1)
        return [int(start), int(start + width)]

    return [
        {
            "contract": P11_V4_EVIDENCE_TRANSFORM_CONTRACT,
            "transform_id": "foa_time_mask_12p5_v1",
            "foa_time_mask": time_mask(0.125, "foa12p5"),
            "semantic_channel_mask": None,
        },
        {
            "contract": P11_V4_EVIDENCE_TRANSFORM_CONTRACT,
            "transform_id": "clap_channel_mask_25p0_v1",
            "foa_time_mask": None,
            "semantic_channel_mask": channel_mask(128, "clap25"),
        },
        {
            "contract": P11_V4_EVIDENCE_TRANSFORM_CONTRACT,
            "transform_id": "combined_foa25_clap50_v1",
            "foa_time_mask": time_mask(0.25, "foa25"),
            "semantic_channel_mask": channel_mask(256, "clap50"),
        },
    ]


def _counterfactual_pair(
    current: Mapping[str, Any],
    *,
    codec: Any,
    patch_codec: ScenePlanEditPatchCodec,
    delta_codec: DeltaSceneSketchCodec,
    seed: int,
) -> tuple[str, str, list[tuple[str, str, dict[str, Any], dict[str, Any]]]]:
    sources = list(current["sources"])
    source = sources[_stable_int(seed, current["sample_id"], "source") % len(sources)]
    source_id = str(source["source_id"])

    operation = "distance_source" if _stable_int(seed, current["sample_id"]) % 2 else "rotate_source"
    if operation == "distance_source":
        choices: list[tuple[str, Any, str]] = [
            ("nearer", 0.75, "Scale its distance by exactly 0.75."),
            ("farther", 1.25, "Scale its distance by exactly 1.25."),
        ]
        specs = [
            {
                "operation": operation,
                "source_id": source_id,
                "distance_factor": float(value),
            }
            for _, value, _ in choices
        ]
    else:
        choices = [
            ("minus_45", -45, "Rotate it by exactly -45 degrees in azimuth."),
            ("plus_45", 45, "Rotate it by exactly +45 degrees in azimuth."),
        ]
        specs = [
            {
                "operation": operation,
                "source_id": source_id,
                "delta_azimuth_deg": int(value),
            }
            for _, value, _ in choices
        ]

    targets = [patch_codec.apply(current, spec) for spec in specs]
    encoded = [codec.encode(target)["input_ids"] for target in targets]
    current_ids = codec.encode(current)["input_ids"]
    if torch.equal(encoded[0], encoded[1]) or any(
        torch.equal(value, current_ids) for value in encoded
    ):
        # Distance quantization can collapse at a hard boundary. Rotation is
        # total over the circular azimuth domain and is the fail-closed fallback.
        operation = "rotate_source"
        choices = [
            ("minus_45", -45, "Rotate it by exactly -45 degrees in azimuth."),
            ("plus_45", 45, "Rotate it by exactly +45 degrees in azimuth."),
        ]
        specs = [
            {
                "operation": operation,
                "source_id": source_id,
                "delta_azimuth_deg": int(value),
            }
            for _, value, _ in choices
        ]
        targets = [patch_codec.apply(current, spec) for spec in specs]

    outputs = []
    delta_token_ids = []
    delta_programs = []
    pair_id = f"e_cf_{_json_sha256([current['sample_id'], operation, source_id])[:20]}"
    for (label, _, instruction), raw_spec, target in zip(choices, specs, targets):
        changed = (
            f"sources.{source_id}.trajectory.distance"
            if operation == "distance_source"
            else f"sources.{source_id}.trajectory.azimuth"
        )
        spec = {
            **raw_spec,
            "contract": "same_scene_atomic_patch_v1",
            "changed_paths": [changed],
            "preserve_all_unspecified_fields": True,
            "input_sceneplan_tokens_required": True,
            "input_foa_required": False,
        }
        patch_codec.assert_target(current, spec, target)
        from stable_audio_tools.data.scene_sketch_v1 import compile_delta_scene_sketch

        delta = compile_delta_scene_sketch(current, target, spec, codec)
        delta_token_ids.append(delta_codec.encode(delta, spec)["input_ids"])
        delta_programs.append(delta_codec.decode(delta_token_ids[-1]))
        prompt = (
            "Read the supplied current ScenePlan as the persistent source-ID authority. "
            f"For {source_id}, {instruction} Change no other field. Return exactly "
            "one atomic edit program; do not rewrite the ScenePlan."
        )
        outputs.append((str(label), prompt, spec, target))
    if torch.equal(delta_token_ids[0], delta_token_ids[1]):
        raise RuntimeError("counterfactual pair collapsed its DeltaSketch direction")
    if any(
        program.get("operation") != operation
        or program.get("source_id") != source_id
        for program in delta_programs
    ):
        raise RuntimeError("counterfactual pair changed operation/source authority")
    if [int(program.get("control_direction", 0)) for program in delta_programs] != [
        -1,
        1,
    ]:
        raise RuntimeError("counterfactual pair does not cover both legal directions")
    return pair_id, operation, outputs


def _row(
    *,
    ordinal: int,
    challenge_id: str,
    task: str,
    family: str,
    view_id: str,
    template_id: str,
    base_manifest_ordinal: int,
    base_target_ordinal: int,
    sample_id: str,
    prompt: str,
    known_groups: Sequence[str],
    target_plan: Mapping[str, Any],
    edit_spec: Mapping[str, Any] | None,
    evidence: Mapping[str, Any],
    references: Sequence[Mapping[str, Any]],
    reference_contract: str,
    pair_id: str | None,
    pair_label: str | None,
    selection_role: str,
) -> tuple[Any, ...]:
    return (
        ordinal,
        challenge_id,
        task,
        family,
        view_id,
        template_id,
        base_manifest_ordinal,
        base_target_ordinal,
        sample_id,
        prompt,
        json.dumps(list(known_groups), separators=(",", ":")),
        _compress(target_plan),
        None
        if edit_spec is None
        else json.dumps(edit_spec, ensure_ascii=False, sort_keys=True),
        json.dumps(evidence, sort_keys=True, separators=(",", ":")),
        _compress(list(references)),
        reference_contract,
        pair_id,
        pair_label,
        selection_role,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--codec", type=Path, default=DEFAULT_CODEC)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--base-scenes", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.base_scenes <= 0:
        raise ValueError("--base-scenes must be positive")
    manifest_path = args.manifest.expanduser().resolve(strict=True)
    index_path = args.index.expanduser().resolve(strict=True)
    codec_path = args.codec.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    codec = load_model_sceneplan_codec(codec_path)
    patch_codec = ScenePlanEditPatchCodec(codec)
    delta_codec = DeltaSceneSketchCodec(codec, patch_codec)
    manifest = sqlite3.connect(
        f"file:{manifest_path}?mode=ro&immutable=1", uri=True
    )
    source = sqlite3.connect(f"file:{index_path}?mode=ro&immutable=1", uri=True)
    manifest_metadata = dict(manifest.execute("SELECT key,value FROM metadata"))
    total_base = int(manifest_metadata["base_samples"])
    if args.base_scenes > total_base:
        raise ValueError(f"challenge requests {args.base_scenes}>{total_base} base scenes")
    if manifest_metadata.get("source_index") != str(index_path):
        raise RuntimeError("challenge manifest/index provenance mismatch")

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".sqlite", dir=output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    destination = sqlite3.connect(temporary)
    completed = False
    started = time.perf_counter()
    counts: Counter[str] = Counter()
    template_ids: set[str] = set()
    pending: list[tuple[Any, ...]] = []
    output_ordinal = 0
    try:
        destination.executescript(
            """
            PRAGMA journal_mode=OFF;
            PRAGMA synchronous=OFF;
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL) WITHOUT ROWID;
            CREATE TABLE rows (
                ordinal INTEGER PRIMARY KEY,
                challenge_id TEXT NOT NULL UNIQUE,
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
                reference_sceneplans_zlib BLOB NOT NULL,
                reference_set_contract TEXT NOT NULL,
                pair_id TEXT,
                pair_label TEXT,
                selection_role TEXT NOT NULL
            );
            CREATE INDEX rows_task_family ON rows(task,family);
            CREATE INDEX rows_base_scene ON rows(base_target_ordinal);
            CREATE INDEX rows_pair ON rows(pair_id);
            CREATE INDEX rows_selection ON rows(selection_role);
            """
        )
        for position in range(args.base_scenes):
            base_rows = manifest.execute(
                """
                SELECT ordinal,task,target_ordinal,prompt,target_sceneplan_zlib,
                       edit_kind,edit_spec_json
                FROM rows WHERE ordinal BETWEEN ? AND ? ORDER BY ordinal
                """,
                (position * 3, position * 3 + 2),
            ).fetchall()
            if len(base_rows) != 3 or [row[1] for row in base_rows] != [
                "generation",
                "understanding",
                "editing",
            ]:
                raise RuntimeError(f"base heldout triplet {position} is invalid")
            target_ordinal = int(base_rows[0][2])
            if any(int(row[2]) != target_ordinal for row in base_rows):
                raise RuntimeError("G/U/E base target ordinal diverged")
            index_row = source.execute(
                "SELECT sample_id,latent_frames_valid FROM samples WHERE ordinal=?",
                (target_ordinal,),
            ).fetchone()
            if index_row is None:
                raise RuntimeError(f"source index lacks ordinal {target_ordinal}")
            sample_id, valid_frames = str(index_row[0]), int(index_row[1])
            current = canonicalize_sceneplan_source_ids(
                codec.project_plan(_decompress(base_rows[0][4]), sample_id=sample_id)
            )
            validate_p11_executor_profile(current)
            if str(current["sample_id"]) != sample_id:
                raise RuntimeError("challenge sample ID drifted")
            identity = _identity_evidence()

            def add(**kwargs: Any) -> None:
                nonlocal output_ordinal
                record = _row(ordinal=output_ordinal, **kwargs)
                pending.append(record)
                counts[f"{kwargs['task']}:{kwargs['family']}"] += 1
                template_ids.add(str(kwargs["template_id"]))
                output_ordinal += 1

            add(
                challenge_id=f"{sample_id}/g/exact",
                task="generation",
                family="exact_compatibility",
                view_id="exact_v1",
                template_id="manifest_renderer_exact_v0",
                base_manifest_ordinal=int(base_rows[0][0]),
                base_target_ordinal=target_ordinal,
                sample_id=sample_id,
                prompt=str(base_rows[0][3]),
                known_groups=[
                    "duration",
                    "room",
                    "source_semantics",
                    "temporal",
                    "geometry",
                    "gain",
                ],
                target_plan=current,
                edit_spec=None,
                evidence=identity,
                references=[current],
                reference_contract="exact_single_target_v1",
                pair_id=None,
                pair_label=None,
                selection_role="compatibility_only",
            )
            for mode in ("numeric_layout", "coarse_numeric"):
                variant = _stable_int(args.seed, sample_id, mode) % 2
                prompt, template_id, groups = _generation_prompt(
                    current, view=mode, template_variant=variant
                )
                references = _reference_completions(
                    current,
                    codec=codec,
                    known_groups=groups,
                    mode=mode,
                    seed=_stable_int(args.seed, sample_id, mode, "references"),
                )
                add(
                    challenge_id=f"{sample_id}/g/{mode}",
                    task="generation",
                    family="generation_numeric_posterior",
                    view_id=f"{mode}_v1",
                    template_id=template_id,
                    base_manifest_ordinal=int(base_rows[0][0]),
                    base_target_ordinal=target_ordinal,
                    sample_id=sample_id,
                    prompt=prompt,
                    known_groups=groups,
                    target_plan=current,
                    edit_spec=None,
                    evidence=identity,
                    references=references,
                    reference_contract=P11_V4_REFERENCE_SET_CONTRACT,
                    pair_id=None,
                    pair_label=None,
                    selection_role="model_selection",
                )

            u_prompt = str(base_rows[1][3])
            add(
                challenge_id=f"{sample_id}/u/exact",
                task="understanding",
                family="exact_compatibility",
                view_id="audio_evidence_exact_v1",
                template_id="manifest_understanding_exact_v0",
                base_manifest_ordinal=int(base_rows[1][0]),
                base_target_ordinal=target_ordinal,
                sample_id=sample_id,
                prompt=u_prompt,
                known_groups=["foa_evidence"],
                target_plan=current,
                edit_spec=None,
                evidence=identity,
                references=[current],
                reference_contract="exact_single_target_v1",
                pair_id=None,
                pair_label=None,
                selection_role="compatibility_only",
            )
            for evidence in _u_evidence_specs(
                valid_frames, seed=_stable_int(args.seed, sample_id, "u")
            ):
                transform_id = str(evidence["transform_id"])
                add(
                    challenge_id=f"{sample_id}/u/{transform_id}",
                    task="understanding",
                    family="understanding_degraded_evidence",
                    view_id=transform_id,
                    template_id="manifest_understanding_exact_v0",
                    base_manifest_ordinal=int(base_rows[1][0]),
                    base_target_ordinal=target_ordinal,
                    sample_id=sample_id,
                    prompt=u_prompt,
                    known_groups=["foa_evidence"],
                    target_plan=current,
                    edit_spec=None,
                    evidence=evidence,
                    references=[current],
                    reference_contract="exact_scene_robustness_target_v1",
                    pair_id=None,
                    pair_label=None,
                    selection_role="robustness_diagnostic",
                )

            exact_e_target = codec.project_plan(
                _decompress(base_rows[2][4]), sample_id=sample_id
            )
            exact_e_spec = json.loads(str(base_rows[2][6]))
            patch_codec.assert_target(current, exact_e_spec, exact_e_target)
            add(
                challenge_id=f"{sample_id}/e/exact",
                task="editing",
                family="exact_compatibility",
                view_id="atomic_instruction_exact_v1",
                template_id="manifest_editing_exact_v0",
                base_manifest_ordinal=int(base_rows[2][0]),
                base_target_ordinal=target_ordinal,
                sample_id=sample_id,
                prompt=str(base_rows[2][3]),
                known_groups=["input_sceneplan_and_instruction"],
                target_plan=exact_e_target,
                edit_spec=exact_e_spec,
                evidence=identity,
                references=[exact_e_target],
                reference_contract="exact_single_target_v1",
                pair_id=None,
                pair_label=None,
                selection_role="compatibility_only",
            )
            pair_id, operation, pair = _counterfactual_pair(
                current,
                codec=codec,
                patch_codec=patch_codec,
                delta_codec=delta_codec,
                seed=args.seed,
            )
            for label, prompt, spec, target in pair:
                add(
                    challenge_id=f"{sample_id}/e/{pair_id}/{label}",
                    task="editing",
                    family="editing_counterfactual_causality",
                    view_id=f"{operation}_{label}_v1",
                    template_id=f"eval_reserved/e_{operation}/{label}",
                    base_manifest_ordinal=int(base_rows[2][0]),
                    base_target_ordinal=target_ordinal,
                    sample_id=sample_id,
                    prompt=prompt,
                    known_groups=["input_sceneplan_and_instruction"],
                    target_plan=target,
                    edit_spec=spec,
                    evidence=identity,
                    references=[target],
                    reference_contract="exact_counterfactual_target_v1",
                    pair_id=pair_id,
                    pair_label=label,
                    selection_role="model_selection",
                )
            if len(pending) >= 1000:
                destination.executemany(
                    "INSERT INTO rows VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    pending,
                )
                pending.clear()
        if pending:
            destination.executemany(
                "INSERT INTO rows VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                pending,
            )

        source_metadata = dict(source.execute("SELECT key,value FROM metadata"))
        metadata = {
            "schema": P11_V4_CHALLENGE_SCHEMA,
            "schema_version": str(P11_V4_CHALLENGE_VERSION),
            "contract": P11_V4_CHALLENGE_CONTRACT,
            "source_manifest": str(manifest_path),
            "source_manifest_sha256": _sha256_file(manifest_path),
            "source_index": str(index_path),
            "source_index_sha256": _sha256_file(index_path),
            "source_index_split": str(source_metadata.get("split")),
            "source_index_contract_revision": str(
                source_metadata.get("contract_revision")
            ),
            "codec_path": str(codec_path),
            "codec_fingerprint": codec.fingerprint,
            "p11_v4_data_contract": P11_V4_DATA_CONTRACT,
            "p11_v4_sequence_contract": P11_V4_SEQUENCE_CONTRACT,
            "p10_release": "p10-sceneplan-dit-v11-step150000",
            "p10_checkpoint": str(P10_CHECKPOINT),
            "p10_checkpoint_sha256": P10_CHECKPOINT_SHA256,
            "p10_nominal_max_duration_seconds": "15",
            "p10_max_grid_seconds": f"{648 * 1024 / 44_100:.9f}",
            "p10_max_latent_frames": "648",
            "p10_source_count": "1-4",
            "p10_motion_profile": "static,linear",
            "base_scenes": str(args.base_scenes),
            "rows": str(output_ordinal),
            "rows_per_base_scene": "10",
            "row_counts_json": json.dumps(counts, sort_keys=True),
            "seed": str(args.seed),
            "generation_reference_contract": P11_V4_REFERENCE_SET_CONTRACT,
            "generation_reference_sets_are_exhaustive": "false",
            "underspecified_hidden_exact_target_model_selection": "forbidden",
            "u_degradation_scope": "synthetic_latent_and_clap_representation_stress_only",
            "u_degradation_is_real_acoustic_benchmark": "false",
            "challenge_template_partition": "heldout_eval_reserved_v1",
            "exact_compatibility_reuses_manifest_surface": "true",
            "exact_compatibility_is_template_generalization_gate": "false",
            "eval_reserved_template_ids_json": json.dumps(
                sorted(value for value in template_ids if value.startswith("eval_reserved/"))
            ),
            "eval_reserved_templates_may_enter_training": "false",
            "builder": str(Path(__file__).resolve()),
            "builder_sha256": _sha256_file(Path(__file__).resolve()),
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
        manifest.close()
        source.close()
        if not completed:
            temporary.unlink(missing_ok=True)
    os.replace(temporary, output)
    report = {
        "status": "BUILT",
        "schema": P11_V4_CHALLENGE_SCHEMA,
        "schema_version": P11_V4_CHALLENGE_VERSION,
        "contract": P11_V4_CHALLENGE_CONTRACT,
        "output": str(output),
        "output_sha256": _sha256_file(output),
        "base_scenes": args.base_scenes,
        "rows": output_ordinal,
        "counts": dict(sorted(counts.items())),
        "elapsed_seconds": time.perf_counter() - started,
    }
    report_path = output.with_suffix(".build.json")
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
