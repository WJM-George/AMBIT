#!/usr/bin/env python3
"""Build a train-only, multi-target P11-v4 curriculum overlay."""

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

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.t2a.data.build_sceneplan_p11_v4_challenge import (  # noqa: E402
    P10_CHECKPOINT,
    P10_CHECKPOINT_SHA256,
    _coarse_activity,
    _reference_completions,
    _source_words,
    _trajectory_words,
)
from stable_audio_tools.data.model_sceneplan_codec import (  # noqa: E402
    load_model_sceneplan_codec,
)
from stable_audio_tools.data.scene_sketch_v1 import (  # noqa: E402
    DeltaSceneSketchCodec,
    compile_delta_scene_sketch,
)
from stable_audio_tools.data.sceneplan_edit_patch import (  # noqa: E402
    ScenePlanEditPatchCodec,
)
from stable_audio_tools.data.sceneplan_p11_single_turn import (  # noqa: E402
    canonicalize_sceneplan_source_ids,
    validate_p11_executor_profile,
)
from stable_audio_tools.data.sceneplan_p11_v4_challenge import (  # noqa: E402
    P11_V4_EVIDENCE_TRANSFORM_CONTRACT,
)
from stable_audio_tools.data.sceneplan_p11_v4_curriculum import (  # noqa: E402
    P11_V4_CURRICULUM_CONTRACT,
    P11_V4_CURRICULUM_PARTITION,
    P11_V4_CURRICULUM_SCHEMA,
    P11_V4_CURRICULUM_VERSION,
)
from stable_audio_tools.data.sceneplan_p11_v4_dataset import (  # noqa: E402
    P11_V4_DATA_CONTRACT,
    P11_V4_SEQUENCE_CONTRACT,
)


DEFAULT_MANIFEST = Path(
    "/mnt/sdb/audio_dataset/sceneplan_v2_1p124m/p11_single_turn_15s_v2/"
    "manifests/p11_train_pilot90_v6.sqlite"
)
DEFAULT_INDEX = Path(
    "/mnt/sdb/audio_dataset/sceneplan_v2_1p124m/revisions/"
    "speech_expansion_noalign_15s_v1/training_index/train.sqlite"
)
DEFAULT_CODEC = Path(
    "/mnt/sdb/audio_dataset/sceneplan_v2_1p124m/p11_single_turn_15s_v2/"
    "model_sceneplan_codec_v4"
)
DEFAULT_OUTPUT = Path(
    "/mnt/sdb/audio_dataset/sceneplan_v2_1p124m/p11_single_turn_15s_v2/"
    "p11_v4_curriculum/"
    "p11_train_pilot90_curriculum_v7_seed42.sqlite"
)
DEFAULT_HELDOUT_CHALLENGE = REPO_ROOT / (
    "artifacts/sceneplan_p11/challenges/"
    "p11_v4_heldout_challenge_v1_20260901.sqlite"
)
EDITING_INSTRUCTION_SURFACE_CONTRACT = (
    "paired_train_only_multisurface_semantic_direction_exact_p10_numeric_v3"
)

_EDIT_PROMPT_WRAPPERS = (
    (
        "Treat the supplied ScenePlan as immutable except for one numeric control. "
        "On {source_id}, {instruction}. Keep every unmentioned field equivalent "
        "and emit one atomic edit operation."
    ),
    (
        "Using the current plan's persistent IDs, apply this single change to "
        "{source_id}: {instruction}. Preserve all remaining content, timing, and "
        "geometry; answer with only the atomic operation."
    ),
    (
        "Make one P10-legal numeric edit for {source_id}. {instruction}. The current "
        "ScenePlan is authoritative; preserve every other field and emit only the "
        "atomic edit."
    ),
    (
        "Apply this local control to {source_id}: {instruction}. Do not alter room, "
        "content, timing, or any other trajectory coordinate; return one atomic patch."
    ),
    (
        "Keep the plan fixed except at {source_id}. {instruction}. Use the persistent "
        "source ID and output the single executable edit operation."
    ),
    (
        "For {source_id} alone, {instruction}. All unspecified ScenePlan values must "
        "remain identical; respond with one canonical atomic edit."
    ),
    (
        "Execute exactly one numeric command on {source_id}: {instruction}. Preserve "
        "the rest of the scene and emit no rewritten plan."
    ),
    (
        "The requested edit targets {source_id}. {instruction}. Leave all other "
        "sources and controls untouched and produce only one P10-compatible patch."
    ),
)

_EDIT_DIRECTION_SURFACES = {
    "distance_source": (
        (
            "bring it closer by multiplying every distance by 0.75",
            "move it farther by multiplying every distance by 1.25",
        ),
        (
            "reduce every distance by 25 percent; equivalently scale it by 0.75",
            "increase every distance by 25 percent; equivalently scale it by 1.25",
        ),
        (
            "set the distance scale to x0.75, the legal nearer endpoint",
            "set the distance scale to x1.25, the legal farther endpoint",
        ),
        (
            "contract its radial distance with factor 0.75",
            "expand its radial distance with factor 1.25",
        ),
        (
            "apply the negative distance direction by multiplying by 0.75",
            "apply the positive distance direction by multiplying by 1.25",
        ),
        (
            "replace every path distance d with 0.75 times d",
            "replace every path distance d with 1.25 times d",
        ),
        (
            "use DISTANCE-NEAR with the exact legal scale 0.75",
            "use DISTANCE-FAR with the exact legal scale 1.25",
        ),
        (
            "decrease its distance to 75 percent of the current value",
            "increase its distance to 125 percent of the current value",
        ),
    ),
    "rotate_source": (
        (
            "SUBTRACT 45 degrees from its azimuth; use exact delta -45 degrees",
            "ADD 45 degrees to its azimuth; use exact delta +45 degrees",
        ),
        (
            "apply AZIMUTH-MINUS and change the angle by minus forty-five degrees (-45)",
            "apply AZIMUTH-PLUS and change the angle by plus forty-five degrees (+45)",
        ),
        (
            "set its signed azimuth delta to -45 degrees",
            "set its signed azimuth delta to +45 degrees",
        ),
        (
            "move its azimuth in the negative direction by 45 degrees",
            "move its azimuth in the positive direction by 45 degrees",
        ),
        (
            "replace every azimuth theta with theta minus 45 degrees",
            "replace every azimuth theta with theta plus 45 degrees",
        ),
        (
            "use the legal negative rotation endpoint, delta azimuth -45",
            "use the legal positive rotation endpoint, delta azimuth +45",
        ),
        (
            "subtract forty-five degrees from the full trajectory azimuth",
            "add forty-five degrees to the full trajectory azimuth",
        ),
        (
            "apply the signed angular offset -45 degrees in azimuth",
            "apply the signed angular offset +45 degrees in azimuth",
        ),
    ),
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


def _identity_evidence() -> dict[str, Any]:
    return {
        "contract": P11_V4_EVIDENCE_TRANSFORM_CONTRACT,
        "transform_id": "identity_v1",
        "foa_time_mask": None,
        "semantic_channel_mask": None,
    }


def _train_generation_prompt(
    plan: Mapping[str, Any], *, mode: str, variant: int
) -> tuple[str, str, list[str]]:
    duration = float(plan["duration_sec"])
    room = str(plan["room"]["type"])
    if mode == "numeric_layout":
        clauses = []
        for source in plan["sources"]:
            activity = source["activity"]
            clauses.append(
                f"{_source_words(source)}, heard from "
                f"{float(activity['onset_sec']):.2f} through "
                f"{float(activity['offset_sec']):.2f} seconds"
            )
        endings = (
            "Stage every source with any renderer-legal static or straight-line path.",
            "Invent a sensible renderer-supported static or linear spatial arrangement.",
            "Select one coherent 3D staging; several exact layouts would be acceptable.",
        )
        prompt = (
            f"Design an executable {duration:.2f}-second FOA scene for a {room}. "
            + "; ".join(clauses)
            + ". "
            + endings[variant % len(endings)]
            + " Preserve the listed content and timing and emit a complete canonical ScenePlan."
        )
        return (
            prompt,
            f"train_reserved/g_numeric_layout/{variant % len(endings):02d}",
            ["duration", "room", "source_semantics", "temporal"],
        )
    if mode != "coarse_numeric":
        raise ValueError(mode)
    clauses = [
        f"{_source_words(source)}, {_coarse_activity(source, duration)}, "
        f"{_trajectory_words(source)}"
        for source in plan["sources"]
    ]
    endings = (
        "Pick exact frame boundaries and exact coordinates that remain inside those descriptions.",
        "Fill in renderer-legal precise timing and geometry without changing any named content.",
        "Choose one precise realization of the stated coarse temporal and spatial relations.",
    )
    prompt = (
        f"Draft a {duration:.2f}-second FOA ScenePlan in a {room}. "
        + "; ".join(clauses)
        + ". "
        + endings[variant % len(endings)]
        + " Return a complete executable plan with canonical source IDs."
    )
    return (
        prompt,
        f"train_reserved/g_coarse_numeric/{variant % len(endings):02d}",
        [
            "duration",
            "room",
            "source_semantics",
            "coarse_temporal",
            "coarse_geometry",
        ],
    )


def _train_evidence_specs(valid_frames: int, *, seed: int) -> list[dict[str, Any]]:
    def time_mask(ratio: float, salt: str) -> list[int]:
        width = max(1, min(valid_frames, int(round(valid_frames * ratio))))
        start = _stable_int(seed, salt) % (valid_frames - width + 1)
        return [int(start), int(start + width)]

    def channel_mask(width: int, salt: str) -> list[int]:
        start = _stable_int(seed, salt) % (512 - width + 1)
        return [int(start), int(start + width)]

    raw = [
        ("foa08_a", time_mask(0.08, "foa08a"), None),
        ("foa08_b", time_mask(0.08, "foa08b"), None),
        ("foa20_a", time_mask(0.20, "foa20a"), None),
        ("clap12p5", None, channel_mask(64, "clap64")),
        ("clap37p5", None, channel_mask(192, "clap192")),
        ("combined10_20", time_mask(0.10, "c10"), channel_mask(102, "c20")),
        ("combined20_35", time_mask(0.20, "c20"), channel_mask(179, "c35")),
        ("combined30_45", time_mask(0.30, "c30"), channel_mask(230, "c45")),
    ]
    return [
        {
            "contract": P11_V4_EVIDENCE_TRANSFORM_CONTRACT,
            "transform_id": f"train_{name}_v1",
            "foa_time_mask": foa,
            "semantic_channel_mask": semantic,
        }
        for name, foa, semantic in raw
    ]


def _edit_spec(
    *, operation: str, source_id: str, numeric_value: float
) -> dict[str, Any]:
    if operation == "rotate_source":
        payload: dict[str, Any] = {"delta_azimuth_deg": int(numeric_value)}
        changed = f"sources.{source_id}.trajectory.azimuth"
    elif operation == "distance_source":
        payload = {"distance_factor": float(numeric_value)}
        changed = f"sources.{source_id}.trajectory.distance"
    else:
        raise ValueError(operation)
    return {
        "operation": operation,
        "source_id": source_id,
        **payload,
        "contract": "same_scene_atomic_patch_v1",
        "changed_paths": [changed],
        "preserve_all_unspecified_fields": True,
        "input_sceneplan_tokens_required": True,
        "input_foa_required": False,
    }


def _train_edit_rows(
    current: Mapping[str, Any],
    *,
    codec: Any,
    patch_codec: ScenePlanEditPatchCodec,
    delta_codec: DeltaSceneSketchCodec,
    seed: int,
) -> list[dict[str, Any]]:
    sources = list(current["sources"])
    source = sources[_stable_int(seed, current["sample_id"], "e-source") % len(sources)]
    source_id = str(source["source_id"])
    # The atomic patch codec is the P10-facing capability authority.  Its
    # complete legal numeric vocabulary is rotation +/-45 and distance
    # x0.75/x1.25; curriculum diversity therefore comes from train-only
    # paraphrases, never from unsupported magnitudes.
    values = {
        "rotate_source": [-45.0, 45.0],
        "distance_source": [0.75, 1.25],
    }
    outputs: list[dict[str, Any]] = []
    current_ids = codec.encode(current)["input_ids"]
    for operation, numeric_values in values.items():
        direction_tokens: list[torch.Tensor] = []
        delta_programs: list[dict[str, Any]] = []
        target_hashes: set[str] = set()
        pair_id = f"train_e_{operation}_{_json_sha256([current['sample_id'], source_id])[:16]}"
        surfaces = _EDIT_DIRECTION_SURFACES[operation]
        surface_indices = [
            _stable_int(
                seed,
                current["sample_id"],
                operation,
                f"surface-p{paraphrase_index}",
            )
            % len(surfaces)
            for paraphrase_index in range(2)
        ]
        if surface_indices[1] == surface_indices[0]:
            surface_indices[1] = (surface_indices[1] + 1) % len(surfaces)
        for value_index, numeric_value in enumerate(numeric_values):
            spec = _edit_spec(
                operation=operation,
                source_id=source_id,
                numeric_value=numeric_value,
            )
            target = patch_codec.apply(current, spec)
            patch_codec.assert_target(current, spec, target)
            validate_p11_executor_profile(target)
            target_ids = codec.encode(target)["input_ids"]
            if torch.equal(target_ids, current_ids):
                raise RuntimeError(
                    f"train edit {operation}/{numeric_value} collapsed to identity"
                )
            digest = _json_sha256(target)
            target_hashes.add(digest)
            delta = compile_delta_scene_sketch(current, target, spec, codec)
            owner_token = delta_codec.encode(delta, spec)["input_ids"]
            delta_program = delta_codec.decode(owner_token)
            negative = (
                numeric_value < 0.0
                if operation == "rotate_source"
                else numeric_value < 1.0
            )
            direction_index = 0 if negative else 1
            label = (
                f"az_{int(numeric_value):+d}"
                if operation == "rotate_source"
                else f"distance_{numeric_value:.2f}"
            )
            prompts = tuple(
                _EDIT_PROMPT_WRAPPERS[surface_index].format(
                    source_id=source_id,
                    instruction=surfaces[surface_index][direction_index],
                )
                for surface_index in surface_indices
            )
            for paraphrase_index, prompt in enumerate(prompts):
                surface_index = surface_indices[paraphrase_index]
                direction_tokens.append(owner_token)
                delta_programs.append(delta_program)
                outputs.append(
                    {
                        "operation": operation,
                        "pair_id": pair_id,
                        "pair_label": f"{label}_p{paraphrase_index}",
                        "prompt": prompt,
                        "template_id": (
                            f"train_reserved/e_{operation}/paired_surface_"
                            f"{surface_index:02d}"
                        ),
                        "edit_spec": spec,
                        "target": target,
                        "target_variant": value_index * 2 + paraphrase_index,
                    }
                )
        if len(target_hashes) != 2:
            raise RuntimeError("legal E magnitudes collapsed to one target")
        # DeltaSketch-v2 owns operation, source owner, and the categorical P10
        # endpoint direction.  Paraphrases of one endpoint must therefore be
        # token-identical, while the negative/positive pair must differ only in
        # its final direction branch.  DeltaThought retains continuous state
        # reasoning and can never rewrite SceneSketch semantic content.
        if not (
            torch.equal(direction_tokens[0], direction_tokens[1])
            and torch.equal(direction_tokens[2], direction_tokens[3])
            and not torch.equal(direction_tokens[0], direction_tokens[2])
        ):
            raise RuntimeError("paired E DeltaSketch direction tokens are invalid")
        if any(
            program.get("operation") != operation
            or program.get("source_id") != source_id
            for program in delta_programs
        ):
            raise RuntimeError("paired E variants changed operation/source authority")
        if [int(program.get("control_direction", 0)) for program in delta_programs] != [
            -1,
            -1,
            1,
            1,
        ]:
            raise RuntimeError("paired E variants do not cover both legal directions")
    return outputs


def _row(
    *,
    ordinal: int,
    curriculum_id: str,
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
    target_variant: int,
    pair_id: str | None,
    pair_label: str | None,
) -> tuple[Any, ...]:
    return (
        ordinal,
        curriculum_id,
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
        int(target_variant),
        pair_id,
        pair_label,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--codec", type=Path, default=DEFAULT_CODEC)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--heldout-challenge", type=Path, default=DEFAULT_HELDOUT_CHALLENGE)
    parser.add_argument("--base-scenes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--g-completions", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.base_scenes <= 0:
        raise ValueError("--base-scenes must be positive")
    if args.g_completions != 4:
        raise ValueError("curriculum-v1 fixes four G completions per prompt")
    manifest_path = args.manifest.expanduser().resolve(strict=True)
    index_path = args.index.expanduser().resolve(strict=True)
    codec_path = args.codec.expanduser().resolve(strict=True)
    challenge_path = args.heldout_challenge.expanduser().resolve(strict=True)
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
    heldout = sqlite3.connect(
        f"file:{challenge_path}?mode=ro&immutable=1", uri=True
    )
    manifest_metadata = dict(manifest.execute("SELECT key,value FROM metadata"))
    total_base = int(manifest_metadata["base_samples"])
    if args.base_scenes > total_base:
        raise ValueError(
            f"curriculum requests {args.base_scenes}>{total_base} base scenes"
        )
    if manifest_metadata.get("source_index") != str(index_path):
        raise RuntimeError("curriculum manifest/index provenance mismatch")
    heldout_ids = {
        str(row[0]) for row in heldout.execute("SELECT DISTINCT sample_id FROM rows")
    }
    heldout_edit_prompts = {
        " ".join(str(row[0]).split())
        for row in heldout.execute(
            "SELECT prompt FROM rows WHERE family='editing_counterfactual_causality'"
        )
    }

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
    output_rows: list[tuple[Any, ...]] = []
    sample_ids: set[str] = set()
    try:
        destination.executescript(
            """
            PRAGMA journal_mode=OFF;
            PRAGMA synchronous=OFF;
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL) WITHOUT ROWID;
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
            CREATE INDEX rows_pair ON rows(pair_id);
            CREATE INDEX rows_template ON rows(template_id);
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
                raise RuntimeError(f"base train triplet {position} is invalid")
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
            if sample_id in heldout_ids:
                raise RuntimeError("held-out sample leaked into train curriculum")
            sample_ids.add(sample_id)
            current = canonicalize_sceneplan_source_ids(
                codec.project_plan(_decompress(base_rows[0][4]), sample_id=sample_id)
            )
            validate_p11_executor_profile(current)
            identity = _identity_evidence()

            g_rows: list[dict[str, Any]] = [
                {
                    "curriculum_id": f"{sample_id}/g/exact",
                    "task": "generation",
                    "family": "exact_compatibility",
                    "view_id": "exact_v1",
                    "template_id": "manifest_renderer_exact_v0",
                    "base_manifest_ordinal": int(base_rows[0][0]),
                    "base_target_ordinal": target_ordinal,
                    "sample_id": sample_id,
                    "prompt": str(base_rows[0][3]),
                    "known_groups": [
                        "duration",
                        "room",
                        "source_semantics",
                        "temporal",
                        "geometry",
                        "gain",
                    ],
                    "target_plan": current,
                    "edit_spec": None,
                    "evidence": identity,
                    "target_variant": 0,
                    "pair_id": None,
                    "pair_label": None,
                }
            ]
            for mode in ("numeric_layout", "coarse_numeric"):
                prompt_variant = _stable_int(
                    args.seed, sample_id, mode, "prompt"
                ) % 3
                prompt, template_id, known_groups = _train_generation_prompt(
                    current, mode=mode, variant=prompt_variant
                )
                completions = _reference_completions(
                    current,
                    codec=codec,
                    known_groups=known_groups,
                    mode=mode,
                    seed=_stable_int(
                        args.seed, sample_id, mode, "train-completions"
                    ),
                    count=args.g_completions,
                )
                if len({_json_sha256(value) for value in completions}) != args.g_completions:
                    raise RuntimeError("G curriculum completions are not unique")
                for target_variant, target in enumerate(completions):
                    g_rows.append(
                        {
                            "curriculum_id": (
                                f"{sample_id}/g/{mode}/target_{target_variant:02d}"
                            ),
                            "task": "generation",
                            "family": "generation_multitarget_posterior",
                            "view_id": f"{mode}_train_v1",
                            "template_id": template_id,
                            "base_manifest_ordinal": int(base_rows[0][0]),
                            "base_target_ordinal": target_ordinal,
                            "sample_id": sample_id,
                            "prompt": prompt,
                            "known_groups": known_groups,
                            "target_plan": target,
                            "edit_spec": None,
                            "evidence": identity,
                            "target_variant": target_variant,
                            "pair_id": f"{sample_id}/g/{mode}",
                            "pair_label": f"target_{target_variant:02d}",
                        }
                    )

            u_prompt = str(base_rows[1][3])
            u_rows: list[dict[str, Any]] = []
            for variant, evidence in enumerate(
                [identity]
                + _train_evidence_specs(
                    valid_frames,
                    seed=_stable_int(args.seed, sample_id, "u-train"),
                )
            ):
                transform_id = str(evidence["transform_id"])
                u_rows.append(
                    {
                        "curriculum_id": f"{sample_id}/u/{transform_id}",
                        "task": "understanding",
                        "family": (
                            "exact_compatibility"
                            if transform_id == "identity_v1"
                            else "understanding_train_evidence_stress"
                        ),
                        "view_id": transform_id,
                        "template_id": "manifest_understanding_exact_v0",
                        "base_manifest_ordinal": int(base_rows[1][0]),
                        "base_target_ordinal": target_ordinal,
                        "sample_id": sample_id,
                        "prompt": u_prompt,
                        "known_groups": ["foa_evidence"],
                        "target_plan": current,
                        "edit_spec": None,
                        "evidence": evidence,
                        "target_variant": variant,
                        "pair_id": f"{sample_id}/u/evidence",
                        "pair_label": transform_id,
                    }
                )

            exact_e_target = codec.project_plan(
                _decompress(base_rows[2][4]), sample_id=sample_id
            )
            exact_e_spec = json.loads(str(base_rows[2][6]))
            patch_codec.assert_target(current, exact_e_spec, exact_e_target)
            e_rows: list[dict[str, Any]] = [
                {
                    "curriculum_id": f"{sample_id}/e/exact",
                    "task": "editing",
                    "family": "exact_compatibility",
                    "view_id": "atomic_instruction_exact_v1",
                    "template_id": "manifest_editing_exact_v0",
                    "base_manifest_ordinal": int(base_rows[2][0]),
                    "base_target_ordinal": target_ordinal,
                    "sample_id": sample_id,
                    "prompt": str(base_rows[2][3]),
                    "known_groups": ["input_sceneplan_and_instruction"],
                    "target_plan": exact_e_target,
                    "edit_spec": exact_e_spec,
                    "evidence": identity,
                    "target_variant": 0,
                    "pair_id": None,
                    "pair_label": None,
                }
            ]
            for edit in _train_edit_rows(
                current,
                codec=codec,
                patch_codec=patch_codec,
                delta_codec=delta_codec,
                seed=args.seed,
            ):
                if " ".join(str(edit["prompt"]).split()) in heldout_edit_prompts:
                    raise RuntimeError(
                        "train Editing instruction duplicated a reserved held-out prompt"
                    )
                e_rows.append(
                    {
                        "curriculum_id": (
                            f"{sample_id}/e/{edit['pair_id']}/{edit['pair_label']}"
                        ),
                        "task": "editing",
                        "family": "editing_numeric_delta_curriculum",
                        "view_id": f"{edit['operation']}_train_v1",
                        "template_id": edit["template_id"],
                        "base_manifest_ordinal": int(base_rows[2][0]),
                        "base_target_ordinal": target_ordinal,
                        "sample_id": sample_id,
                        "prompt": edit["prompt"],
                        "known_groups": ["input_sceneplan_and_instruction"],
                        "target_plan": edit["target"],
                        "edit_spec": edit["edit_spec"],
                        "evidence": identity,
                        "target_variant": int(edit["target_variant"]),
                        "pair_id": edit["pair_id"],
                        "pair_label": edit["pair_label"],
                    }
                )

            if not (len(g_rows) == len(u_rows) == len(e_rows) == 9):
                raise RuntimeError("curriculum-v1 must have 9 balanced rows per task")
            for triplet in zip(g_rows, u_rows, e_rows):
                for record in triplet:
                    ordinal = len(output_rows)
                    output_rows.append(_row(ordinal=ordinal, **record))
                    counts[f"{record['task']}:{record['family']}"] += 1
                    template_ids.add(str(record["template_id"]))

        destination.executemany(
            "INSERT INTO rows VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            output_rows,
        )
        task_counts = {
            task: int(count)
            for task, count in destination.execute(
                "SELECT task,COUNT(*) FROM rows GROUP BY task"
            )
        }
        expected_per_task = args.base_scenes * 9
        if task_counts != {
            "generation": expected_per_task,
            "understanding": expected_per_task,
            "editing": expected_per_task,
        }:
            raise RuntimeError(f"curriculum task balance changed: {task_counts}")
        if sample_ids & heldout_ids:
            raise RuntimeError("train/heldout sample leakage detected after build")

        source_metadata = dict(source.execute("SELECT key,value FROM metadata"))
        metadata = {
            "schema": P11_V4_CURRICULUM_SCHEMA,
            "schema_version": str(P11_V4_CURRICULUM_VERSION),
            "contract": P11_V4_CURRICULUM_CONTRACT,
            "partition": P11_V4_CURRICULUM_PARTITION,
            "source_manifest": str(manifest_path),
            "source_manifest_sha256": _sha256_file(manifest_path),
            "source_index": str(index_path),
            "source_index_sha256": _sha256_file(index_path),
            "source_index_split": str(source_metadata.get("split")),
            "codec_path": str(codec_path),
            "codec_fingerprint": codec.fingerprint,
            "p11_v4_data_contract": P11_V4_DATA_CONTRACT,
            "p11_v4_sequence_contract": P11_V4_SEQUENCE_CONTRACT,
            "p10_release": "p10-sceneplan-dit-v11-step150000",
            "p10_checkpoint": str(P10_CHECKPOINT),
            "p10_checkpoint_sha256": P10_CHECKPOINT_SHA256,
            "p10_max_latent_frames": "648",
            "p10_source_count": "1-4",
            "p10_motion_profile": "static,linear",
            "base_scenes": str(args.base_scenes),
            "rows": str(len(output_rows)),
            "rows_per_base_scene": "27",
            "rows_per_task_per_base_scene": "9",
            "row_counts_json": json.dumps(counts, sort_keys=True),
            "seed": str(args.seed),
            "generation_targets_per_underspecified_prompt": "4",
            "generation_reference_sets_are_exhaustive": "false",
            "generation_single_target_for_underspecified_prompt": "false",
            "u_degradation_scope": "synthetic_train_representation_stress_only",
            "editing_uses_complete_p10_atomic_numeric_vocabulary": "true",
            "editing_instruction_surface_contract": (
                EDITING_INSTRUCTION_SURFACE_CONTRACT
            ),
            "editing_instruction_surface_count": str(len(_EDIT_PROMPT_WRAPPERS)),
            "editing_instruction_surfaces_per_pair_identity": "2",
            "heldout_edit_prompt_exact_overlap": "0",
            "template_partition": P11_V4_CURRICULUM_PARTITION,
            "train_template_ids_json": json.dumps(sorted(template_ids)),
            "eval_reserved_templates_present": "false",
            "heldout_challenge": str(challenge_path),
            "heldout_challenge_sha256": _sha256_file(challenge_path),
            "heldout_sample_overlap": "0",
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
        heldout.close()
        if not completed:
            temporary.unlink(missing_ok=True)
    os.replace(temporary, output)
    report = {
        "status": "BUILT",
        "schema": P11_V4_CURRICULUM_SCHEMA,
        "schema_version": P11_V4_CURRICULUM_VERSION,
        "contract": P11_V4_CURRICULUM_CONTRACT,
        "output": str(output),
        "output_sha256": _sha256_file(output),
        "base_scenes": args.base_scenes,
        "rows": len(output_rows),
        "task_rows": args.base_scenes * 9,
        "counts": dict(sorted(counts.items())),
        "heldout_sample_overlap": 0,
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
