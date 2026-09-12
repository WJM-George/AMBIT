#!/usr/bin/env python3
"""Build the resumable 9.6M-row P11-v4 full curriculum and DDP8 layout.

The builder has two explicit phases. ``part`` expands one contiguous range of
the frozen 4.8M manifest without retaining the corpus in memory. ``merge``
streams completed parts into the exact world-size-8, batch-8/GPU ordering used
by training. No GPU work is performed by this script.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
import sqlite3
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.t2a.data.build_sceneplan_p11_v4_challenge import (  # noqa: E402
    P10_CHECKPOINT,
    P10_CHECKPOINT_SHA256,
    _reference_completions,
)
from scripts.t2a.data.build_sceneplan_p11_v4_curriculum import (  # noqa: E402
    EDITING_INSTRUCTION_SURFACE_CONTRACT,
    _EDIT_DIRECTION_SURFACES,
    _EDIT_PROMPT_WRAPPERS,
    _compress,
    _edit_spec,
    _identity_evidence,
    _json_sha256,
    _stable_int,
    _train_evidence_specs,
    _train_generation_prompt,
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


ROOT = Path("/mnt/sdb/audio_dataset/sceneplan_v2_1p124m")
P11_ROOT = ROOT / "p11_single_turn_15s_v2"
DEFAULT_MANIFEST = P11_ROOT / "manifests/p11_train_4p8m_v6.sqlite"
DEFAULT_INDEX = (
    ROOT
    / "revisions/speech_expansion_noalign_15s_v1/training_index/train.sqlite"
)
DEFAULT_CODEC = P11_ROOT / "model_sceneplan_codec_v4"
DEFAULT_CHALLENGE = REPO_ROOT / (
    "artifacts/sceneplan_p11/challenges/"
    "p11_v4_heldout_challenge_v1_20260901.sqlite"
)
DEFAULT_OUTPUT = P11_ROOT / (
    "p11_v4_curriculum/p11_train_9p6m_ddp8_batch8_seed42_v2.sqlite"
)

PART_SCHEMA = "stable_audio_tools.sceneplan_p11_v4_full_curriculum_part"
PART_VERSION = 2
FULL_CONTRACT = "p10_v11_full9p6m_gue_executable_thought_v2"
SELECTION_CONTRACT = (
    "exact4p8m_plus_pairbalanced_g_multitarget1p6m_u_stress1p6m_"
    "e_pair1p6m_v2"
)
ORDERING_CONTRACT = "p11_v4_full9p6m_ddp8_batch8_supercycle_v2"
AUGMENTATION_SELECTOR_CONTRACT = "adjacent_pair_stable_hash_choose_one_v1"
WORLD_SIZE = 8
LOCAL_BATCH_SIZE = 8
GLOBAL_BATCH_SIZE = WORLD_SIZE * LOCAL_BATCH_SIZE
BASE_SCENES_PER_SUPERCYCLE = 32

FINAL_COLUMNS = (
    "curriculum_id",
    "task",
    "family",
    "view_id",
    "template_id",
    "base_manifest_ordinal",
    "base_target_ordinal",
    "sample_id",
    "prompt",
    "known_field_groups_json",
    "target_sceneplan_zlib",
    "edit_spec_json",
    "evidence_transform_json",
    "target_variant",
    "pair_id",
    "pair_label",
)
ROW_PLACEHOLDERS = ",".join("?" for _ in range(2 + len(FINAL_COLUMNS)))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _decompress(payload: bytes) -> Any:
    import zlib

    return json.loads(zlib.decompress(payload))


def _chunks(values: Sequence[int], size: int) -> Iterable[Sequence[int]]:
    for start in range(0, len(values), int(size)):
        yield values[start : start + int(size)]


def _readonly(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(
        f"file:{path}?mode=ro&immutable=1", uri=True
    )
    connection.execute("PRAGMA query_only=ON")
    return connection


def _record(
    *,
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
        str(curriculum_id),
        str(task),
        str(family),
        str(view_id),
        str(template_id),
        int(base_manifest_ordinal),
        int(base_target_ordinal),
        str(sample_id),
        str(prompt),
        json.dumps(list(known_groups), separators=(",", ":")),
        _compress(target_plan),
        None
        if edit_spec is None
        else json.dumps(edit_spec, ensure_ascii=False, sort_keys=True),
        json.dumps(evidence, sort_keys=True, separators=(",", ":")),
        int(target_variant),
        None if pair_id is None else str(pair_id),
        None if pair_label is None else str(pair_label),
    )


def _queue_row(kind: str, index: int, record: tuple[Any, ...]) -> tuple[Any, ...]:
    return (str(kind), int(index), *record)


def _augmentation_selected(position: int, *, seed: int) -> bool:
    """Select exactly one scene per adjacent pair without source-order bias.

    The frozen 1.6M source index has meaningful parity: even and odd positions
    differ strongly in room and content family.  Selecting every even row would
    therefore bias all G multi-target and E numeric-pair supervision.  Hashing
    the pair index chooses one member deterministically while preserving the
    exact 50% cardinality required by the 9.6M layout.
    """

    position = int(position)
    if position < 0:
        raise ValueError("full augmentation position must be non-negative")
    selected_parity = _stable_int(
        int(seed), AUGMENTATION_SELECTOR_CONTRACT, position // 2
    ) % 2
    return position % 2 == selected_parity


def _augmentations_before(position: int, *, seed: int) -> int:
    """Return the exact number of selected scenes before ``position``."""

    position = int(position)
    if position < 0:
        raise ValueError("full augmentation position must be non-negative")
    selected = position // 2
    if position % 2 and _augmentation_selected(position - 1, seed=seed):
        selected += 1
    return selected


def _g_queue_start(position: int, *, seed: int) -> int:
    # Every position contributes one exact row; each preceding selected scene
    # contributes two additional same-prompt numeric completions.
    return int(position + 2 * _augmentations_before(position, seed=seed))


def _full_edit_pair(
    current: Mapping[str, Any],
    *,
    codec: Any,
    patch_codec: ScenePlanEditPatchCodec,
    delta_codec: DeltaSceneSketchCodec,
    position: int,
    seed: int,
) -> list[dict[str, Any]]:
    sources = list(current["sources"])
    source = sources[
        _stable_int(seed, current["sample_id"], "full-e-source") % len(sources)
    ]
    source_id = str(source["source_id"])
    operation = (
        "rotate_source"
        if ((position // 2 + seed) % 2 == 0)
        else "distance_source"
    )
    numeric_values = (
        (-45.0, 45.0)
        if operation == "rotate_source"
        else (0.75, 1.25)
    )
    pair_id = (
        f"full_e_{operation}_"
        f"{_json_sha256([current['sample_id'], source_id])[:16]}"
    )
    outputs: list[dict[str, Any]] = []
    direction_tokens: list[torch.Tensor] = []
    current_ids = codec.encode(current)["input_ids"]
    target_hashes: set[str] = set()
    for direction_index, numeric_value in enumerate(numeric_values):
        spec = _edit_spec(
            operation=operation,
            source_id=source_id,
            numeric_value=float(numeric_value),
        )
        target = patch_codec.apply(current, spec)
        patch_codec.assert_target(current, spec, target)
        validate_p11_executor_profile(target)
        if torch.equal(codec.encode(target)["input_ids"], current_ids):
            raise RuntimeError("full E numeric endpoint collapsed to identity")
        target_hashes.add(_json_sha256(target))
        delta = compile_delta_scene_sketch(current, target, spec, codec)
        token = delta_codec.encode(delta, spec)["input_ids"]
        decoded = delta_codec.decode(token)
        expected_direction = -1 if direction_index == 0 else 1
        if (
            decoded.get("operation") != operation
            or decoded.get("source_id") != source_id
            or int(decoded.get("control_direction", 0)) != expected_direction
        ):
            raise RuntimeError("full E pair changed operation/owner/direction")
        direction_tokens.append(token)
        surface_index = _stable_int(
            seed,
            current["sample_id"],
            operation,
            "full-surface",
            direction_index,
        ) % len(_EDIT_DIRECTION_SURFACES[operation])
        wrapper_index = _stable_int(
            seed,
            current["sample_id"],
            operation,
            "full-wrapper",
            direction_index,
        ) % len(_EDIT_PROMPT_WRAPPERS)
        instruction = _EDIT_DIRECTION_SURFACES[operation][surface_index][
            direction_index
        ]
        prompt = _EDIT_PROMPT_WRAPPERS[wrapper_index].format(
            source_id=source_id,
            instruction=instruction,
        )
        label = (
            f"az_{int(numeric_value):+d}"
            if operation == "rotate_source"
            else f"distance_{numeric_value:.2f}"
        )
        outputs.append(
            {
                "operation": operation,
                "pair_id": pair_id,
                "pair_label": label,
                "prompt": prompt,
                "template_id": (
                    f"train_reserved/e_{operation}/paired_surface_"
                    f"{surface_index:02d}"
                ),
                "edit_spec": spec,
                "target": target,
                "target_variant": direction_index,
            }
        )
    if torch.equal(direction_tokens[0], direction_tokens[1]):
        raise RuntimeError("full E pair directions collapsed to one token program")
    if len(target_hashes) != 2:
        raise RuntimeError("full E pair endpoints collapsed to one ScenePlan")
    return outputs


def _part_expected_counts(start: int, end: int, *, seed: int) -> dict[str, int]:
    selected = sum(
        1
        for position in range(start, end)
        if _augmentation_selected(position, seed=seed)
    )
    scenes = end - start
    return {
        "generation": scenes + 2 * selected,
        "understanding": 2 * scenes,
        "editing_exact": scenes,
        "editing_pair": 2 * selected,
    }


def _build_part(args: argparse.Namespace) -> None:
    if args.seed != 42:
        raise ValueError("the canonical full curriculum fixes seed 42")
    if args.num_parts <= 0 or not 0 <= args.part_index < args.num_parts:
        raise ValueError("part must satisfy 0 <= part-index < num-parts")
    manifest_path = args.manifest.expanduser().resolve(strict=True)
    index_path = args.index.expanduser().resolve(strict=True)
    codec_path = args.codec.expanduser().resolve(strict=True)
    challenge_path = args.heldout_challenge.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not args.overwrite:
        raise FileExistsError(output)

    manifest = _readonly(manifest_path)
    index = _readonly(index_path)
    heldout = _readonly(challenge_path)
    manifest_metadata = dict(manifest.execute("SELECT key,value FROM metadata"))
    available = int(manifest_metadata.get("base_samples", -1))
    total_scenes = available if args.base_scenes is None else int(args.base_scenes)
    if total_scenes <= 0 or total_scenes > available:
        raise ValueError("base-scenes must lie within the source manifest")
    if total_scenes % BASE_SCENES_PER_SUPERCYCLE:
        raise ValueError(
            f"base-scenes must be divisible by {BASE_SCENES_PER_SUPERCYCLE}"
        )
    if manifest_metadata.get("source_index") != str(index_path):
        raise RuntimeError("full curriculum manifest/index provenance mismatch")
    supplied_hashes = (
        ("manifest", str(args.manifest_sha256), manifest_path),
        ("index", str(args.index_sha256), index_path),
        ("heldout", str(args.heldout_sha256), challenge_path),
    )
    if any(
        len(value) != 64 or any(character not in "0123456789abcdef" for character in value)
        for _, value, _ in supplied_hashes
    ):
        raise ValueError("source SHA-256 arguments must be lowercase hex digests")
    # Logical parts are commonly built by 8 CPU workers. Hash the multi-GB
    # immutable inputs only once in part zero; every part records the same
    # launcher-supplied values, and merge recomputes all three before publish.
    if args.part_index == 0:
        for label, expected, path in supplied_hashes:
            if _sha256_file(path) != expected:
                raise RuntimeError(f"provided source {label} SHA-256 is stale")

    start = total_scenes * args.part_index // args.num_parts
    end = total_scenes * (args.part_index + 1) // args.num_parts
    if start == end:
        raise RuntimeError("logical part selected no base scenes")
    heldout_ids = {
        str(row[0]) for row in heldout.execute("SELECT DISTINCT sample_id FROM rows")
    }
    heldout_edit_prompts = {
        " ".join(str(row[0]).split())
        for row in heldout.execute(
            "SELECT prompt FROM rows "
            "WHERE family='editing_counterfactual_causality'"
        )
    }

    codec = load_model_sceneplan_codec(codec_path)
    patch_codec = ScenePlanEditPatchCodec(codec)
    delta_codec = DeltaSceneSketchCodec(codec, patch_codec)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".sqlite", dir=output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    destination = sqlite3.connect(temporary)
    completed = False
    started = time.perf_counter()
    counts: Counter[str] = Counter()
    templates: set[str] = set()
    inserted = 0
    try:
        destination.executescript(
            f"""
            PRAGMA journal_mode=OFF;
            PRAGMA synchronous=OFF;
            CREATE TABLE metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            ) WITHOUT ROWID;
            CREATE TABLE rows (
                queue_kind TEXT NOT NULL,
                queue_index INTEGER NOT NULL,
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
                pair_label TEXT,
                PRIMARY KEY(queue_kind,queue_index)
            ) WITHOUT ROWID;
            """
        )
        manifest_cursor = manifest.execute(
            """
            SELECT ordinal,task,target_ordinal,prompt,target_sceneplan_zlib,
                   edit_kind,edit_spec_json
            FROM rows WHERE ordinal>=? AND ordinal<? ORDER BY ordinal
            """,
            (start * 3, end * 3),
        )
        position = start
        while True:
            chunk = manifest_cursor.fetchmany(6144)
            if not chunk:
                break
            if len(chunk) % 3:
                raise RuntimeError("manifest chunk split a canonical G/U/E triplet")
            target_ordinals = [int(chunk[offset][2]) for offset in range(0, len(chunk), 3)]
            index_rows: dict[int, tuple[str, int]] = {}
            for ordinal_chunk in _chunks(target_ordinals, 900):
                placeholders = ",".join("?" for _ in ordinal_chunk)
                index_rows.update(
                    {
                        int(ordinal): (str(sample_id), int(valid_frames))
                        for ordinal, sample_id, valid_frames in index.execute(
                            f"SELECT ordinal,sample_id,latent_frames_valid "
                            f"FROM samples WHERE ordinal IN ({placeholders})",
                            tuple(ordinal_chunk),
                        )
                    }
                )
            if len(index_rows) != len(target_ordinals):
                raise RuntimeError("source index did not resolve every base scene")

            output_rows: list[tuple[Any, ...]] = []
            for offset in range(0, len(chunk), 3):
                base_rows = chunk[offset : offset + 3]
                expected_ordinals = [position * 3 + value for value in range(3)]
                if [int(row[0]) for row in base_rows] != expected_ordinals:
                    raise RuntimeError("base manifest triplet ordinals are not canonical")
                if [str(row[1]) for row in base_rows] != [
                    "generation",
                    "understanding",
                    "editing",
                ]:
                    raise RuntimeError("base manifest task triplet changed")
                target_ordinal = int(base_rows[0][2])
                if any(int(row[2]) != target_ordinal for row in base_rows):
                    raise RuntimeError("base G/U/E target ordinal diverged")
                sample_id, valid_frames = index_rows[target_ordinal]
                if sample_id in heldout_ids:
                    raise RuntimeError("held-out sample leaked into full curriculum")
                current = canonicalize_sceneplan_source_ids(
                    codec.project_plan(
                        _decompress(base_rows[0][4]), sample_id=sample_id
                    )
                )
                validate_p11_executor_profile(current)
                u_target = canonicalize_sceneplan_source_ids(
                    codec.project_plan(
                        _decompress(base_rows[1][4]), sample_id=sample_id
                    )
                )
                if _json_sha256(u_target) != _json_sha256(current):
                    raise RuntimeError("base exact G/U ScenePlans differ")
                identity = _identity_evidence()

                g_index = _g_queue_start(position, seed=args.seed)
                exact_g = _record(
                    curriculum_id=f"{sample_id}/g/exact",
                    task="generation",
                    family="exact_compatibility",
                    view_id="exact_v1",
                    template_id="manifest_renderer_exact_v0",
                    base_manifest_ordinal=int(base_rows[0][0]),
                    base_target_ordinal=target_ordinal,
                    sample_id=sample_id,
                    prompt=str(base_rows[0][3]),
                    known_groups=(
                        "duration",
                        "room",
                        "source_semantics",
                        "temporal",
                        "geometry",
                        "gain",
                    ),
                    target_plan=current,
                    edit_spec=None,
                    evidence=identity,
                    target_variant=0,
                    pair_id=None,
                    pair_label=None,
                )
                output_rows.append(_queue_row("generation", g_index, exact_g))
                counts["generation"] += 1
                templates.add("manifest_renderer_exact_v0")

                augment_scene = _augmentation_selected(
                    position, seed=args.seed
                )
                if augment_scene:
                    mode = (
                        "numeric_layout"
                        if ((position // 2 + args.seed) % 2 == 0)
                        else "coarse_numeric"
                    )
                    prompt_variant = _stable_int(
                        args.seed, sample_id, mode, "full-prompt"
                    ) % 3
                    prompt, template_id, known_groups = _train_generation_prompt(
                        current, mode=mode, variant=prompt_variant
                    )
                    reference_set = _reference_completions(
                        current,
                        codec=codec,
                        known_groups=known_groups,
                        mode=mode,
                        seed=_stable_int(
                            args.seed, sample_id, mode, "full-completions"
                        ),
                        count=3,
                    )
                    if len({_json_sha256(value) for value in reference_set}) != 3:
                        raise RuntimeError("full G completions are not unique")
                    completions = reference_set[1:]
                    pair_id = f"{sample_id}/g/{mode}/full"
                    for variant, target in enumerate(completions):
                        record = _record(
                            curriculum_id=(
                                f"{sample_id}/g/{mode}/full_target_{variant + 1:02d}"
                            ),
                            task="generation",
                            family="generation_multitarget_posterior",
                            view_id=f"{mode}_full_train_v1",
                            template_id=template_id,
                            base_manifest_ordinal=int(base_rows[0][0]),
                            base_target_ordinal=target_ordinal,
                            sample_id=sample_id,
                            prompt=prompt,
                            known_groups=known_groups,
                            target_plan=target,
                            edit_spec=None,
                            evidence=identity,
                            target_variant=variant + 1,
                            pair_id=pair_id,
                            pair_label=f"target_{variant + 1:02d}",
                        )
                        output_rows.append(
                            _queue_row("generation", g_index + 1 + variant, record)
                        )
                        counts["generation"] += 1
                        templates.add(template_id)

                exact_u = _record(
                    curriculum_id=f"{sample_id}/u/identity_v1",
                    task="understanding",
                    family="exact_compatibility",
                    view_id="identity_v1",
                    template_id="manifest_understanding_exact_v0",
                    base_manifest_ordinal=int(base_rows[1][0]),
                    base_target_ordinal=target_ordinal,
                    sample_id=sample_id,
                    prompt=str(base_rows[1][3]),
                    known_groups=("foa_evidence",),
                    target_plan=current,
                    edit_spec=None,
                    evidence=identity,
                    target_variant=0,
                    pair_id=f"{sample_id}/u/full_evidence",
                    pair_label="identity_v1",
                )
                output_rows.append(
                    _queue_row("understanding", 2 * position, exact_u)
                )
                evidence_specs = _train_evidence_specs(
                    valid_frames,
                    seed=_stable_int(args.seed, sample_id, "full-u-train"),
                )
                evidence_index = (position + args.seed) % len(evidence_specs)
                evidence = evidence_specs[evidence_index]
                transform_id = str(evidence["transform_id"])
                stress_u = _record(
                    curriculum_id=f"{sample_id}/u/full_{transform_id}",
                    task="understanding",
                    family="understanding_train_evidence_stress",
                    view_id=transform_id,
                    template_id="manifest_understanding_exact_v0",
                    base_manifest_ordinal=int(base_rows[1][0]),
                    base_target_ordinal=target_ordinal,
                    sample_id=sample_id,
                    prompt=str(base_rows[1][3]),
                    known_groups=("foa_evidence",),
                    target_plan=current,
                    edit_spec=None,
                    evidence=evidence,
                    target_variant=evidence_index + 1,
                    pair_id=f"{sample_id}/u/full_evidence",
                    pair_label=transform_id,
                )
                output_rows.append(
                    _queue_row("understanding", 2 * position + 1, stress_u)
                )
                counts["understanding"] += 2
                templates.add("manifest_understanding_exact_v0")

                exact_e_target = codec.project_plan(
                    _decompress(base_rows[2][4]), sample_id=sample_id
                )
                exact_e_spec = json.loads(str(base_rows[2][6]))
                patch_codec.assert_target(current, exact_e_spec, exact_e_target)
                exact_e = _record(
                    curriculum_id=f"{sample_id}/e/exact",
                    task="editing",
                    family="exact_compatibility",
                    view_id="atomic_instruction_exact_v1",
                    template_id="manifest_editing_exact_v0",
                    base_manifest_ordinal=int(base_rows[2][0]),
                    base_target_ordinal=target_ordinal,
                    sample_id=sample_id,
                    prompt=str(base_rows[2][3]),
                    known_groups=("input_sceneplan_and_instruction",),
                    target_plan=exact_e_target,
                    edit_spec=exact_e_spec,
                    evidence=identity,
                    target_variant=0,
                    pair_id=None,
                    pair_label=None,
                )
                output_rows.append(_queue_row("editing_exact", position, exact_e))
                counts["editing_exact"] += 1
                templates.add("manifest_editing_exact_v0")

                if augment_scene:
                    edits = _full_edit_pair(
                        current,
                        codec=codec,
                        patch_codec=patch_codec,
                        delta_codec=delta_codec,
                        position=position,
                        seed=args.seed,
                    )
                    for direction_index, edit in enumerate(edits):
                        normalized_prompt = " ".join(str(edit["prompt"]).split())
                        if normalized_prompt in heldout_edit_prompts:
                            raise RuntimeError(
                                "full E prompt duplicated a reserved held-out prompt"
                            )
                        record = _record(
                            curriculum_id=(
                                f"{sample_id}/e/{edit['pair_id']}/"
                                f"{edit['pair_label']}"
                            ),
                            task="editing",
                            family="editing_numeric_delta_curriculum",
                            view_id=f"{edit['operation']}_full_train_v1",
                            template_id=str(edit["template_id"]),
                            base_manifest_ordinal=int(base_rows[2][0]),
                            base_target_ordinal=target_ordinal,
                            sample_id=sample_id,
                            prompt=str(edit["prompt"]),
                            known_groups=("input_sceneplan_and_instruction",),
                            target_plan=edit["target"],
                            edit_spec=edit["edit_spec"],
                            evidence=identity,
                            target_variant=int(edit["target_variant"]),
                            pair_id=str(edit["pair_id"]),
                            pair_label=str(edit["pair_label"]),
                        )
                        output_rows.append(
                            _queue_row(
                                "editing_pair",
                                2
                                * _augmentations_before(
                                    position, seed=args.seed
                                )
                                + direction_index,
                                record,
                            )
                        )
                        counts["editing_pair"] += 1
                        templates.add(str(edit["template_id"]))

                position += 1

            destination.executemany(
                f"INSERT INTO rows VALUES ({ROW_PLACEHOLDERS})", output_rows
            )
            inserted += len(output_rows)
            destination.commit()
            print(
                json.dumps(
                    {
                        "event": "progress",
                        "part_index": args.part_index,
                        "base_position": position,
                        "base_end": end,
                        "rows": inserted,
                        "elapsed_sec": time.perf_counter() - started,
                    }
                ),
                flush=True,
            )

        if position != end:
            raise RuntimeError(f"part stopped at base {position}, expected {end}")
        expected_counts = _part_expected_counts(start, end, seed=args.seed)
        if dict(counts) != expected_counts:
            raise RuntimeError(
                f"part queue counts changed: {dict(counts)} != {expected_counts}"
            )
        actual_counts = {
            str(kind): int(count)
            for kind, count in destination.execute(
                "SELECT queue_kind,COUNT(*) FROM rows GROUP BY queue_kind"
            )
        }
        if actual_counts != expected_counts:
            raise RuntimeError("part SQLite queue counts differ from memory audit")
        source_metadata = dict(index.execute("SELECT key,value FROM metadata"))
        imported_builder = REPO_ROOT / (
            "scripts/t2a/data/build_sceneplan_p11_v4_curriculum.py"
        )
        metadata = {
            "schema": PART_SCHEMA,
            "schema_version": str(PART_VERSION),
            "contract": FULL_CONTRACT,
            "selection_contract": SELECTION_CONTRACT,
            "augmentation_selector_contract": AUGMENTATION_SELECTOR_CONTRACT,
            "partition": P11_V4_CURRICULUM_PARTITION,
            "source_manifest": str(manifest_path),
            "source_manifest_sha256": str(args.manifest_sha256),
            "source_index": str(index_path),
            "source_index_sha256": str(args.index_sha256),
            "source_index_split": str(source_metadata.get("split")),
            "codec_path": str(codec_path),
            "codec_fingerprint": codec.fingerprint,
            "heldout_challenge": str(challenge_path),
            "heldout_challenge_sha256": str(args.heldout_sha256),
            "heldout_sample_overlap": "0",
            "heldout_edit_prompt_exact_overlap": "0",
            "eval_reserved_templates_present": "false",
            "p11_v4_data_contract": P11_V4_DATA_CONTRACT,
            "p11_v4_sequence_contract": P11_V4_SEQUENCE_CONTRACT,
            "p10_release": "p10-sceneplan-dit-v11-step150000",
            "p10_checkpoint": str(P10_CHECKPOINT),
            "p10_checkpoint_sha256": P10_CHECKPOINT_SHA256,
            "p10_max_latent_frames": "648",
            "p10_source_count": "1-4",
            "p10_motion_profile": "static,linear",
            "seed": str(args.seed),
            "total_base_scenes": str(total_scenes),
            "part_index": str(args.part_index),
            "num_parts": str(args.num_parts),
            "base_start": str(start),
            "base_end": str(end),
            "base_scenes": str(end - start),
            "rows": str(inserted),
            "queue_counts_json": json.dumps(expected_counts, sort_keys=True),
            "train_template_ids_json": json.dumps(sorted(templates)),
            "editing_instruction_surface_contract": (
                EDITING_INSTRUCTION_SURFACE_CONTRACT
            ),
            "evidence_transform_contract": P11_V4_EVIDENCE_TRANSFORM_CONTRACT,
            "builder": str(Path(__file__).resolve()),
            "builder_sha256": _sha256_file(Path(__file__).resolve()),
            "imported_curriculum_builder_sha256": _sha256_file(imported_builder),
            "elapsed_sec": f"{time.perf_counter() - started:.6f}",
        }
        destination.executemany(
            "INSERT INTO metadata(key,value) VALUES (?,?)",
            sorted(metadata.items()),
        )
        destination.commit()
        completed = True
    finally:
        destination.close()
        manifest.close()
        index.close()
        heldout.close()
        if not completed:
            temporary.unlink(missing_ok=True)
    os.replace(temporary, output)
    print(
        json.dumps(
            {
                "schema": PART_SCHEMA,
                "schema_version": PART_VERSION,
                "status": "PASS",
                "output": str(output),
                "output_sha256": _sha256_file(output),
                "part_index": args.part_index,
                "num_parts": args.num_parts,
                "base_start": start,
                "base_end": end,
                "rows": inserted,
                "queue_counts": expected_counts,
            },
            indent=2,
            sort_keys=True,
        )
    )


PART_IDENTITY_KEYS = (
    "schema",
    "schema_version",
    "contract",
    "selection_contract",
    "augmentation_selector_contract",
    "partition",
    "source_manifest",
    "source_manifest_sha256",
    "source_index",
    "source_index_sha256",
    "source_index_split",
    "codec_path",
    "codec_fingerprint",
    "heldout_challenge",
    "heldout_challenge_sha256",
    "eval_reserved_templates_present",
    "p11_v4_data_contract",
    "p11_v4_sequence_contract",
    "p10_release",
    "p10_checkpoint",
    "p10_checkpoint_sha256",
    "p10_max_latent_frames",
    "p10_source_count",
    "p10_motion_profile",
    "seed",
    "total_base_scenes",
    "num_parts",
    "builder_sha256",
    "imported_curriculum_builder_sha256",
)


class _QueueReader:
    def __init__(self, parts: Sequence[Path], kind: str) -> None:
        self.parts = list(parts)
        self.kind = str(kind)
        self.part_index = -1
        self.connection: sqlite3.Connection | None = None
        self.cursor: sqlite3.Cursor | None = None
        self.last_queue_index = -1
        self.rows = 0

    def _advance_part(self) -> bool:
        if self.connection is not None:
            self.connection.close()
            self.connection = None
            self.cursor = None
        self.part_index += 1
        if self.part_index >= len(self.parts):
            return False
        self.connection = _readonly(self.parts[self.part_index])
        columns = ",".join(FINAL_COLUMNS)
        self.cursor = self.connection.execute(
            f"SELECT queue_index,{columns} FROM rows "
            "WHERE queue_kind=? ORDER BY queue_index",
            (self.kind,),
        )
        return True

    def take(self) -> tuple[Any, ...]:
        while self.cursor is not None or self._advance_part():
            assert self.cursor is not None
            row = self.cursor.fetchone()
            if row is None:
                self._advance_part()
                continue
            queue_index = int(row[0])
            if queue_index != self.last_queue_index + 1:
                raise RuntimeError(
                    f"{self.kind} queue is not contiguous at {queue_index}; "
                    f"expected {self.last_queue_index + 1}"
                )
            self.last_queue_index = queue_index
            self.rows += 1
            return tuple(row[1:])
        raise RuntimeError(f"{self.kind} queue exhausted early")

    def assert_exhausted(self, expected: int) -> None:
        if self.rows != int(expected):
            raise RuntimeError(
                f"{self.kind} consumed {self.rows}, expected {expected}"
            )
        try:
            self.take()
        except RuntimeError as error:
            if "exhausted early" not in str(error):
                raise
        else:
            raise RuntimeError(f"{self.kind} queue contains extra rows")
        if self.connection is not None:
            self.connection.close()
            self.connection = None


def _take(reader: _QueueReader, count: int) -> list[tuple[Any, ...]]:
    return [reader.take() for _ in range(int(count))]


def _take_edit_pair(reader: _QueueReader) -> list[tuple[Any, ...]]:
    rows = _take(reader, 2)
    if any(str(row[1]) != "editing" for row in rows):
        raise RuntimeError("editing-pair queue yielded a non-E row")
    if rows[0][14] is None or rows[0][14] != rows[1][14]:
        raise RuntimeError("editing-pair queue split a pair identity")
    specs = [json.loads(str(row[11])) for row in rows]
    operations = {str(spec["operation"]) for spec in specs}
    owners = {str(spec["source_id"]) for spec in specs}
    if len(operations) != 1 or len(owners) != 1:
        raise RuntimeError("editing pair changed operation or source owner")
    operation = operations.pop()
    if operation == "rotate_source":
        values = [float(spec["delta_azimuth_deg"]) for spec in specs]
        expected = [-45.0, 45.0]
    elif operation == "distance_source":
        values = [float(spec["distance_factor"]) for spec in specs]
        expected = [0.75, 1.25]
    else:
        raise RuntimeError(f"unsupported full E pair operation {operation}")
    if values != expected:
        raise RuntimeError(f"editing pair values {values} != {expected}")
    return rows


def _local_order_key(
    record: tuple[Any, ...], *, seed: int, step: int, rank: int
) -> int:
    return _stable_int(seed, "full-ddp-local", step, rank, str(record[0]))


def _update_order_digest(digest: Any, ordinal: int, record: tuple[Any, ...]) -> None:
    digest.update(int(ordinal).to_bytes(8, "little", signed=False))
    for value in record:
        if value is None:
            payload = b"\xff"
        elif isinstance(value, bytes):
            payload = value
        else:
            payload = str(value).encode("utf-8")
        digest.update(len(payload).to_bytes(8, "little", signed=False))
        digest.update(payload)


def _merge_parts(args: argparse.Namespace) -> None:
    inputs = [path.expanduser().resolve(strict=True) for path in args.inputs]
    if not inputs or len(set(inputs)) != len(inputs):
        raise ValueError("full curriculum parts must be non-empty and unique")
    part_rows: list[tuple[Path, dict[str, str], int, dict[str, int]]] = []
    for path in inputs:
        connection = _readonly(path)
        try:
            metadata = dict(connection.execute("SELECT key,value FROM metadata"))
            rows = int(connection.execute("SELECT COUNT(*) FROM rows").fetchone()[0])
            queue_counts = {
                str(kind): int(count)
                for kind, count in connection.execute(
                    "SELECT queue_kind,COUNT(*) FROM rows GROUP BY queue_kind"
                )
            }
            forbidden = int(
                connection.execute(
                    "SELECT COUNT(*) FROM rows "
                    "WHERE template_id LIKE 'eval_reserved/%'"
                ).fetchone()[0]
            )
        finally:
            connection.close()
        if forbidden:
            raise RuntimeError(f"{path} contains reserved eval templates")
        if metadata.get("schema") != PART_SCHEMA:
            raise RuntimeError(f"{path} is not a full-curriculum part")
        if metadata.get("schema_version") != str(PART_VERSION):
            raise RuntimeError(f"{path} full-curriculum part version changed")
        if rows != int(metadata.get("rows", -1)):
            raise RuntimeError(f"{path} row metadata is stale")
        if queue_counts != json.loads(metadata["queue_counts_json"]):
            raise RuntimeError(f"{path} queue-count metadata is stale")
        part_rows.append((path, metadata, rows, queue_counts))

    reference = part_rows[0][1]
    mismatches = {
        str(path): {
            key: (reference.get(key), metadata.get(key))
            for key in PART_IDENTITY_KEYS
            if metadata.get(key) != reference.get(key)
        }
        for path, metadata, _, _ in part_rows
    }
    mismatches = {path: value for path, value in mismatches.items() if value}
    if mismatches:
        raise RuntimeError(f"full curriculum part contracts differ: {mismatches}")
    num_parts = int(reference["num_parts"])
    if len(part_rows) != num_parts:
        raise RuntimeError(f"received {len(part_rows)} of {num_parts} parts")
    ordered_parts = sorted(part_rows, key=lambda value: int(value[1]["part_index"]))
    if [int(value[1]["part_index"]) for value in ordered_parts] != list(
        range(num_parts)
    ):
        raise RuntimeError("full curriculum part-index coverage is incomplete")
    expected_start = 0
    for path, metadata, rows, queue_counts in ordered_parts:
        start = int(metadata["base_start"])
        end = int(metadata["base_end"])
        if start != expected_start or end <= start:
            raise RuntimeError(f"full curriculum base coverage breaks at {path}")
        expected_counts = _part_expected_counts(
            start, end, seed=int(reference["seed"])
        )
        if queue_counts != expected_counts or rows != sum(expected_counts.values()):
            raise RuntimeError(f"{path} part counts violate the selection contract")
        expected_start = end
    total_scenes = int(reference["total_base_scenes"])
    if expected_start != total_scenes:
        raise RuntimeError("full curriculum parts do not cover every base scene")
    if total_scenes % BASE_SCENES_PER_SUPERCYCLE:
        raise RuntimeError("full curriculum cannot form complete DDP supercycles")

    manifest_path = Path(reference["source_manifest"]).resolve(strict=True)
    index_path = Path(reference["source_index"]).resolve(strict=True)
    challenge_path = Path(reference["heldout_challenge"]).resolve(strict=True)
    for path, expected in (
        (manifest_path, reference["source_manifest_sha256"]),
        (index_path, reference["source_index_sha256"]),
        (challenge_path, reference["heldout_challenge_sha256"]),
    ):
        if _sha256_file(path) != expected:
            raise RuntimeError(f"full curriculum source changed after part build: {path}")
    current_builder_sha = _sha256_file(Path(__file__).resolve())
    if current_builder_sha != reference["builder_sha256"]:
        raise RuntimeError("full curriculum builder changed after part generation")

    paths = [value[0] for value in ordered_parts]
    readers = {
        kind: _QueueReader(paths, kind)
        for kind in (
            "generation",
            "understanding",
            "editing_exact",
            "editing_pair",
        )
    }
    expected_queue_counts = {
        "generation": 2 * total_scenes,
        "understanding": 2 * total_scenes,
        "editing_exact": total_scenes,
        "editing_pair": total_scenes,
    }
    aggregate_counts = Counter()
    for _, _, _, values in ordered_parts:
        aggregate_counts.update(values)
    if dict(aggregate_counts) != expected_queue_counts:
        raise RuntimeError(
            f"full queue totals changed: {dict(aggregate_counts)}"
        )

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
    completed = False
    started = time.perf_counter()
    ordinal = 0
    global_step = 0
    task_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    templates: set[str] = set()
    order_digest = hashlib.sha256()
    pair_local_batches = 0
    total_local_batches = 0
    try:
        destination.executescript(
            """
            PRAGMA journal_mode=OFF;
            PRAGMA synchronous=OFF;
            CREATE TABLE metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
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
            """
        )
        insert_buffer: list[tuple[Any, ...]] = []
        supercycles = total_scenes // BASE_SCENES_PER_SUPERCYCLE
        for supercycle in range(supercycles):
            for pattern in ("A", "B", "C"):
                rank_rows: list[list[tuple[Any, ...]]] = []
                for rank in range(WORLD_SIZE):
                    if pattern == "A":
                        local = (
                            _take(readers["generation"], 3)
                            + _take(readers["understanding"], 3)
                            + _take_edit_pair(readers["editing_pair"])
                        )
                        has_pair = True
                    elif pattern == "B":
                        local = (
                            _take(readers["generation"], 3)
                            + _take(readers["understanding"], 2)
                            + _take_edit_pair(readers["editing_pair"])
                            + _take(readers["editing_exact"], 1)
                        )
                        has_pair = True
                    else:
                        local = (
                            _take(readers["generation"], 2)
                            + _take(readers["understanding"], 3)
                            + _take(readers["editing_exact"], 3)
                        )
                        has_pair = False
                    if len(local) != LOCAL_BATCH_SIZE:
                        raise RuntimeError("full DDP local batch size changed")
                    local_counts = Counter(str(row[1]) for row in local)
                    if set(local_counts) != {"generation", "understanding", "editing"}:
                        raise RuntimeError("full DDP local batch lacks G/U/E")
                    local = sorted(
                        local,
                        key=lambda row: _local_order_key(
                            row, seed=int(reference["seed"]), step=global_step, rank=rank
                        ),
                    )
                    rank_rows.append(local)
                    total_local_batches += 1
                    pair_local_batches += int(has_pair)

                # DistributedSampler(shuffle=False) gives rank r the strided
                # positions r,r+8,...; column-major emission preserves each
                # rank-local batch assembled above.
                for local_position in range(LOCAL_BATCH_SIZE):
                    for rank in range(WORLD_SIZE):
                        record = rank_rows[rank][local_position]
                        insert_buffer.append((ordinal, *record))
                        _update_order_digest(order_digest, ordinal, record)
                        task_counts[str(record[1])] += 1
                        family_counts[f"{record[1]}:{record[2]}"] += 1
                        templates.add(str(record[4]))
                        ordinal += 1
                if len(insert_buffer) >= 8192:
                    destination.executemany(
                        "INSERT INTO rows VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        insert_buffer,
                    )
                    destination.commit()
                    insert_buffer.clear()
                global_step += 1
            if supercycle % 1000 == 0:
                print(
                    json.dumps(
                        {
                            "event": "merge_progress",
                            "supercycles": supercycle + 1,
                            "supercycles_total": supercycles,
                            "global_steps": global_step,
                            "rows": ordinal,
                            "elapsed_sec": time.perf_counter() - started,
                        }
                    ),
                    flush=True,
                )
        if insert_buffer:
            destination.executemany(
                "INSERT INTO rows VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                insert_buffer,
            )
            destination.commit()
        expected_rows = total_scenes * 6
        expected_task_counts = {
            "generation": 2 * total_scenes,
            "understanding": 2 * total_scenes,
            "editing": 2 * total_scenes,
        }
        if ordinal != expected_rows or dict(task_counts) != expected_task_counts:
            raise RuntimeError(
                f"full layout counts changed: rows={ordinal}, tasks={task_counts}"
            )
        for kind, reader in readers.items():
            reader.assert_exhausted(expected_queue_counts[kind])
        expected_steps = expected_rows // GLOBAL_BATCH_SIZE
        if global_step != expected_steps:
            raise RuntimeError("full global optimizer-step count changed")
        if pair_local_batches != total_local_batches * 2 // 3:
            raise RuntimeError("full E-pair supercycle coverage changed")
        if any(template.startswith("eval_reserved/") for template in templates):
            raise RuntimeError("reserved eval template entered full layout")

        destination.executescript(
            """
            CREATE INDEX rows_task_family ON rows(task,family);
            CREATE INDEX rows_base_scene ON rows(base_target_ordinal);
            CREATE INDEX rows_pair ON rows(pair_id);
            CREATE INDEX rows_template ON rows(template_id);
            """
        )
        rank_counts = {
            "generation": total_scenes // 4,
            "understanding": total_scenes // 4,
            "editing": total_scenes // 4,
        }
        metadata = {
            "schema": P11_V4_CURRICULUM_SCHEMA,
            "schema_version": str(P11_V4_CURRICULUM_VERSION),
            "contract": P11_V4_CURRICULUM_CONTRACT,
            "full_scale_contract": FULL_CONTRACT,
            "selection_contract": SELECTION_CONTRACT,
            "augmentation_selector_contract": AUGMENTATION_SELECTOR_CONTRACT,
            "partition": P11_V4_CURRICULUM_PARTITION,
            "source_manifest": str(manifest_path),
            "source_manifest_sha256": reference["source_manifest_sha256"],
            "source_index": str(index_path),
            "source_index_sha256": reference["source_index_sha256"],
            "source_index_split": reference["source_index_split"],
            "codec_path": reference["codec_path"],
            "codec_fingerprint": reference["codec_fingerprint"],
            "p11_v4_data_contract": P11_V4_DATA_CONTRACT,
            "p11_v4_sequence_contract": P11_V4_SEQUENCE_CONTRACT,
            "p10_release": reference["p10_release"],
            "p10_checkpoint": reference["p10_checkpoint"],
            "p10_checkpoint_sha256": reference["p10_checkpoint_sha256"],
            "p10_max_latent_frames": reference["p10_max_latent_frames"],
            "p10_source_count": reference["p10_source_count"],
            "p10_motion_profile": reference["p10_motion_profile"],
            "base_scenes": str(total_scenes),
            "rows": str(expected_rows),
            "rows_per_base_scene": "6",
            "average_rows_per_task_per_base_scene": "2",
            "row_counts_json": json.dumps(family_counts, sort_keys=True),
            "seed": reference["seed"],
            "generation_targets_per_underspecified_prompt": "2",
            "generation_reference_sets_are_exhaustive": "false",
            "generation_single_target_for_underspecified_prompt": "false",
            "u_degradation_scope": "synthetic_train_representation_stress_only",
            "editing_uses_complete_p10_atomic_numeric_vocabulary": "true",
            "editing_numeric_pair_rows": str(total_scenes),
            "editing_exact_rows": str(total_scenes),
            "editing_instruction_surface_contract": (
                EDITING_INSTRUCTION_SURFACE_CONTRACT
            ),
            "heldout_challenge": str(challenge_path),
            "heldout_challenge_sha256": reference["heldout_challenge_sha256"],
            "heldout_sample_overlap": "0",
            "heldout_edit_prompt_exact_overlap": "0",
            "template_partition": P11_V4_CURRICULUM_PARTITION,
            "train_template_ids_json": json.dumps(sorted(templates)),
            "eval_reserved_templates_present": "false",
            "ordering_contract": ORDERING_CONTRACT,
            "ordering_seed": reference["seed"],
            "ordering_world_size": str(WORLD_SIZE),
            "ordering_batch_size": str(LOCAL_BATCH_SIZE),
            "ordering_global_batch_size": str(GLOBAL_BATCH_SIZE),
            "global_optimizer_steps": str(expected_steps),
            "distributed_sampler_contract": (
                "strided_shuffle_false_drop_last_false_v1"
            ),
            "ddp_local_batches": str(total_local_batches),
            "ddp_local_batches_with_all_tasks": str(total_local_batches),
            "ddp_local_batches_with_complete_pair": str(pair_local_batches),
            "max_consecutive_local_batches_without_complete_pair": "1",
            "ddp_rank_task_counts_json": json.dumps(
                [rank_counts for _ in range(WORLD_SIZE)], sort_keys=True
            ),
            "ddp_rank_task_counts_identical": "true",
            "row_order_sha256": order_digest.hexdigest(),
            "source_part_count": str(num_parts),
            "builder": str(Path(__file__).resolve()),
            "builder_sha256": current_builder_sha,
            "imported_curriculum_builder_sha256": reference[
                "imported_curriculum_builder_sha256"
            ],
        }
        destination.executemany(
            "INSERT INTO metadata(key,value) VALUES (?,?)",
            sorted(metadata.items()),
        )
        destination.commit()
        destination.execute("PRAGMA optimize")
        destination.commit()
        actual = destination.execute(
            "SELECT COUNT(*),MIN(ordinal),MAX(ordinal) FROM rows"
        ).fetchone()
        if actual != (expected_rows, 0, expected_rows - 1):
            raise RuntimeError(f"full output is incomplete: {actual}")
        completed = True
    finally:
        destination.close()
        if not completed:
            temporary.unlink(missing_ok=True)
    os.replace(temporary, output)
    report = {
        "schema": "stable_audio_tools.sceneplan_p11_v4_full_curriculum_build",
        "schema_version": 1,
        "status": "PASS",
        "output": str(output),
        "output_sha256": _sha256_file(output),
        "rows": total_scenes * 6,
        "base_scenes": total_scenes,
        "task_counts": dict(task_counts),
        "family_counts": dict(family_counts),
        "global_optimizer_steps": global_step,
        "ordering_contract": ORDERING_CONTRACT,
        "source_parts": num_parts,
        "source_part_sha256": {
            str(path): _sha256_file(path) for path in paths
        },
        "row_order_sha256": order_digest.hexdigest(),
        "elapsed_sec": time.perf_counter() - started,
    }
    report_path = output.with_suffix(".build.json")
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    part = subparsers.add_parser("part", help="build one resumable CPU part")
    part.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    part.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    part.add_argument("--codec", type=Path, default=DEFAULT_CODEC)
    part.add_argument("--heldout-challenge", type=Path, default=DEFAULT_CHALLENGE)
    part.add_argument("--output", type=Path, required=True)
    part.add_argument("--manifest-sha256", required=True)
    part.add_argument("--index-sha256", required=True)
    part.add_argument("--heldout-sha256", required=True)
    part.add_argument("--part-index", type=int, required=True)
    part.add_argument("--num-parts", type=int, required=True)
    part.add_argument("--base-scenes", type=int)
    part.add_argument("--seed", type=int, default=42)
    part.add_argument("--overwrite", action="store_true")
    part.set_defaults(function=_build_part)

    merge = subparsers.add_parser(
        "merge", help="stream parts into the canonical DDP8 layout"
    )
    merge.add_argument("--inputs", type=Path, nargs="+", required=True)
    merge.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    merge.add_argument("--overwrite", action="store_true")
    merge.set_defaults(function=_merge_parts)
    return parser


def main() -> None:
    args = _parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
