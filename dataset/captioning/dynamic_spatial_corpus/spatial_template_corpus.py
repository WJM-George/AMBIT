#!/usr/bin/env python3
"""Spatial-caption template corpus with total-coverage stage renderers.

The checked-in JSON contains the legacy ~1000 paraphrase templates.  Universal
renderers cover every valid synthesis-manifest row and are used for stages where
the legacy constraints would otherwise fall back or drop examples.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

try:
    from .spatial_caption_slots import (
        ClipSlots,
        direction_source_phrase,
        extract_clip_slots,
        join_source_phrases,
    )
except ImportError:  # Direct script/import via sys.path.
    from spatial_caption_slots import (
        ClipSlots,
        direction_source_phrase,
        extract_clip_slots,
        join_source_phrases,
    )

DEFAULT_CORPUS_PATH = Path(__file__).with_name("spatial_template_corpus.json")

STAGE_NAMES = {
    1: "direction",
    2: "elevation",
    3: "distance",
    4: "room",
    5: "motion",
    6: "multi_source",
    7: "full",
}

STAGE_DESCRIPTIONS = {
    1: "Content + horizontal direction (WHAT + WHERE azimuth).",
    2: "Add vertical placement (above / below / level).",
    3: "Add distance / proximity cues.",
    4: "Add room acoustics and environment.",
    5: "Add source motion (pan / trajectory).",
    6: "Multi-source mixing (pair and 3-4 source layouts).",
    7: "Full spatial caption (all cues combined).",
}

CORPUS_VOCABULARY = {
    "azimuth_dirs": [
        "front",
        "front-left",
        "left",
        "rear-left",
        "behind",
        "rear-right",
        "right",
        "front-right",
    ],
    "elevations": ["level", "above", "below"],
    "distance_words": [
        "very close",
        "nearby",
        "a short distance away",
        "far away",
    ],
    "mix_types": ["single", "pair", "multi"],
    "motions": ["static", "dynamic"],
}


def _sentence(text: str) -> str:
    value = re.sub(r"\s+", " ", text).strip()
    value = re.sub(r"\s+([,.;:!?])", r"\1", value)
    if value and value[-1] not in ".!?":
        value += "."
    return value


def _stable_rng(stage: int, row: dict[str, Any]) -> random.Random:
    identity = str(row.get("id") or row.get("foa_path") or row.get("path") or "")
    digest = hashlib.sha256(f"{stage}:{identity}".encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _room_phrase(clip: ClipSlots) -> str:
    if clip.free_field:
        return "Recorded outdoors"
    if clip.reverb and clip.reverb != "unspecified reverberation":
        acoustics = re.sub(r"\s+room$", " acoustics", clip.reverb)
        return f"Recorded in {clip.room} with {acoustics}"
    return f"Recorded in {clip.room}"


def _source_phrase(source, stage: int) -> str:
    if stage == 1:
        return f"{source.label} {direction_source_phrase(source.dir)}"
    if stage == 2:
        return source.dir_elev
    if stage <= 4:
        return source.dir_elev_dist
    if source.is_dynamic:
        phrase = source.motion_phrase
        if source.dist:
            phrase += f", {source.dist}"
        return phrase
    return source.dir_elev_dist


def _render_universal(
    stage: int,
    row: dict[str, Any],
    rng: Optional[random.Random] = None,
) -> tuple[str, str]:
    if stage not in STAGE_NAMES:
        raise ValueError(f"stage must be in 1..7, got {stage}")
    clip = extract_clip_slots(row)
    rng = rng or _stable_rng(stage, row)
    phrases = [_source_phrase(source, stage) for source in clip.sources]

    if len(phrases) == 1:
        scene = phrases[0]
    else:
        lead = rng.choice(("The scene contains", "You hear", "The spatial mix places"))
        scene = f"{lead} {join_source_phrases(phrases)}"

    sentences = [_sentence(scene)]
    if stage >= 4:
        sentences.append(_sentence(_room_phrase(clip)))

    template_variant = rng.randrange(4)
    template_id = f"universal_s{stage}_{template_variant:02d}"
    return " ".join(sentences), template_id


def render_stage1(
    row: dict[str, Any], rng: Optional[random.Random] = None
) -> tuple[str, str]:
    return _render_universal(1, row, rng)


def render_stage3(row: dict[str, Any], rng: Optional[random.Random] = None):
    return _render_universal(3, row, rng)


def render_stage4(row: dict[str, Any], rng: Optional[random.Random] = None):
    return _render_universal(4, row, rng)


def render_stage5(row: dict[str, Any], rng: Optional[random.Random] = None):
    return _render_universal(5, row, rng)


def render_stage6(row: dict[str, Any], rng: Optional[random.Random] = None):
    return _render_universal(6, row, rng)


def render_stage7(row: dict[str, Any], rng: Optional[random.Random] = None):
    return _render_universal(7, row, rng)


STAGE_UNIVERSAL_RENDERERS = {
    1: render_stage1,
    3: render_stage3,
    4: render_stage4,
    5: render_stage5,
    6: render_stage6,
    7: render_stage7,
}

# Compatibility for older caption scripts that used the ambiguous "stage12"
# name for the first curriculum stage.
render_stage12 = render_stage1


@dataclass(frozen=True)
class TemplateSpec:
    template_id: str
    stage: int
    pattern: str
    tags: tuple[str, ...] = ()
    constraints: dict[str, Any] = field(default_factory=dict)
    required_slots: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TemplateSpec":
        return cls(
            template_id=str(value["template_id"]),
            stage=int(value["stage"]),
            pattern=str(value["pattern"]),
            tags=tuple(value.get("tags") or ()),
            constraints=dict(value.get("constraints") or {}),
            required_slots=tuple(value.get("required_slots") or ()),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "template_id": self.template_id,
            "stage": self.stage,
            "pattern": self.pattern,
            "tags": list(self.tags),
            "constraints": self.constraints,
            "required_slots": list(self.required_slots),
        }

    def matches(self, clip: ClipSlots) -> bool:
        slots = clip.to_slot_dict()
        for required in self.required_slots:
            if required not in slots or slots[required] in (None, ""):
                return False

        constraints = self.constraints
        for key in ("mix_type", "dir", "elev"):
            allowed = constraints.get(key)
            if allowed is None:
                continue
            allowed = allowed if isinstance(allowed, (list, tuple, set)) else [allowed]
            value = slots.get(key)
            if value not in allowed:
                return False

        motion = constraints.get("motion")
        if motion is not None and clip.sources[0].motion != motion:
            return False
        if "free_field" in constraints and clip.free_field != bool(constraints["free_field"]):
            return False
        if clip.n_sources < int(constraints.get("n_sources_min", 0)):
            return False
        if clip.n_sources > int(constraints.get("n_sources_max", 10**9)):
            return False
        return True

    def render(self, clip: ClipSlots) -> str:
        return _sentence(self.pattern.format_map(clip.to_slot_dict()))


def load_corpus(path: str | Path = DEFAULT_CORPUS_PATH) -> list[TemplateSpec]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        raw_templates = payload
    elif isinstance(payload, dict) and isinstance(payload.get("templates"), list):
        raw_templates = payload["templates"]
    elif isinstance(payload, dict) and isinstance(payload.get("stages"), dict):
        raw_templates = [
            template
            for stage in payload["stages"].values()
            for template in stage.get("templates", [])
        ]
    else:
        raise ValueError(f"invalid spatial template corpus: {path}")
    specs = [TemplateSpec.from_dict(value) for value in raw_templates]
    ids = [spec.template_id for spec in specs]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate template_id in corpus: {path}")
    return specs


def select_template_for_row(
    row: dict[str, Any],
    corpus: Optional[Sequence[TemplateSpec]] = None,
    *,
    stage: int,
    rng: Optional[random.Random] = None,
    exclude_tags: Iterable[str] = (),
) -> TemplateSpec:
    clip = extract_clip_slots(row)
    corpus = list(corpus) if corpus is not None else load_corpus()
    excluded = set(exclude_tags)
    candidates = [
        spec
        for spec in corpus
        if spec.stage == stage
        and not excluded.intersection(spec.tags)
        and spec.matches(clip)
    ]
    if not candidates:
        raise LookupError(
            f"no stage-{stage} template matches id={clip.id!r}, "
            f"mix_type={clip.mix_type!r}, n_sources={clip.n_sources}"
        )
    return (rng or _stable_rng(stage, row)).choice(candidates)


def render_for_row(template: TemplateSpec, row: dict[str, Any]) -> str:
    """Render a previously selected template against one manifest row."""

    return template.render(extract_clip_slots(row))


def render_stage_caption(
    stage: int,
    row: dict[str, Any],
    rng: Optional[random.Random] = None,
) -> tuple[str, str]:
    """Render one curriculum caption.

    Stage 2 intentionally keeps the legacy corpus for the original elevation
    ablation. All other stages use total-coverage structured renderers.
    """

    if stage == 2:
        template = select_template_for_row(row, stage=stage, rng=rng)
        return render_for_row(template, row), template.template_id
    renderer = STAGE_UNIVERSAL_RENDERERS.get(stage)
    if renderer is None:
        raise ValueError(f"stage must be in 1..7, got {stage}")
    return renderer(row, rng)


def corpus_to_dict(corpus: Sequence[TemplateSpec]) -> dict[str, Any]:
    stages: dict[str, dict[str, Any]] = {}
    for stage in range(1, 8):
        templates = [spec.to_dict() for spec in corpus if spec.stage == stage]
        stages[str(stage)] = {
            "name": STAGE_NAMES[stage],
            "description": STAGE_DESCRIPTIONS[stage],
            "count": len(templates),
            "templates": templates,
        }
    flattened = [spec.to_dict() for spec in corpus]
    return {
        "version": 1,
        "kind": "spatial_caption_template_corpus",
        "total": len(flattened),
        "vocabulary": CORPUS_VOCABULARY,
        "stages": stages,
        "templates": flattened,
    }


def save_corpus(
    corpus: Sequence[TemplateSpec],
    path: str | Path = DEFAULT_CORPUS_PATH,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(
        json.dumps(corpus_to_dict(corpus), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


def merge_corpus(
    base: Sequence[TemplateSpec],
    additions: Sequence[TemplateSpec],
) -> list[TemplateSpec]:
    merged: dict[str, TemplateSpec] = {spec.template_id: spec for spec in base}
    patterns = {(spec.stage, spec.pattern) for spec in base}
    for spec in additions:
        if spec.template_id in merged:
            raise ValueError(f"duplicate template_id: {spec.template_id}")
        if (spec.stage, spec.pattern) in patterns:
            continue
        merged[spec.template_id] = spec
        patterns.add((spec.stage, spec.pattern))
    return sorted(merged.values(), key=lambda spec: (spec.stage, spec.template_id))


def generate_template_corpus(
    target_total: int = 1000,
    seed: int = 42,
) -> list[TemplateSpec]:
    """Return the checked-in reproducible corpus.

    The original combinatorial generator produced the committed JSON. Keeping
    that artifact canonical avoids silently changing training captions when a
    phrase family is edited.
    """

    del seed
    corpus = load_corpus(DEFAULT_CORPUS_PATH)
    if len(corpus) < int(target_total):
        raise RuntimeError(
            f"checked-in corpus has {len(corpus)} templates, below requested "
            f"target_total={target_total}"
        )
    return corpus
