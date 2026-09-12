#!/usr/bin/env python3
"""Normalize synthesis-manifest rows into caption-template slots.

The manifest is the source of truth.  This module contains no model-specific
logic and can therefore be reused by template captions, ScenePlan generation,
tests, and future discrete spatial-token compilers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

AZIMUTH_DIRS = (
    "front",
    "front-left",
    "left",
    "rear-left",
    "behind",
    "rear-right",
    "right",
    "front-right",
)
ELEVATIONS = ("level", "above", "below")
DISTANCE_WORDS = ("very close", "nearby", "a short distance away", "far away")
MIX_TYPES = ("single", "pair", "multi")
MOTIONS = ("static", "dynamic")


def _clean_label(value: Any) -> str:
    text = str(value or "a sound").strip().rstrip(".")
    return text or "a sound"


def elevation_short(value: str) -> str:
    return {
        "above": "above",
        "below": "below",
        "level": "at ear level",
    }.get(value, str(value or "at ear level"))


def elevation_long(value: str) -> str:
    return {
        "above": "elevated above the listener",
        "below": "below the listener",
        "level": "at the listener's ear level",
    }.get(value, str(value or "at the listener's ear level"))


def elevation_prep(value: str) -> str:
    return {
        "above": "from above",
        "below": "from below",
        "level": "at ear level",
    }.get(value, str(value or "at ear level"))


def join_source_phrases(phrases: Iterable[str]) -> str:
    items = [str(item).strip().rstrip(".") for item in phrases if str(item).strip()]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return f"{', '.join(items[:-1])}, and {items[-1]}"


def direction_source_phrase(value: str) -> str:
    if value == "behind":
        return "from behind"
    return f"from the {value}"


def direction_endpoint(value: str) -> str:
    if value == "behind":
        return "behind"
    return f"the {value}"


@dataclass(frozen=True)
class SourceSlots:
    label: str
    category: str = ""
    dataset: str = ""
    motion: str = "static"
    dir: str = "front"
    elev: str = "level"
    dist: str = "nearby"
    az_deg: float | None = None
    el_deg: float | None = None
    dist_m: float | None = None
    end_dir: str | None = None
    end_elev: str | None = None
    end_dist: str | None = None
    move_from: str | None = None
    move_to: str | None = None

    @property
    def is_dynamic(self) -> bool:
        return self.motion == "dynamic"

    @property
    def dir_elev(self) -> str:
        return f"{self.label} {direction_source_phrase(self.dir)}, {elevation_short(self.elev)}"

    @property
    def dir_elev_dist(self) -> str:
        return f"{self.dir_elev}, {self.dist}"

    @property
    def motion_phrase(self) -> str:
        if not self.is_dynamic:
            return self.dir_elev_dist
        start = self.move_from or self.dir
        end = self.move_to or self.end_dir or self.dir
        elevation = f", {elevation_short(self.elev)}"
        if self.end_elev and self.end_elev != self.elev:
            elevation = (
                f", changing elevation from {elevation_short(self.elev)} "
                f"to {elevation_short(self.end_elev)}"
            )
        return (
            f"{self.label} moving from {direction_endpoint(start)} "
            f"to {direction_endpoint(end)}{elevation}"
        )

    def to_slot_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "category": self.category,
            "dataset": self.dataset,
            "motion": self.motion,
            "dir": self.dir,
            "elev": self.elev,
            "dist": self.dist,
            "elev_short": elevation_short(self.elev),
            "elev_long": elevation_long(self.elev),
            "elev_prep": elevation_prep(self.elev),
            "move_from": self.move_from or self.dir,
            "move_to": self.move_to or self.end_dir or self.dir,
            "source": self.motion_phrase if self.is_dynamic else self.dir_elev_dist,
            "source_dir_elev": self.dir_elev,
            "source_dir_elev_dist": self.dir_elev_dist,
            "source_motion": self.motion_phrase,
        }


@dataclass(frozen=True)
class ClipSlots:
    id: str
    foa_path: str
    mix_type: str
    sources: tuple[SourceSlots, ...]
    room: str
    reverb: str
    room_outdoor: str
    free_field: bool
    spatial_caption: str = ""
    raw: dict[str, Any] = field(default_factory=dict, compare=False, repr=False)

    @property
    def n_sources(self) -> int:
        return len(self.sources)

    def to_slot_dict(self) -> dict[str, Any]:
        first = self.sources[0] if self.sources else SourceSlots("a sound")
        slots = first.to_slot_dict()
        slots.update(
            {
                "id": self.id,
                "foa_path": self.foa_path,
                "mix_type": self.mix_type,
                "n_sources": self.n_sources,
                "room": self.room,
                "room_desc": self.room,
                "reverb": self.reverb,
                "room_outdoor": self.room_outdoor,
                "free_field": self.free_field,
                "spatial_caption": self.spatial_caption,
            }
        )

        static_phrases = []
        motion_phrases = []
        full_phrases = []
        for index, source in enumerate(self.sources):
            source_slots = source.to_slot_dict()
            slots[f"label_{index}"] = source.label
            slots[f"dir_{index}"] = source.dir
            for key, value in source_slots.items():
                slots[f"source_{index}_{key}"] = value
            static_phrases.append(source.dir_elev)
            motion_phrases.append(source.motion_phrase)
            full_phrases.append(
                source.motion_phrase if source.is_dynamic else source.dir_elev_dist
            )

        pair = join_source_phrases(full_phrases[:2])
        multi = join_source_phrases(full_phrases)
        slots.update(
            {
                "source_0": full_phrases[0] if full_phrases else "a sound",
                "source_1": full_phrases[1] if len(full_phrases) > 1 else "",
                "source_2": full_phrases[2] if len(full_phrases) > 2 else "",
                "source_pair": pair,
                "source_multi": multi,
                "source_pair_dir_elev": join_source_phrases(static_phrases[:2]),
                "source_multi_dir_elev": join_source_phrases(static_phrases),
                "source_pair_motion": join_source_phrases(motion_phrases[:2]),
                "source_multi_motion": join_source_phrases(motion_phrases),
                "source_pair_full": pair,
                "source_multi_full": multi,
            }
        )
        return slots


def _parse_source(raw: dict[str, Any]) -> SourceSlots:
    start = raw.get("start") or {}
    end = raw.get("end") or {}
    move = raw.get("move") or {}
    return SourceSlots(
        label=_clean_label(raw.get("label") or raw.get("category")),
        category=str(raw.get("category") or ""),
        dataset=str(raw.get("dataset") or ""),
        motion=str(raw.get("motion") or "static"),
        dir=str(start.get("dir") or "front"),
        elev=str(start.get("elev") or "level"),
        dist=str(start.get("dist_word") or "nearby"),
        az_deg=start.get("az_deg"),
        el_deg=start.get("el_deg"),
        dist_m=start.get("dist_m"),
        end_dir=end.get("dir"),
        end_elev=end.get("elev"),
        end_dist=end.get("dist_word"),
        move_from=move.get("from"),
        move_to=move.get("to"),
    )


def extract_clip_slots(row: dict[str, Any]) -> ClipSlots:
    """Convert one manifest row to a validated, template-friendly object."""

    raw_sources = row.get("sources") or []
    sources = tuple(_parse_source(source) for source in raw_sources if isinstance(source, dict))
    if not sources:
        # Caption generation should remain total for diagnostic/incomplete rows.
        sources = (SourceSlots(_clean_label(row.get("label") or row.get("category"))),)

    room = row.get("room") or {}
    free_field = bool(room.get("free_field", False))
    room_desc = str(room.get("desc") or room.get("type") or "an unspecified space")
    reverb = str(room.get("reverb") or "unspecified reverberation")
    room_outdoor = "outdoors" if free_field else f"in {room_desc}"
    mix_type = str(row.get("mix_type") or ("single" if len(sources) == 1 else "pair" if len(sources) == 2 else "multi"))

    return ClipSlots(
        id=str(row.get("id") or ""),
        foa_path=str(row.get("foa_path") or row.get("path") or ""),
        mix_type=mix_type,
        sources=sources,
        room=room_desc,
        reverb=reverb,
        room_outdoor=room_outdoor,
        free_field=free_field,
        spatial_caption=str(row.get("spatial_caption") or ""),
        raw=row,
    )
