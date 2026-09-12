"""Training-only attribution of codec targets to request-sourced fields.

This module does not parse user requests or add information at inference. An
incomplete request needs reviewed field provenance and a complete source
binding. A relative constraint is never classified as a unique numeric label.
The categories describe the existing witness supervision, not its correctness.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class TokenField:
    family: str
    origin: str
    source_id: str | None = None
    fields: tuple[str, ...] = ()

    @property
    def category(self) -> str:
        return f"{self.family}/{self.origin}"


def _numeric_origin(fields: Sequence[str], dependencies: Sequence[Mapping]) -> str:
    # A specified event duration constrains onset and offset jointly. Neither
    # endpoint becomes a specified absolute number merely through this relation.
    if any(c.get("op") == "numeric" and c.get("field") in fields for c in dependencies):
        return "explicit_numeric"
    return "relative" if dependencies else "free"


def annotate_target_tokens(
    codec,
    tokens: Sequence[int],
    *,
    fully_specified: bool,
    provenance: Sequence[Mapping] | None = None,
    source_bindings: Mapping[str, str] | None = None,
) -> list[TokenField]:
    """Attribute every unpadded token of a legal static/linear codec target.

    ``fully_specified`` is only for renderers audited to mention every encoded
    field. It must not be inferred from the existence of a witness plan. For
    incomplete requests, source bindings map annotation keys to *this view's*
    complete source IDs; position and time must follow the same mapping as text.
    Static coordinates jointly encode their start and end constraints.
    """
    tokens = [int(x) for x in tokens]
    plan = codec.decode(tokens, sample_id="supervision_audit")
    source_ids = {s["source_id"] for s in plan["sources"]}
    by_field = {}
    reverse_binding = {}
    if not fully_specified:
        if provenance is None or source_bindings is None:
            raise ValueError("Incomplete requests require provenance and source bindings")
        if set(source_bindings.values()) != source_ids or len(source_bindings) != len(source_ids):
            raise ValueError("Source binding must be a bijection to the encoded sources")
        reverse_binding = {sid: key for key, sid in source_bindings.items()}
        for row in provenance:
            key = (row["source"], row["field"])
            if key in by_field:
                raise ValueError(f"Duplicate field provenance: {key}")
            if row["source"] is not None and row["source"] not in source_bindings:
                raise ValueError("Field provenance refers to an unbound source")
            if row["origin"] not in ("free_completion", "request_constrained", "canonical_schema_policy"):
                raise ValueError("Unknown field provenance origin")
            constraints = row.get("constraints", [])
            if (row["origin"] == "request_constrained") != bool(constraints):
                raise ValueError("Field origin disagrees with its request constraints")
            by_field[key] = constraints

    result = [TokenField("structure", "schema") for _ in tokens]
    cursor = 0

    def expect(tag):
        nonlocal cursor
        if cursor >= len(tokens) or tokens[cursor] != codec.token_to_id[tag]:
            raise ValueError(f"Unexpected codec token at {cursor}; expected {tag}")
        cursor += 1

    def value(family, fields, source_id=None):
        nonlocal cursor
        if family == "count":
            origin = "requested"
        elif fully_specified:
            origin = "explicit_numeric" if family in ("time", "space") else "requested"
        else:
            key = reverse_binding[source_id] if source_id is not None else None
            dependencies = []
            for field in fields:
                if (key, field) not in by_field:
                    raise ValueError(f"Missing provenance for {(key, field)}")
                dependencies.extend(by_field[key, field])
            if family in ("time", "space"):
                origin = _numeric_origin(fields, dependencies)
            else:
                origin = "requested" if dependencies else "free"
        if cursor >= len(tokens):
            raise ValueError("Truncated value token")
        result[cursor] = TokenField(family, origin, source_id, tuple(fields))
        cursor += 1

    def text(tag, family, field, source_id):
        expect(tag)
        expect("<text_begin>")
        length = 0
        while cursor < len(tokens) and tokens[cursor] in codec.text_ids:
            value(family, (field,), source_id)
            length += 1
        if not length:
            raise ValueError("Empty text field")
        # The boundary token remains structural. It still receives the current
        # CE objective, and must retain nonzero weight in any later intervention.
        expect("<text_end>")

    def position(tag, endpoints, source_id):
        expect(tag)
        for marker, coordinate in (("<azimuth_bin>", "azimuth_deg"),
                                   ("<elevation_bin>", "elevation_deg"),
                                   ("<distance_bin>", "distance_m")):
            expect(marker)
            value("space", tuple(f"{p}.{coordinate}" for p in endpoints), source_id)

    expect("<plan_bos>")
    expect("<duration_frames>"); value("time", ("duration_sec",))
    expect("<room>"); value("room", ("room",))
    expect("<num_sources>"); value("count", ("source_count",))
    for source in plan["sources"]:
        sid = source["source_id"]
        expect("<source_begin>")
        expect(f"<source_slot_{int(sid.removeprefix('source_'))}>")
        expect("<kind>")
        # Kind belongs to the same requested core identity. It has no separate
        # provenance row in the current annotation schema.
        value("kind", ("core_text",), sid)
        if source["kind"] == "speech":
            text("<speaker_description>", "core_text", "core_text", sid)
            text("<transcript>", "transcript", "transcript", sid)
        else:
            text("<description>", "core_text", "core_text", sid)
        expect("<activity_begin>")
        expect("<onset_frame>"); value("time", ("onset_sec",), sid)
        expect("<offset_frame>"); value("time", ("offset_sec",), sid)
        expect("<activity_end>")
        expect("<trajectory_begin>"); value("motion", ("motion",), sid)
        motion = source["trajectory"]["type"]
        if motion == "static":
            position("<position>", ("start", "end"), sid)
        elif motion == "linear":
            position("<start>", ("start",), sid)
            position("<end>", ("end",), sid)
        else:
            raise ValueError("Request supervision currently supports static/linear only")
        expect("<trajectory_end>"); expect("<source_end>")
    expect("<plan_eos>")
    if cursor != len(tokens):
        raise ValueError("Unexpected trailing tokens")
    return result
