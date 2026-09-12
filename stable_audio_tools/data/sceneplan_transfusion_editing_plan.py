"""Observable source ordering for Editing AR's model-facing ScenePlans.

Persistent renderer slots are random provenance, not recoverable audio labels.
Keep the frozen pair inventory intact and canonicalize only the model view.
"""

from __future__ import annotations

import copy
import json
from typing import Any, Mapping, Sequence

from .model_sceneplan import validate_model_sceneplan


EDITING_AR_PLAN_POLICY = "codec_v4_onset_kind_content_contiguous_slots_v1"


def canonicalize_editing_plan(
    plan: Mapping[str, Any], *, codec: Any
) -> tuple[dict[str, Any], dict[str, str]]:
    """Relabel a copy without changing any acoustic or semantic field.

    Sort by codec-visible onset, kind, then the complete codec-visible source
    content excluding its old id. Quantization comes before ordering so the
    labels cannot depend on distinctions absent from the output vocabulary.
    Identical projected sources have identical targets under either ordering.
    Return the persistent-to-model id map for offline provenance only.
    """

    validate_model_sceneplan(plan)
    projected = codec.project_plan(plan)

    def key(source: Mapping[str, Any]) -> tuple[float, str, str]:
        content = {name: value for name, value in source.items() if name != "source_id"}
        return (
            float(source["activity"]["onset_sec"]),
            str(source["kind"]),
            json.dumps(content, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        )

    original = {str(source["source_id"]): source for source in plan["sources"]}
    result = copy.deepcopy(dict(plan))
    result["sources"] = []
    id_map = {}
    for slot, source in enumerate(sorted(projected["sources"], key=key)):
        old_id = str(source["source_id"])
        new_id = f"source_{slot}"
        renamed = copy.deepcopy(original[old_id])
        renamed["source_id"] = new_id
        result["sources"].append(renamed)
        id_map[old_id] = new_id
    validate_model_sceneplan(result)
    return result, id_map


def editing_ar_allowed_next_ids(
    codec: Any,
    prefix: Sequence[int],
    *,
    fixed_duration_sec: float | None = None,
) -> set[int]:
    """Apply codec grammar, assigning each next source its sequential slot.

    This uses only the already generated prefix; it never consults a teacher
    plan, source-plan metadata, or an offline persistent-to-model mapping.
    """

    allowed = set(codec.allowed_next_ids(prefix, fixed_duration_sec=fixed_duration_sec))
    tokens = codec.token_to_id
    slot_ids = [int(tokens[f"<source_slot_{slot}>"]) for slot in range(4)]
    if allowed.intersection(slot_ids):
        if not allowed.issubset(slot_ids):
            raise RuntimeError("Editing AR codec mixed source slots with other tokens")
        slot = sum(int(token) == int(tokens["<source_begin>"]) for token in prefix) - 1
        if not 0 <= slot < len(slot_ids) or slot_ids[slot] not in allowed:
            raise RuntimeError("Editing AR source slots are not contiguous")
        return {slot_ids[slot]}
    return allowed


__all__ = [
    "EDITING_AR_PLAN_POLICY",
    "canonicalize_editing_plan",
    "editing_ar_allowed_next_ids",
]
