from __future__ import annotations
from typing import Any, Sequence


def editing_ar_allowed_next_ids(
    codec: Any, prefix: Sequence[int], *, fixed_duration_sec: float | None = None
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
        slot = sum((int(token) == int(tokens["<source_begin>"]) for token in prefix)) - 1
        if not 0 <= slot < len(slot_ids) or slot_ids[slot] not in allowed:
            raise RuntimeError("Editing AR source slots are not contiguous")
        return {slot_ids[slot]}
    return allowed
