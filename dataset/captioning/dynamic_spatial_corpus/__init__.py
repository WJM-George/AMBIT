"""Dynamic spatial-caption templates and normalized manifest slots."""

from .spatial_caption_slots import ClipSlots, SourceSlots, extract_clip_slots
from .spatial_template_corpus import (
    TemplateSpec,
    load_corpus,
    render_stage_caption,
    select_template_for_row,
)

__all__ = [
    "ClipSlots",
    "SourceSlots",
    "TemplateSpec",
    "extract_clip_slots",
    "load_corpus",
    "render_stage_caption",
    "select_template_for_row",
]
