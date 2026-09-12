"""Preserve source-preference values while optimizing only the clean input.

This is an experimental loss component, not a promoted training recipe.
The donor still defines each example's margin, but receives no direct gradient
from this term. Shared parameters can still change donor predictions indirectly.
The native role masks and global eligible-field reduction remain unchanged.
"""
from scripts.t2a.experiments.ar_source_grounding_v1.field_loss import (
    CONTENT_FIELDS, target_field_masks, source_preference_terms as original_terms,
)

METHOD_CONTRACT = 'editing_ar_unchanged_source_field_positive_preference_v1'


def source_preference_terms(clean_token_ce, donor_token_ce, masks, *, margin=.1):
    return original_terms(clean_token_ce, donor_token_ce.detach(), masks, margin=margin)
