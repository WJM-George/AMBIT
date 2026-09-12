"""Execution-linked self-distillation for the complete Generation EVENT policy.

Current research entry points and retained controls are in docs/transfusion_opsd.md.
Native Generation/Editing adapters remain reusable; Editing OPSD is unvalidated.
"""

from .objectives import DiffusionTargets, RewardScore, forward_kl
from .adapters import GenerationObservation, EditingObservation, TransfusionOPSDAdapter
from .event_trainer import EventFitConfig, EventFitBatch, EventOPSDTrainer, EventRefinementTrainer

__all__ = [
    "DiffusionTargets", "RewardScore", "forward_kl",
    "GenerationObservation", "EditingObservation", "TransfusionOPSDAdapter",
    "EventFitConfig", "EventFitBatch", "EventOPSDTrainer", "EventRefinementTrainer",
]
