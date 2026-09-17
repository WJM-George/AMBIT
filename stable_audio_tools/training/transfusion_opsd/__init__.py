"""On-policy self-distillation for the shared AR + DiT Transformer.

The current recipe is Editing: STE off, hidden request-side GT, fixed 40k
reference retention, and same-plan RF teachers. See docs/OPSD.md.
"""

from .objectives import DiffusionTargets, RewardScore, forward_kl
from .adapters import GenerationObservation, EditingObservation, TransfusionOPSDAdapter
from .event_trainer import EventFitConfig, EventFitBatch, EventOPSDTrainer, EventRefinementTrainer

__all__ = [
    "DiffusionTargets", "RewardScore", "forward_kl",
    "GenerationObservation", "EditingObservation", "TransfusionOPSDAdapter",
    "EventFitConfig", "EventFitBatch", "EventOPSDTrainer", "EventRefinementTrainer",
]
