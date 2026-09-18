"""On-policy self-distillation for the shared AR + DiT Transformer.

The current mainline is Editing v3: STE off, exclusive per-branch
execution-or-GT fallback, fixed 40k reference retention, and same-plan RF
teachers. See docs/OPSD.md and docs/TRANSFUSION_OPSD_MAINLINE.md.
"""

from .objectives import DiffusionTargets, RewardScore, forward_kl
from .adapters import GenerationObservation, EditingObservation, TransfusionOPSDAdapter
from .event_trainer import EventFitConfig, EventFitBatch, EventOPSDTrainer, EventRefinementTrainer

__all__ = [
    "DiffusionTargets", "RewardScore", "forward_kl",
    "GenerationObservation", "EditingObservation", "TransfusionOPSDAdapter",
    "EventFitConfig", "EventFitBatch", "EventOPSDTrainer", "EventRefinementTrainer",
]
