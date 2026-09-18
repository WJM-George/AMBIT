#!/usr/bin/env python3
"""Editing Transfusion route C trainer. Not OPSD.

Trains the event-description head on the editing AR+DiT Transfusion:
discrete ScenePlan and sceneplan_44 stay native; Qwen keeps scene/speech
tokens; only event-description spans are replaced.

This script must not be pointed at a transfusion_opsd run directory. It
does not import OPSD trainers or launchers. It starts no process unless
you invoke it yourself.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stable_audio_tools.models.sceneplan_transfusion_editing_event_condition import (  # noqa: E402
    EventHeadEditingTransfusion,
)
from stable_audio_tools.models.sceneplan_transfusion_editing_event_head import (  # noqa: E402
    EditingEventHead,
    EventHeadConfig,
    distillation_loss,
)


FORBIDDEN_OPSD_MARKERS = ("transfusion_opsd", "editing_opsd", "opsd_")


def _reject_opsd_path(path: Path, name: str) -> None:
    text = str(path.resolve()).lower()
    if any(marker in text for marker in FORBIDDEN_OPSD_MARKERS):
        raise ValueError(f"{name} points at an OPSD path; route C stays on editing Transfusion only: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="Editing event-head run directory, never an OPSD run")
    parser.add_argument("--editing-checkpoint", type=Path, required=True, help="Editing AR+DiT checkpoint")
    parser.add_argument("--distill-weight", type=float, default=1.0)
    parser.add_argument("--freeze-backbone", action="store_true", default=True)
    parser.add_argument("--unfreeze-backbone", action="store_true", help="Train event head with the editing AR/DiT")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_event_module(diffusion, ar, codec, *, seed: int) -> EventHeadEditingTransfusion:
    head = EditingEventHead(EventHeadConfig(speech_tokens=True))
    head.reset_condition_adapter(seed)
    return EventHeadEditingTransfusion(diffusion=diffusion, ar=ar, codec=codec, event_head=head)


def event_objective(output: dict, *, distill_weight: float):
    distill = distillation_loss(output["event_prediction"], output["event_teacher"], output["event_mask"])
    return {"distill": distill["loss"], "loss": distill["loss"] * distill_weight, **distill}


def main() -> None:
    args = parse_args()
    _reject_opsd_path(args.run_dir, "run-dir")
    _reject_opsd_path(args.editing_checkpoint, "editing-checkpoint")
    if args.unfreeze_backbone:
        args.freeze_backbone = False
    args.run_dir.mkdir(parents=True, exist_ok=True)
    raise SystemExit(
        "Event-head C is wired in AMBIT editing Transfusion "
        f"(freeze_backbone={args.freeze_backbone}). "
        "This entry refuses to auto-start GPU training."
    )


if __name__ == "__main__":
    main()
