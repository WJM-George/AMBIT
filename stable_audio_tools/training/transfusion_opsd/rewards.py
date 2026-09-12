"""Offline plan feedback and a limited, differentiable FOA spatial verifier.

M2D is intentionally absent. Multi-source mixture intensity cannot identify
each source: spatial supervision below uses only plan-defined solo windows.
Full semantic/speech/preservation protection must be composed by the caller.
"""
from __future__ import annotations

import json
from typing import Mapping, Sequence

import torch
from torch import Tensor

from .objectives import RewardScore


class UnobservableAudio(ValueError):
    """The fixed evidence windows do not support this spatial query."""


class ScenePlanReward:
    """Exact, codec-projected field feedback for the existing synthetic tasks.

    The reference is offline supervision; no observation/model method accepts
    it. Source slots must already follow the selected route's canonical order.
    Use independent task metrics for real requests or semantically equivalent
    alternative descriptions; this exact verifier does not judge paraphrases.
    """

    def __init__(self, reference: Mapping, *, codec,
                 primary_fields: Sequence[str] = ("source_count", "trajectory"),
                 protected_fields: Sequence[str] = ("kind", "description", "speaker_description", "transcript",
                                                    "activity", "gain_db", "room", "duration_sec")):
        self.codec = codec
        self.reference = codec.project_plan(reference)
        self.primary_fields = tuple(primary_fields)
        self.protected_fields = tuple(protected_fields)
        if not self.primary_fields:
            raise ValueError("at least one primary plan field is required")
        known = {"source_count", "trajectory", "kind", "description", "speaker_description",
                 "transcript", "activity", "gain_db", "room", "duration_sec"}
        if not set(self.primary_fields + self.protected_fields).issubset(known):
            raise ValueError("unknown ScenePlan reward field")

    def __call__(self, prediction: Mapping | None) -> RewardScore:
        keys = ("source_count", "source_slots", *self.protected_fields)
        if prediction is None:
            return RewardScore(0., {key: 1. for key in keys})
        candidate = self.codec.project_plan(prediction)
        reference = {source["source_id"]: source for source in self.reference["sources"]}
        sources = {source["source_id"]: source for source in candidate["sources"]}
        def field_score(field):
            if field == "source_count":
                return float(len(reference) == len(sources))
            if field in {"duration_sec", "room"}:
                return float(candidate[field] == self.reference[field])
            def canonical(value):
                return json.dumps(value, sort_keys=True, ensure_ascii=False)
            return sum(float(sid in sources and canonical(source.get(field)) == canonical(sources[sid].get(field)))
                       for sid, source in reference.items()) / len(reference)
        utility = sum(field_score(field) for field in self.primary_fields) / len(self.primary_fields)
        costs = {field: 1 - field_score(field) for field in self.protected_fields}
        costs.update(source_count=1 - field_score("source_count"), source_slots=float(reference.keys() != sources.keys()))
        return RewardScore(utility, costs)


class FoaSpatialReward:
    """Own-plan DOA execution reward with fixed, observable time windows.

    This is a spatial pilot scorer, NOT an audio quality or semantic evaluator.
    Silence cannot delete windows or raise coverage. Distance is not inferred
    from mixture loudness. A caller may add independent costs via extra_costs.
    """

    def __init__(self, plan, *, model_num_samples: int, min_solo_fraction: float = 0.05,
                 energy_floor: float = 1e-6, extra_costs=None):
        from ...data.model_sceneplan import compile_model_44_controls
        if not 0 < min_solo_fraction <= 1 or energy_floor <= 0:
            raise ValueError("invalid spatial calibration")
        self.num_samples = model_num_samples
        self.energy_floor = energy_floor
        self.extra_costs = extra_costs
        # The final partial frame is padded for analysis, so the denominator is
        # fixed by the plan, never by output energy or source disappearance.
        controls = compile_model_44_controls(plan, model_num_samples=model_num_samples)
        active = torch.as_tensor(controls["source_event_frame_ids"]) != 0
        self.any_active = active.any(0)
        self.solo = active.sum(0) == 1
        self.coverage = float(self.solo.sum() / self.any_active.sum().clamp_min(1))
        if not self.solo.any() or self.coverage < min_solo_fraction:
            raise UnobservableAudio("insufficient plan-defined solo windows; mixture DOA cannot supervise individual sources")
        features = torch.as_tensor(controls["source_trajectory_features"])
        directions = torch.stack((features[..., 1] * features[..., 3],
                                  features[..., 0] * features[..., 3], features[..., 2]), -1)
        self.expected_direction = (directions * active[..., None]).sum(0)

    def _measure(self, waveform: Tensor):
        from ...data.foa_intensity import foa_to_intensity_trajectory
        if waveform.ndim != 3 or waveform.shape[:2] != (1, 4) or waveform.shape[-1] != self.num_samples:
            raise ValueError("spatial reward expects exact-length [1,4,N] WYZX FOA")
        if not torch.isfinite(waveform).all():
            raise ValueError("non-finite output is a failed generation")
        frames = len(self.solo)
        padded = torch.nn.functional.pad(waveform[0].float(), (0, frames * 1024 - self.num_samples))
        trajectory = foa_to_intensity_trajectory(padded)
        energy = padded[0].view(frames, 1024).square().mean(-1)
        level = energy / (energy + self.energy_floor)
        direction = self.expected_direction.to(waveform.device)
        solo, active = self.solo.to(waveform.device), self.any_active.to(waveform.device)
        alignment = ((trajectory[:, :3] * direction).sum(-1).clamp(-1, 1) + 1) / 2
        utility = (alignment * level)[solo].mean()
        silence = (1 - level)[active].mean()
        leakage = energy[~active].sum() / energy.sum().clamp_min(self.energy_floor)
        return utility, silence, leakage

    def differentiable(self, waveform: Tensor) -> Tensor:
        return self._measure(waveform)[0]

    @torch.no_grad()
    def score(self, waveform: Tensor) -> RewardScore:
        utility, silence, leakage = self._measure(waveform)
        costs = {"silence": float(silence), "activity_leakage": float(leakage),
                 "clipping": float((waveform.abs() > 1).float().mean()),
                 "unobservable_fraction": 1 - self.coverage}
        if self.extra_costs is not None:
            extra = dict(self.extra_costs(waveform))
            if costs.keys() & extra.keys():
                raise ValueError("independent protection names collide with spatial costs")
            costs.update(extra)
        return RewardScore(float(utility), costs)
