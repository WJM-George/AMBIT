"""CLAP44 joint AR/RF behavior without old plans or caption caches.

CLAP is pretrained with audio/text positives and negatives. During joint AR,
its frozen *source audio* features provide a modest distillation target for
the source adapter. This objective is distinct from CLAP pretraining and does
not introduce a new pool of possibly false semantic negatives.
"""
from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Mapping

import torch
from torch import nn
from torch.nn import functional as F

from scripts.t2a.train.train_sceneplan_transfusion_editing_ar_joint_full import JointEditingModule, _move_joint_batch

CLAP44_AR_VARIANTS = {"latent_only", "global_only", "sequence_only", "global_and_sequence"}


def normalized_joint_loss(ar_sum,rf_sum,auxiliary_sum,denominators,*,world_size,lambda_ar,lambda_rf,lambda_auxiliary):
    """DDP averages ranks: multiply local numerators by world/global count."""
    if denominators.shape != (3,) or not bool(torch.isfinite(denominators).all() and (denominators>0).all()) or world_size<1:
        raise ValueError("invalid global joint objective denominators")
    return world_size*(lambda_ar*ar_sum/denominators[0]+lambda_rf*rf_sum/denominators[1]+lambda_auxiliary*auxiliary_sum/denominators[2])


def source_feature_distillation(query, teacher, *, semantic_dim=512, scene_weight=.25):
    if query.ndim != 2 or query.shape != teacher.shape or not 0 < semantic_dim < query.shape[-1] or scene_weight < 0:
        raise ValueError("invalid CLAP44 source distillation tensors/settings")
    if not bool(torch.isfinite(query).all() and torch.isfinite(teacher).all()):
        raise ValueError("non-finite CLAP44 source distillation input")
    teacher = teacher.detach().float()
    semantic = F.cosine_similarity(query[:, :semantic_dim].float(), teacher[:, :semantic_dim], dim=-1)
    scene = F.cosine_similarity(query[:, semantic_dim:].float(), teacher[:, semantic_dim:], dim=-1)
    return (1-semantic) + scene_weight*(1-scene), {"semantic_cosine": semantic.detach(), "scene_cosine": scene.detach()}


def shuffle_source_features(features: Mapping[str, Any], permutation: torch.Tensor) -> dict[str, Any]:
    return {k: value.index_select(0, permutation.to(value.device)) if isinstance(value, torch.Tensor) else value for k,value in features.items()}


class CLAP44JointEditingModule(JointEditingModule):
    """Exactly the existing aligned AR/RF branches, plus a source-only teacher."""

    planning_pretrain = False

    def forward(self, *, source_foa_latent, source_attention_mask, **kwargs):
        if any("m2d" in key.lower() for key in kwargs):
            raise ValueError("CLAP44 joint forward cannot consume M2D tensors")
        if "source_clap_features" in kwargs:
            raise ValueError("CLAP44 joint training derives features from its own source")
        encoder = self.ar.source_clap_model
        features = None if encoder is None else encoder.source_features(source_foa_latent, source_attention_mask)
        logits, prediction, query = super().forward(source_foa_latent=source_foa_latent, source_attention_mask=source_attention_mask, source_clap_features=features, **kwargs)
        teacher = query.detach() if features is None else features["global"].to(query)
        return logits, prediction, query, teacher


class CLAP44ARPretrainModule(CLAP44JointEditingModule):
    """Learn editing plans while replaying P10 RF without an editing reference.

    The P10 warmstart adds zero-initialized reference columns. Keeping the RF
    reference zero makes this generation replay, not qualified Editing DiT.
    The AR branch still receives the real source audio and raw instruction.
    """

    planning_pretrain = True

    def forward(self, *, source_foa_latent, source_attention_mask, plan_input_ids,
                plan_attention_mask, raw_edit_requests, metadata, noised_target,
                timesteps, rf_padding_mask):
        encoder = self.ar.source_clap_model
        features = None if encoder is None else encoder.source_features(source_foa_latent, source_attention_mask)
        context, mask = self.ar.encode_edit_instructions(raw_edit_requests, device=source_foa_latent.device)
        output = self.ar(source_foa_latent, source_attention_mask, plan_input_ids,
                         plan_attention_mask, context, mask,
                         **({"source_clap_features": features} if features is not None else {}),
                         return_source_contrastive_query=features is not None)
        if features is None:
            logits, query = output, source_foa_latent.new_empty((0, 768))
        else:
            logits, query = output
        inputs = replay_conditioning(self, metadata, source_foa_latent.device)
        prediction = self.diffusion.model(noised_target, timesteps, **inputs,
                                         cfg_dropout_prob=0., padding_mask=rf_padding_mask)
        teacher = query.detach() if features is None else features["global"].to(query)
        return logits, prediction, query, teacher


def replay_conditioning(module, metadata, device):
    conditioning = dict(module.diffusion.conditioner(metadata, device))
    if module.planning_pretrain:
        source, mask = conditioning["source_foa_latent"]
        conditioning["source_foa_latent"] = [torch.zeros_like(source), mask]
    return module.diffusion.get_conditioning_inputs(conditioning)


def optimizer_groups(module: CLAP44JointEditingModule, *, ar_lr: float, shared_lr: float, dit_lr: float):
    """Own adapters and one shared block stack; no frozen CLAP/Qwen weights."""
    ar = module.ar
    adapters = [ar.source_audio_adapter, ar.plan_adapter, ar.source_semantic_bridge]
    def unique(modules):
        values = {}
        for item in modules:
            for p in item.parameters():
                if p.requires_grad: values[id(p)] = p
        return list(values.values())
    specific = unique(adapters) + [ar.source_audio_type_embedding, ar.plan_type_embedding]
    shared = unique([ar.shared_transformer.layers])
    seen = {id(p) for p in specific+shared}
    remaining = [p for p in unique([module.diffusion]) if id(p) not in seen]
    groups = [specific, shared, remaining]
    flat = [p for group in groups for p in group]
    if len({id(p) for p in flat}) != len(flat) or {id(p) for p in flat} != {id(p) for p in module.parameters() if p.requires_grad}:
        raise RuntimeError("CLAP44 joint optimizer must cover every trainable parameter exactly once")
    if ar.source_clap_model is not None and any(p.requires_grad for p in ar.source_clap_model.parameters()):
        raise RuntimeError("CLAP44 must stay frozen during joint AR/RF")
    names = ["editing_ar_adapters", "shared_transformer_blocks", "editing_dit_and_conditioners"]
    return [{"params": group, "lr": lr, "base_lr": lr, "group_name": name} for group,lr,name in zip(groups, (ar_lr,shared_lr,dit_lr),names)], {name: sum(p.numel() for p in group) for name,group in zip(names,groups)}


@torch.no_grad()
def validation_metrics(module, loader, *, device, max_batches=32, scene_weight=.25):
    """Teacher-forced diagnostics only. Full free AR/audio gates are separate."""
    was_training = module.training
    module.eval()
    totals = torch.zeros(12, dtype=torch.float64)
    try:
        for batch_number, batch in enumerate(loader):
            if batch_number >= max_batches: break
            ar, target, metadata, rf_mask = _move_joint_batch(batch, device)
            source, mask = ar["source_foa_latent"], ar["source_attention_mask"]
            permutation = torch.arange(source.shape[0], device=device).roll(1)
            if source.shape[0] < 2:
                raise ValueError("reference shuffle diagnostics require at least two rows")
            context, context_mask = module.ar.encode_edit_instructions(ar["raw_edit_requests"], device=device)
            autocast = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
            with autocast:
                encoder = module.ar.source_clap_model
                features = None if encoder is None else encoder.source_features(source, mask)
                clean_kwargs = {} if features is None else {"source_clap_features": features}
                zero_kwargs = {} if features is None else {"source_clap_features": features, "source_clap_keep_mask": torch.zeros(source.shape[0], dtype=torch.bool, device=device)}
                donor_kwargs = {} if features is None else {"source_clap_features": shuffle_source_features(features, permutation)}
                clean = module.ar(source, mask, ar["plan_input_ids"], ar["plan_attention_mask"], context, context_mask, return_source_contrastive_query=features is not None, **clean_kwargs)
                zero = module.ar(torch.zeros_like(source), mask, ar["plan_input_ids"], ar["plan_attention_mask"], context, context_mask, **zero_kwargs)
                donor = module.ar(source[permutation], mask[permutation], ar["plan_input_ids"], ar["plan_attention_mask"], context, context_mask, **donor_kwargs)
            distillation_sum = semantic_sum = scene_sum = 0.
            if features is None: logits = clean
            else:
                logits, query = clean
                values, diag = source_feature_distillation(query, features["global"].to(query), semantic_dim=encoder.config.semantic_dim, scene_weight=scene_weight)
                distillation_sum = float(values.sum()); semantic_sum = float(diag["semantic_cosine"].sum()); scene_sum = float(diag["scene_cosine"].sum())
            labels = ar["plan_labels"]; valid = labels.ne(-100)
            losses = [float(F.cross_entropy(x.float().flatten(0,1), labels.flatten(), ignore_index=-100, reduction="sum")) for x in (logits,zero,donor)]
            prediction = logits.argmax(-1)
            generator = torch.Generator(device=device).manual_seed(8_100_000+batch_number)
            noise = torch.randn(target.shape, generator=generator, device=device)
            inputs = replay_conditioning(module, metadata, device)
            autocast = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
            with autocast:
                rf = module.diffusion.model(.5*(target+noise), torch.full((target.shape[0],), .5, device=device), **inputs, cfg_dropout_prob=0., padding_mask=rf_mask)
            rf_sum = float((((rf.float()-(noise-target))**2)*rf_mask[:,None]).sum())
            totals += torch.tensor([*losses, int(valid.sum()), int((prediction.eq(labels)&valid).sum()), int((prediction.eq(labels)|~valid).all(1).sum()), len(labels), rf_sum, int(rf_mask.sum())*64, distillation_sum,semantic_sum,scene_sum], dtype=torch.float64)
    finally:
        module.train(was_training)
    v = totals.tolist()
    return {"clean_ar_ce": v[0]/max(v[3],1), "zero_reference_ar_ce": v[1]/max(v[3],1), "shuffled_reference_ar_ce": v[2]/max(v[3],1), "tokens":v[3], "token_accuracy":v[4]/max(v[3],1), "teacher_forced_sequence_exact":v[5]/max(v[6],1), "sequences":v[6], "rf_mse":v[7]/max(v[8],1), "rf_mode":"p10_generation_zero_reference" if module.planning_pretrain else "editing_reference", "source_feature_distillation":v[9]/max(v[6],1), "source_semantic_cosine":v[10]/max(v[6],1), "source_scene_cosine":v[11]/max(v[6],1), "quality_gate_passed":False}
