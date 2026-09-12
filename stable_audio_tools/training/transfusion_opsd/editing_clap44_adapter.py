"""OPSD boundary for the existing structured shared AR/Editing DiT candidate.

The source, request, native slot memory and desired-gain controls follow the
existing model. No ground-truth source plan or target audio enters planning.
Sampling stops gradients; fitting recomputes native planning and RF forwards.
"""
from __future__ import annotations

import copy
import math
from contextlib import nullcontext

import torch
from torch import nn

from .adapters import EditingObservation, RenderCondition, TransfusionOPSDAdapter
from .objectives import sample_ar


class StructuredEditingOPSDAdapter(TransfusionOPSDAdapter):
    contract = 'structured_native_CLAP44_editing_execution_coupled_OPSD_v1'

    def __init__(self, *, ar, diffusion, codec, audio_autoencoder=None,
                 fork=True, cfg_scale=1.):
        # The older adapter enables every registered AR parameter and omits
        # desired gains. Keep that historical implementation intact.
        nn.Module.__init__(self)
        clap = getattr(ar, 'source_clap_model', None)
        if (getattr(ar, 'source_semantic_mode', None) != 'clap44_audio_caption_aux'
                or not isinstance(clap, nn.Module)
                or not hasattr(ar, 'source_structure')
                or not hasattr(ar.plan_adapter, 'slot_residual')):
            raise ValueError('Expected the existing structured native CLAP44 Editing AR')
        prompt = ar.instruction_conditioner
        backbone = getattr(prompt, 'model', None)
        frozen = [m for m in (clap, backbone, audio_autoencoder) if isinstance(m, nn.Module)]
        if any(p.requires_grad for m in frozen for p in m.parameters()):
            raise ValueError('Native CLAP, Qwen and FOA VAE dependencies must already be frozen')
        if bool(getattr(prompt, 'enable_grad', False)):
            raise ValueError('The Qwen backbone must remain frozen')
        if fork:
            memo = {id(m): m for m in frozen}
            pretransform = getattr(diffusion, 'pretransform', None)
            if pretransform is not None:
                memo[id(pretransform)] = pretransform
            ar, diffusion = copy.deepcopy((ar, diffusion), memo)
        if ar.shared_transformer is not diffusion.model.model.transformer:
            raise ValueError('Editing AR and DiT must share the same Transformer object')
        if ar.instruction_conditioner is not diffusion.conditioner.conditioners['prompt']:
            raise ValueError('Editing request and DiT prompt projections must remain shared')
        if not math.isfinite(cfg_scale) or cfg_scale < 0:
            raise ValueError('Editing CFG must be finite and nonnegative')
        local = [m for m in diffusion.conditioner.conditioners.values() if hasattr(m, 'gain_projection_weight')]
        if len(local) != 1 or getattr(local[0], 'editing_gain_mode', None) != 'provided':
            raise ValueError('The current structured Editing checkpoint requires its trained desired-gain conditioner')
        self.mode, self.ar, self.diffusion, self.codec = 'editing', ar, diffusion, codec
        self.cfg_scale, self.cfg_rescale_phi = float(cfg_scale), 0.
        self.__dict__['audio_autoencoder'] = audio_autoencoder
        self.__dict__['source_pipeline'] = None
        diffusion.pretransform = None
        frozen_ids = {id(p) for m in frozen for p in m.parameters()}
        for parameter in self.parameters():
            parameter.requires_grad_(id(parameter) not in frozen_ids)
        # This unused contrastive query is excluded by the current joint recipe.
        ar.source_semantic_bridge.source_to_caption.requires_grad_(False)
        self.eval()
        self.assert_frozen_dependencies()

    def assert_frozen_dependencies(self):
        dependencies = [self.ar.source_clap_model, self.prompt_conditioner.model, self.audio_autoencoder]
        for module in dependencies:
            if module is not None and (module.training or any(p.requires_grad for p in module.parameters())):
                raise RuntimeError('An Editing OPSD observer dependency was unfrozen or entered training mode')

    def train(self, mode=True):
        nn.Module.train(self, mode)
        for module in (self.ar.source_clap_model, self.prompt_conditioner.model, self.audio_autoencoder):
            if module is not None:
                module.eval()
        return self

    @classmethod
    def from_editing_pipeline(cls, pipeline, **kwargs):
        return cls(ar=pipeline.editing_ar, diffusion=pipeline.diffusion, codec=pipeline.codec,
                   audio_autoencoder=pipeline.audio_autoencoder, **kwargs)

    def frozen_copy(self):
        result = type(self)(ar=self.ar, diffusion=self.diffusion, codec=self.codec,
            audio_autoencoder=self.audio_autoencoder, cfg_scale=self.cfg_scale)
        return result.eval().requires_grad_(False)

    def observe_editing(self, *, sample_id, request, source_foa_latent,
                        source_attention_mask, model_num_samples):
        if not isinstance(request, str) or not request.strip():
            raise ValueError('Editing needs the raw nonempty edit instruction')
        source = source_foa_latent.detach().to(self.device, dtype=torch.float32)
        mask = source_attention_mask.detach().to(self.device, dtype=torch.bool)
        if source.ndim != 3 or model_num_samples > source.shape[-1] * 1024:
            raise ValueError('The source latent bucket cannot cover the declared audio duration')
        return EditingObservation(sample_id, request, source, mask, model_num_samples)

    def _amp(self):
        return torch.autocast('cuda', dtype=torch.bfloat16) if self.device.type == 'cuda' else nullcontext()

    def _encode_requests(self, texts):
        return self.ar.encode_edit_instructions(texts, device=self.device)

    def _check_observation(self, observation):
        if not isinstance(observation, EditingObservation) or observation.source_m2d_audio_embedding is not None:
            raise ValueError('Use native source FOA plus request; M2D input is not accepted')
        self.assert_frozen_dependencies()

    def _forward_ar(self, observation, token_ids, context, context_mask, *, return_query=False,
                    source_features=None, return_structure=False):
        self._check_observation(observation)
        if return_query:
            raise ValueError('The current structured model does not use a source-caption query')
        tokens = token_ids.to(self.device)
        return self.ar(observation.source_foa_latent, observation.source_attention_mask,
            tokens, torch.ones_like(tokens, dtype=torch.bool), context, context_mask,
            source_clap_features=source_features, return_structure=return_structure)

    def _ar_logits(self, observation, token_ids, text, *, return_query=False):
        with self._amp():
            context, mask = self._encode_requests([text])
            return self._forward_ar(observation, token_ids, context, mask, return_query=return_query)

    def student_with_structure(self, observation, token_ids):
        """Expose the existing native heads for their paired preservation losses."""
        with self._amp():
            context, mask = self._encode_requests([observation.request])
            return self._forward_ar(observation, token_ids, context, mask, return_structure=True)

    @torch.no_grad()
    def sample_plan(self, observation, *, privileged_text=None, **kwargs):
        self._check_observation(observation)
        text = observation.request if privileged_text is None else privileged_text
        with self._amp():
            context, mask = self._encode_requests([text])
            features = self.ar.source_clap_model.source_features(
                observation.source_foa_latent, observation.source_attention_mask)
            return sample_ar(lambda tokens: self._forward_ar(observation, tokens, context, mask,
                source_features=features), self.codec, device=self.device,
                allowed_fn=lambda prefix: self.allowed_next_ids(observation, prefix), **kwargs)

    @torch.no_grad()
    def native_plan(self, observation, *, max_plan_tokens=512):
        """Default inference, separately from stochastic teacher collection."""
        from ...models.sceneplan_transfusion_editing_clap44_pipeline import ScenePlanTransfusionEditingCLAP44Pipeline
        self._check_observation(observation)
        view = ScenePlanTransfusionEditingCLAP44Pipeline(diffusion=self.diffusion,
            editing_ar=self.ar, codec=self.codec)
        plans, tokens = view.generate_new_sceneplans(observation.source_foa_latent,
            observation.source_attention_mask, [observation.request],
            duration_sec=observation.model_num_samples/44100, max_plan_tokens=max_plan_tokens)
        return plans[0], tokens[0]

    def render_condition(self, observation, plan):
        from ...models.sceneplan_transfusion_editing_pipeline import ScenePlanTransfusionEditingDiTPipeline
        from ...models.sceneplan_editing_gain_adapter import install_pipeline_gain_metadata
        from ...data.sceneplan_transfusion_editing_dataset import make_editing_dit_cfg_unknown_metadata
        self._check_observation(observation)
        view = ScenePlanTransfusionEditingDiTPipeline(diffusion=self.diffusion)
        install_pipeline_gain_metadata(view)
        rows = view._conditioning_metadata(observation.source_foa_latent,
            observation.source_attention_mask, [plan], [observation.model_num_samples])
        negative = [make_editing_dit_cfg_unknown_metadata(row) for row in rows]
        if any(a['source_foa_latent'] is not b['source_foa_latent'] for a,b in zip(rows,negative)):
            raise RuntimeError('Editing CFG must retain the same source audio')
        return RenderCondition(plan, rows, negative, observation.source_attention_mask, observation.model_num_samples)
