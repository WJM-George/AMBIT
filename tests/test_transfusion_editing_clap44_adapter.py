import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from stable_audio_tools.training.transfusion_opsd.editing_clap44_adapter import StructuredEditingOPSDAdapter


class ToyAR(nn.Module):
    def __init__(self, diffusion, clap, prompt):
        super().__init__()
        self.editing_dit = diffusion.model.model
        self.instruction_conditioner = prompt
        self.source_clap_model = clap
        self.source_semantic_mode = 'clap44_audio_caption_aux'
        self.source_structure = nn.Linear(2,2)
        self.plan_adapter = nn.Module()
        self.plan_adapter.slot_residual = nn.Linear(2,2)
        self.source_semantic_bridge = nn.Module()
        self.source_semantic_bridge.source_to_caption = nn.Linear(2,2)

    @property
    def shared_transformer(self):
        return self.editing_dit.transformer


def fixture():
    diffusion = nn.Module()
    diffusion.model = nn.Module()
    diffusion.model.model = nn.Module()
    diffusion.model.model.transformer = nn.Linear(2,2)
    diffusion.conditioner = nn.Module()
    prompt = nn.Linear(2,2)
    prompt.model = nn.Linear(2,2).eval().requires_grad_(False)
    prompt.enable_grad = False
    local = nn.Module()
    local.gain_projection_weight = nn.Parameter(torch.zeros(32,1))
    local.editing_gain_mode = 'provided'
    diffusion.conditioner.conditioners = nn.ModuleDict(dict(prompt=prompt,local=local))
    clap = nn.Linear(2,2).eval().requires_grad_(False)
    ar = ToyAR(diffusion,clap,prompt).eval()
    return SimpleNamespace(editing_ar=ar,diffusion=diffusion,codec=object(),audio_autoencoder=None)


def test_fork_preserves_shared_structure_without_mutating_parent_or_unfreezing_observer():
    pipeline = fixture()
    before = {k:v.clone() for k,v in pipeline.editing_ar.state_dict().items()}
    adapter = StructuredEditingOPSDAdapter.from_editing_pipeline(pipeline)
    assert adapter.ar.shared_transformer is adapter.diffusion.model.model.transformer
    assert adapter.ar.shared_transformer is not pipeline.editing_ar.shared_transformer
    assert adapter.ar.source_clap_model is pipeline.editing_ar.source_clap_model
    assert adapter.prompt_conditioner is adapter.diffusion.conditioner.conditioners['prompt']
    assert adapter.prompt_conditioner is not pipeline.editing_ar.instruction_conditioner
    adapter.train()
    adapter.assert_frozen_dependencies()
    assert not any(p.requires_grad for p in adapter.ar.source_semantic_bridge.source_to_caption.parameters())
    with torch.no_grad():
        next(adapter.ar.shared_transformer.parameters()).add_(1.)
    assert all(torch.equal(before[k],v) for k,v in pipeline.editing_ar.state_dict().items())
    teacher = adapter.frozen_copy()
    assert teacher.ar.shared_transformer is teacher.diffusion.model.model.transformer
    assert teacher.ar.shared_transformer is not adapter.ar.shared_transformer
    assert not any(p.requires_grad for p in teacher.parameters())


def test_refuses_foreign_or_unfrozen_encoder_and_missing_trained_gain():
    pipeline = fixture()
    pipeline.editing_ar.source_clap_model.requires_grad_(True)
    with pytest.raises(ValueError,match='already be frozen'):
        StructuredEditingOPSDAdapter.from_editing_pipeline(pipeline)
    pipeline = fixture()
    pipeline.diffusion.conditioner.conditioners['local'].editing_gain_mode = 'zero'
    with pytest.raises(ValueError,match='desired-gain'):
        StructuredEditingOPSDAdapter.from_editing_pipeline(pipeline)


def test_observation_only_accepts_covered_native_source_and_raw_instruction():
    adapter = StructuredEditingOPSDAdapter.from_editing_pipeline(fixture())
    latent = torch.randn(1,64,432,requires_grad=True)
    mask = torch.arange(432)[None] < 431
    observation = adapter.observe_editing(sample_id='test',request='Move the speaker left.',
        source_foa_latent=latent,source_attention_mask=mask,model_num_samples=431*1024)
    assert not observation.source_foa_latent.requires_grad
    assert observation.source_m2d_audio_embedding is None
    with pytest.raises(ValueError,match='bucket cannot cover'):
        adapter.observe_editing(sample_id='test',request='Move left.',source_foa_latent=latent,
            source_attention_mask=torch.ones_like(mask),model_num_samples=500*1024)
