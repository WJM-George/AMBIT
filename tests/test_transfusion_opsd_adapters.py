import copy
import re

import pytest
import torch
from torch import nn

from stable_audio_tools.data.model_sceneplan_codec_v4 import ModelScenePlanCodecV4, create_model_sceneplan_codec_v4_artifact
from stable_audio_tools.models.diffusion import DiTWrapper, ConditionedDiffusionModelWrapper
from stable_audio_tools.models.sceneplan_transfusion_generation_ar import ScenePlanTransfusionGenerationAR, DiscreteScenePlanAdapter
from stable_audio_tools.models.sceneplan_transfusion_editing_ar import ScenePlanTransfusionEditingAR, EditingScenePlanAdapter
from stable_audio_tools.models.sceneplan_transfusion_editing_m2d_clap import EditingARSourceSemanticBridge
from stable_audio_tools.models.sceneplan_transfusion_editing_pipeline import ScenePlanTransfusionEditingPipeline
from stable_audio_tools.training.transfusion_opsd.adapters import TransfusionOPSDAdapter, GenerationObservation, EditingObservation
from stable_audio_tools.training.transfusion_opsd.objectives import ARRollout, euler_rollout, forward_kl
from stable_audio_tools.training.transfusion_opsd.rewards import ScenePlanReward, FoaSpatialReward, UnobservableAudio


class Tokenizer:
    def __call__(self, text, *, padding=False, max_length=None, return_tensors=None, **kwargs):
        many = isinstance(text, list)
        texts = text if many else [text]
        rows = []
        for value in texts:
            spans = [(m.start(), m.end()) for m in re.finditer(r"\S+", value)]
            ids = [1 + sum(map(ord, value[a:b])) % 31 for a, b in spans]
            rows.append((ids, spans))
        length = max_length if padding == "max_length" else max(len(row[0]) for row in rows)
        result = {"input_ids": [], "attention_mask": [], "offset_mapping": []}
        for ids, offsets in rows:
            n = len(ids)
            result["input_ids"].append(ids + [0] * (length - n))
            result["attention_mask"].append([1] * n + [0] * (length - n))
            result["offset_mapping"].append(offsets + [(0, 0)] * (length - n))
        if not many:
            result = {key: value[0] for key, value in result.items()}
        return {key: torch.tensor(value) for key, value in result.items()} if return_tensors else result


class Prompt(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer, self.enable_grad = Tokenizer(), False
        self.__dict__["model"] = nn.Embedding(32, 8).requires_grad_(False)
        self.proj_out = nn.Linear(8, 32)
        self.event_role_embed, self.speech_role_embed = nn.Embedding(6, 32), nn.Embedding(6, 32)

    def forward(self, rows, device):
        ids = torch.stack([row["input_ids"] for row in rows]).to(device)
        mask = torch.stack([row["attention_mask"] for row in rows]).to(device).bool()
        length = int(mask.sum(-1).max())
        ids, mask = ids[:, :length], mask[:, :length]
        with torch.no_grad():
            base = self.model(ids)
        return self.proj_out(base) + self.event_role_embed.weight[0] + self.speech_role_embed.weight[0], mask


class Conditioner(nn.Module):
    def __init__(self, prompt, mode):
        super().__init__()
        self.conditioners = nn.ModuleDict({"prompt": prompt})
        self.scale = nn.Parameter(torch.tensor(.2))
        self.mode = mode

    def forward(self, rows, device):
        context, mask = self.conditioners["prompt"]([row["prompt"] for row in rows], device)
        controls = torch.stack([row["sceneplan_44"]["source_trajectory_features"][:, :, 1].sum(0) for row in rows]).to(device)
        control = controls[:, None] * self.scale
        if self.mode == "editing":
            source = torch.stack([row["source_foa_latent"] for row in rows])
            control = torch.cat((control, source), dim=1)
        return {"prompt": [context, mask], "controls": [control, torch.ones_like(controls, dtype=torch.bool)]}


class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Conv1d(64, 4, 1)
    def decode(self, z):
        return self.decoder(z).repeat_interleave(1024, dim=-1)


@pytest.fixture(scope="module")
def codec(tmp_path_factory):
    root = tmp_path_factory.mktemp("opsd-codec") / "codec"
    create_model_sceneplan_codec_v4_artifact(root)
    return ModelScenePlanCodecV4(root)


def plan():
    return {"sample_id": "one", "duration_sec": 2., "room": {"type": "dry"}, "sources": [{
        "source_id": "source_0", "kind": "sound", "description": "A dog barking.", "gain_db": 0.,
        "activity": {"onset_sec": 0., "offset_sec": 2.},
        "trajectory": {"type": "static", "position": {"azimuth_deg": 0., "elevation_deg": 0., "distance_m": 2.}},
    }]}




def make_bundle(mode, codec, monkeypatch):
    torch.manual_seed(19)
    monkeypatch.setattr("stable_audio_tools.models.sceneplan_transfusion_generation_ar.GENERATION_AR_HIDDEN_DIM", 64)
    monkeypatch.setattr("stable_audio_tools.models.sceneplan_transfusion_editing_ar.EDITING_AR_HIDDEN_DIM", 64)
    prompt = Prompt()
    model = DiTWrapper(diffusion_objective="rectified_flow", io_channels=64, embed_dim=64,
        cond_token_dim=32, input_concat_dim=65 if mode == "editing" else 1,
        depth=1, num_heads=2, zero_init_branch_outputs=False, activation_checkpointing=False)
    diffusion = ConditionedDiffusionModelWrapper(model, Conditioner(prompt, mode), io_channels=64,
        sample_rate=44100, min_input_length=1, diffusion_objective="rectified_flow",
        cross_attn_cond_ids=["prompt"], input_concat_ids=["controls"])
    cls = ScenePlanTransfusionGenerationAR if mode == "generation" else ScenePlanTransfusionEditingAR
    ar = cls.__new__(cls)
    nn.Module.__init__(ar)
    ar.pad_id, ar.vocab_size, ar.activation_checkpointing = codec.pad_id, codec.vocab_size, False
    if mode == "generation":
        ar.p10_dit, ar.prompt_conditioner = model.model, prompt
        ar.ar_adapter = DiscreteScenePlanAdapter(vocab_size=codec.vocab_size, hidden_dim=64, pad_id=codec.pad_id)
    else:
        ar.editing_dit, ar.instruction_conditioner = model.model, prompt
        ar.source_audio_adapter = nn.Linear(64, 64)
        ar.source_audio_type_embedding = nn.Parameter(torch.zeros(64))
        ar.plan_type_embedding = nn.Parameter(torch.zeros(64))
        ar.plan_adapter = EditingScenePlanAdapter(vocab_size=codec.vocab_size, hidden_dim=64, pad_id=codec.pad_id)
        ar.source_semantic_bridge = EditingARSourceSemanticBridge(mode="caption_aux", hidden_dim=64)
    ar.eval().requires_grad_(False)
    diffusion.eval().requires_grad_(False)
    vae = VAE().eval().requires_grad_(False)
    if mode == "editing":
        pipeline = ScenePlanTransfusionEditingPipeline(diffusion=diffusion, editing_ar=ar, codec=codec, audio_autoencoder=vae)
        adapter = TransfusionOPSDAdapter.from_editing_pipeline(pipeline)
    else:
        adapter = TransfusionOPSDAdapter(mode=mode, ar=ar, diffusion=diffusion, codec=codec, audio_autoencoder=vae)
    return adapter, ar, diffusion


@pytest.mark.parametrize("mode", ["generation", "editing"])
def test_real_ar_dit_classes_compiler_and_shared_projection_gradients(codec, monkeypatch, mode):
    adapter, original_ar, original_diffusion = make_bundle(mode, codec, monkeypatch)
    assert not any(p.requires_grad for p in original_ar.parameters())
    assert adapter.ar.shared_transformer is adapter.diffusion.model.model.transformer
    assert adapter.ar.shared_transformer is not original_ar.shared_transformer
    assert adapter.prompt_conditioner.model is original_diffusion.conditioner.conditioners["prompt"].model
    assert not any(p.requires_grad for p in adapter.prompt_conditioner.model.parameters())
    snapshot = adapter.frozen_copy()
    assert not any(p.requires_grad for p in snapshot.parameters())
    assert not {id(p) for p in snapshot.parameters()} & {id(p) for p in adapter.parameters()}
    assert snapshot.ar.shared_transformer is snapshot.diffusion.model.model.transformer
    assert snapshot.prompt_conditioner.model is adapter.prompt_conditioner.model
    assert snapshot.audio_autoencoder is adapter.audio_autoencoder
    source = torch.randn(1, 64, 432)
    mask = torch.arange(432)[None] < 87
    observation = (GenerationObservation("one", "Render a dog barking.") if mode == "generation" else
                   EditingObservation("one", "Move the dog to the front.", source, mask, 88200))
    ids = codec.encode(plan())["input_ids"][None]
    student = adapter.student_logits(observation, ids[:, :-1])
    teacher = adapter.teacher_logits(observation, ids[:, :-1], privileged_text="Verified desired scene: dog at the front.")
    assert not teacher.requires_grad
    assert torch.isfinite(student).all() and student.shape[-1] == 4096
    # The exact frozen Generation encode path remains value-identical while
    # the new wrapper restores gradients through its trainable projections.
    if mode == "generation":
        old_context = original_ar.encode_requests([observation.request], device="cpu")
        new_context = adapter._encode_requests([observation.request])
        torch.testing.assert_close(old_context[0], new_context[0], atol=0, rtol=0)
    condition = adapter.render_condition(observation, plan())
    noise = torch.randn(1, 64, condition.mask.shape[-1])
    time = torch.tensor([.3])
    velocity = adapter.velocity_function(condition, differentiable=True)(noise, time)
    assert velocity.shape == noise.shape and torch.isfinite(velocity).all()
    partition = adapter.dependency_parameters()
    shared = {id(p) for _, p in partition["shared"]}
    assert id(adapter.diffusion.model.model.to_cond_embed[0].weight) in shared
    assert id(adapter.prompt_conditioner.proj_out.weight) in shared
    assert id(adapter.diffusion.model.model.transformer.project_in.weight) not in shared
    student.square().mean().backward()
    assert adapter.prompt_conditioner.proj_out.weight.grad.abs().sum() > 0
    assert adapter.diffusion.model.model.to_cond_embed[0].weight.grad.abs().sum() > 0
    assert adapter.diffusion.model.model.transformer.project_in.weight.grad is None
    adapter.zero_grad(set_to_none=True)
    velocity.square().mean().backward()
    assert adapter.prompt_conditioner.proj_out.weight.grad.abs().sum() > 0
    assert adapter.diffusion.model.model.transformer.project_in.weight.grad.abs().sum() > 0
    latent = noise.detach().requires_grad_(True)
    waveform = adapter.decode_for_reward(latent, condition.model_num_samples)
    waveform.square().mean().backward()
    assert latent.grad.abs().sum() > 0
    assert all(p.grad is None for p in adapter.audio_autoencoder.parameters())
    if mode == "editing":
        assert condition.positive[0]["source_foa_latent"] is condition.negative[0]["source_foa_latent"]
        rollout = ARRollout(tuple(ids[0].tolist()), (), (), True, 42, .8, 0)
        assert adapter.decode_plan(observation, rollout)["duration_sec"] == 2.


@pytest.mark.parametrize("mode", ["generation", "editing"])
def test_sampler_reuses_existing_generation_and_editing_cfg_paths(codec, monkeypatch, mode):
    adapter, _, _ = make_bundle(mode, codec, monkeypatch)
    source = torch.randn(1, 64, 432)
    mask = torch.arange(432)[None] < 87
    observation = (GenerationObservation("one", "dog") if mode == "generation" else
                   EditingObservation("one", "move dog", source, mask, 88200))
    condition = adapter.render_condition(observation, plan())
    noise = torch.randn(1, 64, condition.mask.shape[-1])
    if mode == "editing":
        from stable_audio_tools.models.sceneplan_transfusion_editing_pipeline import ScenePlanTransfusionEditingDiTPipeline
        adapter.cfg_scale = 1.5
        native = ScenePlanTransfusionEditingDiTPipeline(diffusion=adapter.diffusion).sample_edited_latents(
            source, mask, [plan()], model_num_samples=[88200], steps=3, cfg_scale=1.5, initial_noise=noise)
    else:
        from stable_audio_tools.inference.sampling import sample_diffusion
        pos = adapter.diffusion.get_conditioning_inputs(adapter.diffusion.conditioner(condition.positive, adapter.device))
        neg = adapter.diffusion.get_conditioning_inputs(adapter.diffusion.conditioner(condition.negative, adapter.device), negative=True)
        native = sample_diffusion(model=adapter.diffusion.model, noise=noise,
            cond_inputs={**pos, **neg}, diffusion_objective="rectified_flow", steps=3,
            cfg_scale=3., rescale_cfg=True, cfg_rescale_phi=.4, padding_mask=condition.mask,
            dist_shift=adapter.diffusion.sampling_dist_shift, sampler_type="euler", decode=False,
            disable_tqdm=True)
    trace = euler_rollout(adapter.velocity_function(condition, differentiable=False), noise,
        condition.mask, adapter.schedule(3, condition.mask.shape[-1]))
    torch.testing.assert_close(trace.states[-1], native, atol=1e-6, rtol=1e-6)


def test_spatial_reward_direction_gradient_silence_and_multisource_observability():
    p = plan()
    reward = FoaSpatialReward(p, model_num_samples=88200)
    wave = torch.zeros(1, 4, 88200)
    wave[:, 0], wave[:, 3] = .1, .14
    wave.requires_grad_(True)
    value = reward.differentiable(wave)
    value.backward()
    assert value > .99 and torch.isfinite(wave.grad).all()
    left = wave.detach().clone(); left[:, 3] *= -1
    assert reward.score(wave.detach()).utility > reward.score(left).utility
    assert reward.score(torch.zeros_like(wave)).utility == 0.
    p["sources"].append(copy.deepcopy(p["sources"][0]))
    p["sources"][1]["source_id"] = "source_1"
    with pytest.raises(UnobservableAudio):
        FoaSpatialReward(p, model_num_samples=88200)


def test_plan_reward_protects_actual_transcript_field_and_rejects_typos(codec):
    reference = plan()
    source = reference["sources"][0]
    source.pop("description")
    source.update(kind="speech", speaker_description="A woman speaking.", transcript="hello world")
    reward = ScenePlanReward(reference, codec=codec)
    candidate = copy.deepcopy(reference)
    candidate["sources"][0]["transcript"] = "goodbye world"
    assert reward(candidate).costs["transcript"] > 0
    with pytest.raises(ValueError, match="unknown"):
        ScenePlanReward(reference, codec=codec, protected_fields=("speech",))
