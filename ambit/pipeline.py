"""Mainline generation and editing, sharing one runtime and local loader."""

from copy import deepcopy
from types import MethodType
import torch
from safetensors.torch import load_file
from .audio import InferenceResult, SAMPLE_RATE, MAX_SAMPLES, validate_foa
from .bundle import Bundle, validate_request, validate_sampling
from ._runtime.sceneplan.codec import ModelScenePlanCodecV4
from ._runtime.sceneplan.schema import make_sceneplan_cfg_unknown_metadata
from ._runtime.sceneplan.plan_conditioning import compile_plan_condition
from ._runtime.models.factory import create_model_from_config


def _load_state(module, path, *, shared_prefixes=()):
    """Require every owned tensor; shared modules are loaded exactly once."""
    state = load_file(str(path), device="cpu")
    expected = {key for key in module.state_dict() if not key.startswith(shared_prefixes)}
    if set(state) != expected:
        missing = sorted(expected - set(state))[:5]
        extra = sorted(set(state) - expected)[:5]
        raise ValueError(f"Incompatible {path.name}: missing={missing}, unexpected={extra}")
    result = module.load_state_dict(state, strict=False)
    if result.unexpected_keys or any(
        (not key.startswith(shared_prefixes) for key in result.missing_keys)
    ):
        raise ValueError(f"Incomplete model state: {path.name}")
    return module.eval().requires_grad_(False)


def _device(value):
    device = torch.device(value)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Supported devices are cpu and cuda[:index]")
    if device.type == "cuda" and (not torch.cuda.is_available()):
        raise ValueError("CUDA is unavailable; use --device cpu for CPU execution")
    return device


def _common(bundle, device):
    renderer_config = deepcopy(bundle.components["renderer"]["config"])
    conditioning = renderer_config["model"]["conditioning"]
    for item in conditioning["configs"]:
        if item["type"] == "qwen_text":
            item["config"]["model_path"] = str(bundle.text_encoder)
    renderer = create_model_from_config(renderer_config)
    _load_state(renderer, bundle.components["renderer"]["weights"])
    vae = create_model_from_config(bundle.components["vae"]["config"])
    _load_state(vae, bundle.components["vae"]["weights"])
    codec = ModelScenePlanCodecV4(bundle.codec)
    renderer.to(device=device, dtype=torch.float32)
    vae.to(device=device, dtype=torch.float32)
    if (
        vae.sample_rate != SAMPLE_RATE
        or vae.io_channels != 4
        or vae.downsampling_ratio != 1024
        or (vae.latent_dim != 64)
    ):
        raise ValueError("Expected the native 44.1 kHz, four-channel, hop-1024 FOA VAE")
    return (renderer, vae, codec)


def _autocast(device):
    return torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")


@torch.no_grad()
def _euler(model, noise, *, steps, model_kwargs):
    schedule = torch.linspace(1, 0, steps + 1, device=noise.device)
    value = noise
    for current, following in zip(schedule[:-1], schedule[1:]):
        time = current * torch.ones(value.shape[0], device=value.device, dtype=value.dtype)
        value = value + (following - current) * model(value, time, **model_kwargs)
    return value


class GenerationPipeline:
    """Text → grammar-constrained ScenePlan → rectified-flow FOA audio."""

    @classmethod
    def from_bundle(cls, bundle, *, device="cuda"):
        from ._runtime.models.generation_ar import ScenePlanTransfusionGenerationAR
        from ._runtime.models.ar_lora import AdaptedGenerationAR
        from ._runtime.models.text import TextSpanHead
        from ._runtime.models.heads import QualitativeExecutionHead
        from ._runtime.models.heads import SourceInventoryHead
        from ._runtime.inference.precision import configure_float32_ar

        bundle = bundle if isinstance(bundle, Bundle) else Bundle.open(bundle, task="generation")
        if bundle.task != "generation":
            raise ValueError("GenerationPipeline requires a generation bundle")
        obj = cls()
        obj.device = _device(device)
        (obj.diffusion, obj.vae, obj.codec) = _common(bundle, obj.device)
        base = ScenePlanTransfusionGenerationAR(
            p10_dit=obj.diffusion.model.model,
            prompt_conditioner=obj.diffusion.conditioner.conditioners["prompt"],
            pad_id=obj.codec.pad_id,
        )
        obj.ar = AdaptedGenerationAR(base, obj.codec, **bundle.components["ar"]["config"])
        _load_state(
            obj.ar,
            bundle.components["ar"]["weights"],
            shared_prefixes=("p10_dit.", "prompt_conditioner."),
        )
        obj.ar.to(obj.device)
        configure_float32_ar(obj.ar)
        for key, name, head in (
            ("copy", "text", TextSpanHead),
            ("qualitative", "controls", QualitativeExecutionHead),
            ("inventory", "sources", SourceInventoryHead),
        ):
            component = bundle.components[key]
            module = _load_state(head(**component["config"]), component["weights"]).to(obj.device)
            setattr(obj, name, module)
        (obj.steps, obj.cfg_scale) = (bundle.steps, bundle.cfg_scale)
        return obj

    @torch.inference_mode()
    def generate(self, prompt, *, seed=42, steps=None, cfg_scale=None):
        from ._runtime.inference.decoding import generate_plan
        from ._runtime.inference.normalize import normalize_generated_sceneplan
        from ._runtime.models import text as text_module
        from ._runtime.models import heads as planning_heads

        validate_request(prompt, seed)
        steps = self.steps if steps is None else steps
        cfg_scale = self.cfg_scale if cfg_scale is None else cfg_scale
        validate_sampling(steps, cfg_scale)
        (tokens, _) = generate_plan(
            self.ar,
            self.text,
            text_module,
            [prompt],
            self.codec,
            device=self.device,
            max_plan_tokens=512,
            execution_head=self.controls,
            execution_module=planning_heads,
            inventory_head=self.sources,
            inventory_module=planning_heads,
        )
        normalized = normalize_generated_sceneplan(
            self.codec, tokens[0], sample_id="generated", max_tokens=512
        )
        plan = normalized["p10_plan"]
        samples = round(plan["duration_sec"] * SAMPLE_RATE)
        if not 1 <= samples <= MAX_SAMPLES:
            raise ValueError("Generated duration exceeds the supported FOA envelope")
        frames = (samples + 1023) // 1024
        metadata = compile_plan_condition(
            plan,
            tokenizer=self.ar.prompt_conditioner.tokenizer,
            model_num_samples=samples,
            latent_frames_valid=frames,
            latent_crop_length=frames,
        )
        mask = torch.ones(1, frames, device=self.device, dtype=torch.bool)
        metadata.update(
            sample_id="generated",
            model_num_samples=samples,
            seconds_start=0.0,
            seconds_total=samples / SAMPLE_RATE,
            padding_mask=mask[0],
        )
        with _autocast(self.device):
            positive = self.diffusion.conditioner([metadata], self.device)
            inputs = self.diffusion.get_conditioning_inputs(positive)
            if cfg_scale != 1:
                negative = self.diffusion.conditioner(
                    [make_sceneplan_cfg_unknown_metadata(metadata)], self.device
                )
                inputs.update(self.diffusion.get_conditioning_inputs(negative, negative=True))
            noise = torch.randn(
                1,
                64,
                frames,
                generator=torch.Generator(device="cpu").manual_seed(seed),
                device="cpu",
            ).to(self.device)
            latent = _euler(
                self.diffusion.model,
                noise,
                steps=steps,
                model_kwargs=dict(
                    **inputs,
                    cfg_scale=cfg_scale,
                    cfg_dropout_prob=0.0,
                    rescale_cfg=True,
                    scale_phi=0.4,
                    apg_scale=0.0,
                    batch_cfg=True,
                    padding_mask=mask,
                ),
            )
        audio = self.vae.decode(latent.float())[0, :, :samples].float().cpu().numpy()
        return InferenceResult(audio, plan, "generation", prompt, seed)


class EditingPipeline:
    """FOA reference + instruction → new ScenePlan → edited FOA audio."""

    @classmethod
    def from_bundle(cls, bundle, *, device="cuda"):
        from ._runtime.models.editing_ar import ScenePlanTransfusionEditingAR
        from ._runtime.models.clap import CLAP44Config, EditingCLAP44
        from ._runtime.models.editing_pipeline import ScenePlanTransfusionEditingCLAP44Pipeline
        from ._runtime.models.editing_structure import SourceStructure, SlotPlanAdapter, ar_forward
        from ._runtime.models.heads import AudioEventReadout

        bundle = bundle if isinstance(bundle, Bundle) else Bundle.open(bundle, task="editing")
        if bundle.task != "editing":
            raise ValueError("EditingPipeline requires an editing bundle")
        obj = cls()
        device = _device(device)
        (renderer, vae, codec) = _common(bundle, device)
        spec = bundle.components["clap"]
        clap = _load_state(EditingCLAP44(CLAP44Config(**spec["config"])), spec["weights"])
        ar = ScenePlanTransfusionEditingAR(
            editing_dit=renderer.model.model,
            instruction_conditioner=renderer.conditioner.conditioners["prompt"],
            pad_id=codec.pad_id,
            source_clap_model=clap,
            source_semantic_mode="clap44_audio_caption_aux",
        )
        ar.source_structure = SourceStructure(
            AudioEventReadout(**bundle.components["ar"]["config"])
        )
        ar.plan_adapter = SlotPlanAdapter(ar.plan_adapter)
        ar.forward = MethodType(ar_forward, ar)
        _load_state(
            ar,
            bundle.components["ar"]["weights"],
            shared_prefixes=("editing_dit.", "instruction_conditioner.", "source_clap_model."),
        )
        obj.core = (
            ScenePlanTransfusionEditingCLAP44Pipeline(
                diffusion=renderer, editing_ar=ar, codec=codec, audio_autoencoder=vae
            )
            .eval()
            .requires_grad_(False)
            .to(device)
        )
        (obj.steps, obj.cfg_scale) = (bundle.steps, bundle.cfg_scale)
        return obj

    @torch.inference_mode()
    def edit(self, source, prompt, *, seed=42, steps=None, cfg_scale=None):
        validate_request(prompt, seed)
        audio = validate_foa(source)
        steps = self.steps if steps is None else steps
        cfg_scale = self.cfg_scale if cfg_scale is None else cfg_scale
        validate_sampling(steps, cfg_scale)
        output = self.core.edit_audio(
            torch.from_numpy(audio.copy())[None],
            [prompt],
            model_num_samples=[audio.shape[1]],
            sample_rate=SAMPLE_RATE,
            vae_seeds=seed,
            noise_seed=seed,
            max_plan_tokens=512,
            steps=steps,
            cfg_scale=cfg_scale,
        )
        edited = output["edited_foa"][0, :, : audio.shape[1]].float().cpu().numpy()
        return InferenceResult(edited, output["new_sceneplans"][0], "editing", prompt, seed)
