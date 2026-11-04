"""Unified text → AR → ScenePlan → rectified-flow DiT → FOA inference."""

from copy import deepcopy
from types import MethodType, SimpleNamespace
import math
import torch
from torch import nn
from safetensors.torch import load_file
from transformers import AutoModel
from .audio import InferenceResult, SAMPLE_RATE, validate_foa
from .bundle import Bundle, read_json, validate_request, validate_sampling
from ._runtime.models.factory import create_model_from_config
from ._runtime.models.editing_ar import ScenePlanTransfusionEditingAR
from ._runtime.models.editing_structure import SourceStructure, SlotPlanAdapter, ar_forward
from ._runtime.models.heads import AudioEventReadout
from ._runtime.models.clap import CLAP44Config, EditingCLAP44
from ._runtime.models.gain import install_local_gain_adapter, attach_desired_gain_tracks
from ._runtime.models.unified import AudioTranscriptUnifiedTransfusion, ExpandedPlanHead
from ._runtime.models.editing_audio import (
    ScenePlanTransfusionEditingDiTPipeline,
    _align_decoded_sceneplan_to_audio_duration,
)
from ._runtime.sceneplan.codec import ModelScenePlanCodecV4
from ._runtime.sceneplan.editing_grammar import editing_ar_allowed_next_ids
from ._runtime.sceneplan.plan_conditioning import compile_plan_condition
from ._runtime.sceneplan.schema import make_sceneplan_cfg_unknown_metadata


def _context(self, task, instructions, device):
    tokenizer = self.ar.instruction_conditioner.tokenizer
    encoded = tokenizer(
        list(instructions),
        padding=True,
        truncation=False,
        add_special_tokens=True,
        return_tensors="pt",
    )
    if encoded["input_ids"].shape[1] > 512:
        raise ValueError("Request exceeds the supported 512-token context")
    ids = encoded["input_ids"].to(device)
    mask = encoded["attention_mask"].to(device).bool()
    hidden = self.__dict__["_large_qwen"](
        input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True
    ).last_hidden_state
    projected = self.ar_request_projection(
        hidden.to(self.ar_request_projection.weight.dtype)
    ) * mask.unsqueeze(-1)
    context = self.ar.editing_dit.to_cond_embed(projected)
    return context + self.ar_task.weight[task].to(context.dtype)[None, None], mask


def _load_unique(module, path):
    tensors = dict(list(module.named_parameters()) + list(module.named_buffers()))
    state = load_file(str(path))
    if set(tensors) != set(state):
        raise ValueError(
            f"Incomplete weights: missing={sorted(set(tensors) - set(state))[:6]}, extra={sorted(set(state) - set(tensors))[:6]}"
        )
    with torch.no_grad():
        for name, dest in tensors.items():
            src = state[name]
            if dest.shape != src.shape or dest.dtype != src.dtype:
                raise ValueError(f"Incompatible tensor: {name}")
            dest.copy_(src)


def _load(bundle, device):
    cfg = bundle.config
    rc = deepcopy(read_json(bundle.path("renderer_config")))
    for item in rc["model"]["conditioning"]["configs"]:
        if item["type"] == "qwen_text":
            item["config"]["model_path"] = str(bundle.path("text_encoder"))
    diffusion = create_model_from_config(rc)
    for local in diffusion.conditioner.conditioners.values():
        if hasattr(local, "encoder") and hasattr(local.encoder, "event_embedding_dim"):
            install_local_gain_adapter(local)
    codec = ModelScenePlanCodecV4(bundle.path("codec"))
    clap = EditingCLAP44(CLAP44Config(**cfg["clap"])).eval().requires_grad_(False)
    ar = ScenePlanTransfusionEditingAR(
        editing_dit=diffusion.model.model,
        instruction_conditioner=diffusion.conditioner.conditioners["prompt"],
        pad_id=codec.pad_id,
        source_clap_model=clap,
        source_semantic_mode="clap44_audio_caption_aux",
    )
    ar.source_structure = SourceStructure(AudioEventReadout(**cfg.get("structure", {})))
    ar.plan_adapter = SlotPlanAdapter(ar.plan_adapter)
    ar.forward = MethodType(ar_forward, ar)
    model = AudioTranscriptUnifiedTransfusion(
        SimpleNamespace(editing_ar=ar, diffusion=diffusion, codec=codec),
        cfg["transcript"],
        cfg["audio_transcript"],
    )
    ar.plan_adapter.plan_head = ExpandedPlanHead(
        ar.plan_adapter.plan_head, cfg["planner"]["head_blocks"]
    )
    model.ar_request_projection = nn.Linear(cfg["planner"]["hidden_size"], 768)
    _load_unique(model, bundle.path("weights"))
    model.eval().requires_grad_(False).to(device)
    # Frozen HF encoders are deliberately external to the shared trainable module tree.
    model.__dict__["_large_qwen"] = (
        AutoModel.from_pretrained(
            str(bundle.path("planner_encoder")),
            dtype=torch.bfloat16,
            local_files_only=True,
            trust_remote_code=False,
        )
        .eval()
        .requires_grad_(False)
        .to(device)
    )
    model.ar.instruction_conditioner.model.to(device=device)
    model.instruction_context = MethodType(_context, model)
    if ar.shared_transformer is not diffusion.model.model.transformer:
        raise RuntimeError("AR and DiT must share one Transformer")
    vae = create_model_from_config(read_json(bundle.path("vae_config")))
    vae.load_state_dict(load_file(str(bundle.path("vae_weights"))), strict=True)
    vae.eval().requires_grad_(False).to(device)
    return model, vae


class UnifiedPipeline:
    @classmethod
    def from_bundle(cls, bundle, *, device="cuda"):
        task = getattr(cls, "task", "generation")
        bundle = bundle if isinstance(bundle, Bundle) else Bundle.open(bundle, task=task)
        obj = cls()
        obj.device = torch.device(device)
        if obj.device.type != "cuda" or not torch.cuda.is_available():
            raise ValueError("Unified AMBIT requires a CUDA GPU with bfloat16 support")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision("high")
        torch.backends.mha.set_fastpath_enabled(False)
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(20261010)
        obj.model, obj.vae = _load(bundle, obj.device)
        obj.bundle = bundle
        return obj

    def _plan(self, task, prompt, source, mask, duration):
        m = self.model
        tokens = [int(m.codec.bos_id)]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            context = m.instruction_context(task, [prompt], self.device)
            features = None if task == 0 else m.ar.source_clap_model.source_features(source, mask)
            while len(tokens) < 1024:
                allowed = sorted(
                    editing_ar_allowed_next_ids(m.codec, tokens, fixed_duration_sec=duration)
                )
                if not allowed:
                    raise RuntimeError("ScenePlan grammar has no admissible next token")
                if len(allowed) == 1:
                    selected = allowed[0]
                else:
                    ids = torch.tensor([tokens], device=self.device, dtype=torch.long)
                    logits = m.plan_logits(
                        task,
                        ids,
                        torch.ones_like(ids, dtype=torch.bool),
                        [prompt],
                        source,
                        mask,
                        encoded_context=context,
                        source_features=features,
                    )[0, -1]
                    permitted = torch.tensor(allowed, device=self.device)
                    selected = int(permitted[logits[permitted].argmax()])
                tokens.append(selected)
                if selected == m.codec.eos_id:
                    plan = m.codec.decode(tokens, sample_id="generated" if task == 0 else "edited")
                    return (
                        _align_decoded_sceneplan_to_audio_duration(plan, duration)
                        if duration is not None
                        else plan
                    )
        raise RuntimeError("AR did not complete a ScenePlan within 1024 tokens")

    @torch.inference_mode()
    def _run(self, prompt, source_audio, *, seed, steps, cfg_scale):
        validate_request(prompt, seed)
        task = int(source_audio is not None)
        defaults = self.bundle.config["sampling"]["editing" if task else "generation"]
        steps = defaults["steps"] if steps is None else steps
        scale = defaults["cfg_scale"] if cfg_scale is None else cfg_scale
        validate_sampling(steps, scale)
        source = mask = duration = None
        if task:
            audio = validate_foa(source_audio)
            duration = audio.shape[1] / SAMPLE_RATE
            encoder = ScenePlanTransfusionEditingDiTPipeline(
                diffusion=self.model.diffusion, audio_autoencoder=self.vae
            )
            source, mask = encoder.encode_source_foa(
                torch.from_numpy(audio.copy())[None],
                model_num_samples=[audio.shape[1]],
                vae_seeds=[seed],
            )
        plan = self._plan(task, prompt, source, mask, duration)
        n = round(plan["duration_sec"] * SAMPLE_RATE)
        frames = math.ceil(n / 1024)
        if not 0 < frames <= 648:
            raise ValueError("Generated duration exceeds the supported FOA envelope")
        bucket = 432 if frames <= 432 else 648
        valid = torch.arange(bucket) < frames
        reference = torch.zeros(64, bucket) if not task else source[0, :, :bucket].cpu()
        metadata = compile_plan_condition(
            plan,
            tokenizer=self.model.ar.instruction_conditioner.tokenizer,
            model_num_samples=n,
            latent_frames_valid=frames,
            latent_crop_length=bucket,
        )
        metadata.update(
            source_foa_latent=reference,
            model_num_samples=n,
            seconds_start=0.0,
            seconds_total=n / SAMPLE_RATE,
            padding_mask=[valid],
            latent_frames_valid=frames,
            latent_crop_length=bucket,
            latent_bucket_frames=bucket,
        )
        metadata = attach_desired_gain_tracks(metadata)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            positive = self.model.renderer_inputs(task, [metadata], self.device)
            negative = (
                self.model.renderer_inputs(
                    task, [make_sceneplan_cfg_unknown_metadata(metadata)], self.device
                )
                if scale != 1
                else None
            )
        noise_device = torch.device("cpu") if task else self.device
        latent = torch.randn(
            1,
            64,
            bucket,
            generator=torch.Generator(device=noise_device).manual_seed(seed),
            device=noise_device,
        ).to(self.device)
        valid = valid[None].to(self.device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for t in torch.linspace(1.0, 1.0 / steps, steps, device=self.device):
                velocity = self.model.diffusion.model(
                    latent, t[None], **positive, padding_mask=valid, cfg_dropout_prob=0.0
                )
                if negative is not None:
                    uncond = self.model.diffusion.model(
                        latent, t[None], **negative, padding_mask=valid, cfg_dropout_prob=0.0
                    )
                    velocity = uncond + scale * (velocity - uncond)
                if task:
                    latent = latent - velocity.float() / steps
                else:
                    delta = torch.full_like(t, -1.0 / steps)
                    latent = latent + velocity.float() * delta
        audio = self.vae.decode(latent.float())[0, :, :n].float().cpu().numpy()
        return InferenceResult(
            validate_foa(audio), plan, "editing" if task else "generation", prompt, seed
        )

    def generate(self, prompt, *, seed=42, steps=None, cfg_scale=None):
        return self._run(prompt, None, seed=seed, steps=steps, cfg_scale=cfg_scale)

    def edit(self, source, prompt, *, seed=42, steps=None, cfg_scale=None):
        return self._run(prompt, source, seed=seed, steps=steps, cfg_scale=cfg_scale)


class GenerationPipeline(UnifiedPipeline):
    task = "generation"


class EditingPipeline(UnifiedPipeline):
    task = "editing"
