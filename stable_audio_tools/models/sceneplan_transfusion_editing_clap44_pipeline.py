"""Native source FOA + instruction -> full plan -> edited FOA, with CLAP44.

This class reuses the existing DiT/VAE sampler without changing its numerical
implementation. Checkpoint promotion is a separate concern from constructing
a candidate pipeline for evaluation.
"""
from __future__ import annotations

from contextlib import nullcontext
import math
from typing import Any, Sequence

import torch
from torch import Tensor, nn

from .sceneplan_transfusion_editing_pipeline import ScenePlanTransfusionEditingDiTPipeline, _align_decoded_sceneplan_to_audio_duration

CLAP44_PIPELINE_CONTRACT = "source_foa_44100_instruction_clap44_ar_complete_plan_aligned_dit_v1"


class ScenePlanTransfusionEditingCLAP44Pipeline(ScenePlanTransfusionEditingDiTPipeline):
    def __init__(self, *, diffusion, editing_ar, codec, audio_autoencoder: nn.Module | None = None):
        super().__init__(diffusion=diffusion, audio_autoencoder=audio_autoencoder)
        self.editing_ar = editing_ar; self.codec = codec
        if editing_ar.shared_transformer is not diffusion.model.model.transformer:
            raise RuntimeError("CLAP44 AR and Editing DiT must share one Transformer")
        if editing_ar.source_semantic_mode not in {"latent_only", "clap44_audio_caption_aux"}:
            raise ValueError("CLAP44 pipeline cannot load an M2D route")
        encoder = editing_ar.source_clap_model
        if editing_ar.source_semantic_mode != "latent_only" and (encoder is None or any(p.requires_grad for p in encoder.parameters())):
            raise ValueError("CLAP44 pipeline requires its frozen source encoder")

    @torch.no_grad()
    def generate_new_sceneplans(self, source_foa_latent: Tensor, source_attention_mask: Tensor, edit_instructions: Sequence[str], *, duration_sec: float | Sequence[float], max_plan_tokens: int = 512):
        batch, channels, frames = source_foa_latent.shape if source_foa_latent.ndim == 3 else (0,0,0)
        if batch < 1 or channels != 64 or frames not in (432,648) or source_attention_mask.shape != (batch,frames):
            raise ValueError("CLAP44 inference requires [B,64,432|648] and its valid-frame mask")
        durations = [float(duration_sec)]*batch if isinstance(duration_sec,(int,float)) else [float(x) for x in duration_sec]
        if len(durations) != batch or len(edit_instructions) != batch or any(not math.isfinite(x) or x <= 0 or x*44100 > frames*1024 for x in durations):
            raise ValueError("CLAP44 instructions/durations do not match source audio")
        samples = [round(x*44100) for x in durations]
        expected = torch.arange(frames,device=source_attention_mask.device)[None] < torch.tensor([(x+1023)//1024 for x in samples],device=source_attention_mask.device)[:,None]
        if not torch.equal(source_attention_mask.bool(),expected):
            raise ValueError("CLAP44 source mask is not the exact valid-frame prefix")
        source = source_foa_latent.to(self.device, dtype=torch.float32); mask = source_attention_mask.to(self.device, dtype=torch.bool)
        context = torch.autocast("cuda",dtype=torch.bfloat16) if self.device.type == "cuda" else nullcontext()
        with context:
            tokens = self.editing_ar.generate_batch(source,mask,edit_instructions,codec=self.codec,max_plan_tokens=max_plan_tokens,fixed_duration_sec=durations)
        plans = [_align_decoded_sceneplan_to_audio_duration(self.codec.decode(ids.tolist(),sample_id=f"edited_{i:06d}"),duration) for i,(ids,duration) in enumerate(zip(tokens,durations))]
        if len(plans) != batch: raise RuntimeError("CLAP44 AR omitted a complete output plan")
        return plans,tokens

    @torch.no_grad()
    def edit_latents(self, source_foa_latent: Tensor, source_attention_mask: Tensor, edit_instructions: Sequence[str], *, model_num_samples: Sequence[int], max_plan_tokens=512, steps=20, cfg_scale=1., seed: int | Sequence[int]=42, initial_noise: Tensor | None=None):
        samples = [int(x) for x in model_num_samples]
        plans,tokens = self.generate_new_sceneplans(source_foa_latent,source_attention_mask,edit_instructions,duration_sec=[x/44100 for x in samples],max_plan_tokens=max_plan_tokens)
        output = self.sample_edited_latents(source_foa_latent,source_attention_mask,plans,model_num_samples=samples,steps=steps,cfg_scale=cfg_scale,seed=seed,initial_noise=initial_noise)
        return output,plans,tokens

    @torch.no_grad()
    def edit_audio(self, source_foa: Tensor, edit_instructions: Sequence[str], *, model_num_samples: Sequence[int], sample_rate=44100, vae_seeds: int | Sequence[int]=42, max_plan_tokens=512, steps=20, cfg_scale=1., noise_seed: int | Sequence[int]=42, initial_noise: Tensor | None=None) -> dict[str,Any]:
        if sample_rate != 44100: raise ValueError("CLAP44 Editing uses native 44.1 kHz FOA; resampling is not implicit")
        samples = [int(x) for x in model_num_samples]
        source,mask = self.encode_source_foa(source_foa,model_num_samples=samples,vae_seeds=vae_seeds)
        edited,plans,tokens = self.edit_latents(source,mask,edit_instructions,model_num_samples=samples,max_plan_tokens=max_plan_tokens,steps=steps,cfg_scale=cfg_scale,seed=noise_seed,initial_noise=initial_noise)
        audio,audio_mask = self.decode_foa_latents(edited,model_num_samples=samples)
        source_codec,_ = self.decode_foa_latents(source,model_num_samples=samples)
        return {"edited_foa":audio,"source_codec_foa":source_codec,"sample_attention_mask":audio_mask,"new_sceneplans":plans,"new_sceneplan_token_ids":tokens,"source_foa_latent":source,"source_attention_mask":mask,"edited_foa_latent":edited,"pipeline_contract":CLAP44_PIPELINE_CONTRACT}


def load_clap44_joint_candidate(checkpoint, *, device="cpu", load_audio_autoencoder=True):
    """Load an identified candidate for evaluation; never claim promotion.

    A future selected production bundle must additionally bind independent
    full-plan/audio selection evidence. The candidate API intentionally makes
    no such claim and its report always records quality_gate_passed=False.
    """
    from pathlib import Path
    from stable_audio_tools.configuration import load_config
    from stable_audio_tools.data.model_sceneplan_codec_v4 import ModelScenePlanCodecV4
    from .factory import create_model_from_config
    from .sceneplan_transfusion_editing_ar import ScenePlanTransfusionEditingAR
    from .sceneplan_transfusion_editing_clap44_io import file_sha256,load_clap44_checkpoint
    from .sceneplan_transfusion_editing_clap44_joint_io import load_joint_checkpoint,load_ar_specific,codec_artifact_sha256
    from .sceneplan_transfusion_editing_pipeline import _load_frozen_foa_vae
    from .sceneplan_transfusion_editing_provenance import verify_frozen_qwen_runtime
    payload,identity = load_joint_checkpoint(checkpoint)
    contract = payload["run_contract"]
    for key, hasher in (("model_config",file_sha256),("codec",codec_artifact_sha256)):
        if hasher(contract[key]) != contract[f"{key}_sha256"]:
            raise RuntimeError(f"CLAP44 joint {key} identity changed")
    cfg = load_config(contract["model_config"])
    prompt = next(x for x in cfg["model"]["conditioning"]["configs"] if x["id"]=="prompt")
    qwen = verify_frozen_qwen_runtime(prompt["config"]["model_path"])
    if qwen != contract["frozen_qwen_runtime"]:
        raise RuntimeError("CLAP44 joint frozen Qwen runtime changed")
    clap,clap_identity = load_clap44_checkpoint(contract["clap_checkpoint"]["path"],expected_sha256=contract["clap_checkpoint"]["sha256"])
    diffusion = create_model_from_config(cfg); diffusion.pretransform=None
    diffusion.load_state_dict(payload["diffusion_state_dict"],strict=True)
    codec = ModelScenePlanCodecV4(contract["codec"])
    if codec.fingerprint != contract["codec_fingerprint"]:
        raise RuntimeError("CLAP44 codec fingerprint changed")
    variant = contract["variant"]
    ar = ScenePlanTransfusionEditingAR(editing_dit=diffusion.model.model,instruction_conditioner=diffusion.conditioner.conditioners["prompt"],pad_id=codec.pad_id,source_semantic_mode="latent_only" if variant=="latent_only" else "clap44_audio_caption_aux",source_clap_model=None if variant=="latent_only" else clap,source_semantic_dropout=0. if variant=="latent_only" else contract["config"]["source_feature_dropout"],clap44_global_features=variant in {"global_only","global_and_sequence"},clap44_sequence_features=variant in {"sequence_only","global_and_sequence"})
    load_ar_specific(ar,payload["editing_ar_specific_state_dict"]); del payload
    vae=vae_identity=None
    if load_audio_autoencoder:
        vae,vae_identity = _load_frozen_foa_vae(device)
        frontend = clap_identity["contract"]["frontend_files"]
        if frontend.get(vae_identity["config"]) != vae_identity["config_sha256"] or frontend.get(vae_identity["checkpoint"]) != vae_identity["checkpoint_sha256"]:
            raise RuntimeError("CLAP44 pretraining and runtime FOA VAE differ")
    pipeline = ScenePlanTransfusionEditingCLAP44Pipeline(diffusion=diffusion,editing_ar=ar,codec=codec,audio_autoencoder=vae).to(device).eval()
    return pipeline,{"purpose":"candidate_diagnostic_only","checkpoint":str(Path(checkpoint).resolve()),"checkpoint_sha256":identity["checkpoint_sha256"],"checkpoint_step":identity["step"],"run_contract_sha256":identity["run_contract_sha256"],"variant":variant,"clap_checkpoint":contract["clap_checkpoint"],"frozen_vae":vae_identity,"shared_transformer_same_object":True,"runtime_inputs":["source_foa_audio","raw_edit_instruction"],"old_sceneplan_input":False,"source_caption_input":False,"m2d_used":False,"quality_gate_passed":False}


def load_clap44_selected_for_audio_evaluation(selection, *, expected_sha256, device="cpu", load_audio_autoencoder=True):
    """Require replayed plan/RF gates before the separate real-audio stage."""
    from pathlib import Path
    from .sceneplan_transfusion_editing_clap44_selection_io import validate_selected
    value = validate_selected(selection,expected_sha256=expected_sha256)
    pipeline, report = load_clap44_joint_candidate(value["selected_checkpoint"],device=device,load_audio_autoencoder=load_audio_autoencoder)
    if report["checkpoint_sha256"] != value["selected_checkpoint_sha256"] or report["checkpoint_step"] != value["selected_step"]:
        raise RuntimeError("selected CLAP44 checkpoint changed during pipeline loading")
    report.update(
        purpose="selected_for_post_joint_audio_evaluation",
        selection=str(Path(selection).resolve()),
        selection_sha256=expected_sha256,
        plan_rf_gates_passed=True,
        quality_gate_passed=False,
    )
    return pipeline, report


def load_clap44_validated_release(release, *, expected_sha256, device="cpu"):
    """Load the complete audio runtime only after replaying the pinned release."""
    from pathlib import Path
    from .sceneplan_transfusion_editing_clap44_audio_io import validate_release
    seal = validate_release(release,expected_sha256=expected_sha256)
    pipeline, report = load_clap44_joint_candidate(seal["checkpoint"]["path"],device=device,load_audio_autoencoder=True)
    if (report["checkpoint_sha256"] != seal["checkpoint"]["sha256"] or
            report["m2d_used"] is not False or report["shared_transformer_same_object"] is not True):
        raise RuntimeError("validated native release loaded a different checkpoint/runtime")
    report.update(purpose="validated_native_editing_audio_runtime",
        release=str(Path(release).resolve()),release_sha256=expected_sha256,
        independent_test=seal["independent_test"],independent_test_pairs=5000,
        sampling_and_evaluation_policy=seal["policy"],quality_gate_passed=True,
        quality_scope="frozen_5000_pair_distribution_and_evaluation_policy",
        known_limitations=seal["deliverables"]["known_limitations"])
    return pipeline,report
