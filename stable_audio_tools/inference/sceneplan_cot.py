"""Single-turn P11 ScenePlan CoT pipeline with an external P10 executor.

The orchestration layer is intentionally the only place where P11 and P10
meet. P11 emits :class:`ScenePlanExecutionBundle`; P10 consumes that bundle
and returns FOA. Audio-aware Editing first observes the source FOA, predicts
an atomic patch from that observation and the instruction, then sends only
the deterministically revised ScenePlan across the frozen P10 boundary.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Protocol

import torch
from torch import Tensor

from ..data.model_sceneplan import make_sceneplan_cfg_unknown_metadata
from ..data.sceneplan_p11_single_turn import (
    AudioAwareEditPlanningBundle,
    P11_EDITING_CONTRACT,
    P11_MAX_LATENT_FRAMES,
    P11Task,
    ScenePlanExecutionBundle,
    ScenePlanResolver,
)


class ScenePlanExecutor(Protocol):
    """Minimal renderer interface intentionally independent of P11 weights."""

    def render(
        self, bundle: ScenePlanExecutionBundle, *, seed: int | None = None
    ) -> Tensor: ...


@dataclass(frozen=True)
class ScenePlanCoTResult:
    task: P11Task
    sceneplan: Mapping[str, Any]
    execution_bundle: ScenePlanExecutionBundle
    foa: Tensor | None
    foa_role: str | None
    render_seed: int | None
    editing_contract: str | None
    preserves_unedited_waveform: bool
    localized_editing: bool
    audio_aware_edit_bundle: AudioAwareEditPlanningBundle | None = None


class ScenePlanCoTPipeline:
    """Unified single-turn generation, understanding, and revision facade."""

    def __init__(
        self,
        planner,
        *,
        executor: ScenePlanExecutor | None = None,
        resolver: ScenePlanResolver | None = None,
    ) -> None:
        self.planner = planner
        self.executor = executor
        self.resolver = resolver

    def _as_input_latent(self, input_foa: Tensor) -> Tensor:
        value = torch.as_tensor(input_foa)
        if value.ndim != 2:
            raise ValueError("single-turn input FOA must be [4,N] or [64,T]")
        if int(value.shape[0]) == 64:
            latent = value.float()
        elif int(value.shape[0]) == 4:
            device = next(self.planner.parameters()).device
            encoded = self.planner.encode_audio(
                value.to(device=device, dtype=torch.float32).unsqueeze(0)
            )
            if encoded.ndim != 3 or tuple(encoded.shape[:2]) != (1, 64):
                raise RuntimeError(
                    "P11 VAE encoder did not return one [64,T] FOA latent"
                )
            latent = encoded[0]
        else:
            raise ValueError("single-turn input FOA must be [4,N] or [64,T]")
        if not 1 <= int(latent.shape[-1]) <= P11_MAX_LATENT_FRAMES:
            raise ValueError(
                "P11 input FOA latent lies outside "
                f"[1,{P11_MAX_LATENT_FRAMES}] frames"
            )
        if not bool(torch.isfinite(latent).all()):
            raise ValueError("P11 input FOA latent contains non-finite values")
        return latent

    def _execute(self, bundle: ScenePlanExecutionBundle, *, seed: int | None) -> Tensor:
        bundle.assert_external_p10_boundary()
        if not bundle.requires_p10_render:
            raise ValueError("understanding does not invoke the P10 renderer")
        if self.executor is None:
            raise RuntimeError(
                "generation/editing requires a separately loaded P10 executor"
            )
        effective_seed = 0 if seed is None else int(seed)
        rendered = torch.as_tensor(
            self.executor.render(bundle, seed=effective_seed)
        )
        if rendered.ndim != 2 or int(rendered.shape[0]) != 4:
            raise RuntimeError("P10 executor must return one [4,N] FOA waveform")
        if int(rendered.shape[1]) != int(bundle.model_num_samples):
            raise RuntimeError("P10 executor returned the wrong waveform length")
        if not bool(torch.isfinite(rendered).all()):
            raise RuntimeError("P10 executor returned non-finite FOA")
        return rendered

    @staticmethod
    def _result(
        bundle: ScenePlanExecutionBundle,
        foa: Tensor | None,
        *,
        foa_role: str | None = None,
        render_seed: int | None = None,
        audio_aware_edit_bundle: AudioAwareEditPlanningBundle | None = None,
    ) -> ScenePlanCoTResult:
        if (foa is None) != (foa_role is None or render_seed is None):
            raise ValueError("rendered FOA, role, and seed must be present together")
        return ScenePlanCoTResult(
            task=bundle.task,
            sceneplan=bundle.sceneplan,
            execution_bundle=bundle,
            foa=foa,
            foa_role=foa_role,
            render_seed=render_seed,
            editing_contract=bundle.editing_contract,
            preserves_unedited_waveform=bundle.preserves_unedited_waveform,
            localized_editing=bundle.localized_editing,
            audio_aware_edit_bundle=audio_aware_edit_bundle,
        )

    @torch.no_grad()
    def generate(
        self,
        user_text: str,
        *,
        duration_sec: float | None = None,
        sample_id: str = "p11_generation",
        seed: int | None = None,
        temperature: float = 0.0,
    ) -> ScenePlanCoTResult:
        bundle = self.planner.plan_generation(
            user_text,
            duration_sec=duration_sec,
            sample_id=sample_id,
            resolver=self.resolver,
            temperature=temperature,
        )
        effective_seed = 0 if seed is None else int(seed)
        return self._result(
            bundle,
            self._execute(bundle, seed=effective_seed),
            foa_role="generated",
            render_seed=effective_seed,
        )

    @torch.no_grad()
    def understand(
        self,
        input_foa: Tensor,
        *,
        input_semantic: Tensor | None = None,
        input_lexical: Mapping[str, Any] | Tensor | None = None,
        render_closure: bool = False,
        seed: int | None = None,
        prompt: str = (
            "Describe the audible sources, timing, and spatial motion as a ScenePlan."
        ),
        sample_id: str = "p11_understanding",
        temperature: float = 0.0,
    ) -> ScenePlanCoTResult:
        latent = self._as_input_latent(input_foa)
        understanding_kwargs = {
            "input_semantic": input_semantic,
            "prompt": prompt,
            "sample_id": sample_id,
            "resolver": self.resolver,
            "temperature": temperature,
        }
        if input_lexical is not None:
            if not bool(getattr(self.planner, "transfusion_cot_enabled", False)):
                raise ValueError(
                    "speech lexical evidence requires the P11-v4 planner"
                )
            understanding_kwargs["input_lexical"] = input_lexical
        bundle = self.planner.plan_understanding(latent, **understanding_kwargs)
        if bundle.requires_p10_render:
            raise RuntimeError("understanding unexpectedly requested P10 execution")
        if not render_closure:
            return self._result(bundle, None)
        # Closure rendering is an evaluator action, not another planner task.
        # P10 consumes identical metadata while the returned result remains U.
        render_bundle = replace(bundle, task=P11Task.GENERATION)
        effective_seed = 0 if seed is None else int(seed)
        return self._result(
            bundle,
            self._execute(render_bundle, seed=effective_seed),
            foa_role="understanding_p10_cycle",
            render_seed=effective_seed,
        )

    @torch.no_grad()
    def edit(
        self,
        edit_instruction: str,
        *,
        input_foa: Tensor,
        input_semantic: Tensor,
        current_sceneplan: Mapping[str, Any] | None = None,
        input_lexical: Mapping[str, Any] | Tensor | None = None,
        input_foa_ref: str | None = None,
        input_foa_latent_ref: str | None = None,
        sample_id: str = "p11_editing",
        seed: int | None = None,
        temperature: float = 0.0,
    ) -> ScenePlanCoTResult:
        if not hasattr(self.planner, "plan_editing_audio_aware"):
            raise TypeError(
                "Editing requires the active audio-aware P11 planner; "
                "the retired ScenePlan-only route is not accepted"
            )
        latent = self._as_input_latent(input_foa)
        planning_bundle = self.planner.plan_editing_audio_aware(
            edit_instruction,
            input_foa_latent=latent,
            input_semantic=input_semantic,
            input_sceneplan=current_sceneplan,
            input_lexical=input_lexical,
            input_foa_ref=input_foa_ref,
            input_foa_latent_ref=input_foa_latent_ref,
            sample_id=sample_id,
            resolver=self.resolver,
            temperature=temperature,
        )
        bundle = planning_bundle.execution_bundle
        if bundle.editing_contract != P11_EDITING_CONTRACT:
            raise RuntimeError("P11 editing boundary changed")
        effective_seed = 0 if seed is None else int(seed)
        return self._result(
            bundle,
            self._execute(bundle, seed=effective_seed),
            foa_role="edited_same_seed",
            render_seed=effective_seed,
            audio_aware_edit_bundle=planning_bundle,
        )

    @torch.no_grad()
    def edit_from_result(
        self,
        edit_instruction: str,
        *,
        current: ScenePlanCoTResult,
        input_semantic: Tensor,
        input_lexical: Mapping[str, Any] | Tensor | None = None,
        input_foa_ref: str | None = None,
        sample_id: str = "p11_editing",
        temperature: float = 0.0,
    ) -> ScenePlanCoTResult:
        """Canonical edit entry that cannot silently change P10 initial noise."""

        if current.foa is None or current.render_seed is None:
            raise ValueError("same-seed editing requires a previously rendered result")
        return self.edit(
            edit_instruction,
            input_foa=current.foa,
            input_semantic=input_semantic,
            current_sceneplan=current.sceneplan,
            input_lexical=input_lexical,
            input_foa_ref=input_foa_ref,
            sample_id=sample_id,
            seed=current.render_seed,
            temperature=temperature,
        )


class P10ScenePlanDiTExecutor:
    """Adapter around an already trained P10 ScenePlan-DiT Lightning wrapper."""

    def __init__(
        self,
        training_wrapper,
        *,
        device: str | torch.device = "cuda:0",
        steps: int = 100,
        cfg_scale: float = 3.0,
        rescale_cfg: bool = True,
        cfg_rescale_phi: float = 0.4,
        apg_scale: float = 0.0,
    ) -> None:
        self.wrapper = training_wrapper
        self.diffusion = getattr(training_wrapper, "diffusion", None)
        if self.diffusion is None:
            raise TypeError("P10 executor requires a diffusion training wrapper")
        if (
            getattr(training_wrapper, "diffusion_ema", None) is None
            or getattr(training_wrapper, "conditioner_ema", None) is None
        ):
            raise RuntimeError("P10 executor requires EMA DiT and conditioner weights")
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("P10 CUDA executor requested but CUDA is unavailable")
        self.steps = int(steps)
        self.cfg_scale = float(cfg_scale)
        self.rescale_cfg = bool(rescale_cfg)
        self.cfg_rescale_phi = float(cfg_rescale_phi)
        self.apg_scale = float(apg_scale)
        if self.steps <= 0 or self.cfg_scale <= 0.0:
            raise ValueError("P10 sampling steps and CFG scale must be positive")

        self.sample_model = (
            training_wrapper.diffusion_ema.ema_model.to(self.device)
            .eval()
            .requires_grad_(False)
        )
        self.diffusion.conditioner.to(self.device).eval().requires_grad_(False)
        if self.diffusion.pretransform is None:
            raise RuntimeError("P10 executor has no frozen FOA VAE decoder")
        self.diffusion.pretransform.to(self.device).eval().requires_grad_(False)
        self.dtype = next(self.sample_model.parameters()).dtype

    @classmethod
    def from_checkpoints(
        cls,
        *,
        model_config_path: str | Path,
        checkpoint_path: str | Path,
        vae_checkpoint_path: str | Path,
        device: str | torch.device = "cuda:0",
        **sampling_kwargs,
    ) -> "P10ScenePlanDiTExecutor":
        """Load P10 separately; no weight is copied into the P11 planner."""

        from ..configuration import load_config
        from ..models import create_model_from_config
        from ..models.utils import load_ckpt_state_dict
        from ..training.factory import create_training_wrapper_from_config

        config = load_config(Path(model_config_path).expanduser().resolve(strict=True))
        if config.get("model_type") != "diffusion_cond":
            raise ValueError("external executor checkpoint is not a P10 DiT config")
        diffusion_config = config.get("model", {}).get("diffusion", {})
        if list(diffusion_config.get("input_concat_ids") or ()) != ["sceneplan_44"]:
            raise ValueError("P10 executor must use direct sceneplan_44 concatenation")
        model = create_model_from_config(config)
        wrapper = create_training_wrapper_from_config(config, model)
        state = load_ckpt_state_dict(
            str(Path(checkpoint_path).expanduser().resolve(strict=True))
        )
        incompatible = wrapper.load_state_dict(state, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                "P10 checkpoint/model mismatch: "
                f"missing={incompatible.missing_keys[:12]}, "
                f"unexpected={incompatible.unexpected_keys[:12]}"
            )
        vae_state = load_ckpt_state_dict(
            str(Path(vae_checkpoint_path).expanduser().resolve(strict=True))
        )
        if hasattr(model, "load_pretransform_state_dict"):
            model.load_pretransform_state_dict(vae_state, strict=False)
        elif model.pretransform is not None:
            incompatible_vae = model.pretransform.load_state_dict(
                vae_state, strict=False
            )
            if incompatible_vae is not None and incompatible_vae.unexpected_keys:
                raise RuntimeError(
                    "P10 VAE checkpoint has unexpected keys: "
                    f"{incompatible_vae.unexpected_keys[:12]}"
                )
        else:
            raise RuntimeError("P10 executor model has no VAE pretransform")
        return cls(wrapper, device=device, **sampling_kwargs)

    @torch.no_grad()
    def render(
        self, bundle: ScenePlanExecutionBundle, *, seed: int | None = None
    ) -> Tensor:
        bundle.assert_external_p10_boundary()
        if not bundle.requires_p10_render:
            raise ValueError("P10 must not render an understanding-only bundle")
        positive = dict(bundle.p10_metadata)
        negative = make_sceneplan_cfg_unknown_metadata(positive)
        effective_seed = 0 if seed is None else int(seed)
        generator = torch.Generator(device="cpu").manual_seed(effective_seed)
        noise = torch.randn(
            (
                1,
                int(self.diffusion.io_channels),
                int(bundle.latent_frames_valid),
            ),
            generator=generator,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.dtype)
        autocast = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if self.device.type == "cuda"
            else nullcontext()
        )
        from .sampling import sample_diffusion

        with torch.inference_mode(), autocast:
            with self.wrapper.ema_conditioner_context():
                positive_tensors = self.diffusion.conditioner(
                    [positive], self.device
                )
                negative_tensors = self.diffusion.conditioner(
                    [negative], self.device
                )
            cond_inputs = self.diffusion.get_conditioning_inputs(positive_tensors)
            cond_inputs.update(
                self.diffusion.get_conditioning_inputs(
                    negative_tensors, negative=True
                )
            )
            cond_inputs = {
                key: value.to(self.dtype) if isinstance(value, Tensor) else value
                for key, value in cond_inputs.items()
            }
            generated = sample_diffusion(
                model=self.sample_model,
                noise=noise,
                cond_inputs=cond_inputs,
                diffusion_objective=self.diffusion.diffusion_objective,
                steps=self.steps,
                cfg_scale=self.cfg_scale,
                conditioning=[positive],
                sample_rate=44_100,
                pretransform=self.diffusion.pretransform,
                mask_padding_attention=True,
                use_effective_length_for_schedule=False,
                padding_mask=torch.ones(
                    1,
                    bundle.latent_frames_valid,
                    dtype=torch.bool,
                    device=self.device,
                ),
                dist_shift=self.diffusion.sampling_dist_shift,
                sampler_type="euler",
                batch_cfg=True,
                rescale_cfg=self.rescale_cfg,
                cfg_rescale_phi=self.cfg_rescale_phi,
                apg_scale=self.apg_scale,
                decode=True,
                disable_tqdm=True,
            )[0, :, : bundle.model_num_samples]
        output = generated.float().cpu().contiguous()
        if tuple(output.shape) != (4, bundle.model_num_samples):
            raise RuntimeError("P10 executor produced an invalid FOA shape")
        if not bool(torch.isfinite(output).all()):
            raise RuntimeError("P10 executor produced non-finite FOA")
        return output


__all__ = [
    "P10ScenePlanDiTExecutor",
    "ScenePlanCoTPipeline",
    "ScenePlanCoTResult",
    "ScenePlanExecutor",
]
