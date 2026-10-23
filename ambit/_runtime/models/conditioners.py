from torch import nn
from torch.nn import functional as F
import torch
import typing as tp
from enum import Enum


class PaddingMode(str, Enum):
    """Enum for handling padding in text conditioner embeddings."""

    NONE = "none"
    ZERO = "zero"
    LEARNED = "learned"


def _assert_tensor_true(condition: torch.Tensor, message: str) -> None:
    """Fail closed while avoiding a CUDA-to-CPU synchronization."""
    condition = condition.reshape(())
    if condition.device.type == "cuda":
        torch._assert_async(condition, message)
    elif not bool(condition):
        raise ValueError(message)


class Conditioner(nn.Module):
    def __init__(
        self,
        dim: int,
        output_dim: int,
        project_out: bool = False,
        padding_mode: str = "zero",
        project_out_init: str = "default",
    ):
        super().__init__()
        self.dim = dim
        self.output_dim = output_dim
        self.padding_mode = padding_mode
        self.proj_out = (
            nn.Linear(dim, output_dim) if dim != output_dim or project_out else nn.Identity()
        )
        if project_out_init not in {"default", "identity"}:
            raise ValueError(
                f"project_out_init must be 'default' or 'identity', got {project_out_init!r}"
            )
        if project_out_init == "identity":
            if not isinstance(self.proj_out, nn.Linear) or dim != output_dim:
                raise ValueError(
                    "project_out_init='identity' requires a square, enabled output projection"
                )
            nn.init.eye_(self.proj_out.weight)
            if self.proj_out.bias is not None:
                nn.init.zeros_(self.proj_out.bias)
        if padding_mode == "learned" or padding_mode == PaddingMode.LEARNED:
            self.padding_embedding = nn.Parameter(torch.randn(output_dim) * 0.02)

    def apply_padding(self, embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Apply padding handling based on padding_mode.

        Args:
            embeddings: [batch, seq_len, dim] - the embeddings to process
            attention_mask: [batch, seq_len] bool/int, True/1 = valid token

        Returns:
            embeddings with padding handled according to mode
        """
        mode = self.padding_mode
        if isinstance(mode, str):
            mode = PaddingMode(mode)
        if mode == PaddingMode.NONE:
            return embeddings
        elif mode == PaddingMode.ZERO:
            return embeddings * attention_mask.unsqueeze(-1).to(
                device=embeddings.device, dtype=embeddings.dtype
            )
        elif mode == PaddingMode.LEARNED:
            mask_expanded = attention_mask.unsqueeze(-1).bool()
            padding_embedding = self.padding_embedding.to(
                device=embeddings.device, dtype=embeddings.dtype
            )
            return torch.where(
                mask_expanded,
                embeddings,
                padding_embedding.unsqueeze(0).unsqueeze(0).expand_as(embeddings),
            )
        else:
            raise ValueError(f"Unknown padding mode: {mode}")

    def forward(self, x: tp.Any) -> tp.Any:
        raise NotImplementedError()


class ScenePlan44LocalConditioner(Conditioner):
    """Encode four categorical event tracks and four geometric trajectories."""

    def __init__(
        self,
        output_dim: int,
        max_sources: int = 4,
        trajectory_feature_dim: int = 5,
        event_embedding_dim: int = 32,
        trajectory_embedding_dim: int = 32,
        **kwargs,
    ):
        super().__init__(output_dim, output_dim)
        from .sceneplan_conditioning import ScenePlan44Conditioner

        self.encoder = ScenePlan44Conditioner(
            max_sources=int(max_sources),
            trajectory_feature_dim=int(trajectory_feature_dim),
            event_embedding_dim=int(event_embedding_dim),
            trajectory_embedding_dim=int(trajectory_embedding_dim),
            output_dim=int(output_dim),
        )

    @staticmethod
    def _tensor(value, *, device, dtype=None):
        result = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
        return result.to(device=device, dtype=dtype, non_blocking=True)

    def forward(self, controls, device=None):
        events = []
        trajectories = []
        valid_masks = []
        frame_lengths = []
        for index, value in enumerate(controls):
            if not isinstance(value, dict):
                raise TypeError(f"ScenePlan controls[{index}] must be an object")
            event = self._tensor(value["source_event_frame_ids"], device=device, dtype=torch.long)
            trajectory = self._tensor(
                value["source_trajectory_features"],
                device=device,
                dtype=self.encoder.trajectory_encoder[0].weight.dtype,
            )
            valid = self._tensor(value["frame_valid_mask"], device=device, dtype=torch.bool)
            if event.ndim != 2 or int(event.shape[0]) != 4:
                raise ValueError("one ScenePlan event tensor must be [4,T]")
            if trajectory.ndim != 3 or tuple(trajectory.shape[:2]) != tuple(event.shape):
                raise ValueError("one ScenePlan trajectory tensor must be [4,T,5]")
            if tuple(valid.shape) != (int(event.shape[1]),):
                raise ValueError("ScenePlan frame_valid_mask must align with local tracks")
            if bool(value.get("cfg_unknown", False)):
                event = torch.where(
                    valid.unsqueeze(0), torch.full_like(event, -1), torch.zeros_like(event)
                )
                trajectory = torch.zeros_like(trajectory)
            events.append(event)
            trajectories.append(trajectory)
            valid_masks.append(valid)
            frame_lengths.append(int(event.shape[1]))
        max_frames = max(frame_lengths)
        padded_events = [
            F.pad(value, (0, max_frames - int(value.shape[1])), value=0) for value in events
        ]
        padded_trajectories = [
            F.pad(value, (0, 0, 0, max_frames - int(value.shape[1])), value=0)
            for value in trajectories
        ]
        padded_valid = [
            F.pad(value, (0, max_frames - int(value.shape[0])), value=False)
            for value in valid_masks
        ]
        stacked_events = torch.stack(padded_events)
        stacked_valid = torch.stack(padded_valid)
        encoded = self.encoder(
            source_event_frame_ids=stacked_events,
            source_trajectory_features=torch.stack(padded_trajectories),
        )
        return [encoded, stacked_valid]


class MultiConditioner(nn.Module):
    """
    A module that applies multiple conditioners to an input dictionary based on the keys

    Args:
        conditioners: a dictionary of conditioners with keys corresponding to the keys of the conditioning input dictionary (e.g. "prompt")
        default_keys: a dictionary of default keys to use if the key is not in the input dictionary (e.g. {"prompt_t5": "prompt"})
    """

    def __init__(
        self,
        conditioners: tp.Dict[str, Conditioner],
        default_keys: tp.Optional[tp.Dict[str, str]] = None,
        pre_encoded_keys: tp.Optional[tp.List[str]] = None,
    ):
        super().__init__()
        self.conditioners = nn.ModuleDict(conditioners)
        self.default_keys = dict(default_keys or {})
        self.pre_encoded_keys = list(pre_encoded_keys or [])

    def forward(
        self, batch_metadata: tp.List[tp.Dict[str, tp.Any]], device: tp.Union[torch.device, str]
    ) -> tp.Dict[str, tp.Any]:
        output = {}
        for key, conditioner in self.conditioners.items():
            conditioner_inputs = []
            for x in batch_metadata:
                condition_key = key
                if condition_key not in x:
                    if key in self.default_keys:
                        condition_key = self.default_keys[key]
                    else:
                        raise ValueError(f"Conditioner key {key} not found in batch metadata")
                if isinstance(x[condition_key], (list, tuple)) and len(x[condition_key]) == 1:
                    conditioner_input = x[condition_key][0]
                else:
                    conditioner_input = x[condition_key]
                conditioner_inputs.append(conditioner_input)
            if key in self.pre_encoded_keys:
                output[key] = [torch.stack(conditioner_inputs, dim=0).to(device), None]
            else:
                output[key] = conditioner(conditioner_inputs, device)
        for key in self.pre_encoded_keys:
            if key in output:
                continue
            values = []
            for index, metadata in enumerate(batch_metadata):
                if key not in metadata:
                    raise ValueError(
                        f"Pre-encoded conditioner key {key!r} not found in batch metadata row {index}"
                    )
                value = metadata[key]
                if isinstance(value, (list, tuple)) and len(value) == 1:
                    value = value[0]
                if not isinstance(value, torch.Tensor):
                    value = torch.as_tensor(value)
                values.append(value)
            try:
                stacked = torch.stack(values, dim=0)
            except RuntimeError as error:
                raise ValueError(
                    f"Pre-encoded conditioner {key!r} rows do not have one aligned shape"
                ) from error
            output[key] = [stacked.to(device, non_blocking=True), None]
        return output


class QwenTextConditioner(Conditioner):
    """Frozen local text encoder with source-specific event and speech roles."""

    def __init__(
        self,
        output_dim,
        model_path,
        max_length=512,
        hidden_dim=1024,
        project_out=True,
        padding_mode="zero",
        sceneplan_max_sources=4,
        sceneplan_role_scale=1.0,
        fail_on_truncation=True,
        enable_grad=False,
        caption_region_embedding=False,
        source_region_embedding=False,
        sceneplan_role_embedding=True,
    ):
        if (
            enable_grad
            or caption_region_embedding
            or source_region_embedding
            or (not sceneplan_role_embedding)
        ):
            raise ValueError("Only frozen ScenePlan-role text conditioning is supported")
        if not fail_on_truncation or padding_mode != "zero" or sceneplan_max_sources != 4:
            raise ValueError(
                "Text conditioning requires complete captions, zero padding and four slots"
            )
        super().__init__(hidden_dim, output_dim, project_out=project_out, padding_mode=padding_mode)
        from transformers import AutoTokenizer, AutoModelForCausalLM

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, local_files_only=True, trust_remote_code=False
        )
        causal = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            local_files_only=True,
            trust_remote_code=False,
            use_safetensors=True,
        )
        model = causal.model.eval().requires_grad_(False)
        cfg = getattr(model.config, "text_config", model.config)
        if cfg.hidden_size != hidden_dim:
            raise ValueError("Text encoder width differs from the renderer config")
        self.__dict__["model"] = model
        self.max_length = max_length
        self.sceneplan_max_sources = sceneplan_max_sources
        self.sceneplan_role_scale = sceneplan_role_scale
        self.event_role_embed = nn.Embedding(6, output_dim, padding_idx=1)
        self.speech_role_embed = nn.Embedding(6, output_dim, padding_idx=1)

    def forward(self, texts, device):
        if not texts:
            raise ValueError("Empty text batch")
        rows = []
        for value in texts:
            ids = torch.as_tensor(value["input_ids"], dtype=torch.long)
            mask = torch.as_tensor(value["attention_mask"], dtype=torch.bool)
            if (
                ids.ndim != 1
                or mask.shape != ids.shape
                or (not mask.any())
                or (mask[1:] & ~mask[:-1]).any()
            ):
                raise ValueError("Text tokens require a nonempty right-padded mask")
            if value.get("cfg_unknown", False):
                null = next(
                    (
                        n
                        for n in (
                            self.tokenizer.bos_token_id,
                            self.tokenizer.eos_token_id,
                            self.tokenizer.pad_token_id,
                        )
                        if n is not None
                    ),
                    None,
                )
                if null is None:
                    raise ValueError("Text tokenizer has no null token")
                ids = torch.full_like(
                    ids,
                    self.tokenizer.pad_token_id
                    if self.tokenizer.pad_token_id is not None
                    else null,
                )
                ids[0] = null
                mask = torch.zeros_like(mask)
                mask[0] = True
                event = torch.zeros_like(ids)
                event[0] = -1
                speech = event.clone()
            else:
                event = torch.as_tensor(value["event_source_ids"], dtype=torch.long)
                speech = torch.as_tensor(value["speech_source_ids"], dtype=torch.long)
            if event.shape != ids.shape or speech.shape != ids.shape:
                raise ValueError("Text role IDs do not align with tokens")
            if ((event < -1) | (event > 4) | (speech < -1) | (speech > 4)).any() or (
                (event > 0) & (speech > 0)
            ).any():
                raise ValueError("Invalid event/speech role IDs")
            if (~mask & ((event != 0) | (speech != 0))).any():
                raise ValueError("Padded tokens must have zero role IDs")
            if int(mask.sum()) > self.max_length:
                raise ValueError("Caption exceeds the text token limit; truncation is disabled")
            rows.append((ids, mask, event, speech))
        length = max((int(row[1].sum()) for row in rows))
        (ids, mask, event, speech) = [
            torch.stack([row[i][:length] for row in rows]).to(device) for i in range(4)
        ]
        self.model.to(
            device=device,
            dtype=torch.bfloat16 if torch.device(device).type == "cuda" else torch.float32,
        )
        with torch.no_grad():
            hidden = self.model(
                input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True
            ).last_hidden_state
        dtype = self.event_role_embed.weight.dtype
        embeddings = self.proj_out(hidden.to(dtype))
        roles = (self.event_role_embed(event + 1) + self.speech_role_embed(speech + 1)).to(
            embeddings.dtype
        )
        return (self.apply_padding(embeddings + roles * self.sceneplan_role_scale, mask), mask)
