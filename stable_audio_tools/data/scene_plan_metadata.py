"""Dataset metadata provider for canonical ScenePlan and crop-aware Spatial-CoT."""
from __future__ import annotations

from typing import Optional

from stable_audio_tools.data.scene_plan import (
    crop_scene_plan,
    serialize_spatial_dsl,
    tokenize_spatial_cot,
)
from stable_audio_tools.data.t2a_artifacts import IndexedJsonlStore


class ScenePlanMetadata:
    def __init__(self, config: dict):
        self.store = IndexedJsonlStore(
            config["store_dir"],
            require_ready=bool(config.get("require_ready", True)),
            max_open_shards=int(config.get("max_open_shards", 8)),
        )
        self.output_key = str(config.get("output_key", "scene_plan"))
        self.dsl_key = str(config.get("dsl_key", "scene_plan_dsl"))
        self.tokens_key = str(config.get("tokens_key", "scene_plan_tokens"))
        self.include_dsl = bool(config.get("include_dsl", True))
        self.include_tokens = bool(config.get("include_tokens", False))
        self.tokenizer_path: Optional[str] = config.get("tokenizer_path")
        self.codec_path: Optional[str] = config.get("codec_path")
        self.max_tokens = int(config.get("max_tokens", 1024))
        if self.include_tokens and not (self.tokenizer_path or self.codec_path):
            raise ValueError("include_tokens=true requires codec_path or tokenizer_path")
        if self.tokenizer_path and self.codec_path:
            raise ValueError("ScenePlan metadata accepts only one of codec_path/tokenizer_path")
        self._tokenizer = None
        self._codec = None

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_tokenizer"] = None
        state["_codec"] = None
        return state

    def _get_tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_path,
                trust_remote_code=True,
                local_files_only=True,
            )
        return self._tokenizer

    def _get_codec(self):
        if self._codec is None:
            from stable_audio_tools.data.spatial_plan_codec import SpatialPlanCodec

            self._codec = SpatialPlanCodec(self.codec_path)
        return self._codec

    @staticmethod
    def _caption(info: dict) -> str:
        for key in ("prompt_text", "prompt", "text_text", "text", "caption"):
            value = info.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        raise KeyError("ScenePlan metadata requires prompt/text/caption")

    def __call__(self, info: dict, _latents) -> dict:
        full_plan = self.store.get(info["path"])
        timestamps = info.get("timestamps")
        plan = crop_scene_plan(full_plan, timestamps)
        output = {
            self.output_key: plan,
            f"{self.output_key}_sample_id": plan["sample_id"],
            f"{self.output_key}_crop_aligned": True,
        }
        if self.include_dsl or self.include_tokens:
            dsl = serialize_spatial_dsl(plan)
            output[self.dsl_key] = dsl
        if self.include_tokens:
            if self.codec_path:
                codec = self._get_codec()
                output[self.tokens_key] = codec.encode(plan, max_tokens=self.max_tokens)
                output[f"{self.tokens_key}_codec"] = codec.details["codec_name"]
                output[f"{self.tokens_key}_codec_fingerprint"] = codec.fingerprint
            else:
                output[self.tokens_key] = tokenize_spatial_cot(
                    self._caption(info),
                    plan,
                    self._get_tokenizer(),
                    max_tokens=self.max_tokens,
                )
        return output


def create_custom_metadata(config: dict):
    return ScenePlanMetadata(config)


__all__ = ["ScenePlanMetadata", "create_custom_metadata"]
