"""Config-driven composition of optional T2A supervision providers."""
from __future__ import annotations

import importlib.util
from pathlib import Path

from stable_audio_tools.data.scene_plan_metadata import ScenePlanMetadata
from stable_audio_tools.data.spatial_story_metadata import ScenePlanTrackMetadata
from stable_audio_tools.data.spatial_conversation_metadata import (
    SpatialConversationMetadata,
    SpatialFamilyMetadata,
)


_PROVIDER_TYPES = {
    "scene_plan": ScenePlanMetadata,
    "scene_plan_tracks": ScenePlanTrackMetadata,
    "spatial_conversation": SpatialConversationMetadata,
    "spatial_family": SpatialFamilyMetadata,
}


def _provider_from_module(spec: dict):
    module_path = Path(spec.get("module", "")).expanduser().resolve()
    if not module_path.is_file():
        raise FileNotFoundError(f"T2A metadata provider module not found: {module_path}")
    module_name = f"stable_audio_tools_t2a_provider_{abs(hash(str(module_path)))}"
    module_spec = importlib.util.spec_from_file_location(module_name, module_path)
    if module_spec is None or module_spec.loader is None:
        raise ImportError(f"cannot import T2A metadata provider: {module_path}")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    factory = getattr(module, "create_custom_metadata", None)
    if not callable(factory):
        raise AttributeError(
            f"T2A provider {module_path} must export create_custom_metadata(config)"
        )
    return factory(dict(spec.get("config") or {}))


class CompositeT2AMetadata:
    def __init__(self, config: dict):
        specs = config.get("providers")
        if not isinstance(specs, list) or not specs:
            raise ValueError("composite T2A metadata requires a non-empty providers list")
        self.providers = []
        for index, spec in enumerate(specs):
            if not isinstance(spec, dict) or "type" not in spec:
                raise ValueError(f"providers[{index}] must contain a type")
            provider_type = str(spec["type"])
            if provider_type == "python_module":
                self.providers.append(_provider_from_module(spec))
                continue
            provider_class = _PROVIDER_TYPES.get(provider_type)
            if provider_class is None:
                raise ValueError(
                    f"unknown T2A metadata provider {provider_type!r}; "
                    f"expected one of {sorted([*_PROVIDER_TYPES, 'python_module'])}"
                )
            self.providers.append(provider_class(dict(spec.get("config") or {})))

    def __call__(self, info: dict, latents) -> dict:
        merged = {}
        current = dict(info)
        for provider in self.providers:
            output = provider(current, latents)
            collisions = set(merged).intersection(output)
            if collisions:
                raise KeyError(f"T2A metadata providers emitted duplicate keys: {sorted(collisions)}")
            merged.update(output)
            current.update(output)
        return merged


def create_custom_metadata(config: dict):
    return CompositeT2AMetadata(config)


__all__ = ["CompositeT2AMetadata", "create_custom_metadata"]
