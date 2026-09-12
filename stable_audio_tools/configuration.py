"""Configuration loading and validation helpers.

Plain JSON files keep their historical behaviour.  A config may optionally add
``"extends"`` with one path (or a list of paths); parent paths are resolved
relative to the child config and dictionaries are merged recursively.  Lists
and scalar values are replaced, not concatenated.  This keeps experiment
configs explicit while avoiding another model/training code tree for each T2A
variant.
"""
from __future__ import annotations
import os

import copy
import json
import math
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Optional, Sequence, Union

from .paths import expand_config_values


ConfigPath = Union[str, Path]


class ConfigError(ValueError):
    """Raised when a config cannot be resolved or is internally inconsistent."""


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict:
    # Some config objects are complete contracts rather than additive maps
    # (for example modality-id -> loss-weight after replacing the modality
    # list).  ``__replace__`` makes that intent explicit without teaching the
    # generic loader about every experiment-specific field.
    if override.get("__replace__") is True:
        return copy.deepcopy(
            {key: value for key, value in override.items() if key != "__replace__"}
        )
    merged = copy.deepcopy(dict(base))
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], MutableMapping)
            and isinstance(value, Mapping)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _read_json(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except FileNotFoundError as exc:
        raise ConfigError(f"config does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(
            f"invalid JSON in {path}:{exc.lineno}:{exc.colno}: {exc.msg}"
        ) from exc
    if not isinstance(value, dict):
        raise ConfigError(f"top-level config must be an object: {path}")
    return value


def _load_config(path: Path, stack: Sequence[Path]) -> dict:
    path = path.expanduser().resolve()
    if path in stack:
        cycle = " -> ".join(str(item) for item in (*stack, path))
        raise ConfigError(f"config inheritance cycle: {cycle}")

    config = _read_json(path)
    parents = config.pop("extends", None)
    if parents is None:
        return config
    if isinstance(parents, (str, Path)):
        parents = [parents]
    if not isinstance(parents, list) or not all(
        isinstance(parent, (str, Path)) for parent in parents
    ):
        raise ConfigError(f"'extends' must be a path or list of paths: {path}")

    resolved: dict = {}
    next_stack = (*stack, path)
    for parent in parents:
        parent_path = Path(parent).expanduser()
        if not parent_path.is_absolute():
            parent_path = path.parent / parent_path
        resolved = _deep_merge(resolved, _load_config(parent_path, next_stack))
    return _deep_merge(resolved, config)


def load_config(path: ConfigPath) -> dict:
    """Load a plain or inherited JSON config and return a fully resolved copy."""

    return expand_config_values(_load_config(Path(path), ()))


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{name} must be an object")
    return value


def _require_positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigError(f"{name} must be a positive integer, got {value!r}")
    return value


def is_t2a_config(model_config: Mapping[str, Any]) -> bool:
    task = model_config.get("task")
    return isinstance(task, Mapping) and task.get("type") == "t2a"


def _pretransform_dimensions(model_config: Mapping[str, Any]) -> tuple[int, int, int]:
    model = _require_mapping(model_config.get("model"), "model")
    pretransform = _require_mapping(model.get("pretransform"), "model.pretransform")
    pretransform_config = _require_mapping(
        pretransform.get("config"), "model.pretransform.config"
    )
    ratio = _require_positive_int(
        pretransform_config.get("downsampling_ratio"),
        "model.pretransform.config.downsampling_ratio",
    )
    latent_dim = _require_positive_int(
        pretransform_config.get("latent_dim"),
        "model.pretransform.config.latent_dim",
    )
    io_channels = _require_positive_int(
        pretransform_config.get("io_channels"),
        "model.pretransform.config.io_channels",
    )
    return ratio, latent_dim, io_channels


def _validate_dataset_compatibility(
    model_config: Mapping[str, Any],
    dataset_config: Mapping[str, Any],
    *,
    ratio: int,
    latent_length: int,
    allow_missing_speech_timing: bool = False,
) -> None:
    if dataset_config.get("dataset_type") not in {
        "pre_encoded",
        "spatial_family_preencoded",
        "sceneplan_v2_preencoded",
        "sceneplan_transfusion_editing_preencoded",
        "sceneplan_p11_single_turn",
    }:
        raise ConfigError(
            "T2A latent training requires dataset_type='pre_encoded', "
            "'spatial_family_preencoded', 'sceneplan_v2_preencoded', or "
            "'sceneplan_transfusion_editing_preencoded', or "
            "'sceneplan_p11_single_turn'; "
            f"got {dataset_config.get('dataset_type')!r}"
        )

    model_section = _require_mapping(model_config.get("model"), "model")
    diffusion_section = _require_mapping(
        model_section.get("diffusion"), "model.diffusion"
    )
    editing_input_ids = ["sceneplan_44", "source_foa_latent"]
    editing_model = diffusion_section.get("input_concat_ids") == editing_input_ids
    editing_dataset = (
        dataset_config.get("dataset_type")
        == "sceneplan_transfusion_editing_preencoded"
    )
    if editing_model != editing_dataset:
        raise ConfigError(
            "Transfusion Editing model and paired Editing dataset must be used "
            "together; the aligned source_foa_latent branch cannot be omitted "
            "or supplied to a Generation-only model"
        )

    dataset_ratio = dataset_config.get("latent_downsampling_ratio")
    if dataset_ratio is not None and int(dataset_ratio) != ratio:
        raise ConfigError(
            "dataset latent_downsampling_ratio does not match the model "
            f"pretransform: {dataset_ratio} != {ratio}"
        )

    crop_length = dataset_config.get("latent_crop_length")
    if crop_length is not None:
        crop_length = _require_positive_int(
            crop_length, "dataset latent_crop_length"
        )
        if crop_length != latent_length:
            raise ConfigError(
                "dataset latent_crop_length does not match model sample_size / "
                f"downsampling_ratio: {crop_length} != {latent_length}"
            )

    if dataset_config.get("dataset_type") == "sceneplan_v2_preencoded":
        if dataset_config.get("random_crop", False) is not False:
            raise ConfigError("ScenePlan-v2 dataset must set random_crop=false")
        if crop_length not in {432, 648} or latent_length not in {432, 648}:
            raise ConfigError(
                "ScenePlan-v2 latent/batch ceiling must be 432 or 648 frames"
            )
        if int(dataset_config.get("caption_max_tokens", -1)) != 512:
            raise ConfigError("ScenePlan-v2 revised caption_max_tokens must be 512")
        if dataset_config.get("require_complete", False) is not True:
            raise ConfigError("ScenePlan-v2 training requires a frozen complete index")
        bucket_config = dataset_config.get("length_bucket_batching")
        if bucket_config is not None:
            bucket_config = _require_mapping(
                bucket_config, "dataset.length_bucket_batching"
            )
            enabled = bucket_config.get("enabled", False)
            if not isinstance(enabled, bool):
                raise ConfigError(
                    "dataset.length_bucket_batching.enabled must be boolean"
                )
            if enabled:
                if latent_length != 648:
                    raise ConfigError(
                        "ScenePlan length-bucket batching requires the 648-frame envelope"
                    )
                long_batch_size = bucket_config.get("long_batch_size")
                if long_batch_size is not None:
                    _require_positive_int(
                        long_batch_size,
                        "dataset.length_bucket_batching.long_batch_size",
                    )
                seed = bucket_config.get("seed", 0)
                if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
                    raise ConfigError(
                        "dataset.length_bucket_batching.seed must be a non-negative integer"
                    )
                if dataset_config.get("drop_last", False) is not True:
                    raise ConfigError(
                        "ScenePlan distributed length buckets require drop_last=true"
                    )

    if (
        dataset_config.get("dataset_type")
        == "sceneplan_transfusion_editing_preencoded"
    ):
        if dataset_config.get("random_crop", False) is not False:
            raise ConfigError("aligned Transfusion Editing must set random_crop=false")
        if crop_length != 648 or latent_length != 648:
            raise ConfigError(
                "Transfusion Editing must retain the canonical P10 648-frame envelope"
            )
        if int(dataset_config.get("caption_max_tokens", -1)) != 512:
            raise ConfigError("Editing new-plan semantic captions require 512 tokens")
        if dataset_config.get("require_complete", False) is not True:
            raise ConfigError("Editing requires a finalized frozen paired index")
        if dataset_config.get("speech_timing_index_path") is not None:
            raise ConfigError("Editing/new-plan semantic v2 forbids a timing sidecar")
        verify_hashes = dataset_config.get("verify_tensor_hashes_on_access", False)
        if not isinstance(verify_hashes, bool):
            raise ConfigError("verify_tensor_hashes_on_access must be boolean")
        index_sha = dataset_config.get("index_sha256")
        if index_sha is not None and (
            not isinstance(index_sha, str) or len(index_sha) != 64
        ):
            raise ConfigError("Editing index_sha256 must be a 64-character digest")
        bucket_config = dataset_config.get("length_bucket_batching")
        if bucket_config is not None:
            bucket_config = _require_mapping(
                bucket_config, "dataset.length_bucket_batching"
            )
            enabled = bucket_config.get("enabled", False)
            if not isinstance(enabled, bool):
                raise ConfigError(
                    "dataset.length_bucket_batching.enabled must be boolean"
                )
            if enabled:
                long_batch_size = bucket_config.get("long_batch_size")
                if long_batch_size is not None:
                    _require_positive_int(
                        long_batch_size,
                        "dataset.length_bucket_batching.long_batch_size",
                    )
                seed = bucket_config.get("seed", 42)
                if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
                    raise ConfigError(
                        "dataset.length_bucket_batching.seed must be non-negative"
                    )
                if dataset_config.get("drop_last", False) is not True:
                    raise ConfigError(
                        "Editing distributed length buckets require drop_last=true"
                    )

    if dataset_config.get("dataset_type") == "sceneplan_p11_single_turn":
        if crop_length != 648 or latent_length != 648:
            raise ConfigError(
                "canonical P11 must match the current P10 648-frame envelope"
            )
        if dataset_config.get("require_complete", False) is not True:
            raise ConfigError("P11 requires the frozen P9 base index")
        if dataset_config.get("shuffle", False) is not False:
            raise ConfigError(
                "P11 uses a pre-shuffled G/U/E-interleaved manifest and must set shuffle=false"
            )
        for key in ("manifest_path", "codec_path"):
            if not isinstance(dataset_config.get(key), str) or not dataset_config.get(key):
                raise ConfigError(f"P11 dataset requires non-empty {key}")
        _require_positive_int(
            dataset_config.get("index_num_samples"), "P11 index_num_samples"
        )
        require_exact_batch = dataset_config.get(
            "require_exact_batch_size", False
        )
        if not isinstance(require_exact_batch, bool):
            raise ConfigError("P11 require_exact_batch_size must be boolean")
        if require_exact_batch and dataset_config.get("drop_last") is not True:
            raise ConfigError(
                "P11 exact-batch throughput measurement requires drop_last=true"
            )
        curriculum_path = dataset_config.get("p11_v4_curriculum_path")
        curriculum_rows = dataset_config.get(
            "p11_v4_curriculum_expected_rows"
        )
        if (curriculum_path is None) != (curriculum_rows is None):
            raise ConfigError(
                "P11 curriculum path and expected row count must be configured together"
            )
        if curriculum_path is not None:
            if not isinstance(curriculum_path, str) or not curriculum_path:
                raise ConfigError("P11 curriculum path must be a non-empty string")
            _require_positive_int(
                curriculum_rows, "P11 curriculum expected rows"
            )
            ordering_contract = dataset_config.get(
                "p11_v4_curriculum_ordering_contract"
            )
            ordering_batch_size = dataset_config.get(
                "p11_v4_curriculum_ordering_batch_size"
            )
            if (ordering_contract is None) != (ordering_batch_size is None):
                raise ConfigError(
                    "P11 curriculum ordering contract and batch size must be "
                    "configured together"
                )
            if ordering_contract is not None:
                pilot_ordering = (
                    "p11_v4_pair_aware_interleaved_batch8_diverse_instruction_surface_v6"
                )
                screening_ordering = (
                    "p11_v4_matched_triplet_interleaved_batch8_v1"
                )
                ddp8_ordering = (
                    "p11_v4_ddp8_rank_balanced_pair_aware_batch8_v7"
                )
                if ordering_contract not in {
                    pilot_ordering,
                    screening_ordering,
                    ddp8_ordering,
                }:
                    raise ConfigError("unknown P11 curriculum ordering contract")
                if _require_positive_int(
                    ordering_batch_size,
                    "P11 curriculum ordering batch size",
                ) != 8:
                    raise ConfigError(
                        "P11 canonical pair-aware curriculum is frozen at batch size 8"
                    )
                if dataset_config.get("drop_last") is not True or (
                    dataset_config.get("require_exact_batch_size") is not True
                ):
                    raise ConfigError(
                        "P11 pair-aware curriculum requires exact drop-last batches"
                    )
                if ordering_contract == screening_ordering:
                    screening_expected = {
                        "p11_v4_curriculum_contract": (
                            "p10_v11_matched_10k_gue_screening_v1"
                        ),
                        "p11_v4_curriculum_expected_rows": 30_000,
                        "expected_num_samples": 30_000,
                        "in_order": True,
                    }
                    for key, expected in screening_expected.items():
                        if dataset_config.get(key) != expected:
                            raise ConfigError(
                                f"P11 screening {key} must be {expected!r}"
                            )
                if ordering_contract == ddp8_ordering:
                    ddp8_expected = {
                        "p11_v4_curriculum_contract": (
                            "p10_v11_train_only_gue_multitarget_v1"
                        ),
                        "p11_v4_curriculum_expected_rows": 269_568,
                        "in_order": True,
                    }
                    for key, expected in ddp8_expected.items():
                        if dataset_config.get(key) != expected:
                            raise ConfigError(
                                f"P11 DDP8 curriculum {key} must be {expected!r}"
                            )
                    if int(curriculum_rows) % 64:
                        raise ConfigError(
                            "P11 DDP8 curriculum must be divisible by global batch 64"
                        )

    for option in ("min_length_sec", "max_length_sec"):
        value = dataset_config.get(option)
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or value < 0
        ):
            raise ConfigError(f"dataset {option} must be a non-negative number")
    minimum = dataset_config.get("min_length_sec")
    maximum = dataset_config.get("max_length_sec")
    if minimum is not None and maximum is not None and minimum > maximum:
        raise ConfigError("dataset min_length_sec cannot exceed max_length_sec")

    datasets = dataset_config.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise ConfigError("T2A dataset config requires a non-empty 'datasets' list")
    expected_num_samples = dataset_config.get("expected_num_samples")
    if expected_num_samples is not None:
        _require_positive_int(expected_num_samples, "expected_num_samples")
    if dataset_config.get("require_complete", False) and expected_num_samples is None:
        raise ConfigError(
            "require_complete=true requires expected_num_samples"
        )
    for index, dataset in enumerate(datasets):
        dataset = _require_mapping(dataset, f"datasets[{index}]")
        dataset_path = dataset.get("path")
        if not isinstance(dataset_path, (str, list)) or not dataset_path:
            raise ConfigError(f"datasets[{index}].path must be a path or path list")
        if isinstance(dataset_path, list) and not all(
            isinstance(item, str) and item for item in dataset_path
        ):
            raise ConfigError(
                f"datasets[{index}].path list must contain non-empty paths"
            )
    if editing_dataset:
        if len(datasets) != 1:
            raise ConfigError("Editing requires exactly one split index per loader")
        index_rows = dataset_config.get("index_num_samples")
        if index_rows is not None:
            _require_positive_int(index_rows, "Editing index_num_samples")
        sample_ordinals = dataset_config.get("sample_ordinals")
        ordinal_range = dataset_config.get("ordinal_range")
        if sample_ordinals is not None and ordinal_range is not None:
            raise ConfigError(
                "Editing sample_ordinals and ordinal_range are mutually exclusive"
            )
    if dataset_config.get("dataset_type") == "sceneplan_v2_preencoded" and len(datasets) > 1:
        if dataset_config.get("index_num_samples") is not None:
            raise ConfigError(
                "multi-index ScenePlan-v2 puts index_num_samples on each dataset entry"
            )
        if dataset_config.get("sample_ordinals") is not None:
            raise ConfigError(
                "multi-index ScenePlan-v2 puts sample_ordinals on each dataset entry"
            )
        if dataset_config.get("ordinal_range") is not None:
            raise ConfigError(
                "multi-index ScenePlan-v2 puts ordinal_range on each dataset entry"
            )
        child_rows = 0
        for index, dataset in enumerate(datasets):
            rows = _require_positive_int(
                dataset.get("num_samples"), f"datasets[{index}].num_samples"
            )
            child_rows += rows
            weight = dataset.get("weight", 1.0)
            if isinstance(weight, bool) or not isinstance(weight, (int, float)):
                raise ConfigError(f"datasets[{index}].weight must be numeric")
            if float(weight) != 1.0:
                raise ConfigError(
                    "multi-index ScenePlan-v2 currently requires weight=1.0; "
                    "row proportions are explicit and must not be silently resampled"
                )
            ordinal_range = dataset.get("ordinal_range")
            if ordinal_range is not None:
                if (
                    not isinstance(ordinal_range, (list, tuple))
                    or len(ordinal_range) != 2
                    or any(
                        isinstance(value, bool) or not isinstance(value, int)
                        for value in ordinal_range
                    )
                ):
                    raise ConfigError(
                        f"datasets[{index}].ordinal_range must be [start, stop] integers"
                    )
                start, stop = ordinal_range
                index_rows = _require_positive_int(
                    dataset.get("index_num_samples"),
                    f"datasets[{index}].index_num_samples",
                )
                if not 0 <= start < stop <= index_rows:
                    raise ConfigError(
                        f"datasets[{index}].ordinal_range is outside its frozen index"
                    )
                if stop - start != rows:
                    raise ConfigError(
                        f"datasets[{index}].ordinal_range length does not match "
                        f"num_samples: {stop - start} != {rows}"
                    )
        if child_rows != int(expected_num_samples):
            raise ConfigError(
                "multi-index ScenePlan-v2 child rows do not sum to "
                f"expected_num_samples: {child_rows} != {expected_num_samples}"
            )

    model_type = model_config.get("model_type")
    if model_type in {
        "sceneplan_p11",
        "sceneplan_p11_v4",
        "sceneplan_p11_audio_aware_v1",
    }:
        if dataset_config.get("dataset_type") != "sceneplan_p11_single_turn":
            raise ConfigError(
                "P11 requires dataset_type=sceneplan_p11_single_turn"
            )
        for key in ("semantic_cache_path", "semantic_encoder_revision"):
            if not isinstance(dataset_config.get(key), str) or not dataset_config.get(key):
                raise ConfigError(f"sceneplan_p11 dataset requires non-empty {key}")
        if int(dataset_config.get("semantic_dim", -1)) != 512:
            raise ConfigError("sceneplan_p11 semantic_dim must remain 512")
        p11_contract = str(dataset_config.get("p11_contract", "discrete_d0_v1"))
        retired_dataset_fields = {
            "scene_thought_path",
            "scene_thought_required",
            "prompt_view_policy",
        } & set(dataset_config)
        if retired_dataset_fields:
            raise ConfigError(
                "retired core40 dataset fields are forbidden: "
                f"{sorted(retired_dataset_fields)}"
            )
        if model_type in {"sceneplan_p11_v4", "sceneplan_p11_audio_aware_v1"}:
            expected_contract = (
                "audio_aware_sketch_first_transfusion_cot_v2"
                if model_type == "sceneplan_p11_audio_aware_v1"
                else "sketch_first_transfusion_cot_v4"
            )
            if p11_contract != expected_contract:
                raise ConfigError(
                    f"{model_type} requires p11_contract={expected_contract}"
                )
            lexical_mode = str(dataset_config.get("lexical_evidence_mode", "none"))
            if lexical_mode not in {
                "none",
                "frozen_asr_cache_v1",
                "oracle_transcript_wiring_v1",
            }:
                raise ConfigError("P11 dataset lexical evidence mode is invalid")
            if lexical_mode == "frozen_asr_cache_v1":
                for key in ("lexical_cache_path", "lexical_encoder_revision"):
                    if not isinstance(dataset_config.get(key), str) or not dataset_config.get(key):
                        raise ConfigError(
                            f"P11 frozen ASR evidence requires non-empty {key}"
                        )
                lexical_threshold = dataset_config.get(
                    "lexical_confidence_threshold"
                )
                if (
                    not isinstance(lexical_threshold, (int, float))
                    or isinstance(lexical_threshold, bool)
                    or not 0.0 <= float(lexical_threshold) <= 1.0
                ):
                    raise ConfigError(
                        "P11 frozen ASR evidence requires a train-calibrated "
                        "lexical_confidence_threshold within [0,1]"
                    )
            elif dataset_config.get("lexical_cache_path") is not None or (
                dataset_config.get("lexical_encoder_revision") is not None
            ) or dataset_config.get("lexical_confidence_threshold") is not None:
                raise ConfigError(
                    "P11 lexical cache/threshold fields require "
                    "frozen_asr_cache_v1"
                )
            if model_type == "sceneplan_p11_audio_aware_v1":
                if int(dataset_config.get("seed", -1)) != 42:
                    raise ConfigError("active P11 audio-aware data must use seed 42")
                if dataset_config.get("shuffle", False) is not False:
                    raise ConfigError("audio-aware manifest ordering is immutable")
        elif p11_contract != "discrete_d0_v1":
            raise ConfigError(
                "the discrete D0 model requires p11_contract=discrete_d0_v1"
            )

def validate_t2a_config(
    model_config: Mapping[str, Any],
    dataset_config: Optional[Mapping[str, Any]] = None,
    *,
    allow_missing_speech_timing: bool = False,
) -> dict:
    """Validate one opt-in T2A experiment and return a compact resolved summary.

    Configs without ``task.type == "t2a"`` are intentionally ignored by the
    generic training entry point.  Calling this function directly requires a
    T2A config.
    """

    if not is_t2a_config(model_config):
        raise ConfigError("model config is not marked with task.type='t2a'")

    sample_size = _require_positive_int(model_config.get("sample_size"), "sample_size")
    sample_rate = _require_positive_int(model_config.get("sample_rate"), "sample_rate")
    audio_channels = _require_positive_int(
        model_config.get("audio_channels"), "audio_channels"
    )
    if audio_channels != 4:
        raise ConfigError(f"current T2A configs target FOA 4ch, got {audio_channels}ch")

    ratio, latent_dim, pretransform_channels = _pretransform_dimensions(model_config)
    if pretransform_channels != audio_channels:
        raise ConfigError(
            "pretransform io_channels must match audio_channels: "
            f"{pretransform_channels} != {audio_channels}"
        )
    if sample_size % ratio:
        raise ConfigError(
            f"sample_size ({sample_size}) must be divisible by downsampling_ratio ({ratio})"
        )
    latent_length = sample_size // ratio

    # Validate the fixed-window contract before constructing a large model or
    # loading any checkpoint.  Spatial-CoT treats the complete waveform
    # window—including silent intervals—as modeled state; silently changing
    # any one of these values would invalidate every frame-aligned control.
    fixed_window = _require_mapping(
        model_config.get("fixed_audio_window", {}), "fixed_audio_window"
    )
    if bool(fixed_window.get("enabled", False)):
        fixed_rate = _require_positive_int(
            fixed_window.get("sample_rate"), "fixed_audio_window.sample_rate"
        )
        fixed_samples = _require_positive_int(
            fixed_window.get("num_samples"), "fixed_audio_window.num_samples"
        )
        fixed_frames = _require_positive_int(
            fixed_window.get("latent_frames"), "fixed_audio_window.latent_frames"
        )
        fixed_ratio = _require_positive_int(
            fixed_window.get("downsampling_ratio"),
            "fixed_audio_window.downsampling_ratio",
        )
        if fixed_samples != fixed_frames * fixed_ratio:
            raise ConfigError(
                "fixed_audio_window requires num_samples == "
                "latent_frames * downsampling_ratio"
            )
        if (fixed_rate, fixed_samples, fixed_frames, fixed_ratio) != (
            sample_rate,
            sample_size,
            latent_length,
            ratio,
        ):
            raise ConfigError(
                "fixed_audio_window disagrees with the resolved model/VAE "
                "dimensions: "
                f"fixed={(fixed_rate, fixed_samples, fixed_frames, fixed_ratio)} "
                f"resolved={(sample_rate, sample_size, latent_length, ratio)}"
            )
        exact_duration = fixed_samples / float(fixed_rate)
        configured_duration = fixed_window.get("duration_sec", exact_duration)
        if (
            isinstance(configured_duration, bool)
            or not isinstance(configured_duration, (int, float))
            or not math.isfinite(float(configured_duration))
            or not math.isclose(
                float(configured_duration),
                exact_duration,
                rel_tol=0.0,
                abs_tol=1.0e-9,
            )
        ):
            raise ConfigError(
                "fixed_audio_window.duration_sec must equal "
                "num_samples / sample_rate"
            )

    model_type = model_config.get("model_type")
    model = _require_mapping(model_config.get("model"), "model")
    training = _require_mapping(model_config.get("training"), "training")
    route_id = model_config.get("route_id")
    if route_id is not None and route_id not in {
        "continuous_traj",
        "spatial_cot",
        "hybrid_source_tracks",
        "sceneplan_p11",
        "sceneplan_p11_audio_aware_v1",
    }:
        raise ConfigError(f"unknown T2A route_id: {route_id!r}")
    if not training.get("pre_encoded", False):
        raise ConfigError("current T2A stage-2 recipes require training.pre_encoded=true")
    if (
        training.get("optimizer_configs") is None
        and training.get("learning_rate") is None
    ):
        raise ConfigError(
            "T2A training requires optimizer_configs or learning_rate"
        )
    cfg_dropout = training.get("cfg_dropout_prob", 0.1)
    if (
        isinstance(cfg_dropout, bool)
        or not isinstance(cfg_dropout, (int, float))
        or not 0.0 <= cfg_dropout <= 1.0
    ):
        raise ConfigError("training.cfg_dropout_prob must be in [0, 1]")
    mask_loss_weight = training.get("mask_loss_weight", 0.0)
    if (
        isinstance(mask_loss_weight, bool)
        or not isinstance(mask_loss_weight, (int, float))
        or mask_loss_weight < 0
    ):
        raise ConfigError("training.mask_loss_weight must be non-negative")
    if training.get("loss_normalization", "none") not in {
        "none",
        "timestep",
        "sample",
        "sample_channel",
    }:
        raise ConfigError(
            "training.loss_normalization must be none, timestep, sample, "
            "or sample_channel"
        )
    if training.get("use_ema", True):
        ema_beta = training.get("ema_beta", 0.999)
        ema_power = training.get("ema_power", 0.75)
        if (
            isinstance(ema_beta, bool)
            or not isinstance(ema_beta, (int, float))
            or not 0.0 <= ema_beta <= 1.0
        ):
            raise ConfigError("training.ema_beta must be in [0, 1]")
        if (
            isinstance(ema_power, bool)
            or not isinstance(ema_power, (int, float))
            or ema_power <= 0
        ):
            raise ConfigError("training.ema_power must be positive")
        _require_positive_int(
            training.get("ema_update_every", 1),
            "training.ema_update_every",
        )
        update_after = training.get("ema_update_after_step", 1)
        if (
            isinstance(update_after, bool)
            or not isinstance(update_after, int)
            or update_after < 0
        ):
            raise ConfigError(
                "training.ema_update_after_step must be a non-negative integer"
            )
    demo = training.get("demo", {})
    if isinstance(demo, Mapping):
        demo_conditions = demo.get("demo_cond", [])
        if isinstance(demo_conditions, list):
            unsupported_formats = sorted(
                {
                    str(item.get("spatial_format")).lower()
                    for item in demo_conditions
                    if isinstance(item, Mapping)
                    and str(item.get("spatial_format", "foa")).lower() != "foa"
                }
            )
            if unsupported_formats:
                raise ConfigError(
                    "current T2A checkpoints are FOA-only, but demo_cond contains: "
                    f"{unsupported_formats}"
                )

    text_mode: str
    architecture: str
    if model_type == "diffusion_cond":
        architecture = "dit"
        diffusion = _require_mapping(model.get("diffusion"), "model.diffusion")
        if diffusion.get("type") != "dit":
            raise ConfigError("T2A diffusion_cond recipe requires model.diffusion.type='dit'")
        if diffusion.get("diffusion_objective") != "rectified_flow":
            raise ConfigError("T2A DiT recipe requires rectified_flow")
        if not diffusion.get("mask_padding_attention", False):
            raise ConfigError("T2A DiT requires mask_padding_attention=true")

        diffusion_config = _require_mapping(
            diffusion.get("config"), "model.diffusion.config"
        )
        if int(diffusion_config.get("io_channels", -1)) != latent_dim:
            raise ConfigError(
                "DiT io_channels must match VAE latent_dim: "
                f"{diffusion_config.get('io_channels')} != {latent_dim}"
            )
        if int(model.get("io_channels", -1)) != latent_dim:
            raise ConfigError(
                "model.io_channels must match VAE latent_dim: "
                f"{model.get('io_channels')} != {latent_dim}"
            )

        conditioning = _require_mapping(model.get("conditioning"), "model.conditioning")
        cond_dim = _require_positive_int(
            conditioning.get("cond_dim"),
            "model.conditioning.cond_dim",
        )
        if int(diffusion_config.get("cond_token_dim", -1)) != cond_dim:
            raise ConfigError(
                "DiT cond_token_dim must match conditioning.cond_dim: "
                f"{diffusion_config.get('cond_token_dim')} != {cond_dim}"
            )
        embed_dim = _require_positive_int(
            diffusion_config.get("embed_dim"),
            "model.diffusion.config.embed_dim",
        )
        depth = _require_positive_int(
            diffusion_config.get("depth"),
            "model.diffusion.config.depth",
        )
        num_heads = _require_positive_int(
            diffusion_config.get("num_heads"),
            "model.diffusion.config.num_heads",
        )
        if embed_dim % num_heads:
            raise ConfigError(
                f"DiT embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )
        head_dim = embed_dim // num_heads
        context_dim = (
            embed_dim
            if diffusion_config.get("project_cond_tokens", True)
            else cond_dim
        )
        if context_dim % head_dim:
            raise ConfigError(
                "cross-attention context width must be divisible by DiT head "
                f"width: {context_dim} % {head_dim} != 0"
            )
        context_heads = context_dim // head_dim
        if context_heads <= 0 or num_heads % context_heads:
            raise ConfigError(
                "DiT query heads must be divisible by context K/V heads: "
                f"{num_heads} % {context_heads} != 0"
            )
        conditioner_configs = conditioning.get("configs")
        if not isinstance(conditioner_configs, list):
            raise ConfigError("model.conditioning.configs must be a list")
        conditioner_ids = [
            item.get("id")
            for item in conditioner_configs
            if isinstance(item, Mapping) and item.get("id")
        ]
        if len(conditioner_ids) != len(set(conditioner_ids)):
            raise ConfigError("model.conditioning.configs contains duplicate ids")
        by_id = {
            item.get("id"): item
            for item in conditioner_configs
            if isinstance(item, Mapping) and item.get("id")
        }
        sceneplan_v2 = "sceneplan_44" in by_id
        required_ids = (
            {"prompt", "sceneplan_44"}
            if sceneplan_v2
            else {"prompt", "spatial_format", "seconds_start", "seconds_total"}
        )
        missing = sorted(required_ids - set(by_id))
        if missing:
            raise ConfigError(f"T2A DiT is missing conditioners: {missing}")
        if sceneplan_v2:
            if diffusion.get("cross_attention_cond_ids", []) != ["prompt"]:
                raise ConfigError(
                    "ScenePlan v2 cross-attention must contain caption prompt only"
                )
            if diffusion.get("global_cond_ids", []):
                raise ConfigError("ScenePlan v2 does not use legacy seconds globals")
            input_concat_ids = diffusion.get("input_concat_ids", [])
            editing_input_concat = ["sceneplan_44", "source_foa_latent"]
            if input_concat_ids not in (["sceneplan_44"], editing_input_concat):
                raise ConfigError(
                    "ScenePlan v2 input concat must be [sceneplan_44], or the "
                    "Transfusion Editing order [sceneplan_44, source_foa_latent]"
                )
            fusion_config = _require_mapping(
                by_id["sceneplan_44"].get("config"),
                "ScenePlan 4+4 conditioner config",
            )
            fusion_dim = _require_positive_int(
                fusion_config.get("output_dim"),
                "ScenePlan 4+4 output_dim",
            )
            editing_dit = input_concat_ids == editing_input_concat
            expected_input_concat_dim = fusion_dim + (
                int(model.get("io_channels", -1)) if editing_dit else 0
            )
            if (
                int(diffusion_config.get("input_concat_dim", -1))
                != expected_input_concat_dim
            ):
                raise ConfigError(
                    "DiT input_concat_dim does not match its ordered frame "
                    "conditions: "
                    f"{diffusion_config.get('input_concat_dim')} != "
                    f"{expected_input_concat_dim}"
                )
            if editing_dit:
                pre_encoded_keys = conditioning.get("pre_encoded_keys", [])
                if pre_encoded_keys != ["source_foa_latent"]:
                    raise ConfigError(
                        "Transfusion Editing must declare source_foa_latent as "
                        "its sole pre-encoded conditioner"
                    )
                if "source_foa_latent" in by_id:
                    raise ConfigError(
                        "source_foa_latent is a direct aligned tensor and must "
                        "not register a learned conditioner"
                    )
                if (
                    diffusion_config.get("require_explicit_negative_input_concat")
                    is not True
                ):
                    raise ConfigError(
                        "Transfusion Editing CFG must receive an explicit "
                        "[unknown new-plan, same source] negative condition"
                    )
                if training.get("pre_encoded") is not True:
                    raise ConfigError("Editing DiT must train on frozen FOA latents")
                if float(
                    training.get("sceneplan_speech_active_loss_weight", 1.0)
                ) != 1.0:
                    raise ConfigError(
                        "Editing DiT v1 uses only masked rectified-flow MSE"
                    )
                for auxiliary_key in (
                    "sceneplan_sound_transient_loss",
                    "sceneplan_speech_duration_loss",
                    "sceneplan_sound_temporal_difference_loss",
                ):
                    auxiliary = training.get(auxiliary_key)
                    if isinstance(auxiliary, Mapping) and bool(
                        auxiliary.get("enabled", False)
                    ):
                        raise ConfigError(
                            "Editing DiT v1 forbids inherited P10 auxiliary "
                            f"objective {auxiliary_key}"
                        )
            if int(fusion_config.get("max_sources", -1)) != 4:
                raise ConfigError(
                    "ScenePlan 4+4 conditioner requires max_sources=4"
                )
            if int(fusion_config.get("trajectory_feature_dim", -1)) != 5:
                raise ConfigError(
                    "ScenePlan trajectory_feature_dim must be 5"
                )
            per_source_dim = int(
                fusion_config.get("event_embedding_dim", -1)
            ) + int(fusion_config.get("trajectory_embedding_dim", -1))
            if per_source_dim <= 0 or per_source_dim * 4 != fusion_dim:
                raise ConfigError(
                    "ScenePlan output must be the direct concatenation of four "
                    "event+trajectory source blocks"
                )
            if diffusion_config.get("input_concat_cond_cfg") is not True:
                raise ConfigError(
                    "ScenePlan local conditions must participate in inference CFG"
                )
            if "sceneplan_raw_cfg_dropout" in training:
                raise ConfigError(
                    "legacy joint ScenePlan CFG dropout is forbidden"
                )
            if float(cfg_dropout) != 0.0:
                raise ConfigError(
                    "ScenePlan raw branch dropout requires cfg_dropout_prob=0"
                )
            sceneplan_cfg = _require_mapping(
                training.get("sceneplan_cfg_dropout"),
                "training.sceneplan_cfg_dropout",
            )
            expected_cfg_keys = {
                "mode",
                "caption_unknown_prob",
                "structured_unknown_prob",
            }
            if set(sceneplan_cfg) != expected_cfg_keys:
                raise ConfigError(
                    "training.sceneplan_cfg_dropout must contain exactly "
                    f"{sorted(expected_cfg_keys)}"
                )
            if sceneplan_cfg.get("mode") != "independent":
                raise ConfigError(
                    "ScenePlan caption and structured CFG dropout must be independent"
                )
            for key in ("caption_unknown_prob", "structured_unknown_prob"):
                probability = sceneplan_cfg.get(key)
                if (
                    isinstance(probability, bool)
                    or not isinstance(probability, (int, float))
                    or not math.isfinite(float(probability))
                    or not math.isclose(
                        float(probability), 0.15, rel_tol=0.0, abs_tol=1.0e-12
                    )
                ):
                    raise ConfigError(
                        f"training.sceneplan_cfg_dropout.{key} must equal 0.15"
                    )
            speech_weight = training.get(
                "sceneplan_speech_active_loss_weight", 1.0
            )
            if (
                isinstance(speech_weight, bool)
                or not isinstance(speech_weight, (int, float))
                or not math.isfinite(float(speech_weight))
                or float(speech_weight) < 1.0
            ):
                raise ConfigError(
                    "sceneplan_speech_active_loss_weight must be finite and >= 1"
                )

            alignment = diffusion_config.get(
                "sceneplan_frame_text_alignment"
            ) or {}
            alignment = _require_mapping(
                alignment,
                "model.diffusion.config.sceneplan_frame_text_alignment",
            )
            alignment_enabled = alignment.get("enabled", False)
            if not isinstance(alignment_enabled, bool):
                raise ConfigError(
                    "ScenePlan frame/text alignment enabled must be boolean"
                )

            soft_block = diffusion_config.get(
                "sceneplan_soft_block_attention"
            ) or {}
            soft_block = _require_mapping(
                soft_block,
                "model.diffusion.config.sceneplan_soft_block_attention",
            )
            soft_block_enabled = soft_block.get("enabled", False)
            if not isinstance(soft_block_enabled, bool):
                raise ConfigError(
                    "ScenePlan soft-block attention enabled must be boolean"
                )
            allowed_soft_block = {
                "enabled",
                "max_sources",
                "max_bias",
                "layer_start",
                "layer_count",
            }
            unknown_soft_block = set(soft_block) - allowed_soft_block
            if unknown_soft_block:
                raise ConfigError(
                    "unknown ScenePlan soft-block attention settings: "
                    f"{sorted(unknown_soft_block)}"
                )
            chunk_moe = diffusion_config.get("sceneplan_chunk_moe") or {}
            chunk_moe = _require_mapping(
                chunk_moe,
                "model.diffusion.config.sceneplan_chunk_moe",
            )
            chunk_moe_enabled = chunk_moe.get("enabled", False)
            if not isinstance(chunk_moe_enabled, bool):
                raise ConfigError("ScenePlan chunk-MoE enabled must be boolean")
            if alignment_enabled and (soft_block_enabled or chunk_moe_enabled):
                raise ConfigError(
                    "legacy monotonic alignment and the no-align ScenePlan "
                    "attention/MoE upgrade are mutually exclusive"
                )
            allowed_chunk_moe = {
                "enabled",
                "layer_start",
                "layer_count",
                "num_experts",
                "top_k",
                "chunk_size",
                "expert_inner_dim",
                "router_hidden_dim",
                "router_temperature",
                "max_sources",
                "boundary_aware",
                "sparse_scale",
                "load_balance_loss_weight",
                "router_entropy_loss_weight",
                "router_entropy_target",
                "conflict_logit_l2_loss_weight",
                "conflict_gate_min",
                "conflict_gate_max",
                "dropout",
            }
            unknown_chunk_moe = set(chunk_moe) - allowed_chunk_moe
            if unknown_chunk_moe:
                raise ConfigError(
                    "unknown ScenePlan chunk-MoE settings: "
                    f"{sorted(unknown_chunk_moe)}"
                )

            duration_loss = training.get(
                "sceneplan_speech_duration_loss"
            ) or {}
            duration_loss = _require_mapping(
                duration_loss,
                "training.sceneplan_speech_duration_loss",
            )
            duration_enabled = duration_loss.get("enabled", False)
            if not isinstance(duration_enabled, bool):
                raise ConfigError(
                    "ScenePlan speech duration enabled must be boolean"
                )
            if duration_enabled and not alignment_enabled:
                raise ConfigError(
                    "ScenePlan speech duration loss requires frame/text alignment"
                )
            prompt_config = _require_mapping(
                by_id["prompt"].get("config"),
                "ScenePlan prompt conditioner config",
            )
            if alignment_enabled:
                if prompt_config.get("sceneplan_timing_aux") is not True:
                    raise ConfigError(
                        "ScenePlan frame/text alignment requires prompt "
                        "sceneplan_timing_aux=true"
                    )
                if fusion_config.get("sceneplan_timing_aux") is not True:
                    raise ConfigError(
                        "ScenePlan frame/text alignment requires sceneplan_44 "
                        "sceneplan_timing_aux=true"
                    )
                if int(diffusion_config.get("patch_size", 1)) != 1:
                    raise ConfigError(
                        "ScenePlan frame/text alignment currently requires patch_size=1"
                    )
                if int(alignment.get("max_sources", 4)) != 4:
                    raise ConfigError(
                        "ScenePlan frame/text alignment requires max_sources=4"
                    )

            if soft_block_enabled:
                if int(diffusion_config.get("patch_size", 1)) != 1:
                    raise ConfigError(
                        "ScenePlan soft-block attention requires patch_size=1"
                    )
                if int(soft_block.get("max_sources", 4)) != 4:
                    raise ConfigError(
                        "ScenePlan soft-block attention requires max_sources=4"
                    )
                layer_start = soft_block.get("layer_start", 0)
                layer_count = soft_block.get("layer_count", depth)
                if (
                    isinstance(layer_start, bool)
                    or not isinstance(layer_start, int)
                    or isinstance(layer_count, bool)
                    or not isinstance(layer_count, int)
                    or layer_start < 0
                    or layer_count <= 0
                    or layer_start + layer_count > depth
                ):
                    raise ConfigError(
                        "ScenePlan soft-block layer range must fit DiT depth"
                    )
                max_bias = soft_block.get("max_bias", 4.0)
                if (
                    isinstance(max_bias, bool)
                    or not isinstance(max_bias, (int, float))
                    or not math.isfinite(float(max_bias))
                    or float(max_bias) <= 0.0
                ):
                    raise ConfigError(
                        "ScenePlan soft-block max_bias must be finite and positive"
                    )

            if chunk_moe_enabled:
                expected_moe_values = {
                    "layer_start": depth - 4,
                    "layer_count": 4,
                    "num_experts": 4,
                    "top_k": 2,
                    "chunk_size": 4,
                    "expert_inner_dim": embed_dim,
                    "max_sources": 4,
                }
                for key, expected in expected_moe_values.items():
                    value = chunk_moe.get(key, expected)
                    if isinstance(value, bool) or not isinstance(value, int) or value != expected:
                        raise ConfigError(
                            "P10 ScenePlan chunk-MoE requires "
                            f"{key}={expected}, got {value!r}"
                        )
                router_hidden = chunk_moe.get("router_hidden_dim", embed_dim)
                if (
                    isinstance(router_hidden, bool)
                    or not isinstance(router_hidden, int)
                    or router_hidden <= 0
                ):
                    raise ConfigError(
                        "ScenePlan chunk-MoE router_hidden_dim must be positive"
                    )
                boundary_aware = chunk_moe.get("boundary_aware", True)
                if boundary_aware is not True:
                    raise ConfigError(
                        "P10 ScenePlan chunk-MoE must restart chunks at activity boundaries"
                    )
                for key, default, allow_zero in (
                    ("sparse_scale", 0.5, False),
                    ("load_balance_loss_weight", 0.01, True),
                    ("router_entropy_loss_weight", 0.0, True),
                    ("conflict_logit_l2_loss_weight", 0.0, True),
                    ("router_temperature", 1.0, False),
                    ("dropout", 0.0, True),
                ):
                    value = chunk_moe.get(key, default)
                    if (
                        isinstance(value, bool)
                        or not isinstance(value, (int, float))
                        or not math.isfinite(float(value))
                        or (float(value) < 0.0 if allow_zero else float(value) <= 0.0)
                        or (key == "dropout" and float(value) >= 1.0)
                    ):
                        raise ConfigError(
                            f"ScenePlan chunk-MoE {key} has an invalid value"
                        )
                router_entropy_target = chunk_moe.get("router_entropy_target")
                if router_entropy_target is not None and (
                    isinstance(router_entropy_target, bool)
                    or not isinstance(router_entropy_target, (int, float))
                    or not math.isfinite(float(router_entropy_target))
                    or not (
                        0.0
                        < float(router_entropy_target)
                        < math.log(float(expected_moe_values["num_experts"]))
                    )
                ):
                    raise ConfigError(
                        "ScenePlan chunk-MoE router_entropy_target must lie "
                        "strictly between zero and log(num_experts)"
                    )
                conflict_gate_min = chunk_moe.get("conflict_gate_min", 0.0)
                conflict_gate_max = chunk_moe.get("conflict_gate_max", 1.0)
                if not (
                    isinstance(conflict_gate_min, (int, float))
                    and not isinstance(conflict_gate_min, bool)
                    and isinstance(conflict_gate_max, (int, float))
                    and not isinstance(conflict_gate_max, bool)
                    and math.isfinite(float(conflict_gate_min))
                    and math.isfinite(float(conflict_gate_max))
                    and 0.0
                    <= float(conflict_gate_min)
                    < float(conflict_gate_max)
                    <= 1.0
                ):
                    raise ConfigError(
                        "ScenePlan chunk-MoE conflict gate bounds must satisfy "
                        "0 <= min < max <= 1"
                    )
                if int(diffusion_config.get("patch_size", 1)) != 1:
                    raise ConfigError("ScenePlan chunk-MoE requires patch_size=1")
                if diffusion_config.get("timestep_cond_type", "global") != "global":
                    raise ConfigError(
                        "ScenePlan chunk-MoE requires the native global timestep embedding"
                    )

            if soft_block_enabled or chunk_moe_enabled:
                if prompt_config.get("sceneplan_timing_aux") is not True:
                    raise ConfigError(
                        "ScenePlan attention/MoE requires prompt source-role auxiliary"
                    )
                if fusion_config.get("sceneplan_timing_aux") is not True:
                    raise ConfigError(
                        "ScenePlan attention/MoE requires frame source-role auxiliary"
                    )
            if duration_enabled:
                if set(duration_loss) != {"enabled", "weight"}:
                    raise ConfigError(
                        "training.sceneplan_speech_duration_loss must contain "
                        "exactly enabled and weight"
                    )
                duration_weight = duration_loss.get("weight")
                if (
                    isinstance(duration_weight, bool)
                    or not isinstance(duration_weight, (int, float))
                    or not math.isfinite(float(duration_weight))
                    or float(duration_weight) <= 0.0
                ):
                    raise ConfigError(
                        "ScenePlan speech duration weight must be finite and positive"
                    )
                timing_fields = (
                    "require_speech_timing",
                    "speech_timing_index_path",
                    "speech_timing_index_sha256",
                    "expected_speech_timing_rows",
                )
                has_complete_timing_contract = (
                    dataset_config.get("require_speech_timing") is True
                    and isinstance(
                        dataset_config.get("speech_timing_index_path"), str
                    )
                    and bool(dataset_config.get("speech_timing_index_path"))
                    and isinstance(
                        dataset_config.get("speech_timing_index_sha256"), str
                    )
                    and len(dataset_config["speech_timing_index_sha256"]) == 64
                    and isinstance(
                        dataset_config.get("expected_speech_timing_rows"), int
                    )
                    and dataset_config["expected_speech_timing_rows"] > 0
                )
                has_any_timing_field = any(
                    field in dataset_config for field in timing_fields
                )
                teacher_free_validation = (
                    allow_missing_speech_timing and not has_any_timing_field
                )
                if not (
                    has_complete_timing_contract or teacher_free_validation
                ):
                    raise ConfigError(
                        "enabled ScenePlan speech duration loss requires the "
                        "immutable speech timing sidecar path, SHA256, row count, "
                        "and require_speech_timing=true; only an explicitly "
                        "teacher-free validation call may omit all timing fields"
                    )

            sound_temporal = training.get(
                "sceneplan_sound_temporal_difference_loss"
            ) or {}
            sound_temporal = _require_mapping(
                sound_temporal,
                "training.sceneplan_sound_temporal_difference_loss",
            )
            sound_temporal_enabled = sound_temporal.get("enabled", False)
            if not isinstance(sound_temporal_enabled, bool):
                raise ConfigError(
                    "ScenePlan sound temporal-difference enabled must be boolean"
                )
            if sound_temporal_enabled:
                expected_temporal_keys = {
                    "enabled",
                    "scope",
                    "weight",
                    "lags",
                    "smooth_l1_beta",
                }
                if set(sound_temporal) != expected_temporal_keys:
                    raise ConfigError(
                        "training.sceneplan_sound_temporal_difference_loss must "
                        f"contain exactly {sorted(expected_temporal_keys)}"
                    )
                if sound_temporal.get("scope") != "sound_only":
                    raise ConfigError(
                        "ScenePlan sound temporal-difference scope must be sound_only"
                    )
                for key in ("weight", "smooth_l1_beta"):
                    value = sound_temporal.get(key)
                    if (
                        isinstance(value, bool)
                        or not isinstance(value, (int, float))
                        or not math.isfinite(float(value))
                        or float(value) <= 0.0
                    ):
                        raise ConfigError(
                            "ScenePlan sound temporal-difference "
                            f"{key} must be finite and positive"
                        )
                lags = sound_temporal.get("lags")
                if (
                    not isinstance(lags, list)
                    or not lags
                    or len(set(lags)) != len(lags)
                    or any(
                        isinstance(value, bool)
                        or not isinstance(value, int)
                        or value <= 0
                        for value in lags
                    )
                ):
                    raise ConfigError(
                        "ScenePlan sound temporal-difference lags must be unique "
                        "positive integers"
                    )
                rejected_transient = training.get(
                    "sceneplan_sound_transient_loss"
                ) or {}
                if bool(rejected_transient.get("enabled", False)):
                    raise ConfigError(
                        "rejected transient MSE reweighting cannot be combined "
                        "with the clean-latent temporal-difference auxiliary"
                    )
        else:
            if not {"prompt", "spatial_format"}.issubset(
                set(diffusion.get("cross_attention_cond_ids", []))
            ):
                raise ConfigError(
                    "T2A DiT cross_attention_cond_ids must include prompt and spatial_format"
                )
            if not {"seconds_start", "seconds_total"}.issubset(
                set(diffusion.get("global_cond_ids", []))
            ):
                raise ConfigError(
                    "T2A DiT global_cond_ids must include seconds_start and seconds_total"
                )
        expected_global_dim = (
            len(diffusion.get("global_cond_ids", [])) * cond_dim
        )
        if int(diffusion_config.get("global_cond_dim", -1)) != expected_global_dim:
            raise ConfigError(
                "DiT global_cond_dim must equal the concatenated global "
                f"conditioner width: {diffusion_config.get('global_cond_dim')} "
                f"!= {expected_global_dim}"
            )
        text_mode = str(by_id["prompt"].get("type"))
        prompt_config = _require_mapping(
            by_id["prompt"].get("config"),
            "model.conditioning prompt config",
        )
        if text_mode in {"qwen", "qwen_text"}:
            model_path = prompt_config.get("model_path")
            if not isinstance(model_path, str) or not model_path:
                raise ConfigError("Qwen prompt conditioner requires model_path")
            _require_positive_int(
                prompt_config.get("hidden_dim"),
                "Qwen prompt hidden_dim",
            )
            _require_positive_int(
                prompt_config.get("max_length"),
                "Qwen prompt max_length",
            )
            if sceneplan_v2:
                if prompt_config.get("sceneplan_role_embedding") is not True:
                    raise ConfigError(
                        "ScenePlan Qwen requires separate event/speech role embeddings"
                    )
                if int(prompt_config.get("sceneplan_max_sources", -1)) != 4:
                    raise ConfigError(
                        "ScenePlan Qwen role maps require sceneplan_max_sources=4"
                    )
        for conditioner_id, conditioner in by_id.items():
            conditioner_config = _require_mapping(
                conditioner.get("config"),
                f"conditioner {conditioner_id!r} config",
            )
            explicit_output_dim = conditioner_config.get("output_dim")
            expected_output_dim = (
                int(fusion_dim)
                if sceneplan_v2 and conditioner_id == "sceneplan_44"
                else cond_dim
            )
            if explicit_output_dim is not None and int(explicit_output_dim) != expected_output_dim:
                raise ConfigError(
                    f"conditioner {conditioner_id!r} output_dim must equal "
                    f"its configured route width: {explicit_output_dim} != {expected_output_dim}"
                )

    elif model_type in {
        "sceneplan_p11",
        "sceneplan_p11_v4",
        "sceneplan_p11_audio_aware_v1",
    }:
        is_p11_v4 = model_type == "sceneplan_p11_v4"
        is_audio_aware = model_type == "sceneplan_p11_audio_aware_v1"
        architecture = (
            "qwen_audio_aware_sketch_first_transfusion_cot"
            if is_audio_aware
            else "qwen_sketch_first_transfusion_cot"
            if is_p11_v4
            else "qwen_sceneplan_planner"
        )
        text_mode = "qwen_sceneplan_planner"
        expected_route = (
            "sceneplan_p11_audio_aware_v1" if is_audio_aware else "sceneplan_p11"
        )
        if route_id != expected_route:
            raise ConfigError(f"{model_type} requires route_id={expected_route}")
        expected_output = (
            "observed_plan_atomic_patch_revised_plan_v1"
            if is_audio_aware
            else "sceneplan_or_atomic_patch_v1"
        )
        if model.get("output_protocol") != expected_output:
            raise ConfigError(
                f"{model_type} requires output_protocol={expected_output}"
            )
        expected_editing_input = (
            "input_foa_required_old_sceneplan_optional_v1"
            if is_audio_aware
            else "sceneplan_tokens_only_v1"
        )
        if model.get("editing_input_contract") != expected_editing_input:
            raise ConfigError(
                f"{model_type} requires editing_input_contract={expected_editing_input}"
            )

        text = _require_mapping(model.get("text"), "model.text")
        if text.get("mode") != "qwen_sceneplan_planner":
            raise ConfigError(
                "P11 requires model.text.mode=qwen_sceneplan_planner"
            )
        if not isinstance(text.get("model_path"), str) or not text.get("model_path"):
            raise ConfigError("P11 requires a pinned model.text.model_path")
        _require_positive_int(text.get("hidden_size"), "P11 Qwen hidden_size")
        if int(text.get("max_length", -1)) != 512:
            raise ConfigError("P11 Qwen max_length must be 512")
        if int(text.get("plan_max_tokens", -1)) != 1024:
            raise ConfigError("P11 ScenePlan plan_max_tokens must be 1024")
        expected_patch_ceiling = 512 if is_audio_aware else 6
        if int(text.get("patch_max_tokens", -1)) != expected_patch_ceiling:
            raise ConfigError(
                f"{model_type} atomic patch ceiling must be {expected_patch_ceiling}"
            )
        if int(text.get("sequence_length", -1)) != 1024:
            raise ConfigError("P11 fixed sequence length must remain 1024 tokens")
        if text.get("dense_right_padding") is not True:
            raise ConfigError("P11 requires dense right-padding execution")
        if is_audio_aware and text.get("activation_checkpointing") is not True:
            raise ConfigError(
                "audio-aware P11 requires activation checkpointing at batch 8"
            )
        if text.get("discrete_decode_mode", "prefix_recompute") != "prefix_recompute":
            raise ConfigError(
                "canonical P11 requires prefix_recompute discrete decoding; "
                "cached decoding is diagnostic only"
            )
        model_codec_path = text.get("plan_codec_path")
        dataset_codec_path = None if dataset_config is None else dataset_config.get("codec_path")
        if dataset_config is not None and (
            not isinstance(model_codec_path, str)
            or not model_codec_path
            or not isinstance(dataset_codec_path, str)
            or not dataset_codec_path
            or Path(model_codec_path).expanduser().resolve()
            != Path(dataset_codec_path).expanduser().resolve()
        ):
            raise ConfigError(
                "P11 model and dataset must reference the identical ScenePlan codec artifact"
            )
        lora = _require_mapping(text.get("lora"), "model.text.lora")
        _require_positive_int(lora.get("rank"), "P11 LoRA rank")
        _require_positive_int(lora.get("top_layers"), "P11 LoRA top_layers")

        bridge = _require_mapping(model.get("audio_bridge"), "model.audio_bridge")
        if bridge.get("type") != "hybrid_temporal_semantic":
            raise ConfigError("P11 requires the hybrid temporal/semantic audio bridge")
        temporal = _require_mapping(
            bridge.get("temporal"), "model.audio_bridge.temporal"
        )
        if int(temporal.get("channels", -1)) != latent_dim:
            raise ConfigError(
                "P11 temporal bridge channels must match the VAE latent width"
            )
        semantic = _require_mapping(
            bridge.get("semantic"), "model.audio_bridge.semantic"
        )
        if (
            semantic.get("type") != "frozen_feature_resampler"
            or int(semantic.get("input_dim", -1)) != 512
            or semantic.get("required") is not True
        ):
            raise ConfigError("P11 requires the pinned 512-D frozen semantic bridge")
        if dataset_config is not None and semantic.get(
            "encoder_revision"
        ) != dataset_config.get("semantic_encoder_revision"):
            raise ConfigError("P11 model and dataset semantic encoder revisions differ")
        if dataset_config is not None and (
            dataset_config.get("semantic_representation")
            != "windowed_clap_pooler_output"
            or float(dataset_config.get("semantic_window_sec", -1.0)) != 5.0
            or float(dataset_config.get("semantic_hop_sec", -1.0)) != 5.0
        ):
            raise ConfigError(
                "P11 canonical 15-second evidence requires windowed CLAP "
                "with 5-second windows and 5-second hops"
            )

        executor = _require_mapping(model.get("executor"), "model.executor")
        expected_executor = {
            "type": "external_p10_sceneplan_dit",
            "handoff_contract": "external_p10_sceneplan_executor_v1",
            "editing_contract": (
                "audio_aware_observed_patch_revised_v1"
                if is_audio_aware
                else "single_turn_sceneplan_patch_v1"
            ),
            "localized_editing": False,
            "preserves_unedited_waveform": False,
            "capability_contract": "p10_sceneplan_44_capability_v1",
            "semantic_caption_contract": "p10_semantic_caption_v2_only_20260830",
            "semantic_caption_compiler_version": 2,
            "semantic_caption_surface": (
                "<speaker description> who says: <exact transcript>"
            ),
            "transcript_state_authority": "sceneplan.source.transcript",
            "canonical_executor_family": "sceneplan_dit_v11_semantic_v2_15s_300m",
            "canonical_model_config": (
                os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/stable-audio-tools-workspace/stable_audio_tools/configs/"
                "model_configs/txt2audio/t2a/dit/"
                "qwen35_0p8b_300m_model_sceneplan_44_soundexp_noalign_15s_"
                "resume_cosine_40k.json"
            ),
            "canonical_model_config_sha256": (
                "3ebcd2b6b3c9a8b78b9160243f6509fb9eb11a32959b2eeddb86f48bb0b44827"
            ),
            "canonical_checkpoint_step": 150000,
            "canonical_checkpoint": (
                os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/dit/"
                "sceneplan_dit_v11_semantic_v2_protected_resume_150k/"
                "checkpoints/epoch=48-step=150000.ckpt"
            ),
            "canonical_checkpoint_sha256": (
                "be8c90cd1434bd71f73951531175c3674ff0f3173d5db591e2e1c476152ff59e"
            ),
            "p10_max_latent_frames": 648,
            "planner_max_latent_frames": 648,
            "planner_motion_types": ["static", "linear"],
            "gain_policy": "constant_0db_not_conditioned",
            "word_level_timing_supported": False,
        }
        for key, expected in expected_executor.items():
            if executor.get(key) != expected:
                raise ConfigError(
                    f"P11 executor.{key}={executor.get(key)!r}, expected {expected!r}"
                )
        if dataset_config is not None:
            dataset_surface = {
                "p10_semantic_caption_contract": expected_executor[
                    "semantic_caption_contract"
                ],
                "p10_semantic_caption_compiler_version": expected_executor[
                    "semantic_caption_compiler_version"
                ],
                "transcript_state_authority": expected_executor[
                    "transcript_state_authority"
                ],
            }
            for key, expected in dataset_surface.items():
                if dataset_config.get(key) != expected:
                    raise ConfigError(
                        f"P11 dataset {key}={dataset_config.get(key)!r}, "
                        f"expected {expected!r}"
                    )
        optimizer_configs = _require_mapping(
            training.get("optimizer_configs"), "training.optimizer_configs"
        )
        if set(optimizer_configs) != {"p11"}:
            raise ConfigError("P11 requires exactly optimizer_configs.p11")
        if training.get("use_ema") is not True:
            raise ConfigError("P11 canonical training requires EMA")
        thought = model.get("scene_thought") or {}
        if is_p11_v4:
            if thought:
                raise ConfigError(
                    "P11-v4 must not configure the retired core40 scene_thought block"
                )
            if training.get("scene_thought_loss_weights") is not None:
                raise ConfigError(
                    "P11-v4 must not configure retired SceneThought losses"
                )
            transfusion = _require_mapping(
                model.get("transfusion_cot"), "model.transfusion_cot"
            )
            expected_transfusion = {
                "contract": "p11_sketch_first_transfusion_cot_v4",
                "sequence_contract": (
                    "task_evidence_then_discrete_sketch_then_continuous_thought_"
                    "then_assembler_v1"
                ),
                "scene_sketch_contract": "p10_bound_discrete_cot_v1",
                "thought_contract": "p10_numeric_scene_delta_thought_v1",
                "delta_token_contract": "atomic_delta_owner_direction_program_v2",
                "output_contract": "deterministic_sceneplan_or_atomic_patch_v1",
                "assembler": "deterministic_sketch_execution_assembler_v1",
                "semantic_from_execution_state": False,
                "numeric_from_scene_sketch": False,
                "editing_render_seed_policy": "same_each_turn",
            }
            for key, value in expected_transfusion.items():
                if transfusion.get(key) != value:
                    raise ConfigError(
                        f"P11-v4 transfusion_cot.{key}={transfusion.get(key)!r}, "
                        f"expected {value!r}"
                    )
            sct = _require_mapping(
                transfusion.get("sct"), "model.transfusion_cot.sct"
            )
            expected_sct = {
                "enabled": True,
                "contract": "audiochat_style_understanding_then_execution_sct_v1",
                "total_layers": 24,
                "understanding_layers": 16,
                "generation_layers": 8,
                "understanding_lora_top_layers": 8,
                "discrete_head": "understanding_layer_16_v1",
                "continuous_head": "generation_layer_24_v1",
                "generation_gradient_to_understanding_lora": False,
                "text_inference": "understanding_layers_only_v1",
            }
            if set(sct) != set(expected_sct):
                raise ConfigError("P11-v4 SCT keys changed")
            for key, value in expected_sct.items():
                if sct.get(key) != value:
                    raise ConfigError(
                        f"P11-v4 sct.{key}={sct.get(key)!r}, expected {value!r}"
                    )
            lora = _require_mapping(text.get("lora"), "model.text.lora")
            if int(lora.get("top_layers", -1)) != int(
                sct["generation_layers"]
            ):
                raise ConfigError(
                    "P11-v4 text.lora.top_layers must equal SCT generation_layers"
                )
            if int(transfusion.get("scene_sketch_max_tokens", -1)) != 512:
                raise ConfigError("P11-v4 SceneSketch ceiling must be 512")
            if int(transfusion.get("delta_sketch_max_tokens", -1)) != 5:
                raise ConfigError("P11-v4 DeltaSketch ceiling must be five tokens")
            discrete_supervision = _require_mapping(
                transfusion.get("discrete_supervision"),
                "model.transfusion_cot.discrete_supervision",
            )
            if (
                discrete_supervision.get("contract")
                != "scene_sketch_finite_field_boundary_supervision_v1"
                or float(discrete_supervision.get("text_end_weight", 0.0)) < 1.0
                or float(discrete_supervision.get("scene_eos_weight", 0.0)) < 1.0
            ):
                raise ConfigError(
                    "P11-v4 discrete boundary supervision contract changed"
                )
            inventory = _require_mapping(
                discrete_supervision.get("understanding_inventory"),
                (
                    "model.transfusion_cot.discrete_supervision."
                    "understanding_inventory"
                ),
            )
            expected_inventory = {
                "contract": "u_same_decoder_finite_inventory_aux_v1",
                "objective": "grammar_candidate_ce_v1",
                "authority": "scene_sketch_autoregressive_logits_v1",
                "normalization": "weighted_mean_of_field_ce_v1",
                "fields": ["source_count", "room", "kind"],
                "loss_weight": 1.0,
                "source_count_weight": 4.0,
                "room_weight": 1.0,
                "kind_weight": 1.0,
            }
            if inventory != expected_inventory:
                raise ConfigError(
                    "P11-v4 requires the exact same-decoder U inventory objective"
                )
            control_direction = _require_mapping(
                transfusion.get("control_direction"),
                "model.transfusion_cot.control_direction",
            )
            expected_control_direction = {
                "contract": "p10_atomic_edit_control_direction_v1",
                "authority": "delta_sketch_operation_specific_token_v1",
                "operations": ["rotate_source", "distance_source"],
                "decode": "exact_negative_positive_lookup_v1",
                "continuous_parallel_head": False,
                "conditions_delta_thought": True,
            }
            if control_direction != expected_control_direction:
                raise ConfigError(
                    "P11-v4 control_direction must be the exact P10 token-forcing contract"
                )
            lexical = _require_mapping(
                transfusion.get("lexical_evidence"),
                "model.transfusion_cot.lexical_evidence",
            )
            if int(lexical.get("max_tokens", -1)) != 128:
                raise ConfigError("P11-v4 lexical evidence ceiling must be 128")
            if not isinstance(lexical.get("required"), bool):
                raise ConfigError("P11-v4 lexical_evidence.required must be boolean")
            if dataset_config is not None:
                lexical_mode = str(
                    dataset_config.get("lexical_evidence_mode", "none")
                )
                if lexical.get("required") is True:
                    raise ConfigError(
                        "P11-v4 ASR cannot be required on every U row; CLAP/FOA "
                        "are universal while lexical evidence is reliable-speech-only"
                    )
                if lexical_mode == "frozen_asr_cache_v1":
                    if (
                        lexical.get("canonical_source")
                        != "frozen_asr_input_foa_confidence_gated_v1"
                        or lexical.get("injection_policy")
                        not in {
                            # Retained only as the measured negative ablation.
                            "optional_reliable_speech_only_v1",
                            # Canonical reliable-ASR authority boundary.
                            "deterministic_reliable_speech_assembler_v1",
                        }
                    ):
                        raise ConfigError(
                            "frozen ASR data requires a registered reliable-"
                            "speech-only lexical policy"
                        )
                elif lexical.get("injection_policy") is not None:
                    raise ConfigError(
                        "P11-v4 lexical injection policy is legal only with "
                        "frozen_asr_cache_v1"
                    )
            v4_thought = _require_mapping(
                transfusion.get("thought"), "model.transfusion_cot.thought"
            )
            expected_thought = {
                "contract": "p10_numeric_scene_delta_thought_v1",
                "backbone": "shared_qwen_causal_v1",
            }
            for key, value in expected_thought.items():
                if v4_thought.get(key) != value:
                    raise ConfigError(
                        f"P11-v4 thought.{key}={v4_thought.get(key)!r}, "
                        f"expected {value!r}"
                    )
            supported_arms = {
                "sketch_first_transfusion_cot_v4": "rectified_flow",
                "sketch_first_direct_mse_v4": "direct_mse",
            }
            thought_arm = str(v4_thought.get("arm") or "")
            if thought_arm not in supported_arms:
                raise ConfigError(
                    f"unsupported P11-v4 matched arm {thought_arm!r}"
                )
            expected_objective = supported_arms[thought_arm]
            if v4_thought.get("continuous_objective") != expected_objective:
                raise ConfigError(
                    f"P11-v4 arm {thought_arm!r} requires "
                    f"continuous_objective={expected_objective!r}"
                )
            editing_objective = str(
                v4_thought.get("editing_continuous_objective", expected_objective)
            )
            if thought_arm == "sketch_first_transfusion_cot_v4":
                if editing_objective not in {"rectified_flow", "direct_mse"}:
                    raise ConfigError(
                        "P11-v4 Flow G/U requires Editing to use "
                        "rectified_flow or direct_mse"
                    )
            elif editing_objective != "direct_mse":
                raise ConfigError(
                    "P11-v4 Direct-MSE must also use direct Editing"
                )
            editing_output_head = str(
                v4_thought.get("editing_output_head", "shared_velocity_v1")
            )
            if editing_output_head not in {
                "shared_velocity_v1",
                "dedicated_delta_v1",
            }:
                raise ConfigError("P11-v4 Editing output head is unsupported")
            if (
                editing_output_head == "dedicated_delta_v1"
                and editing_objective != "direct_mse"
            ):
                raise ConfigError(
                    "P11-v4 dedicated delta head requires direct-mse Editing"
                )
            if int(v4_thought.get("slot_count", -1)) != 5 or int(
                v4_thought.get("core_dim", -1)
            ) != 15:
                raise ConfigError("P11-v4 requires an exact [5,15] numeric core")
            if int(v4_thought.get("dim", -1)) != int(text.get("hidden_size", -1)):
                raise ConfigError("P11-v4 thought dim must equal Qwen hidden_size")
            inference_steps = _require_positive_int(
                v4_thought.get("inference_steps"), "P11-v4 inference steps"
            )
            training_steps = _require_positive_int(
                v4_thought.get("training_inference_steps"),
                "P11-v4 training inference steps",
            )
            if training_steps != inference_steps:
                raise ConfigError(
                    "P11-v4 must use identical train/inference solver steps"
                )
            if "counterfactual_forcing" in v4_thought:
                raise ConfigError(
                    "retired endpoint-pair objectives are forbidden in P11-v4"
                )
            if editing_objective != "direct_mse" or (
                editing_output_head != "dedicated_delta_v1"
            ):
                raise ConfigError(
                    "P11-v4 Editing requires direct-mse and "
                    "the dedicated delta head"
                )
            weights = _require_mapping(
                training.get("transfusion_cot_loss_weights"),
                "training.transfusion_cot_loss_weights",
            )
            required_weights = {"flow", "solve", "locality", "owner"}
            if set(weights) != required_weights:
                raise ConfigError("P11-v4 Transfusion-CoT loss keys changed")
            flow_weight = float(weights["flow"])
            if thought_arm == "sketch_first_transfusion_cot_v4" and flow_weight <= 0.0:
                raise ConfigError("the canonical P11-v4 Flow arm requires flow weight > 0")
            if thought_arm == "sketch_first_direct_mse_v4" and flow_weight != 0.0:
                raise ConfigError("the P11-v4 Direct-MSE baseline requires flow weight = 0")
        elif is_audio_aware:
            if thought:
                raise ConfigError(
                    "audio-aware P11 must not configure the retired core40 block"
                )
            transfusion = _require_mapping(
                model.get("transfusion_cot"), "model.transfusion_cot"
            )
            expected = {
                "contract": "sceneplan_p11_audio_aware_v1",
                "sequence_contract": (
                    "audio_observation_then_atomic_delta_then_deterministic_revised_v1"
                ),
                "output_contract": "observed_plan_atomic_patch_revised_plan_v1",
                "assembler": "observed_plus_atomic_patch_only_v1",
                "editing_audio_required": True,
                "old_sceneplan_role": "optional_fallible_prior",
                "target_audio_supervision": False,
                "delta_token_contract": "audio_aware_atomic_delta_v1",
                "scene_sketch_max_tokens": 512,
                "delta_sketch_max_tokens": 512,
            }
            for key, value in expected.items():
                if transfusion.get(key) != value:
                    raise ConfigError(
                        f"audio-aware transfusion_cot.{key}="
                        f"{transfusion.get(key)!r}, expected {value!r}"
                    )
            required_operations = [
                "no_op",
                "add_source",
                "remove_source",
                "replace_source",
                "move_source",
                "retime_source",
                "room_change",
                "change_speech_description",
                "change_transcript",
            ]
            if transfusion.get("active_edit_operations") != required_operations:
                raise ConfigError("audio-aware atomic edit vocabulary changed")
            observation = _require_mapping(
                transfusion.get("observation"), "model.transfusion_cot.observation"
            )
            if observation != {
                "shared_with_understanding": True,
                "discrete_authority": "scene_sketch",
                "continuous_authority": "flow_r1_execution_state",
            }:
                raise ConfigError("audio-aware observation authority changed")
            editing = _require_mapping(
                transfusion.get("editing"), "model.transfusion_cot.editing"
            )
            if editing != {
                "discrete_authority": "atomic_delta_sketch",
                "continuous_authority": "deterministic_delta_execution_state",
                "revised_plan_decoder": False,
                "revised_plan_authority": "deterministic_apply_patch_to_observed",
            }:
                raise ConfigError("audio-aware revised-plan authority changed")
            if transfusion.get("control_direction") is not None:
                raise ConfigError(
                    "audio-aware absolute move edits must retire v4 direction forcing"
                )
            if transfusion.get("delta_owner") != {
                "contract": "delta_sketch_decoder_owned_source_slot_v1",
                "objective": "legal_source_ce_plus_hardest_negative_margin_v1",
                "authority": "delta_sketch_owner_token_v1",
                "context": "edit_instruction_plus_audio_observed_sceneplan_v1",
                "inference": "grammar_constrained_argmax_v1",
                "legal_source_inventory": "observed_or_revised_source_mask_v1",
                "loss_normalization": "active_owner_rows_v1",
                "margin": 2.0,
                "margin_weight": 1.0,
            }:
                raise ConfigError("audio-aware DeltaSketch owner contract changed")
            active_thought = _require_mapping(
                transfusion.get("thought"), "model.transfusion_cot.thought"
            )
            if active_thought != {
                "arm": "audio_aware_flow_r1_v1",
                "contract": "p10_numeric_scene_delta_thought_v1",
                "backbone": "shared_qwen_causal_v1",
                "continuous_objective": "rectified_flow",
                "editing_continuous_objective": "direct_mse",
                "editing_output_head": "dedicated_delta_v1",
                "slot_count": 5,
                "core_dim": 15,
                "dim": 1024,
                "inference_steps": 1,
                "training_inference_steps": 1,
                "inference_noise_seed": 42,
            }:
                raise ConfigError("audio-aware Flow-R1/Direct-Delta contract changed")
            lexical = _require_mapping(
                transfusion.get("lexical_evidence"),
                "model.transfusion_cot.lexical_evidence",
            )
            if int(lexical.get("max_tokens", -1)) != 128 or lexical.get("required") is not False:
                raise ConfigError(
                    "audio-aware ASR must remain optional reliable evidence"
                )
            if dataset_config is not None and str(
                dataset_config.get("lexical_evidence_mode", "none")
            ) == "none" and lexical.get("injection_policy") is not None:
                raise ConfigError(
                    "audio-aware lexical injection requires a frozen ASR cache"
                )
            distribution = _require_mapping(
                training.get("editing_evidence_distribution"),
                "training.editing_evidence_distribution",
            )
            if distribution != {
                "no_plan": 0.4,
                "correct_plan": 0.3,
                "corrupt_plan": 0.3,
            }:
                raise ConfigError("audio-aware old-plan evidence distribution changed")
            if int(training.get("seed", -1)) != 42:
                raise ConfigError("audio-aware P11 training seed must be 42")
            weights = _require_mapping(
                training.get("transfusion_cot_loss_weights"),
                "training.transfusion_cot_loss_weights",
            )
            required_weights = {
                "flow",
                "solve",
                "locality",
                "owner",
                "delta_control",
            }
            if set(weights) != required_weights:
                raise ConfigError(
                    "audio-aware P11 Transfusion-CoT loss keys changed"
                )
            if any(float(weights[name]) < 0.0 for name in required_weights):
                raise ConfigError("audio-aware P11 loss weights must be non-negative")
            if float(weights["flow"]) <= 0.0:
                raise ConfigError("audio-aware P11 Flow-R1 requires flow weight > 0")
            if float(weights["owner"]) <= 0.0:
                raise ConfigError("audio-aware P11 owner weight must be positive")
            if float(weights["delta_control"]) <= 0.0:
                raise ConfigError(
                    "audio-aware P11 operation-owned control weight must be positive"
                )
        else:
            if thought:
                raise ConfigError(
                    "the legacy core40 SceneThought route is retired; "
                    "sceneplan_p11 is reserved for the discrete D0 baseline"
                )
            if training.get("scene_thought_loss_weights") is not None:
                raise ConfigError(
                    "the discrete D0 baseline must not configure SceneThought losses"
                )

    else:
        raise ConfigError(f"unsupported T2A model_type: {model_type!r}")

    if dataset_config is not None:
        _validate_dataset_compatibility(
            model_config,
            dataset_config,
            ratio=ratio,
            latent_length=latent_length,
            allow_missing_speech_timing=allow_missing_speech_timing,
        )

    return {
        "task": "t2a",
        "architecture": architecture,
        "model_type": model_type,
        "text_mode": text_mode,
        "route_id": route_id,
        "sample_rate": sample_rate,
        "audio_channels": audio_channels,
        "latent_channels": latent_dim,
        "latent_length": latent_length,
        "downsampling_ratio": ratio,
        "dataset_type": dataset_config.get("dataset_type")
        if dataset_config is not None
        else None,
    }


def validate_training_configs(
    model_config: Mapping[str, Any],
    dataset_config: Optional[Mapping[str, Any]] = None,
    *,
    allow_missing_speech_timing: bool = False,
) -> Optional[dict]:
    """Run task-specific checks only for configs that explicitly opt in."""

    if is_t2a_config(model_config):
        return validate_t2a_config(
            model_config,
            dataset_config,
            allow_missing_speech_timing=allow_missing_speech_timing,
        )
    return None
