#!/usr/bin/env python3
"""Fail-closed consistency checks for ScenePlan Renderer v2 revision 4."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SPEC = (
    REPO_ROOT
    / "stable_audio_tools/configs/dataset_configs/construct_dataset/sceneplan_renderer_v2_1p124m.json"
)
DEFAULT_SCHEMA = REPO_ROOT / "docs/sceneplan_v2/sceneplan_renderer_sample_v2.schema.json"
DEFAULT_COMPILER_SCHEMA = REPO_ROOT / "docs/sceneplan_v2/sceneplan_compiler_output_v2.schema.json"
DEFAULT_SPEECH_SCHEMA = REPO_ROOT / "docs/sceneplan_v2/speech_dry_asset_v2.schema.json"
DEFAULT_DELETE = REPO_ROOT / "docs/sceneplan_v2/deletion_manifest_20260814.json"
DEFAULT_README = REPO_ROOT / "docs/sceneplan_v2/README.md"


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_contract_snapshot(
    spec: dict[str, Any], sources: list[Path], check_filesystem: bool
) -> dict[str, Any]:
    contract_root = Path(spec["storage"]["contract_root"])
    require(
        contract_root.is_absolute() and str(contract_root).startswith("/mnt/sdb/"),
        "contract snapshot root must be on SDB",
    )
    files: dict[str, Any] = {}
    for source in sources:
        source = source.expanduser().resolve(strict=True)
        destination = contract_root / source.name
        source_sha256 = sha256_file(source)
        destination_sha256 = None
        synchronized = False
        if destination.is_file():
            destination_sha256 = sha256_file(destination)
            synchronized = destination_sha256 == source_sha256
        if check_filesystem:
            require(destination.is_file(), f"missing frozen contract copy: {destination}")
            require(synchronized, f"stale frozen contract copy: {destination}")
        files[source.name] = {
            "source": str(source),
            "snapshot": str(destination),
            "sha256": source_sha256,
            "snapshot_sha256": destination_sha256,
            "synchronized": synchronized,
        }
    return {
        "root": str(contract_root),
        "filesystem_enforced": check_filesystem,
        "all_synchronized": all(value["synchronized"] for value in files.values()),
        "files": files,
    }


def integer_sum(values: Any, label: str) -> int:
    require(isinstance(values, dict), f"{label} must be an object")
    total = 0
    for key, value in values.items():
        require(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0,
            f"{label}.{key} must be a non-negative integer",
        )
        total += value
    return total


def validate_spec(spec: dict[str, Any]) -> dict[str, Any]:
    require(spec.get("schema_version") == 2, "build spec must be version 2")
    require(spec.get("contract_revision") == 4, "build contract must be revision 4")
    require(spec.get("dataset_id") == "sceneplan_renderer_v2_1p124m", "dataset id drift")

    audio = spec["audio"]
    require(audio["random_crop"] is False, "random_crop must be false")
    model_rate = int(audio["model_sample_rate_hz"])
    max_samples = int(audio["max_model_num_samples"])
    require(max_samples == 442_368, "maximum model length must remain 442,368 samples")
    require(
        math.isclose(float(audio["max_duration_sec"]), max_samples / model_rate, abs_tol=1e-12),
        "maximum duration must equal samples / sample rate",
    )
    hop = int(audio["vae_hop_samples"])
    frames = math.ceil(max_samples / hop)
    require(frames == int(audio["max_latent_frames_valid"]) == 432, "latent frame maximum drift")
    require(frames * hop == int(audio["max_vae_padded_num_samples"]) == max_samples, "VAE padding drift")

    split_totals: dict[str, int] = {}
    speech_totals: dict[str, int] = {}
    no_speech_totals: dict[str, int] = {}
    by_source_count = {str(index): 0 for index in range(1, 5)}
    for split, split_spec in spec["splits"].items():
        speech = split_spec["joint_quotas"]["speech"]
        no_speech = split_spec["joint_quotas"]["no_speech"]
        require(set(speech) == set(by_source_count), f"{split} speech source-count keys drift")
        require(set(no_speech) == set(by_source_count), f"{split} non-speech source-count keys drift")
        speech_total = integer_sum(speech, f"splits.{split}.speech")
        no_speech_total = integer_sum(no_speech, f"splits.{split}.no_speech")
        split_total = speech_total + no_speech_total
        require(split_total == int(split_spec["scenes"]), f"{split} quotas do not sum to scenes")
        split_totals[split] = split_total
        speech_totals[split] = speech_total
        no_speech_totals[split] = no_speech_total
        for count in by_source_count:
            by_source_count[count] += int(speech[count]) + int(no_speech[count])

    require(
        split_totals == {"train": 1_100_000, "validation": 20_000, "test": 4_000},
        "split totals drift",
    )
    require(sum(split_totals.values()) == 1_124_000, "scene total must be 1.124M")
    require(
        speech_totals == {"train": 500_000, "validation": 10_000, "test": 2_000},
        "speech split totals drift",
    )
    require(
        no_speech_totals == {"train": 600_000, "validation": 10_000, "test": 2_000},
        "non-speech split totals drift",
    )
    require(sum(speech_totals.values()) == 512_000, "speech total must be 512k")
    require(sum(no_speech_totals.values()) == 612_000, "non-speech total must be 612k")
    expected_source_counts = {
        str(key): int(value) for key, value in spec["composition"]["source_count_totals"].items()
    }
    require(by_source_count == expected_source_counts, "joint/source-count totals disagree")
    require(
        by_source_count == {"1": 393_400, "2": 393_400, "3": 224_800, "4": 112_400},
        "35/35/20/10 source ratios drift",
    )

    inventory = spec["speech_inventory"]
    allowed = inventory["allowed_sources"]
    require(set(allowed) == {"libritts", "hifi_tts"}, "speech sources must be LibriTTS + HiFiTTS only")
    expected_roots = {
        "libritts": "/mnt/sdc/speech_dataset/mythicinfinity__libritts",
        "hifi_tts": "/mnt/sdc/speech_dataset/MikhailT__hifi-tts",
    }
    selected_by_split = {"train": 0, "validation": 0, "test": 0}
    selected_by_dataset: dict[str, int] = {}
    for dataset, values in allowed.items():
        require(values["root"] == expected_roots[dataset], f"{dataset} source root drift")
        require(values["structurally_eligible_rows_observed"] > 256_000, f"{dataset} inventory is too small")
        require(values["eligible_unique_normalized_transcripts_observed"] > 256_000, f"{dataset} text inventory is too small")
        selected = values["selected"]
        require(selected == {"train": 250_000, "validation": 5_000, "test": 1_000}, f"{dataset} selected split drift")
        selected_by_dataset[dataset] = integer_sum(selected, f"allowed_sources.{dataset}.selected")
        for split in selected_by_split:
            selected_by_split[split] += int(selected[split])
    require(selected_by_dataset == {"libritts": 256_000, "hifi_tts": 256_000}, "per-dataset speech selection drift")
    require(selected_by_split == inventory["selected_totals"] == speech_totals, "speech source/split totals disagree")
    require(inventory["selected_total"] == sum(selected_by_dataset.values()) == 512_000, "selected speech total drift")
    require(inventory["reuse_across_all_scenes"] is False, "speech reuse must be false")
    require(inventory["speaker_disjoint_across_splits"] is True, "speaker split isolation must be enabled")
    require(
        set(inventory["global_uniqueness"])
        == {"dry_asset_id", "source_dataset_plus_source_id", "source_audio_sha256", "normalized_transcript"},
        "global speech uniqueness keys drift",
    )
    excluded = inventory["excluded_sources"]
    require(
        {"spatial_librispeech", "nvidia_hifitts_2", "bigcomputer_kaggle_notebooks_conversations_hq", "legacy_spatial_speech_foa_tts"}
        <= set(excluded),
        "excluded source families drift",
    )

    composition = spec["composition"]
    require(composition["speech_scenes_have_exactly_one_speech_source"] is True, "speech scenes must have one speech source")
    require(composition["no_speech_scenes_have_zero_speech_sources"] is True, "non-speech scenes must have zero speech sources")
    require(
        composition["speech_corpus_fraction_within_every_split_source_count_cell"]
        == {"libritts": 0.5, "hifi_tts": 0.5},
        "speech corpus balance must remain 50/50 in every split/source-count cell",
    )

    loudness = spec["loudness"]
    require(loudness["background_budget_is_aggregate_not_per_source"] is True, "background loudness must be aggregate")
    require(math.isclose(loudness["speech_linear_weight_median"], 0.6), "speech weight drift")
    require(math.isclose(loudness["aggregate_background_linear_weight_median"], 0.4), "background weight drift")
    require(
        math.isclose(loudness["speech_to_aggregate_background_db_median"], 20 * math.log10(0.6 / 0.4), abs_tol=1e-12),
        "speech/background median dB drift",
    )
    require(loudness["speech_to_aggregate_background_db_range"] == [2.0, 6.0], "speech/background range drift")

    renderer = spec["renderer"]
    require(renderer["scene_source_input_modality"] == "canonical_dry_mono_only", "renderer source domain drift")
    require(renderer["reject_multichannel_or_foa_source_assets"] is True, "renderer must reject FOA sources")
    require(renderer["spatialization_passes_per_source"] == 1, "renderer must spatialize exactly once")
    require(renderer["shared_room_and_listener_per_scene"] is True, "scene sources must share one room")
    require(renderer["foa_channel_order"] == "WYZX_ACN", "renderer FOA order drift")
    require(renderer["foa_normalization"] == "SN3D", "renderer FOA normalization drift")
    require(
        math.isclose(float(renderer["n3d_to_sn3d_first_order_gain"]), 1.0 / math.sqrt(3.0), abs_tol=1e-15),
        "Pyroom N3D-to-SN3D gain drift",
    )
    require(
        renderer["pyroom_fractional_delay_filter_samples"] == 81
        and renderer["residual_algorithmic_delay_samples"] == 40,
        "Pyroom timing contract drift",
    )
    require(renderer["forbid_silent_source_truncation"] is True, "silent source truncation must be forbidden")
    require(renderer["forbid_foa_to_mono_to_foa"] is True, "FOA-to-mono-to-FOA must be forbidden")

    conditioning = spec["conditioning"]
    require(conditioning["caption_input"] == "frozen_qwen_cross_attention", "caption cross-attention drift")
    require(conditioning["source_semantic_token_masks"] == 4, "semantic token-mask count drift")
    require(conditioning["source_motion_activity_token_masks"] == 4, "motion/activity token-mask count drift")
    require(conditioning["speaker_info_token_masks"] == 1, "speaker token-mask count drift")
    require(conditioning["quoted_transcript_token_masks"] == 1, "transcript token-mask count drift")
    require(conditioning["event_embeddings"] == 4, "event embedding count drift")
    require(conditioning["position_activity_embeddings"] == 4, "position embedding count drift")
    require(conditioning["structured_stream_fusion"] == "concat_four_sources_then_project", "source stream fusion drift")
    require(conditioning["sceneplan_token_concat"] is False, "ScenePlan tokens must not be concatenated")

    storage = spec["storage"]
    require(storage["persistent_output_mount"] == "/mnt/sdb", "persistent output mount must be SDB")
    for key, value in storage.items():
        if key.endswith("_root") and isinstance(value, str):
            require(value.startswith("/mnt/sdb/"), f"storage.{key} must be on SDB")
    require(storage["forbid_new_persistent_outputs_on"] == ["/mnt/sdc", "/mnt/sdd"], "forbidden output mounts drift")
    require(spec["quality_control"]["stop_after_stage"] == "P9", "execution stop gate must be P9")

    return {
        "scene_totals": split_totals,
        "speech_totals": speech_totals,
        "no_speech_totals": no_speech_totals,
        "source_count_totals": by_source_count,
        "speech_selected_by_dataset": selected_by_dataset,
        "max_model_num_samples": max_samples,
        "max_vae_padded_num_samples": max_samples,
        "max_latent_frames_valid": frames,
        "persistent_output_mount": storage["persistent_output_mount"],
    }


def validate_schema(schema: dict[str, Any]) -> dict[str, Any]:
    require(schema.get("$schema") == "https://json-schema.org/draft/2020-12/schema", "JSON Schema draft drift")
    require(schema.get("$id") == "stable_audio_tools.sceneplan_renderer_sample.v2", "schema id drift")
    caption = schema["$defs"]["renderer_caption"]["properties"]
    require(caption["compiler_version"]["const"] == 4, "caption compiler revision drift")
    audio = schema["$defs"]["scene_plan"]["properties"]["audio"]["properties"]
    expected_duration = 442_368 / 44_100
    require(math.isclose(audio["duration_sec"]["maximum"], expected_duration, abs_tol=1e-12), "record schema duration max drift")
    require(audio["model_num_samples"]["maximum"] == 442_368, "record schema sample max drift")
    require(audio["latent_frames_valid"]["maximum"] == 432, "record schema latent max drift")
    sources = schema["$defs"]["scene_plan"]["properties"]["sources"]
    require(sources["minItems"] == sources["maxItems"] == 4, "record schema must have four slots")
    asset = schema["$defs"]["asset_ref"]["properties"]
    require(asset["input_audio_domain"]["const"] == "dry_mono", "asset schema must require dry mono")
    require(asset["canonical_channels"]["const"] == 1, "asset schema must require one channel")
    require(asset["spatialization_passes_before_scene"]["const"] == 0, "asset prior spatialization drift")
    lineage = schema["$defs"]["lineage"]["properties"]
    require(lineage["dataset_contract_version"]["const"] == 4, "lineage contract revision drift")
    require(lineage["spatialization_passes"]["const"] == 1, "lineage spatialization count drift")
    require(lineage["foa_normalization"]["const"] == "WYZX_ACN_SN3D", "lineage normalization drift")
    require(lineage["residual_algorithmic_delay_samples"]["const"] == 40, "lineage delay drift")
    target = schema["$defs"]["target"]
    require("vae_encode_seed" in target["required"], "materialized VAE seed lineage drift")
    require(
        target["properties"]["vae_encode_seed"]["type"] == ["integer", "null"],
        "VAE encode seed schema drift",
    )
    return {
        "schema_id": schema["$id"],
        "source_slots": 4,
        "max_model_num_samples": 442_368,
        "max_duration_sec": expected_duration,
        "max_latent_frames": 432,
    }


def validate_speech_asset_schema(schema: dict[str, Any]) -> dict[str, Any]:
    require(schema.get("$id") == "stable_audio_tools.speech_dry_asset.v2", "speech asset schema id drift")
    properties = schema["properties"]
    require(properties["contract_revision"]["const"] == 4, "speech asset contract revision drift")
    require(properties["catalog_partition"]["const"] == "sdb", "speech catalog must be on SDB")
    require(properties["source_dataset"]["enum"] == ["libritts", "hifi_tts"], "speech asset sources drift")
    locator = properties["source_locator"]["properties"]
    require(locator["parquet_path"]["pattern"].startswith("^/mnt/sdc/speech_dataset/"), "speech source root pattern drift")
    audio = properties["audio"]["properties"]
    require(audio["native_channels"]["const"] == 1, "speech asset must be native mono")
    require(audio["model_num_samples"]["maximum"] == 442_368, "speech maximum length drift")
    coverage = properties["coverage"]["properties"]
    require(coverage["random_crop"]["const"] is False, "speech random crop drift")
    require(coverage["complete_selected_transcript"]["const"] is True, "speech completion drift")
    eligibility = properties["scene_source_eligibility"]["properties"]
    require(eligibility["input_audio_domain"]["const"] == "dry_mono", "speech domain drift")
    require(eligibility["spatialization_passes_before_scene"]["const"] == 0, "speech prior spatialization drift")
    require(eligibility["eligible_as_scene_source"]["const"] is True, "speech eligibility drift")
    return {
        "schema_id": schema["$id"],
        "allowed_source_datasets": properties["source_dataset"]["enum"],
        "catalog_partition": "sdb",
        "canonical_channels": 1,
        "max_model_num_samples": 442_368,
    }


def validate_compiler_schema(schema: dict[str, Any]) -> dict[str, Any]:
    require(schema.get("$id") == "stable_audio_tools.sceneplan_compiler_output.v2", "compiler schema id drift")
    properties = schema["properties"]
    require(properties["contract_revision"]["const"] == 4, "compiler contract revision drift")
    require(properties["sequence_length"]["maximum"] == 256, "caption token maximum drift")
    masks = properties["token_masks_4_4_2"]["properties"]
    require(set(masks) == {"source_semantic", "source_motion_activity", "speaker_info", "quoted_transcript"}, "4+4+2 mask keys drift")
    structured = properties["structured_controls"]["properties"]
    require(structured["source_position_activity_features"]["items"]["items"]["minItems"] == 8, "position feature dimension drift")
    return {
        "schema_id": schema["$id"],
        "caption_max_tokens": 256,
        "token_masks": "4+4+2",
        "position_activity_feature_dim": 8,
    }


def validate_deletion_manifest(manifest: dict[str, Any], check_filesystem: bool) -> dict[str, Any]:
    targets = [Path(item["path"]) for item in manifest.get("targets", [])]
    retained = [Path(value) for value in manifest.get("explicitly_retained", [])]
    require(len(targets) == len(set(targets)), "deletion targets must be unique")
    require(not set(targets) & set(retained), "a retained path is also a deletion target")
    filesystem: dict[str, Any] = {}
    if check_filesystem:
        for path in [*targets, *retained]:
            exists = path.exists()
            resolved = path.resolve(strict=False)
            filesystem[str(path)] = {
                "exists": exists,
                "resolved": str(resolved),
                "is_symlink": path.is_symlink(),
                "is_directory": path.is_dir() if exists else None,
            }
            if path in targets and exists:
                require(path.is_dir(), f"deletion target is not a directory: {path}")
                require(not path.is_symlink(), f"deletion target must not be a symlink: {path}")
                require(resolved == path, f"deletion target resolves elsewhere: {path} -> {resolved}")
                require(not os.path.ismount(path), f"deletion target must not be a mount point: {path}")
            if path in retained:
                require(exists, f"retained path is missing: {path}")
    return {
        "status": manifest.get("status"),
        "targets": [str(value) for value in targets],
        "retained_count": len(retained),
        "filesystem": filesystem,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--record-schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--compiler-schema", type=Path, default=DEFAULT_COMPILER_SCHEMA)
    parser.add_argument("--speech-asset-schema", type=Path, default=DEFAULT_SPEECH_SCHEMA)
    parser.add_argument("--deletion-manifest", type=Path, default=DEFAULT_DELETE)
    parser.add_argument("--check-filesystem", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    spec = load_json(args.spec)
    report = {
        "ok": True,
        "spec": validate_spec(spec),
        "record_schema": validate_schema(load_json(args.record_schema)),
        "compiler_schema": validate_compiler_schema(load_json(args.compiler_schema)),
        "speech_asset_schema": validate_speech_asset_schema(load_json(args.speech_asset_schema)),
        "deletion_manifest": validate_deletion_manifest(load_json(args.deletion_manifest), args.check_filesystem),
        "contract_snapshot": validate_contract_snapshot(
            spec,
            [
                args.spec,
                DEFAULT_README,
                args.record_schema,
                args.compiler_schema,
                args.speech_asset_schema,
                Path(__file__),
            ],
            args.check_filesystem,
        ),
    }
    payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
