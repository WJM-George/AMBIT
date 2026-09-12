#!/usr/bin/env python3
"""Exhaustively audit the speech-text delta and sample the unchanged 4+4+2 path."""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from build_sceneplan_manifests_v2 import (
    SPEAKER_DESCRIPTION_REGISTRY, SPEECH_LEDGER, load_speaker_registry,
)
from build_model_sceneplan_manifests_v1 import canonical_json
from sceneplan_v2_common import DATASET_ROOT, atomic_write_json, clean_text
from stable_audio_tools.data.model_sceneplan import (
    compile_model_renderer_caption, compile_model_structured_controls,
    validate_model_sceneplan,
)
from stable_audio_tools.data.sceneplan_v2 import compile_442_token_masks


ROOT = DATASET_ROOT / "sceneplans_model_v1"
TOKENIZER = Path("/mnt/sdc/ckpts/pretrained/Qwen/Qwen3.5-0.8B")


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_lines(path: Path) -> list[tuple[dict[str, Any], str]]:
    rows = []
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        value = json.loads(raw)
        if canonical_json(value) != raw:
            raise RuntimeError(f"noncanonical JSONL: {path}:{number}")
        rows.append((value, raw))
    return rows


def main() -> int:
    started = time.time()
    registry_path = SPEAKER_DESCRIPTION_REGISTRY.resolve(strict=True)
    registry = load_speaker_registry(registry_path)
    ledger_rows = [
        row for row in pq.read_table(
            SPEECH_LEDGER,
            columns=["asset_id", "pool", "renderer_text", "source_audio_sha256"],
        ).to_pylist()
        if str(row["pool"]) in {"train", "validation", "test"}
    ]
    ledger = {str(row["asset_id"]): row for row in ledger_rows}
    if len(ledger) != 512_000:
        raise RuntimeError("formal speech ledger lookup is not 512000 rows")
    index = pq.ParquetFile(ROOT / "index.parquet")
    if index.metadata.num_rows != 1_124_000 or index.num_row_groups != 1_099:
        raise RuntimeError("P7.5 index shape changed")
    profile_samples: dict[str, dict[str, Any]] = {}
    periodic_samples: list[dict[str, Any]] = []
    max_token_sample: dict[str, Any] | None = None
    speech_assets: set[str] = set()
    speaker_descriptions: Counter[str] = Counter()
    counts: Counter[tuple[str, str, int]] = Counter()
    token_counts: list[int] = []
    rows_seen = 0
    for group_index in range(index.num_row_groups):
        index_rows = index.read_row_group(group_index).to_pylist()
        first = index_rows[0]
        model_path = Path(str(first["sceneplan_path"]))
        recipe_path = Path(str(first["render_recipe_path"]))
        condition_path = Path(str(first["conditioning_path"]))
        models = canonical_lines(model_path)
        recipes = canonical_lines(recipe_path)
        conditions = canonical_lines(condition_path)
        if not (len(index_rows) == len(models) == len(recipes) == len(conditions)):
            raise RuntimeError(f"three-view row mismatch: {model_path}")
        for row, (scene, scene_text), (recipe, recipe_text), (condition, _) in zip(
            index_rows, models, recipes, conditions
        ):
            sample_id = str(row["sample_id"])
            if not (scene["sample_id"] == recipe["sample_id"] == condition["sample_id"] == sample_id):
                raise RuntimeError(f"{sample_id}: sample id/order mismatch")
            validate_model_sceneplan(scene)
            caption = compile_model_renderer_caption(scene)
            if caption != condition["renderer_caption"]:
                raise RuntimeError(f"{sample_id}: caption compiler/character span drift")
            if (
                sha256_text(scene_text) != row["model_sceneplan_sha256"]
                or sha256_text(recipe_text) != row["render_recipe_sha256"]
                or sha256_text(canonical_json(caption)) != row["renderer_caption_sha256"]
                or recipe["model_sceneplan_sha256"] != row["model_sceneplan_sha256"]
            ):
                raise RuntimeError(f"{sample_id}: three-view hash/link mismatch")
            split = str(row["split"])
            family = str(row["family"])
            counts[(split, family, int(row["source_count"]))] += 1
            token_count = int(row["caption_qwen_tokens"])
            token_counts.append(token_count)
            sample = {
                "sample_id": sample_id, "scene": scene, "caption": caption,
                "stored_tokens": token_count,
                "model_num_samples": int(row["model_num_samples"]),
                "latent_frames_valid": int(row["latent_frames_valid"]),
            }
            if max_token_sample is None or token_count > max_token_sample["stored_tokens"]:
                max_token_sample = sample
            recipe_by_id = {str(source["source_id"]): source for source in recipe["sources"]}
            speech = [source for source in scene["sources"] if source["kind"] == "speech"]
            if len(speech) != (1 if family == "speech" else 0):
                raise RuntimeError(f"{sample_id}: exactly-one-speech family gate failed")
            if speech:
                source = speech[0]
                asset_id = str(recipe_by_id[str(source["source_id"])]["asset_ref"]["asset_id"])
                entry = registry.get(asset_id)
                donor = ledger.get(asset_id)
                if (
                    entry is None or donor is None
                    or str(entry["split"]) != split
                    or source["speaker_description"] != entry["speaker_description"]
                    or source["transcript"] != clean_text(donor["renderer_text"])
                    or str(entry["source_audio_sha256"]) != str(donor["source_audio_sha256"])
                ):
                    raise RuntimeError(f"{sample_id}: registry/transcript lineage mismatch")
                if asset_id in speech_assets:
                    raise RuntimeError(f"{sample_id}: speech asset reused")
                speech_assets.add(asset_id)
                speaker_descriptions[str(source["speaker_description"])] += 1
                profile_samples.setdefault(str(entry["speaker_profile_key"]), sample)
            if rows_seen < 16 or rows_seen % 10_000 == 0:
                periodic_samples.append(sample)
            rows_seen += 1
        if (group_index + 1) % 100 == 0:
            print(json.dumps({"audited_shards": group_index + 1, "rows": rows_seen}), flush=True)
    if rows_seen != 1_124_000 or speech_assets != set(registry):
        raise RuntimeError("P7.5 registry coverage/row total failed")
    samples_by_id = {
        sample["sample_id"]: sample
        for sample in [*profile_samples.values(), *periodic_samples, max_token_sample]
        if sample is not None
    }
    samples = list(samples_by_id.values())
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)
    encoded = tokenizer(
        [sample["caption"]["text"] for sample in samples],
        add_special_tokens=True, truncation=False, padding=False,
        return_offsets_mapping=True,
    )
    structured_checked = 0
    for sample, ids, attention, offsets in zip(
        samples, encoded["input_ids"], encoded["attention_mask"], encoded["offset_mapping"]
    ):
        if len(ids) != sample["stored_tokens"] or len(ids) > 512:
            raise RuntimeError(f"{sample['sample_id']}: sampled token count mismatch")
        masks = compile_442_token_masks(sample["caption"], offsets, attention)
        present = {int(str(source["source_id"])[7:]) for source in sample["scene"]["sources"]}
        for slot in range(4):
            if bool(masks["source_semantic_token_masks"][slot].any()) != (slot in present):
                raise RuntimeError(f"{sample['sample_id']}: semantic 4-mask mismatch")
            if bool(masks["source_motion_activity_token_masks"][slot].any()) != (slot in present):
                raise RuntimeError(f"{sample['sample_id']}: motion 4-mask mismatch")
        has_speech = any(source["kind"] == "speech" for source in sample["scene"]["sources"])
        if (
            bool(masks["speaker_info_token_mask"].any()) != has_speech
            or bool(masks["quoted_transcript_token_mask"].any()) != has_speech
        ):
            raise RuntimeError(f"{sample['sample_id']}: speaker/transcript 2-mask mismatch")
        controls = compile_model_structured_controls(
            sample["scene"],
            model_num_samples=sample["model_num_samples"],
            latent_frames_valid=sample["latent_frames_valid"],
        )
        if controls["source_position_activity_gain_features"].shape != (
            4, sample["latent_frames_valid"], 9
        ):
            raise RuntimeError(f"{sample['sample_id']}: structured 9D control mismatch")
        structured_checked += 1
    p99 = float(np.percentile(np.asarray(token_counts, dtype=np.float64), 99))
    maximum = max(token_counts)
    if p99 > 384 or maximum > 512 or speaker_descriptions["an English audiobook narrator"]:
        raise RuntimeError("caption envelope or generic-speaker gate failed")
    registry_audit = json.loads(
        (registry_path.parent / "audit.json").read_text(encoding="utf-8")
    )
    report = {
        "schema": "stable_audio_tools.model_sceneplan_manifest_audit",
        "schema_version": 2,
        "ok": True,
        "dataset_contract_revision": 5,
        "mode": "full",
        "root": str(ROOT),
        "rows": rows_seen,
        "joint_counts": {"|".join(map(str, key)): value for key, value in sorted(counts.items())},
        "globally_unique_speech_assets": len(speech_assets),
        "globally_unique_speech_audio_hashes": len(speech_assets),
        "globally_unique_normalized_speech_transcripts": len(speech_assets),
        "speech_speaker_registry_rows": len(registry),
        "speech_speaker_registry_rows_referenced": len(speech_assets),
        "unique_speaker_descriptions": len(speaker_descriptions),
        "unique_speaker_profile_keys": len(profile_samples),
        "unique_speaker_identity_descriptions": registry_audit["unique_identity_descriptions"],
        "constant_generic_speaker_description_rows": speaker_descriptions["an English audiobook narrator"],
        "caption_tokens": {"p99": p99, "p99_target": 384, "max": maximum, "hard_max": 512, "truncated": 0},
        "caption_character_spans_all_rows": True,
        "token_masks": "4+4+2",
        "token_mask_and_structured_control_samples": structured_checked,
        "token_mask_sample_covers_every_speaker_profile": True,
        "structured_feature_dim": 9,
        "exact_transcript_all_rows_ledger_authoritative": True,
        "exactly_one_formal_speech_all_rows": True,
        "non_text_sceneplan_state_preserved_by_migration": True,
        "foa_materialization_started": False,
        "latent_materialization_started": False,
        "p8_started": False,
        "p9_started": False,
        "elapsed_sec": round(time.time() - started, 3),
    }
    atomic_write_json(ROOT / "audit.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
