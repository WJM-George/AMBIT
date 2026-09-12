#!/usr/bin/env python3
"""Freeze a matched ScenePlan DiT checkpoint-comparison evaluation contract.

The benchmark uses one fixed source-disjoint panel with equal numbers of
single-source music, sound, and speech examples.  The exact same rows and noise
seeds are used for every checkpoint.  The historical default remains five rows
per domain; larger diagnostic panels are selected deterministically and
balanced over every room/motion stratum available to that domain.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import zlib
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq


DATASET_ROOT = Path("/mnt/sdb/audio_dataset/sceneplan_v2_1p124m")
RUN_ROOT = Path(
    "/mnt/sdb/model_archives/p10_pre_v11_20260831/sceneplan_dit_fail/"
    "sceneplan_dit_v4_r8_300m"
)
DEFAULT_OUTPUT = RUN_ROOT / "evaluation/p10_ckpt_5k_10k_15k_sceneplan44_v1"
CHECKPOINT_STEPS = (5_000, 10_000, 15_000)
LISTENING_SEED = "sceneplan-p10-listening-panel-44-v1-20260821"
NOISE_SEED = "sceneplan-p10-common-noise-44-v1-20260821"

# Easy, frequent, reader-interpretable sound events selected from the frozen
# source-disjoint P9 test set.  Their raw labels and A2T descriptions were
# audited previously, and the current ScenePlan text is re-read below rather
# than copied from an older evaluation artifact.
SIMPLE_SOUND_SAMPLE_IDS = (
    "spv2_test_no_speech_1_0000208",  # steady engine idle
    "spv2_test_no_speech_1_0000624",  # cat meowing
    "spv2_test_no_speech_1_0000204",  # mechanical keyboard typing
    "spv2_test_no_speech_1_0000670",  # toilet flushing
    "spv2_test_no_speech_1_0000256",  # electronic police siren
)
MIXED_MUSIC_SAMPLE_IDS = (
    "spv2_test_no_speech_1_0000369",  # solo blues harmonica
    "spv2_test_no_speech_1_0000045",  # chiptune electronic song
    "spv2_test_no_speech_1_0000579",  # distorted glitch electronic music
    "spv2_test_no_speech_1_0000235",  # dark ambient vocal music
    "spv2_test_no_speech_1_0000403",  # upbeat accordion folk tune
)
INSTRUMENTAL_MUSIC_SAMPLE_IDS = (
    "spv2_test_no_speech_1_0000021",  # solo acoustic guitar
    "spv2_test_no_speech_1_0000631",  # solo melancholic piano
    "spv2_test_no_speech_1_0000023",  # solo violin
    "spv2_test_no_speech_1_0000037",  # instrumental electronic dance music
    "spv2_test_no_speech_1_0000311",  # cinematic orchestral music
)
VOCAL_MUSIC_PATTERN = re.compile(
    r"\b(?:sings?|singing|singer|vocal(?:s|ist)?|choir|choral|raps?|rapping|"
    r"rapper|lyrics?|chant(?:s|ing)?|hums?|humming|a cappella|spoken[- ]word|"
    r"speech|speaks?|talks?|talking|whispers?|narrat(?:or|ion|es?|ing)|voice|"
    r"phrase|words?)\b",
    re.IGNORECASE,
)
INSTRUMENTAL_MUSIC_EVIDENCE_PATTERN = re.compile(
    r"\b(?:music(?:al)?|instrumental|melod(?:y|ic)|harmonica|accordion|guitar|"
    r"piano|violin|cello|harp|flute|clarinet|saxophone|trumpet|trombone|organ|"
    r"keyboard|xylophone|marimba|banjo|mandolin|ukulele|drum(?:s|ming)?|"
    r"percussion|orchestra(?:l)?|symphon(?:y|ic)|synth(?:esizer)?|chiptune|"
    r"jazz|blues|classical|rock|heavy metal|folk|country|reggae|hip[- ]hop|"
    r"techno|house|trance|dance track|soundtrack)\b",
    re.IGNORECASE,
)
MIN_SPEECH_EVAL_WORDS = 4
SPEECH_SAMPLE_IDS = (
    "spv2_test_speech_1_0000000",  # adult male, dry, linear
    "spv2_test_speech_1_0000006",  # mature male, reverberant, static
    "spv2_test_speech_1_0000044",  # young female, dry, static
    "spv2_test_speech_1_0000001",  # elderly female, moderate, static
    "spv2_test_speech_1_0000620",  # young female, dry, linear, short
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _write_json(path: Path, value: Any) -> None:
    _atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    _atomic_text(
        path,
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
    )


def _stable_rank(sample_id: str) -> str:
    return hashlib.sha256(f"{LISTENING_SEED}\0{sample_id}".encode()).hexdigest()


def _noise_seed(sample_id: str) -> int:
    digest = hashlib.sha256(f"{NOISE_SEED}\0{sample_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def _read_p9_rows(index_path: Path) -> list[dict[str, Any]]:
    connection = sqlite3.connect(f"file:{index_path}?mode=ro&immutable=1", uri=True)
    try:
        output = []
        for ordinal, sample_id, model_num_samples, valid_frames, caption_blob, plan_blob in connection.execute(
            """
            SELECT ordinal, sample_id, model_num_samples, latent_frames_valid,
                   renderer_caption_zlib, scene_plan_zlib
            FROM samples ORDER BY ordinal
            """
        ):
            caption = json.loads(zlib.decompress(caption_blob))
            plan = json.loads(zlib.decompress(plan_blob))
            if len(plan["sources"]) != 1:
                continue
            source = plan["sources"][0]
            domain = str(source["kind"])
            semantic_text = (
                str(source["transcript"])
                if domain == "speech"
                else str(source["description"])
            )
            output.append(
                {
                    "ordinal": int(ordinal),
                    "sample_id": str(sample_id),
                    "domain": domain,
                    "model_num_samples": int(model_num_samples),
                    "latent_frames_valid": int(valid_frames),
                    "duration_sec": float(plan["duration_sec"]),
                    "room_type": str(plan["room"]["type"]),
                    "motion_type": str(source["trajectory"]["type"]),
                    "semantic_text": semantic_text,
                    "renderer_caption": str(caption["text"]),
                    "scene_plan": plan,
                    "noise_seed": _noise_seed(str(sample_id)),
                }
            )
        return output
    finally:
        connection.close()


def _parquet_rows(pattern: str) -> Iterable[dict[str, Any]]:
    paths = sorted(DATASET_ROOT.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"no parquet files match {DATASET_ROOT / pattern}")
    for path in paths:
        yield from pq.read_table(path).to_pylist()


def _reference_map() -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in _parquet_rows("materialized/manifests/test/*.parquet"):
        output[str(row["sample_id"])] = {
            "reference_foa_path": str(row["foa_path"]),
            "reference_foa_sha256": str(row["foa_sha256"]),
        }
    return output


def _source_map() -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    paths = sorted(
        (DATASET_ROOT / "sceneplans_model_v1/test").glob(
            "render-recipes-test-*.jsonl"
        )
    )
    if not paths:
        raise FileNotFoundError(
            DATASET_ROOT / "sceneplans_model_v1/test/render-recipes-test-*.jsonl"
        )
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                recipe = json.loads(line)
                sources = recipe.get("sources") or []
                if len(sources) != 1:
                    continue
                sample_id = str(recipe["sample_id"])
                if sample_id in output:
                    raise RuntimeError(
                        f"{path}:{line_number}: duplicate sample_id {sample_id}"
                    )
                asset = sources[0]["asset_ref"]
                output[sample_id] = {
                    "source_asset_id": str(asset["asset_id"]),
                    "source_audio_sha256": str(asset["identity_hash"]),
                    # Non-speech assets are ordinary files. Speech donors live
                    # inside frozen LibriTTS/HiFiTTS parquet shards, so row
                    # provenance is their lossless locator.
                    "source_dry_audio_path": (
                        str(asset["dry_audio_path"])
                        if asset.get("dry_audio_path") is not None
                        else None
                    ),
                    "source_parquet_path": (
                        str(asset["parquet_path"])
                        if asset.get("parquet_path") is not None
                        else None
                    ),
                    "source_row_group": (
                        int(asset["row_group"])
                        if asset.get("row_group") is not None
                        else None
                    ),
                    "source_row_in_group": (
                        int(asset["row_in_group"])
                        if asset.get("row_in_group") is not None
                        else None
                    ),
                    "source_native_sample_rate_hz": int(asset["native_sample_rate_hz"]),
                    "source_native_num_samples": int(asset["native_num_samples"]),
                }
    return output


def _spoken_language_map() -> dict[str, bool]:
    path = DATASET_ROOT / (
        "source_annotations/nonspeech_instruct_v2/registry/"
        "source_description_registry.parquet"
    )
    table = pq.read_table(path, columns=["source_audio_sha256", "spoken_language_background"])
    return {
        str(row["source_audio_sha256"]): bool(row["spoken_language_background"])
        for row in table.to_pylist()
    }


def _pick_one(
    candidates: list[dict[str, Any]],
    used: set[str],
    *,
    room: str | None = None,
    motion: str | None = None,
    duration_min: float | None = None,
    duration_max: float | None = None,
) -> dict[str, Any]:
    eligible = []
    for row in candidates:
        if row["sample_id"] in used:
            continue
        if row["domain"] != "speech" and row.get("spoken_language_background"):
            continue
        if room is not None and row["room_type"] != room:
            continue
        if motion is not None and row["motion_type"] != motion:
            continue
        if duration_min is not None and row["duration_sec"] < duration_min:
            continue
        if duration_max is not None and row["duration_sec"] >= duration_max:
            continue
        eligible.append(row)
    if not eligible:
        raise RuntimeError(
            f"listening stratum is empty: room={room}, motion={motion}, "
            f"duration=[{duration_min},{duration_max})"
        )
    selected = min(eligible, key=lambda row: _stable_rank(row["sample_id"]))
    used.add(selected["sample_id"])
    return selected


def _listening_panel(
    rows: list[dict[str, Any]],
    *,
    music_sample_ids: tuple[str, ...] = MIXED_MUSIC_SAMPLE_IDS,
    instrumental_music_only: bool = False,
) -> list[dict[str, Any]]:
    by_domain = {
        domain: [row for row in rows if row["domain"] == domain]
        for domain in ("music", "sound", "speech")
    }
    selected_ids = {
        "music": music_sample_ids,
        "sound": SIMPLE_SOUND_SAMPLE_IDS,
        "speech": SPEECH_SAMPLE_IDS,
    }
    panel = []
    for domain in ("music", "sound", "speech"):
        by_sample_id = {row["sample_id"]: row for row in by_domain[domain]}
        for panel_index, sample_id in enumerate(selected_ids[domain], start=1):
            if sample_id not in by_sample_id:
                raise RuntimeError(f"frozen {domain} test row is missing: {sample_id}")
            selected = dict(by_sample_id[sample_id])
            if domain != "speech" and selected.get("spoken_language_background"):
                raise RuntimeError(f"{domain} row contains spoken language: {sample_id}")
            if (
                domain == "music"
                and instrumental_music_only
                and VOCAL_MUSIC_PATTERN.search(selected["semantic_text"])
            ):
                raise RuntimeError(
                    f"instrumental panel contains vocal-language evidence: {sample_id}"
                )
            selected["panel_index"] = panel_index
            selected["panel_id"] = f"{domain}_{panel_index:02d}"
            panel.append(selected)
    return panel


def _balanced_panel(
    rows: list[dict[str, Any]],
    *,
    rows_per_domain: int,
    instrumental_music_only: bool,
) -> list[dict[str, Any]]:
    """Select an even, deterministic panel over domain-specific room/motion cells."""

    panel: list[dict[str, Any]] = []
    panel_width = max(2, len(str(rows_per_domain)))
    for domain in ("music", "sound", "speech"):
        eligible = [
            row
            for row in rows
            if row["domain"] == domain
            and (domain == "speech" or not row.get("spoken_language_background"))
            and not (
                domain == "speech"
                and len(re.findall(r"[A-Za-z0-9']+", row["semantic_text"]))
                < MIN_SPEECH_EVAL_WORDS
            )
            and not (
                domain == "music"
                and instrumental_music_only
                and (
                    VOCAL_MUSIC_PATTERN.search(row["semantic_text"])
                    or not INSTRUMENTAL_MUSIC_EVIDENCE_PATTERN.search(
                        row["semantic_text"]
                    )
                )
            )
        ]
        if len(eligible) < rows_per_domain:
            raise RuntimeError(
                f"not enough eligible {domain} rows: {len(eligible)} < {rows_per_domain}"
            )
        strata = sorted({(row["room_type"], row["motion_type"]) for row in eligible})
        if not strata:
            raise RuntimeError(f"no room/motion strata for {domain}")
        buckets = {
            stratum: sorted(
                [
                    row
                    for row in eligible
                    if (row["room_type"], row["motion_type"]) == stratum
                ],
                key=lambda row: _stable_rank(row["sample_id"]),
            )
            for stratum in strata
        }
        selected: list[dict[str, Any]] = []
        # Round-robin gives each available room/motion cell either floor(N/K)
        # or ceil(N/K) rows while stable hash ranking removes catalogue-order bias.
        for index in range(rows_per_domain):
            stratum = strata[index % len(strata)]
            if not buckets[stratum]:
                raise RuntimeError(
                    f"{domain} stratum exhausted before quota: {stratum}"
                )
            selected.append(dict(buckets[stratum].pop(0)))
        for panel_index, row in enumerate(selected, start=1):
            row["panel_index"] = panel_index
            row["panel_id"] = f"{domain}_{panel_index:0{panel_width}d}"
            panel.append(row)
    return panel


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument(
        "--checkpoint-steps",
        type=int,
        nargs="+",
        default=list(CHECKPOINT_STEPS),
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=(
            Path(__file__).resolve().parents[3]
            / "stable_audio_tools/configs/model_configs/txt2audio/t2a/dit/"
            "qwen35_0p8b_300m_model_sceneplan_44.json"
        ),
    )
    parser.add_argument("--hash-checkpoints", action="store_true")
    parser.add_argument(
        "--rows-per-domain",
        type=int,
        default=5,
        help=(
            "Equal rows selected for music, sound, and speech. Five preserves "
            "the historical hand-frozen panel; larger values use balanced "
            "deterministic room/motion selection."
        ),
    )
    parser.add_argument(
        "--music-profile",
        choices=("mixed", "instrumental"),
        default="mixed",
        help="Use the original mixed music panel or a voice-free instrumental panel.",
    )
    args = parser.parse_args()
    output_root = args.output_root.expanduser().resolve()
    run_root = args.run_root.expanduser().resolve(strict=True)
    model_config = args.model_config.expanduser().resolve(strict=True)
    checkpoint_steps = tuple(int(step) for step in args.checkpoint_steps)
    rows_per_domain = int(args.rows_per_domain)
    if rows_per_domain < 2:
        raise ValueError("rows-per-domain must be at least two")
    if (
        not checkpoint_steps
        or len(set(checkpoint_steps)) != len(checkpoint_steps)
        or any(step <= 0 for step in checkpoint_steps)
    ):
        raise ValueError("checkpoint steps must be unique positive integers")
    index_path = DATASET_ROOT / "training_index/test.sqlite"
    if not index_path.is_file():
        raise FileNotFoundError(index_path)
    checkpoints_by_step: dict[int, Path] = {}
    for step in checkpoint_steps:
        candidates = sorted((run_root / "checkpoints").glob(f"*step={step}.ckpt"))
        if len(candidates) != 1:
            raise RuntimeError(
                f"expected one checkpoint at step {step}, found {candidates}"
            )
        checkpoints_by_step[step] = candidates[0]

    rows = _read_p9_rows(index_path)
    references = _reference_map()
    sources = _source_map()
    spoken = _spoken_language_map()
    for row in rows:
        sample_id = row["sample_id"]
        if sample_id not in references or sample_id not in sources:
            raise RuntimeError(f"{sample_id}: test provenance is incomplete")
        row.update(references[sample_id])
        row.update(sources[sample_id])
        if row["domain"] == "speech":
            row["spoken_language_background"] = False
        else:
            source_hash = row["source_audio_sha256"]
            if source_hash not in spoken:
                raise RuntimeError(f"{sample_id}: spoken-language label is missing")
            row["spoken_language_background"] = spoken[source_hash]
        if not Path(row["reference_foa_path"]).is_file():
            raise FileNotFoundError(row["reference_foa_path"])
        source_locator = (
            row["source_dry_audio_path"]
            if row["source_dry_audio_path"] is not None
            else row["source_parquet_path"]
        )
        if source_locator is None or not Path(source_locator).is_file():
            raise FileNotFoundError(source_locator)

    counts = Counter(row["domain"] for row in rows)
    expected = {"music": 350, "sound": 350, "speech": 700}
    if dict(counts) != expected:
        raise RuntimeError(f"single-source test counts changed: {dict(counts)} != {expected}")
    instrumental_music_only = args.music_profile == "instrumental"
    music_sample_ids = (
        INSTRUMENTAL_MUSIC_SAMPLE_IDS
        if instrumental_music_only
        else MIXED_MUSIC_SAMPLE_IDS
    )
    if rows_per_domain == 5:
        panel = _listening_panel(
            rows,
            music_sample_ids=music_sample_ids,
            instrumental_music_only=instrumental_music_only,
        )
    else:
        panel = _balanced_panel(
            rows,
            rows_per_domain=rows_per_domain,
            instrumental_music_only=instrumental_music_only,
        )
    panel_counts = Counter(row["domain"] for row in panel)
    expected_panel_counts = {
        "music": rows_per_domain,
        "sound": rows_per_domain,
        "speech": rows_per_domain,
    }
    if dict(panel_counts) != expected_panel_counts:
        raise RuntimeError(f"listening panel counts changed: {dict(panel_counts)}")
    if len({row["sample_id"] for row in panel}) != len(panel):
        raise RuntimeError("evaluation panel contains duplicate sample IDs")

    checkpoints = []
    for step, path in checkpoints_by_step.items():
        item: dict[str, Any] = {
            "step": step,
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
        }
        if args.hash_checkpoints:
            item["sha256"] = _sha256_file(path)
        checkpoints.append(item)

    speaker_audit_path = DATASET_ROOT / "sceneplans_model_v1/audit.json"
    speaker_audit = json.loads(speaker_audit_path.read_text(encoding="utf-8"))
    if not (
        speaker_audit.get("ok") is True
        and int(speaker_audit.get("speech_speaker_registry_rows", -1)) == 512_000
        and int(speaker_audit.get("constant_generic_speaker_description_rows", -1)) == 0
    ):
        raise RuntimeError("speaker-registry P9 audit is not all-pass")
    contract = {
        "schema": "stable_audio_tools.sceneplan_dit_p10_checkpoint_evaluation_contract",
        "schema_version": 5,
        "status": "FROZEN_SCENEPLAN_44_CFG_RESCALE_V2",
        "purpose": (
            f"matched {len(panel)}-row comparison of the corrected registry-driven "
            f"ScenePlan DiT at steps {list(checkpoint_steps)}"
        ),
        "speaker_conditioning": {
            "registry_rows": speaker_audit["speech_speaker_registry_rows"],
            "unique_speaker_profiles": speaker_audit["unique_speaker_profile_keys"],
            "unique_speaker_descriptions": speaker_audit["unique_speaker_descriptions"],
            "generic_constant_rows": speaker_audit["constant_generic_speaker_description_rows"],
            "audit": str(speaker_audit_path),
            "audit_sha256": _sha256_file(speaker_audit_path),
            "exact_transcript_ledger_authoritative": True,
        },
        "test_set": {
            "index": str(index_path.resolve()),
            "index_sha256": _sha256_file(index_path),
            "all_rows": 4_000,
            "evaluation_subset": f"fixed {len(panel)}-row single-source panel only",
            "evaluation_rows": len(panel),
            "domain_counts": expected_panel_counts,
            "eligible_single_source_counts_used_only_for_selection": expected,
            "source_disjoint": True,
            "speech_minimum_reference_words": MIN_SPEECH_EVAL_WORDS,
        },
        "checkpoints": checkpoints,
        "sampling": {
            "architecture": "semantic_cross_attention_plus_direct_sceneplan_4+4",
            "model_config": str(model_config),
            "vae_checkpoint": "/mnt/sdc/ckpts/compareVAE_ckpt/unwrapped_wdmix_1350000.ckpt",
            "weights": "EMA DiT plus EMA trainable 4+4 conditioner",
            "sampler": "euler_rectified_flow",
            "steps": 100,
            "cfg_scale": 3.0,
            "rescale_cfg": True,
            "cfg_rescale_phi": 0.4,
            "apg_scale": 0.0,
            "negative_condition": "caption and structured controls both unknown at inference",
            "training_cfg_dropout": "caption and structured controls dropped independently at 15% each",
            "common_noise_seed_namespace": NOISE_SEED,
            "same_noise_per_sample_across_checkpoints": True,
            "raw_output": "float32 WAV, four-channel WYZX/ACN/SN3D, no normalization or clipping",
            "listening_preview": "fixed virtual-stereo FOA decode, loudness normalized; never used for objective metrics",
        },
        "listening_panel": {
            "music_profile": args.music_profile,
            "music_sample_ids": [
                row["sample_id"] for row in panel if row["domain"] == "music"
            ],
            "instrumental_music_lexical_gate": (
                {
                    "vocal_exclusion": VOCAL_MUSIC_PATTERN.pattern,
                    "music_evidence_required": INSTRUMENTAL_MUSIC_EVIDENCE_PATTERN.pattern,
                }
                if instrumental_music_only
                else None
            ),
            "rows_per_domain": rows_per_domain,
            "total_prompts": len(panel),
            "total_outputs_across_checkpoints": len(panel) * len(checkpoint_steps),
            "selection_seed": LISTENING_SEED,
            "selection": (
                "historical hand-frozen five-row panel"
                if rows_per_domain == 5
                else (
                    "deterministic stable-hash selection balanced over all "
                    "domain-specific room/motion strata; non-speech rows exclude "
                    "registry-positive spoken-language backgrounds"
                    + (
                        "; instrumental music additionally passes the "
                        "vocal-language lexical gate"
                        if instrumental_music_only
                        else ""
                    )
                )
            ),
            "blind_order": "deterministic randomized checkpoint labels generated after all outputs exist",
        },
        "metrics": {
            "music_sound_primary": [
                "CLAP text-audio cosine on W channel (higher)",
                "FD-CLAP between reference and generated W embeddings (lower)",
                "FAD-VGGish between reference and generated W audio (lower)",
                "paired KL-PANN softmax on W channel (lower)",
                "azimuth circular MAE, elevation MAE, and spherical DoA angular error from FOA active intensity (lower)",
            ],
            "speech_primary": [
                "WER and CER from full generated W channel against exact transcript (lower)",
                "UTMOS naturalness proxy on W channel (higher)",
                "azimuth/elevation/spherical DoA errors and valid-direction coverage (lower/higher)",
                "activity temporal IoU and onset/offset errors (higher/lower)",
            ],
            "speech_diagnostic_not_gate": [
                "PESQ-WB after active-region extraction and bounded-lag alignment against the exact donor (higher)",
                "SI-SDR after active-region extraction and bounded-lag alignment against the exact donor (higher)",
                "STOI after active-region extraction and bounded-lag alignment against the exact donor (higher)",
                "WavLM speaker-verification cosine against the exact donor; the baseline has neither identity nor diverse speaker-attribute conditioning",
                "the same paired metrics on frozen-VAE reconstructions, reported as the codec ceiling",
            ],
            "qc": [
                "finite samples", "raw peak", "raw clipped fraction", "RMS", "silence/activity leakage"
            ],
            "uncertainty": (
                f"per-sample values plus descriptive summaries; N={rows_per_domain}/domain "
                "is a checkpoint-selection diagnostic, so distributional scores are "
                "matched-comparison estimates rather than paper-scale population results"
            ),
            "not_used": [
                "PESQ-WB/SI-SDR/STOI as hard generation gates (valid speech can differ in phase, pitch contour, and timing)",
                "distance-in-metres error until a calibrated FOA distance estimator is frozen",
            ],
        },
        "paper_protocols": [
            "https://arxiv.org/abs/2410.14945",
            "https://arxiv.org/abs/2507.07318",
            "https://arxiv.org/abs/2410.11299",
            "https://arxiv.org/abs/2406.02430",
            "https://aclanthology.org/2025.acl-long.313/",
            "https://github.com/BytedanceSpeech/seed-tts-eval",
            "https://github.com/haoheliu/audioldm_eval",
        ],
    }
    output_root.mkdir(parents=True, exist_ok=True)
    panel_filename = (
        "listening_panel_15.jsonl"
        if len(panel) == 15
        else f"evaluation_panel_{len(panel)}.jsonl"
    )
    panel_path = output_root / panel_filename
    _write_jsonl(panel_path, panel)
    contract["test_set"]["panel_filename"] = panel_filename
    contract["test_set"]["panel_sha256"] = _sha256_file(panel_path)
    _write_json(output_root / "EVAL_CONTRACT.json", contract)
    summary = {
        "status": "PASS",
        "output_root": str(output_root),
        "evaluation_rows": len(panel),
        "evaluation_counts": dict(panel_counts),
        "eligible_single_source_counts": dict(counts),
        "spoken_language_background_in_listening_music_sound": sum(
            bool(row["spoken_language_background"])
            for row in panel
            if row["domain"] != "speech"
        ),
        "room_motion_counts": {
            domain: {
                f"{room}/{motion}": count
                for (room, motion), count in sorted(
                    Counter(
                        (row["room_type"], row["motion_type"])
                        for row in panel
                        if row["domain"] == domain
                    ).items()
                )
            }
            for domain in ("music", "sound", "speech")
        },
    }
    _write_json(output_root / "BUILD_SUMMARY.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
