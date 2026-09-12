#!/usr/bin/env python3
"""Score the frozen P10-v11 public baselines on their compatible test lanes.

The v11 contract deliberately has different model-compatible panels:

* general text-to-audio systems: 745 Music and 745 Sound rows;
* Sound specialists: the same 745 Sound rows;
* Qwen3-TTS: 469 single-source Speech rows.

Music/Sound use the frozen mono quality view (FOA W, native mono, or the
precomputed stereo arithmetic mean).  Public mono/stereo systems never receive
spatial scores.  Expensive metric families are independent arms so they can be
run concurrently on separate GPUs, followed by a deterministic merge.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from faster_whisper import WhisperModel
from stable_audio_tools.training.metrics.fad_metrics import (
    load_clap_model,
    load_panns_model,
    load_vggish_model,
)

from score_p10_60k_cross_system import (
    _atomic_json,
    _atomic_text,
    _edit_distance,
    _kl,
    _load_quality_mono,
    _load_utmos,
    _low_rank_frechet,
    _normalize_words,
    _summary,
    _transcribe,
    _utmos_score,
)


DEFAULT_ROOT = Path(
    "/mnt/sdb/audio_dataset/sceneplan_v2_1p124m/revisions/"
    "speech_expansion_noalign_15s_v1/evaluation/"
    "p10_v11_150k_stratified_test_3000_semantic_v2/cross_system_baselines"
)
DEFAULT_WHISPER = Path(
    "/mnt/sdb/audio_dataset/sceneplan_v2_1p124m/models/"
    "faster-distil-whisper-large-v3"
)
REFERENCE_ID = "ground_truth"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _load_inputs(root: Path) -> dict[str, Any]:
    complete = root / "BASELINE_GENERATION_COMPLETE"
    complete.resolve(strict=True)
    contract_path = (root / "BENCHMARK_CONTRACT.json").resolve(strict=True)
    manifest_path = (root / "generation_requests.jsonl").resolve(strict=True)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    requests = _read_jsonl(manifest_path)
    expected = int(contract["generation_request_count"])
    if len(requests) != expected:
        raise RuntimeError(f"manifest count changed: {len(requests)} != {expected}")

    display_names = {REFERENCE_ID: "Ground truth"}
    rows_by_baseline: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in requests:
        baseline_id = row["baseline_id"]
        display_names[baseline_id] = row["baseline_display_name"]
        rows_by_baseline[baseline_id].append(row)
        Path(row["quality_w_path"]).resolve(strict=True)
        Path(row["reference_foa_path"]).resolve(strict=True)

    domain_meta: dict[str, dict[str, dict[str, Any]]] = {
        "music": {},
        "sound": {},
    }
    domain_sources: dict[str, dict[str, dict[str, str]]] = {
        "music": defaultdict(dict),
        "sound": defaultdict(dict),
    }
    for row in requests:
        score_domains = row["score_domains"]
        # A Sound-specialist model may receive a mixed Music+Sound prompt, but
        # it is admitted only to the Sound table.  Its 366 mixed rows must not
        # create a partial and therefore incomparable Music lane.
        if row["evaluation_lane"] == "sound_specialist_no_speech":
            score_domains = ["sound"]
        for domain in score_domains:
            if domain not in domain_meta:
                continue
            panel_id = row["panel_id"]
            meta = {
                "panel_id": panel_id,
                "semantic_prompt": row["semantic_prompt"],
                "reference_foa_path": row["reference_foa_path"],
            }
            previous = domain_meta[domain].setdefault(panel_id, meta)
            if previous != meta:
                raise RuntimeError(f"inconsistent panel metadata: {domain}/{panel_id}")
            domain_sources[domain][row["baseline_id"]][panel_id] = row[
                "quality_w_path"
            ]

    for domain in ("music", "sound"):
        expected_ids = set(domain_meta[domain])
        if len(expected_ids) != 745:
            raise RuntimeError(
                f"frozen {domain} lane changed: {len(expected_ids)} != 745"
            )
        for system_id, items in domain_sources[domain].items():
            if set(items) != expected_ids:
                raise RuntimeError(
                    f"incomplete compatible lane: {domain}/{system_id} "
                    f"{len(items)} != {len(expected_ids)}"
                )

    speech_rows = rows_by_baseline.get("qwen3_tts_1p7b_voice_design", [])
    if len(speech_rows) != 469:
        raise RuntimeError(f"frozen Speech lane changed: {len(speech_rows)} != 469")

    return {
        "contract": contract,
        "contract_path": contract_path,
        "manifest_path": manifest_path,
        "requests": requests,
        "display_names": display_names,
        "domain_meta": domain_meta,
        "domain_sources": domain_sources,
        "speech_rows": speech_rows,
    }


def _lineage(inputs: dict[str, Any], arm: str) -> dict[str, Any]:
    return {
        "schema": "sceneplan_foa.p10_v11_public_baseline_metric_partial",
        "schema_version": 1,
        "status": "PASS",
        "arm": arm,
        "contract_sha256": _sha256(inputs["contract_path"]),
        "manifest_sha256": _sha256(inputs["manifest_path"]),
    }


def _audio_items(inputs: dict[str, Any]) -> dict[str, str]:
    items: dict[str, str] = {}
    for domain in ("music", "sound"):
        for panel_id, row in inputs["domain_meta"][domain].items():
            items.setdefault(f"reference:{panel_id}", row["reference_foa_path"])
        for system_id, system_items in inputs["domain_sources"][domain].items():
            for panel_id, path in system_items.items():
                key = f"system:{system_id}:{panel_id}"
                previous = items.setdefault(key, path)
                if previous != path:
                    raise RuntimeError(f"inconsistent quality path for {key}")
    return items


def _domain_feature_keys(
    inputs: dict[str, Any], domain: str, system_id: str
) -> tuple[list[str], list[str]]:
    panel_ids = sorted(inputs["domain_meta"][domain])
    if system_id == REFERENCE_ID:
        return panel_ids, [f"reference:{panel_id}" for panel_id in panel_ids]
    return panel_ids, [
        f"system:{system_id}:{panel_id}" for panel_id in panel_ids
    ]


def _domain_system_order(inputs: dict[str, Any], domain: str) -> list[str]:
    return [REFERENCE_ID] + sorted(inputs["domain_sources"][domain])


def _feature_space_frechet(left: torch.Tensor, right: torch.Tensor) -> float:
    """Compute Fréchet distance in feature space when observations exceed D.

    ``_low_rank_frechet`` is efficient for one embedding per clip (N <= D),
    but VGGish emits many windows per clip and therefore has N much larger
    than its 128-dimensional feature space.  Forming the equivalent N-by-N
    cross matrix is wasteful; the standard D-by-D covariance formulation is
    numerically equivalent and bounded here at 128-by-128.
    """

    left = left.to(torch.float64)
    right = right.to(torch.float64)
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1]:
        raise ValueError("Fréchet inputs must be [N,D] with the same D")
    if left.shape[0] < 2 or right.shape[0] < 2:
        raise ValueError("Fréchet distance requires at least two observations")
    left_mean = left.mean(dim=0)
    right_mean = right.mean(dim=0)
    left_centered = left - left_mean
    right_centered = right - right_mean
    left_cov = left_centered.transpose(0, 1) @ left_centered / (left.shape[0] - 1)
    right_cov = right_centered.transpose(0, 1) @ right_centered / (
        right.shape[0] - 1
    )
    left_values, left_vectors = torch.linalg.eigh(left_cov)
    left_sqrt = (
        left_vectors
        * left_values.clamp_min(0.0).sqrt().unsqueeze(0)
    ) @ left_vectors.transpose(0, 1)
    middle = left_sqrt @ right_cov @ left_sqrt
    middle = (middle + middle.transpose(0, 1)) * 0.5
    trace_sqrt = torch.linalg.eigvalsh(middle).clamp_min(0.0).sqrt().sum()
    value = (
        (left_mean - right_mean).square().sum()
        + torch.trace(left_cov)
        + torch.trace(right_cov)
        - 2.0 * trace_sqrt
    )
    return float(value.clamp_min(0.0))


@torch.inference_mode()
def _score_clap(inputs: dict[str, Any], device: torch.device) -> dict[str, Any]:
    items = _audio_items(inputs)
    ordered_items = sorted(items.items())
    model = load_clap_model("630k-audioset-fusion-best.pt", device=str(device))
    audio_embeddings: dict[str, torch.Tensor] = {}
    for start in range(0, len(ordered_items), 8):
        batch = ordered_items[start : start + 8]
        waves = torch.cat(
            [
                _load_quality_mono(path, 48_000, target_samples=480_000).to(device)
                for _, path in batch
            ],
            dim=0,
        )
        embeddings = model.get_audio_embedding_from_data(
            x=waves, use_tensor=True
        ).float()
        embeddings = F.normalize(embeddings, dim=-1).cpu()
        for (key, _), embedding in zip(batch, embeddings):
            audio_embeddings[key] = embedding
        print(
            json.dumps(
                {
                    "event": "clap_audio",
                    "completed": min(start + len(batch), len(ordered_items)),
                    "total": len(ordered_items),
                }
            ),
            flush=True,
        )

    captions: dict[str, str] = {}
    for domain in ("music", "sound"):
        for panel_id, row in inputs["domain_meta"][domain].items():
            previous = captions.setdefault(panel_id, row["semantic_prompt"])
            if previous != row["semantic_prompt"]:
                raise RuntimeError(f"caption mismatch for {panel_id}")
    caption_ids = sorted(captions)
    text_kwargs: dict[str, Any] = {"use_tensor": True}
    if callable(getattr(model, "tokenize", None)):
        text_kwargs["tokenizer"] = lambda values: model.tokenize(
            values,
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
    text_tensor = model.get_text_embedding(
        [captions[panel_id] for panel_id in caption_ids], **text_kwargs
    ).float()
    text_tensor = F.normalize(text_tensor, dim=-1).cpu()
    text_embeddings = dict(zip(caption_ids, text_tensor))
    del model
    gc.collect()
    torch.cuda.empty_cache()

    domains: dict[str, Any] = {}
    for domain in ("music", "sound"):
        panel_ids = sorted(inputs["domain_meta"][domain])
        reference = torch.stack(
            [audio_embeddings[f"reference:{panel_id}"] for panel_id in panel_ids]
        )
        domains[domain] = {}
        for system_id in _domain_system_order(inputs, domain):
            _, keys = _domain_feature_keys(inputs, domain, system_id)
            generated = torch.stack([audio_embeddings[key] for key in keys])
            text_scores = [
                float(audio_embeddings[key] @ text_embeddings[panel_id])
                for panel_id, key in zip(panel_ids, keys)
            ]
            paired_scores = [
                float(
                    audio_embeddings[key]
                    @ audio_embeddings[f"reference:{panel_id}"]
                )
                for panel_id, key in zip(panel_ids, keys)
            ]
            domains[domain][system_id] = {
                "display_name": inputs["display_names"][system_id],
                "rows": len(panel_ids),
                "clap_text_audio_cosine": _summary(text_scores),
                "paired_generated_reference_clap_cosine": _summary(paired_scores),
                "fd_clap": _low_rank_frechet(generated, reference),
            }
    return {
        **_lineage(inputs, "clap"),
        "domains": domains,
        "protocol": "LAION-CLAP 630k AudioSet fusion; deterministic 10 s W/mono quality view, peak-normalized to -1 dBFS.",
    }


@torch.inference_mode()
def _score_vggish(inputs: dict[str, Any], device: torch.device) -> dict[str, Any]:
    items = _audio_items(inputs)
    ordered_items = sorted(items.items())
    model, processor, backend = load_vggish_model(str(device))
    if backend != "torchaudio" or processor is None:
        raise RuntimeError("frozen benchmark requires torchaudio VGGish preprocessing")
    embeddings: dict[str, torch.Tensor] = {}
    for index, (key, path) in enumerate(ordered_items, start=1):
        waveform = _load_quality_mono(path, 16_000)[0]
        examples = processor(waveform)
        chunks = [
            model(examples[start : start + 32].to(device)).float().cpu()
            for start in range(0, int(examples.shape[0]), 32)
        ]
        embeddings[key] = torch.cat(chunks, dim=0)
        if index % 25 == 0 or index == len(ordered_items):
            print(
                json.dumps(
                    {"event": "vggish", "completed": index, "total": len(ordered_items)}
                ),
                flush=True,
            )
    del model
    gc.collect()
    torch.cuda.empty_cache()

    domains: dict[str, Any] = {}
    for domain in ("music", "sound"):
        panel_ids = sorted(inputs["domain_meta"][domain])
        reference = torch.cat(
            [embeddings[f"reference:{panel_id}"] for panel_id in panel_ids], dim=0
        )
        domains[domain] = {}
        for system_id in _domain_system_order(inputs, domain):
            _, keys = _domain_feature_keys(inputs, domain, system_id)
            generated = torch.cat([embeddings[key] for key in keys], dim=0)
            domains[domain][system_id] = {
                "display_name": inputs["display_names"][system_id],
                "rows": len(panel_ids),
                "fad_vggish": _feature_space_frechet(generated, reference),
            }
    return {
        **_lineage(inputs, "vggish"),
        "domains": domains,
        "protocol": "VGGish Fréchet distance over the complete variable-duration W/mono quality view.",
    }


@torch.inference_mode()
def _score_panns(inputs: dict[str, Any], device: torch.device) -> dict[str, Any]:
    items = _audio_items(inputs)
    ordered_items = sorted(items.items())
    model = load_panns_model(str(device))
    features: dict[str, dict[str, torch.Tensor]] = {}
    for start in range(0, len(ordered_items), 8):
        batch = ordered_items[start : start + 8]
        waves = torch.cat(
            [
                _load_quality_mono(path, 32_000, target_samples=320_000).to(device)
                for _, path in batch
            ],
            dim=0,
        )
        embedding = model(waves).float()
        probability = torch.softmax(model.fc_audioset(embedding).float(), dim=-1)
        for (key, _), item_embedding, item_probability in zip(
            batch, embedding, probability
        ):
            features[key] = {
                "embedding": item_embedding.cpu(),
                "probability": item_probability.cpu(),
            }
        print(
            json.dumps(
                {
                    "event": "panns",
                    "completed": min(start + len(batch), len(ordered_items)),
                    "total": len(ordered_items),
                }
            ),
            flush=True,
        )
    del model
    gc.collect()
    torch.cuda.empty_cache()

    domains: dict[str, Any] = {}
    for domain in ("music", "sound"):
        panel_ids = sorted(inputs["domain_meta"][domain])
        reference = torch.stack(
            [features[f"reference:{panel_id}"]["embedding"] for panel_id in panel_ids]
        )
        domains[domain] = {}
        for system_id in _domain_system_order(inputs, domain):
            _, keys = _domain_feature_keys(inputs, domain, system_id)
            generated = torch.stack([features[key]["embedding"] for key in keys])
            paired_kl = [
                _kl(
                    features[f"reference:{panel_id}"]["probability"],
                    features[key]["probability"],
                )
                for panel_id, key in zip(panel_ids, keys)
            ]
            domains[domain][system_id] = {
                "display_name": inputs["display_names"][system_id],
                "rows": len(panel_ids),
                "fd_pann": _low_rank_frechet(generated, reference),
                "paired_kl_pann_softmax": _summary(paired_kl),
            }
    return {
        **_lineage(inputs, "panns"),
        "domains": domains,
        "protocol": "PANN AudioSet embedding FD and paired softmax KL on the deterministic 10 s W/mono quality view.",
    }


def _speech_output(
    row: dict[str, Any], system_id: str, path: str, model: WhisperModel, utmos: Any,
    device: torch.device,
) -> dict[str, Any]:
    waveform = _load_quality_mono(path, 16_000)[0].numpy().astype(np.float32)
    transcription = _transcribe(model, waveform)
    reference_words = _normalize_words(row["transcript"])
    hypothesis_words = _normalize_words(transcription)
    reference_chars = list("".join(reference_words))
    hypothesis_chars = list("".join(hypothesis_words))
    word_errors = _edit_distance(reference_words, hypothesis_words)
    char_errors = _edit_distance(reference_chars, hypothesis_chars)
    return {
        "system_id": system_id,
        "panel_id": row["panel_id"],
        "speech_seen_speaker": bool(row["speech_seen_speaker"]),
        "speech_speaker_key": row["speech_speaker_key"],
        "length_bucket": row["length_bucket"],
        "word_errors": word_errors,
        "reference_words": len(reference_words),
        "char_errors": char_errors,
        "reference_chars": len(reference_chars),
        "wer": word_errors / max(len(reference_words), 1),
        "cer": char_errors / max(len(reference_chars), 1),
        "utmos": _utmos_score(utmos, waveform, device) if utmos is not None else None,
        "asr": transcription,
        "reference": row["transcript"],
    }


def _score_speech_shard(
    inputs: dict[str, Any], *, device_index: int, shard_index: int, num_shards: int
) -> dict[str, Any]:
    if not 0 <= shard_index < num_shards:
        raise ValueError("invalid Speech shard")
    rows = [
        row
        for index, row in enumerate(inputs["speech_rows"])
        if index % num_shards == shard_index
    ]
    whisper_path = DEFAULT_WHISPER.resolve(strict=True)
    model = WhisperModel(
        str(whisper_path),
        device="cuda",
        device_index=device_index,
        compute_type="float16",
    )
    device = torch.device(f"cuda:{device_index}")
    utmos, utmos_error = _load_utmos(device)
    outputs = []
    for index, row in enumerate(rows, start=1):
        systems = {
            REFERENCE_ID: row["reference_foa_path"],
            row["baseline_id"]: row["quality_w_path"],
        }
        for system_id, path in systems.items():
            outputs.append(_speech_output(row, system_id, path, model, utmos, device))
        print(
            json.dumps(
                {
                    "event": "speech",
                    "shard": shard_index,
                    "completed_rows": index,
                    "total_rows": len(rows),
                }
            ),
            flush=True,
        )
    return {
        **_lineage(inputs, "speech"),
        "shard_index": shard_index,
        "num_shards": num_shards,
        "rows": len(rows),
        "outputs": outputs,
        "utmos_error": utmos_error,
        "protocol": "faster-distil-whisper-large-v3 corpus WER/CER and UTMOS22-strong on full variable-duration waveforms.",
    }


def _aggregate_speech(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def aggregate(chosen: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "rows": len(chosen),
            "wer": _summary([row["wer"] for row in chosen]),
            "cer": _summary([row["cer"] for row in chosen]),
            "corpus_wer": sum(row["word_errors"] for row in chosen)
            / max(sum(row["reference_words"] for row in chosen), 1),
            "corpus_cer": sum(row["char_errors"] for row in chosen)
            / max(sum(row["reference_chars"] for row in chosen), 1),
            "utmos": _summary([row["utmos"] for row in chosen]),
        }

    return {
        **aggregate(rows),
        "subgroups": {
            "seen_speaker": aggregate(
                [row for row in rows if row["speech_seen_speaker"]]
            ),
            "unseen_speaker": aggregate(
                [row for row in rows if not row["speech_seen_speaker"]]
            ),
            "latent_le_432": aggregate(
                [row for row in rows if row["length_bucket"] == 432]
            ),
            "latent_433_648": aggregate(
                [row for row in rows if row["length_bucket"] == 648]
            ),
        },
        "per_output": sorted(rows, key=lambda row: row["panel_id"]),
    }


def _fmt(value: float | None, digits: int = 3) -> str:
    return "N/A" if value is None else f"{value:.{digits}f}"


def _merge(inputs: dict[str, Any], root: Path, num_speech_shards: int) -> dict[str, Any]:
    partial = root / "metrics/partials"
    clap = json.loads((partial / "CLAP.json").read_text(encoding="utf-8"))
    vggish = json.loads((partial / "VGGISH.json").read_text(encoding="utf-8"))
    panns = json.loads((partial / "PANNS.json").read_text(encoding="utf-8"))
    expected_contract = _sha256(inputs["contract_path"])
    expected_manifest = _sha256(inputs["manifest_path"])
    for report in (clap, vggish, panns):
        if report["contract_sha256"] != expected_contract:
            raise RuntimeError(f"contract lineage mismatch in {report['arm']}")
        if report["manifest_sha256"] != expected_manifest:
            raise RuntimeError(f"manifest lineage mismatch in {report['arm']}")

    audio: dict[str, Any] = {}
    for domain in ("music", "sound"):
        audio[domain] = {}
        systems = set(clap["domains"][domain])
        if systems != set(vggish["domains"][domain]) or systems != set(
            panns["domains"][domain]
        ):
            raise RuntimeError(f"metric-system mismatch for {domain}")
        for system_id in sorted(systems):
            audio[domain][system_id] = {
                **clap["domains"][domain][system_id],
                **vggish["domains"][domain][system_id],
                **panns["domains"][domain][system_id],
            }

    speech_outputs: list[dict[str, Any]] = []
    utmos_errors = []
    for shard in range(num_speech_shards):
        report = json.loads(
            (partial / f"SPEECH_SHARD_{shard:02d}_OF_{num_speech_shards:02d}.json").read_text(
                encoding="utf-8"
            )
        )
        if report["contract_sha256"] != expected_contract:
            raise RuntimeError(f"Speech contract lineage mismatch in shard {shard}")
        speech_outputs.extend(report["outputs"])
        if report.get("utmos_error"):
            utmos_errors.append(report["utmos_error"])
    speech: dict[str, Any] = {}
    for system_id in (REFERENCE_ID, "qwen3_tts_1p7b_voice_design"):
        rows = [row for row in speech_outputs if row["system_id"] == system_id]
        if len(rows) != 469 or len({row["panel_id"] for row in rows}) != 469:
            raise RuntimeError(f"incomplete Speech scores for {system_id}: {len(rows)}")
        speech[system_id] = {
            "display_name": inputs["display_names"][system_id],
            **_aggregate_speech(rows),
        }

    report = {
        "schema": "sceneplan_foa.p10_v11_public_baseline_metrics",
        "schema_version": 1,
        "status": "PASS",
        "contract_sha256": expected_contract,
        "manifest_sha256": expected_manifest,
        "domain_counts": {"music": 745, "sound": 745, "speech": 469},
        "audio_domains": audio,
        "speech": speech,
        "utmos_errors": sorted(set(utmos_errors)),
        "quality_protocol": "FOA reference uses W; native mono unchanged; stereo uses the frozen arithmetic-mean quality view; metric preprocessing peak-normalizes to -1 dBFS.",
        "spatial_protocol": "Public mono/stereo baselines receive no spatial score. Native-FOA spatial evaluation is a separate table.",
        "metric_protocols": {
            "clap": clap["protocol"],
            "fad_vggish": vggish["protocol"],
            "panns": panns["protocol"],
        },
    }
    _atomic_json(root / "metrics/PUBLIC_BASELINE_METRICS.json", report)

    table_root = root / "tables"
    for domain in ("music", "sound"):
        lines = [
            f"# {domain.title()} public-baseline evaluation (frozen)",
            "",
            "| System | CLAP ↑ | Paired CLAP ↑ | FD-CLAP ↓ | FAD-VGGish ↓ | FD-PANN ↓ | KL-PANN ↓ |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        order = [REFERENCE_ID] + [
            system_id for system_id in audio[domain] if system_id != REFERENCE_ID
        ]
        for system_id in order:
            row = audio[domain][system_id]
            lines.append(
                "| {name} | {clap} | {paired} | {fd_clap} | {fad} | {fd_pann} | {kl} |".format(
                    name=row["display_name"],
                    clap=_fmt(row["clap_text_audio_cosine"]["mean"]),
                    paired=_fmt(row["paired_generated_reference_clap_cosine"]["mean"]),
                    fd_clap=_fmt(row["fd_clap"]),
                    fad=_fmt(row["fad_vggish"]),
                    fd_pann=_fmt(row["fd_pann"]),
                    kl=_fmt(row["paired_kl_pann_softmax"]["mean"]),
                )
            )
        lines.extend(
            [
                "",
                "All rows use the same compatible 745-item domain slice. Public systems are mono/stereo and therefore have no FOA spatial score.",
                "",
            ]
        )
        _atomic_text(table_root / f"PUBLIC_{domain.upper()}_TABLE.md", "\n".join(lines))

    lines = [
        "# Speech public-baseline evaluation (frozen)",
        "",
        "| System | WER ↓ | Seen-spk WER ↓ | Unseen-spk WER ↓ | CER ↓ | UTMOS ↑ |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for system_id in (REFERENCE_ID, "qwen3_tts_1p7b_voice_design"):
        row = speech[system_id]
        lines.append(
            "| {name} | {wer} | {seen} | {unseen} | {cer} | {utmos} |".format(
                name=row["display_name"],
                wer=_fmt(row["corpus_wer"]),
                seen=_fmt(row["subgroups"]["seen_speaker"]["corpus_wer"]),
                unseen=_fmt(row["subgroups"]["unseen_speaker"]["corpus_wer"]),
                cer=_fmt(row["corpus_cer"]),
                utmos=_fmt(row["utmos"]["mean"]),
            )
        )
    lines.extend(
        [
            "",
            "Qwen3-TTS receives the same exact transcript and speaker description on 469 single-source Speech rows; it is not a spatial scene generator.",
            "",
        ]
    )
    _atomic_text(table_root / "PUBLIC_SPEECH_TABLE.md", "\n".join(lines))
    (root / "PUBLIC_BASELINE_METRICS_COMPLETE").write_text(
        "PASS\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--arm", choices=("clap", "vggish", "panns", "speech", "merge"), required=True
    )
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()

    root = args.benchmark_root.expanduser().resolve(strict=True)
    inputs = _load_inputs(root)
    partial = root / "metrics/partials"
    partial.mkdir(parents=True, exist_ok=True)
    if args.arm == "merge":
        report = _merge(inputs, root, args.num_shards)
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "metrics": str(root / "metrics/PUBLIC_BASELINE_METRICS.json"),
                    "tables": str(root / "tables"),
                }
            ),
            flush=True,
        )
        return 0

    if not torch.cuda.is_available():
        raise RuntimeError("metric scoring requires CUDA")
    device = torch.device(f"cuda:{args.device_index}")
    if args.arm == "clap":
        report = _score_clap(inputs, device)
        output = partial / "CLAP.json"
    elif args.arm == "vggish":
        report = _score_vggish(inputs, device)
        output = partial / "VGGISH.json"
    elif args.arm == "panns":
        report = _score_panns(inputs, device)
        output = partial / "PANNS.json"
    else:
        report = _score_speech_shard(
            inputs,
            device_index=args.device_index,
            shard_index=args.shard_index,
            num_shards=args.num_shards,
        )
        output = partial / (
            f"SPEECH_SHARD_{args.shard_index:02d}_OF_{args.num_shards:02d}.json"
        )
    _atomic_json(output, report)
    print(json.dumps({"status": "PASS", "arm": args.arm, "output": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
