#!/usr/bin/env python3
"""Score the final internal-8k or OOD-3k P10 cross-system benchmark.

Music and Sound use three compact, standard metric families: LAION-CLAP
semantic similarity, VGGish FAD, and paired PANN posterior KL.  Speech uses
corpus WER, corpus CER, and UTMOS.  Internal results are additionally stratified
by one through four sources; OOD v1 is single-source and reports one full lane.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from faster_whisper import WhisperModel


REPO_ROOT = Path(__file__).resolve().parents[4]
repo_root_text = str(REPO_ROOT)
# The benchmark may be launched from a machine that also keeps an archived
# workspace on PYTHONPATH.  Always make the checkout containing this scorer
# authoritative so the loader contract and the scorer cannot silently diverge.
sys.path = [entry for entry in sys.path if entry != repo_root_text]
sys.path.insert(0, repo_root_text)

from stable_audio_tools.training.metrics.fad_metrics import (
    load_clap_model,
    load_panns_model,
    load_vggish_model,
)
from scripts.t2a.eval.baselines.score_p10_60k_cross_system import (
    _atomic_json,
    _atomic_text,
    _edit_distance,
    _kl,
    _load_utmos,
    _normalize_words,
    _summary,
    _transcribe,
    _utmos_score,
)
from scripts.t2a.eval.baselines.score_p10_v11_stratified_3000_public_baselines import (  # noqa: E501
    _feature_space_frechet,
)


DEFAULT_WHISPER = Path(
    os.environ.get("AMBIT_DATA_ROOT", "data") + "/sceneplan_v2_1p124m/models/"
    "faster-distil-whisper-large-v3"
)
REFERENCE_ID = "ground_truth"
OURS_ID = "ours_p10_150k"
DOMAINS = ("music", "sound", "speech")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _speech_source(scene_plan: dict[str, Any]) -> dict[str, Any] | None:
    sources = [source for source in scene_plan["sources"] if source["kind"] == "speech"]
    if len(sources) > 1:
        raise RuntimeError("a frozen ScenePlan has more than one formal Speech source")
    return sources[0] if sources else None


def _load_inputs(root: Path, benchmark_kind: str) -> dict[str, Any]:
    contract_path = (root / "BENCHMARK_CONTRACT.json").resolve(strict=True)
    manifest_path = (root / "generation_requests.jsonl").resolve(strict=True)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    requests = _read_jsonl(manifest_path)
    if len(requests) != int(contract["generation_request_count"]):
        raise RuntimeError("baseline request count changed")
    if _sha256(manifest_path) != contract["generation_manifest_sha256"]:
        raise RuntimeError("baseline generation manifest SHA256 changed")
    (root / "BASELINE_GENERATION_COMPLETE").resolve(strict=True)

    if benchmark_kind == "internal8k":
        if contract["schema"] != "sceneplan_foa.p10_final_8000_public_baseline_contract":
            raise RuntimeError("benchmark root is not the frozen internal 8k contract")
        source_eval = Path(contract["source_eval_root"]).resolve(strict=True)
        (source_eval / "INFERENCE_COMPLETE").resolve(strict=True)
        panel_path = Path(contract["source_panel_path"]).resolve(strict=True)
        if _sha256(panel_path) != contract["source_panel_sha256"]:
            raise RuntimeError("internal panel SHA256 changed")
        panel = _read_jsonl(panel_path)
        if len(panel) != 8000:
            raise RuntimeError("internal panel row count changed")
    elif benchmark_kind == "ood3k":
        if contract["schema"] != "sceneplan_foa.p10_ood_3000_benchmark_contract":
            raise RuntimeError("benchmark root is not the frozen OOD 3k contract")
        (root / "P10_GENERATION_COMPLETE").resolve(strict=True)
        panel_path = Path(contract["source_panel_path"]).resolve(strict=True)
        if _sha256(panel_path) != contract["source_panel_sha256"]:
            raise RuntimeError("OOD panel SHA256 changed")
        panel = _read_jsonl(panel_path)
        if len(panel) != 3000:
            raise RuntimeError("OOD panel row count changed")
        source_eval = None
    else:
        raise ValueError(f"unsupported benchmark kind: {benchmark_kind}")

    display_names = {
        REFERENCE_ID: "Ground truth",
        OURS_ID: "ScenePlan DiT (ours, 150k)",
    }
    conditioning = {
        REFERENCE_ID: "reference",
        OURS_ID: "ScenePlan",
    }
    for baseline in contract["baselines"]:
        display_names[baseline["id"]] = baseline["display_name"]
        conditioning[baseline["id"]] = "raw text"

    panel_meta: dict[str, dict[str, Any]] = {}
    paths: dict[str, dict[str, str]] = defaultdict(dict)
    domain_system_ids: dict[str, dict[str, set[str]]] = {
        domain: defaultdict(set) for domain in DOMAINS
    }
    for row in panel:
        panel_id = str(row["panel_id"])
        if panel_id in panel_meta:
            raise RuntimeError(f"duplicate panel ID: {panel_id}")
        if benchmark_kind == "internal8k":
            source_kinds = tuple(str(value) for value in row["source_kinds"])
            reference_path = str(Path(row["reference_foa_path"]).resolve(strict=True))
            reference_sha = row["reference_foa_sha256"]
            speech = _speech_source(row["scene_plan"])
            ours_meta_path = (
                source_eval
                / "outputs"
                / "step_150000"
                / row["domain"]
                / panel_id
                / "metadata.json"
            ).resolve(strict=True)
            ours_meta = json.loads(ours_meta_path.read_text(encoding="utf-8"))
            if not (
                ours_meta.get("status") == "PASS"
                and ours_meta.get("panel_id") == panel_id
                and ours_meta.get("reference_foa_sha256") == reference_sha
            ):
                raise RuntimeError(f"invalid internal P10 metadata: {ours_meta_path}")
            ours_path = str(Path(ours_meta["generated_foa_path"]).resolve(strict=True))
            meta = {
                "panel_id": panel_id,
                "source_count": int(row["source_count"]),
                "source_kinds": source_kinds,
                "semantic_prompt": row["semantic_text"],
                "reference_path": reference_path,
                "reference_sha256": reference_sha,
                "transcript": None if speech is None else speech["transcript"],
                "speech_seen_speaker": row.get("speech_seen_speaker"),
                "length_bucket": int(row["length_bucket"]),
            }
        else:
            source_kinds = (str(row["domain"]),)
            reference_path = str(Path(row["reference_audio_path"]).resolve(strict=True))
            reference_sha = row["reference_audio_sha256"]
            ours_meta_path = (
                root / "outputs" / OURS_ID / panel_id / "generation.json"
            ).resolve(strict=True)
            ours_meta = json.loads(ours_meta_path.read_text(encoding="utf-8"))
            if not (
                ours_meta.get("status") == "PASS"
                and ours_meta.get("panel_id") == panel_id
                and ours_meta.get("reference_audio_sha256") == reference_sha
            ):
                raise RuntimeError(f"invalid OOD P10 metadata: {ours_meta_path}")
            ours_path = str(Path(ours_meta["quality_w_path"]).resolve(strict=True))
            meta = {
                "panel_id": panel_id,
                "source_count": 1,
                "source_kinds": source_kinds,
                "semantic_prompt": row["semantic_text"],
                "reference_path": reference_path,
                "reference_sha256": reference_sha,
                "transcript": row.get("exact_transcript"),
                "speech_seen_speaker": None,
                "length_bucket": 432,
            }
        panel_meta[panel_id] = meta
        paths[REFERENCE_ID][panel_id] = reference_path
        paths[OURS_ID][panel_id] = ours_path
        for domain in set(source_kinds):
            domain_system_ids[domain][REFERENCE_ID].add(panel_id)
            domain_system_ids[domain][OURS_ID].add(panel_id)

    for request in requests:
        baseline_id = str(request["baseline_id"])
        panel_id = str(request["panel_id"])
        if panel_id not in panel_meta:
            raise RuntimeError(f"baseline request is outside panel: {panel_id}")
        path = str(Path(request["quality_w_path"]).resolve(strict=True))
        previous = paths[baseline_id].setdefault(panel_id, path)
        if previous != path:
            raise RuntimeError(f"inconsistent baseline output path: {baseline_id}/{panel_id}")
        for domain in request["score_domains"]:
            if domain not in DOMAINS:
                raise RuntimeError(f"unknown score domain: {domain}")
            domain_system_ids[domain][baseline_id].add(panel_id)

    all_domain_ids = {
        domain: {
            panel_id
            for panel_id, meta in panel_meta.items()
            if domain in meta["source_kinds"]
        }
        for domain in DOMAINS
    }
    qwen_id = "qwen3_tts_1p7b_voice_design"
    for domain in DOMAINS:
        for system_id, panel_ids in domain_system_ids[domain].items():
            if system_id == qwen_id and benchmark_kind == "internal8k":
                expected = {
                    panel_id
                    for panel_id in all_domain_ids["speech"]
                    if panel_meta[panel_id]["source_count"] == 1
                }
            else:
                expected = all_domain_ids[domain]
            if panel_ids != expected:
                raise RuntimeError(
                    f"incomplete compatible lane {domain}/{system_id}: "
                    f"{len(panel_ids)} != {len(expected)}"
                )

    expected_domains = {
        "music": 4013,
        "sound": 4013,
        "speech": 5000,
    } if benchmark_kind == "internal8k" else {domain: 1000 for domain in DOMAINS}
    actual_domains = {domain: len(ids) for domain, ids in all_domain_ids.items()}
    if actual_domains != expected_domains:
        raise RuntimeError(f"domain counts changed: {actual_domains}")

    return {
        "benchmark_kind": benchmark_kind,
        "root": root,
        "contract": contract,
        "contract_path": contract_path,
        "manifest_path": manifest_path,
        "panel_path": panel_path,
        "panel_meta": panel_meta,
        "paths": dict(paths),
        "domain_system_ids": domain_system_ids,
        "all_domain_ids": all_domain_ids,
        "display_names": display_names,
        "conditioning": conditioning,
    }


def _lineage(inputs: dict[str, Any], arm: str) -> dict[str, Any]:
    return {
        "schema": "sceneplan_foa.p10_final_content_metric_partial",
        "schema_version": 1,
        "status": "PASS",
        "benchmark_kind": inputs["benchmark_kind"],
        "arm": arm,
        "contract_sha256": _sha256(inputs["contract_path"]),
        "manifest_sha256": _sha256(inputs["manifest_path"]),
        "panel_sha256": _sha256(inputs["panel_path"]),
    }


def _strata(inputs: dict[str, Any]) -> tuple[str, ...]:
    return ("all", "source_1", "source_2", "source_3", "source_4") if inputs[
        "benchmark_kind"
    ] == "internal8k" else ("all",)


def _stratum_ids(
    inputs: dict[str, Any], domain: str, system_id: str, stratum: str
) -> list[str]:
    panel_ids = inputs["domain_system_ids"][domain][system_id]
    if stratum == "all":
        return sorted(panel_ids)
    source_count = int(stratum.rsplit("_", 1)[1])
    return sorted(
        panel_id
        for panel_id in panel_ids
        if inputs["panel_meta"][panel_id]["source_count"] == source_count
    )


def _quality_items(inputs: dict[str, Any]) -> dict[str, str]:
    items: dict[str, str] = {}
    for domain in ("music", "sound"):
        for system_id, panel_ids in inputs["domain_system_ids"][domain].items():
            for panel_id in panel_ids:
                key = f"{system_id}:{panel_id}"
                path = inputs["paths"][system_id][panel_id]
                previous = items.setdefault(key, path)
                if previous != path:
                    raise RuntimeError(f"quality path changed for {key}")
    return items


def _load_metric_mono(
    inputs: dict[str, Any],
    key: str,
    path: str | Path,
    target_rate: int,
    *,
    target_samples: int | None = None,
) -> torch.Tensor:
    """Load the frozen content view without guessing from channel count.

    Native internal P10/reference files are contracted WYZX FOA and therefore
    use W.  Public outputs and OOD natural references are content recordings,
    so every available channel is arithmetic-mean downmixed.  In particular,
    a four-channel natural recording must not be mistaken for FOA merely from
    its shape, and MusicCaps may legitimately contain six-channel material.
    """

    system_id = key.split(":", 1)[0]
    audio, sample_rate = torchaudio.load(str(path))
    native_foa_system_ids = set(
        inputs.get("native_foa_system_ids", {REFERENCE_ID, OURS_ID})
    )
    native_foa = (
        inputs["benchmark_kind"] == "internal8k"
        and system_id in native_foa_system_ids
    )
    if native_foa:
        if int(audio.shape[0]) != 4:
            raise RuntimeError(
                f"contracted native FOA must have four WYZX channels: {path} {audio.shape}"
            )
        mono = audio[:1]
    else:
        if int(audio.shape[0]) < 1:
            raise RuntimeError(f"content input has no audio channels: {path} {audio.shape}")
        mono = audio.mean(dim=0, keepdim=True)
    peak = mono.abs().max().clamp_min(1.0e-8)
    mono = mono / peak * (10.0 ** (-1.0 / 20.0))
    if sample_rate != target_rate:
        mono = torchaudio.functional.resample(mono, sample_rate, target_rate)
    if target_samples is not None:
        if int(mono.shape[-1]) < target_samples:
            mono = F.pad(mono, (0, target_samples - int(mono.shape[-1])))
        else:
            mono = mono[:, :target_samples]
    return mono.clamp(-1.0, 1.0)


def _clap_windows(inputs: dict[str, Any], key: str, path: str) -> torch.Tensor:
    waveform = _load_metric_mono(inputs, key, path, 48_000)[0]
    window = 480_000
    if int(waveform.numel()) <= window:
        return F.pad(waveform, (0, window - int(waveform.numel())))[None]
    starts = list(range(0, int(waveform.numel()) - window + 1, window))
    tail = int(waveform.numel()) - window
    if tail not in starts:
        starts.append(tail)
    return torch.stack([waveform[start : start + window] for start in starts])


@torch.inference_mode()
def _score_clap(inputs: dict[str, Any], device: torch.device) -> dict[str, Any]:
    items = sorted(_quality_items(inputs).items())
    model = load_clap_model("630k-audioset-fusion-best.pt", device=str(device))
    embeddings: dict[str, torch.Tensor] = {}
    for start in range(0, len(items), 8):
        batch = items[start : start + 8]
        windows: list[torch.Tensor] = []
        owners: list[str] = []
        for key, path in batch:
            values = _clap_windows(inputs, key, path)
            windows.extend(value for value in values)
            owners.extend([key] * int(values.shape[0]))
        tensor = torch.stack(windows).to(device)
        values = model.get_audio_embedding_from_data(
            x=tensor, use_tensor=True
        ).float()
        values = F.normalize(values, dim=-1).cpu()
        by_owner: dict[str, list[torch.Tensor]] = defaultdict(list)
        for owner, value in zip(owners, values):
            by_owner[owner].append(value)
        for key, _ in batch:
            embeddings[key] = F.normalize(
                torch.stack(by_owner[key]).mean(dim=0), dim=0
            )
        print(
            json.dumps(
                {"event": "clap_audio", "completed": min(start + 8, len(items)), "total": len(items)}
            ),
            flush=True,
        )

    captions = {
        panel_id: meta["semantic_prompt"]
        for panel_id, meta in inputs["panel_meta"].items()
        if any(domain in meta["source_kinds"] for domain in ("music", "sound"))
    }
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
    text_values = model.get_text_embedding(
        [captions[panel_id] for panel_id in caption_ids], **text_kwargs
    ).float()
    text_values = F.normalize(text_values, dim=-1).cpu()
    text_embeddings = dict(zip(caption_ids, text_values))
    del model
    gc.collect()
    torch.cuda.empty_cache()

    domains: dict[str, Any] = {}
    for domain in ("music", "sound"):
        domains[domain] = {}
        for system_id in sorted(inputs["domain_system_ids"][domain]):
            result = {
                "display_name": inputs["display_names"][system_id],
                "conditioning": inputs["conditioning"][system_id],
                "strata": {},
            }
            for stratum in _strata(inputs):
                panel_ids = _stratum_ids(inputs, domain, system_id, stratum)
                if not panel_ids:
                    result["strata"][stratum] = None
                    continue
                text_scores = [
                    float(embeddings[f"{system_id}:{panel_id}"] @ text_embeddings[panel_id])
                    for panel_id in panel_ids
                ]
                paired_scores = [
                    float(
                        embeddings[f"{system_id}:{panel_id}"]
                        @ embeddings[f"{REFERENCE_ID}:{panel_id}"]
                    )
                    for panel_id in panel_ids
                ]
                result["strata"][stratum] = {
                    "rows": len(panel_ids),
                    "clap": _summary(text_scores),
                    "paired_reference_clap": _summary(paired_scores),
                }
            domains[domain][system_id] = result
    return {
        **_lineage(inputs, "clap"),
        "domains": domains,
        "protocol": (
            "LAION-CLAP 630k AudioSet fusion; internal native FOA uses W, "
            "while public and OOD content recordings use an arithmetic-mean "
            "downmix across all available channels; "
            "clips longer than 10 s use normalized mean embeddings of the "
            "first and tail-aligned 10 s windows."
        ),
    }


@torch.inference_mode()
def _score_vggish(inputs: dict[str, Any], device: torch.device) -> dict[str, Any]:
    items = sorted(_quality_items(inputs).items())
    model, processor, backend = load_vggish_model(str(device))
    if backend != "torchaudio" or processor is None:
        raise RuntimeError("final benchmark requires torchaudio VGGish preprocessing")
    embeddings: dict[str, torch.Tensor] = {}
    for index, (key, path) in enumerate(items, start=1):
        waveform = _load_metric_mono(inputs, key, path, 16_000)[0]
        examples = processor(waveform)
        chunks = [
            model(examples[start : start + 32].to(device)).float().cpu()
            for start in range(0, int(examples.shape[0]), 32)
        ]
        embeddings[key] = torch.cat(chunks, dim=0)
        if index % 50 == 0 or index == len(items):
            print(json.dumps({"event": "vggish", "completed": index, "total": len(items)}), flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()

    domains: dict[str, Any] = {}
    for domain in ("music", "sound"):
        domains[domain] = {}
        for system_id in sorted(inputs["domain_system_ids"][domain]):
            result = {
                "display_name": inputs["display_names"][system_id],
                "conditioning": inputs["conditioning"][system_id],
                "strata": {},
            }
            for stratum in _strata(inputs):
                panel_ids = _stratum_ids(inputs, domain, system_id, stratum)
                if not panel_ids:
                    result["strata"][stratum] = None
                    continue
                reference = torch.cat(
                    [embeddings[f"{REFERENCE_ID}:{panel_id}"] for panel_id in panel_ids]
                )
                generated = torch.cat(
                    [embeddings[f"{system_id}:{panel_id}"] for panel_id in panel_ids]
                )
                result["strata"][stratum] = {
                    "rows": len(panel_ids),
                    "fad_vggish": _feature_space_frechet(generated, reference),
                }
            domains[domain][system_id] = result
    return {
        **_lineage(inputs, "vggish"),
        "domains": domains,
        "protocol": (
            "VGGish FAD on the complete variable-duration content view; "
            "internal native FOA uses W and other recordings use an "
            "arithmetic-mean downmix across all available channels."
        ),
    }


@torch.inference_mode()
def _score_panns(inputs: dict[str, Any], device: torch.device) -> dict[str, Any]:
    items = sorted(_quality_items(inputs).items())
    model = load_panns_model(str(device))
    probabilities: dict[str, torch.Tensor] = {}
    target_seconds = 15.1 if inputs["benchmark_kind"] == "internal8k" else 10.0
    target_samples = int(round(target_seconds * 32_000))
    for start in range(0, len(items), 8):
        batch = items[start : start + 8]
        waves = torch.cat(
            [
                _load_metric_mono(
                    inputs,
                    key,
                    path,
                    32_000,
                    target_samples=target_samples,
                ).to(device)
                for key, path in batch
            ]
        )
        embedding = model(waves).float()
        probability = torch.softmax(model.fc_audioset(embedding).float(), dim=-1)
        for (key, _), value in zip(batch, probability):
            probabilities[key] = value.cpu()
        print(
            json.dumps({"event": "panns", "completed": min(start + 8, len(items)), "total": len(items)}),
            flush=True,
        )
    del model
    gc.collect()
    torch.cuda.empty_cache()

    domains: dict[str, Any] = {}
    for domain in ("music", "sound"):
        domains[domain] = {}
        for system_id in sorted(inputs["domain_system_ids"][domain]):
            result = {
                "display_name": inputs["display_names"][system_id],
                "conditioning": inputs["conditioning"][system_id],
                "strata": {},
            }
            for stratum in _strata(inputs):
                panel_ids = _stratum_ids(inputs, domain, system_id, stratum)
                if not panel_ids:
                    result["strata"][stratum] = None
                    continue
                values = [
                    _kl(
                        probabilities[f"{REFERENCE_ID}:{panel_id}"],
                        probabilities[f"{system_id}:{panel_id}"],
                    )
                    for panel_id in panel_ids
                ]
                result["strata"][stratum] = {
                    "rows": len(panel_ids),
                    "kl_pann": _summary(values),
                }
            domains[domain][system_id] = result
    return {
        **_lineage(inputs, "panns"),
        "domains": domains,
        "protocol": (
            "Paired KL between PANN AudioSet softmax posteriors on the full "
            f"content envelope padded to {target_seconds:.1f} s; internal "
            "native FOA uses W and other recordings use an arithmetic-mean "
            "downmix across all available channels."
        ),
    }


def _speech_tasks(inputs: dict[str, Any]) -> list[dict[str, Any]]:
    output = []
    for system_id in sorted(inputs["domain_system_ids"]["speech"]):
        for panel_id in sorted(inputs["domain_system_ids"]["speech"][system_id]):
            meta = inputs["panel_meta"][panel_id]
            if not meta.get("transcript"):
                raise RuntimeError(f"Speech row has no transcript: {panel_id}")
            output.append(
                {
                    "system_id": system_id,
                    "panel_id": panel_id,
                    "path": inputs["paths"][system_id][panel_id],
                    "transcript": meta["transcript"],
                    "source_count": meta["source_count"],
                    "speech_seen_speaker": meta.get("speech_seen_speaker"),
                    "length_bucket": meta["length_bucket"],
                }
            )
    return output


def _score_speech_shard(
    inputs: dict[str, Any], *, device_index: int, shard_index: int, num_shards: int
) -> dict[str, Any]:
    tasks = [
        task
        for index, task in enumerate(_speech_tasks(inputs))
        if index % num_shards == shard_index
    ]
    model = WhisperModel(
        str(DEFAULT_WHISPER.resolve(strict=True)),
        device="cuda",
        device_index=device_index,
        compute_type="float16",
    )
    device = torch.device(f"cuda:{device_index}")
    utmos, utmos_error = _load_utmos(device)
    outputs = []
    for index, task in enumerate(tasks, start=1):
        waveform = _load_metric_mono(
            inputs,
            f"{task['system_id']}:{task['panel_id']}",
            task["path"],
            16_000,
        )[0].numpy().astype(np.float32)
        transcription = _transcribe(model, waveform)
        reference_words = _normalize_words(task["transcript"])
        hypothesis_words = _normalize_words(transcription)
        reference_chars = list("".join(reference_words))
        hypothesis_chars = list("".join(hypothesis_words))
        word_errors = _edit_distance(reference_words, hypothesis_words)
        char_errors = _edit_distance(reference_chars, hypothesis_chars)
        outputs.append(
            {
                **{key: task[key] for key in ("system_id", "panel_id", "source_count", "speech_seen_speaker", "length_bucket")},
                "word_errors": word_errors,
                "reference_words": len(reference_words),
                "char_errors": char_errors,
                "reference_chars": len(reference_chars),
                "wer": word_errors / max(len(reference_words), 1),
                "cer": char_errors / max(len(reference_chars), 1),
                "utmos": _utmos_score(utmos, waveform, device) if utmos is not None else None,
                "asr": transcription,
                "reference": task["transcript"],
            }
        )
        if index % 10 == 0 or index == len(tasks):
            print(
                json.dumps(
                    {
                        "event": "speech",
                        "shard": shard_index,
                        "completed": index,
                        "total": len(tasks),
                    }
                ),
                flush=True,
            )
    return {
        **_lineage(inputs, "speech"),
        "shard_index": shard_index,
        "num_shards": num_shards,
        "tasks": len(tasks),
        "outputs": outputs,
        "utmos_error": utmos_error,
        "protocol": "faster-distil-whisper-large-v3 corpus WER/CER and UTMOS22-strong on full waveforms.",
    }


def _aggregate_speech(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"rows": 0, "corpus_wer": None, "corpus_cer": None, "utmos": _summary([])}
    return {
        "rows": len(rows),
        "corpus_wer": sum(row["word_errors"] for row in rows)
        / max(sum(row["reference_words"] for row in rows), 1),
        "corpus_cer": sum(row["char_errors"] for row in rows)
        / max(sum(row["reference_chars"] for row in rows), 1),
        "utterance_wer": _summary([row["wer"] for row in rows]),
        "utterance_cer": _summary([row["cer"] for row in rows]),
        "utmos": _summary([row["utmos"] for row in rows]),
    }


def _fmt(value: float | None, digits: int = 3) -> str:
    return "N/A" if value is None or not math.isfinite(float(value)) else f"{float(value):.{digits}f}"


def _content_cell(row: dict[str, Any] | None) -> str:
    if row is None:
        return "N/A"
    return "{}/{}/{}".format(
        _fmt(row["clap"]["mean"]),
        _fmt(row["fad_vggish"]),
        _fmt(row["kl_pann"]["mean"]),
    )


def _speech_cell(row: dict[str, Any] | None) -> str:
    if row is None or not row.get("rows"):
        return "N/A"
    return "{}/{}/{}".format(
        _fmt(row["corpus_wer"]),
        _fmt(row["corpus_cer"]),
        _fmt(row["utmos"]["mean"]),
    )


def _merge(inputs: dict[str, Any], root: Path, num_speech_shards: int) -> dict[str, Any]:
    partial = root / "metrics/final_content/partials"
    reports = {
        name: json.loads((partial / f"{name.upper()}.json").read_text(encoding="utf-8"))
        for name in ("clap", "vggish", "panns")
    }
    expected = {
        "contract_sha256": _sha256(inputs["contract_path"]),
        "manifest_sha256": _sha256(inputs["manifest_path"]),
        "panel_sha256": _sha256(inputs["panel_path"]),
    }
    for report in reports.values():
        for key, value in expected.items():
            if report[key] != value:
                raise RuntimeError(f"metric lineage mismatch: {report['arm']} {key}")

    audio_domains: dict[str, Any] = {}
    for domain in ("music", "sound"):
        systems = set(reports["clap"]["domains"][domain])
        if systems != set(reports["vggish"]["domains"][domain]) or systems != set(
            reports["panns"]["domains"][domain]
        ):
            raise RuntimeError(f"metric system mismatch: {domain}")
        audio_domains[domain] = {}
        for system_id in sorted(systems):
            strata = {}
            for stratum in _strata(inputs):
                clap = reports["clap"]["domains"][domain][system_id]["strata"][stratum]
                vggish = reports["vggish"]["domains"][domain][system_id]["strata"][stratum]
                panns = reports["panns"]["domains"][domain][system_id]["strata"][stratum]
                if clap is None or vggish is None or panns is None:
                    if any(value is not None for value in (clap, vggish, panns)):
                        raise RuntimeError(f"partial stratum metrics: {domain}/{system_id}/{stratum}")
                    strata[stratum] = None
                else:
                    if len({clap["rows"], vggish["rows"], panns["rows"]}) != 1:
                        raise RuntimeError(f"stratum row mismatch: {domain}/{system_id}/{stratum}")
                    strata[stratum] = {**clap, **vggish, **panns}
            audio_domains[domain][system_id] = {
                "display_name": inputs["display_names"][system_id],
                "conditioning": inputs["conditioning"][system_id],
                "strata": strata,
            }

    speech_outputs: list[dict[str, Any]] = []
    utmos_errors = []
    for shard in range(num_speech_shards):
        path = partial / f"SPEECH_SHARD_{shard:02d}_OF_{num_speech_shards:02d}.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        for key, value in expected.items():
            if report[key] != value:
                raise RuntimeError(f"Speech lineage mismatch: shard={shard} {key}")
        speech_outputs.extend(report["outputs"])
        if report.get("utmos_error"):
            utmos_errors.append(report["utmos_error"])
    expected_tasks = _speech_tasks(inputs)
    expected_pairs = {(task["system_id"], task["panel_id"]) for task in expected_tasks}
    observed_pairs = {(row["system_id"], row["panel_id"]) for row in speech_outputs}
    if len(speech_outputs) != len(expected_pairs) or observed_pairs != expected_pairs:
        raise RuntimeError("Speech scoring is incomplete or duplicated")
    speech: dict[str, Any] = {}
    for system_id in sorted(inputs["domain_system_ids"]["speech"]):
        system_rows = [row for row in speech_outputs if row["system_id"] == system_id]
        strata = {"all": _aggregate_speech(system_rows)}
        if inputs["benchmark_kind"] == "internal8k":
            for source_count in range(1, 5):
                chosen = [row for row in system_rows if int(row["source_count"]) == source_count]
                strata[f"source_{source_count}"] = (
                    _aggregate_speech(chosen) if chosen else None
                )
        speech[system_id] = {
            "display_name": inputs["display_names"][system_id],
            "conditioning": inputs["conditioning"][system_id],
            "strata": strata,
        }

    report = {
        "schema": "sceneplan_foa.p10_final_content_metrics",
        "schema_version": 1,
        "status": "PASS",
        "benchmark_kind": inputs["benchmark_kind"],
        **expected,
        "audio_domains": audio_domains,
        "speech": speech,
        "metric_protocols": {name: value["protocol"] for name, value in reports.items()},
        "speech_protocol": "faster-distil-whisper-large-v3 corpus WER/CER and UTMOS22-strong on full waveforms.",
        "utmos_errors": sorted(set(utmos_errors)),
        "conditioning_disclosure": (
            "Ours uses its native ScenePlan interface; public baselines receive raw "
            "source descriptions only. This is a native-interface capability comparison, "
            "not equal conditioning bandwidth."
        ),
        "spatial_disclosure": "Public mono/stereo baselines receive no spatial score.",
    }
    metric_root = root / "metrics/final_content"
    _atomic_json(metric_root / "CONTENT_METRICS.json", report)

    table_root = root / "tables/final"
    table_root.mkdir(parents=True, exist_ok=True)
    for domain in ("music", "sound"):
        lines = [
            f"# {domain.title()} content benchmark",
            "",
            "Each metric cell is `CLAP ↑ / FAD-VGGish ↓ / KL-PANN ↓`.",
            "",
        ]
        if inputs["benchmark_kind"] == "internal8k":
            lines.extend(
                [
                    "| System | Input | All | 1 source | 2 sources | 3 sources | 4 sources |",
                    "|---|---|---:|---:|---:|---:|---:|",
                ]
            )
            strata = ("all", "source_1", "source_2", "source_3", "source_4")
        else:
            lines.extend(["| System | Input | N | CLAP ↑ | FAD ↓ | KL ↓ |", "|---|---|---:|---:|---:|---:|"])
            strata = ("all",)
        for system_id, value in audio_domains[domain].items():
            if inputs["benchmark_kind"] == "internal8k":
                cells = [_content_cell(value["strata"][stratum]) for stratum in strata]
                lines.append(
                    f"| {value['display_name']} | {value['conditioning']} | "
                    + " | ".join(cells)
                    + " |"
                )
            else:
                row = value["strata"]["all"]
                lines.append(
                    "| {name} | {cond} | {n} | {clap} | {fad} | {kl} |".format(
                        name=value["display_name"], cond=value["conditioning"], n=row["rows"],
                        clap=_fmt(row["clap"]["mean"]), fad=_fmt(row["fad_vggish"]),
                        kl=_fmt(row["kl_pann"]["mean"]),
                    )
                )
        lines.extend(["", report["conditioning_disclosure"], ""])
        _atomic_text(table_root / f"{domain.upper()}_CONTENT_TABLE.md", "\n".join(lines))

    lines = ["# Speech content benchmark", ""]
    if inputs["benchmark_kind"] == "internal8k":
        lines.extend(
            [
                "Each metric cell is `WER ↓ / CER ↓ / UTMOS ↑`.",
                "",
                "| System | Input | All | 1 source | 2 sources | 3 sources | 4 sources |",
                "|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for value in speech.values():
            cells = [
                _speech_cell(value["strata"].get(stratum))
                for stratum in ("all", "source_1", "source_2", "source_3", "source_4")
            ]
            lines.append(
                f"| {value['display_name']} | {value['conditioning']} | "
                + " | ".join(cells)
                + " |"
            )
    else:
        lines.extend(["| System | Input | N | WER ↓ | CER ↓ | UTMOS ↑ |", "|---|---|---:|---:|---:|---:|"])
        for value in speech.values():
            row = value["strata"]["all"]
            lines.append(
                "| {name} | {cond} | {n} | {wer} | {cer} | {utmos} |".format(
                    name=value["display_name"], cond=value["conditioning"], n=row["rows"],
                    wer=_fmt(row["corpus_wer"]), cer=_fmt(row["corpus_cer"]),
                    utmos=_fmt(row["utmos"]["mean"]),
                )
            )
    lines.extend(["", report["conditioning_disclosure"], ""])
    _atomic_text(table_root / "SPEECH_CONTENT_TABLE.md", "\n".join(lines))
    (metric_root / "CONTENT_METRICS_COMPLETE").write_text("PASS\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--benchmark-kind", choices=("internal8k", "ood3k"), required=True)
    parser.add_argument("--arm", choices=("clap", "vggish", "panns", "speech", "merge"), required=True)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()

    root = args.benchmark_root.expanduser().resolve(strict=True)
    inputs = _load_inputs(root, args.benchmark_kind)
    partial = root / "metrics/final_content/partials"
    partial.mkdir(parents=True, exist_ok=True)
    if args.arm == "merge":
        report = _merge(inputs, root, args.num_shards)
        print(json.dumps({"status": report["status"], "metrics": str(root / "metrics/final_content/CONTENT_METRICS.json")}))
        return 0
    if not torch.cuda.is_available():
        raise RuntimeError("final content scoring requires CUDA")
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
        output = partial / f"SPEECH_SHARD_{args.shard_index:02d}_OF_{args.num_shards:02d}.json"
    _atomic_json(output, report)
    print(json.dumps({"status": "PASS", "arm": args.arm, "output": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
