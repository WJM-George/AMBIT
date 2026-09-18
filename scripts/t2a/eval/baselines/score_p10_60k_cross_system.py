#!/usr/bin/env python3
"""Score runnable baselines and P10 checkpoints on one frozen matched panel.

The quality lane is mono: FOA systems use W, mono systems are unchanged, and
stereo systems use the precomputed arithmetic-mean quality view. Spatial scores
are copied only from the native-FOA evaluator; mono/stereo baselines remain N/A.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from faster_whisper import WhisperModel

from stable_audio_tools.training.metrics.fad_metrics import (
    load_clap_model,
    load_panns_model,
    load_vggish_model,
)


DEFAULT_BENCHMARK = Path(os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/baselines/p10_60k_15row_v1")
DEFAULT_SOURCE = Path(
    os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/archives/"
    "sceneplan_dit_v7_sao_300m_from_scratch_160k/evaluation/"
    "p10_gt_vae_20k_40k_60k_instrumental_music_v1"
)
DEFAULT_WHISPER = Path(
    os.environ.get("AMBIT_DATA_ROOT", "data") + "/sceneplan_v2_1p124m/models/"
    "faster-distil-whisper-large-v3"
)
REFERENCE_ID = "ground_truth"
VAE_ID = "vae_codec_ceiling"


DISPLAY_NAMES = {
    REFERENCE_ID: "Ground truth",
    VAE_ID: "FOA VAE reconstruction (ceiling)",
    "stable_audio_open_1_0": "Stable Audio Open 1.0",
    "tangoflux": "TangoFlux",
    "audiox_turbo": "AudioX-Turbo",
    "mmaudio_large_44k_v2_text_only": "MMAudio-L v2 (text-only)",
    "woosh_flow": "Woosh-Flow",
    "qwen3_tts_1p7b_voice_design": "Qwen3-TTS 1.7B VoiceDesign",
}


def _ours_id(step: int) -> str:
    return f"ours_sceneplan_foa_{int(step)}"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _summary(values: list[float | None]) -> dict[str, Any]:
    finite = [float(value) for value in values if value is not None and np.isfinite(value)]
    if not finite:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "std": None,
            "mean_ci95_low": None,
            "mean_ci95_high": None,
            "mean_ci95_method": None,
        }
    values_array = np.asarray(finite, dtype=np.float64)
    if len(finite) > 1:
        rng = np.random.default_rng(20260824)
        bootstrap = values_array[
            rng.integers(0, len(finite), size=(10_000, len(finite)))
        ].mean(axis=1)
        ci_low, ci_high = np.quantile(bootstrap, (0.025, 0.975))
        std = float(values_array.std(ddof=1))
    else:
        ci_low = ci_high = values_array[0]
        std = None
    return {
        "count": len(finite),
        "mean": float(values_array.mean()),
        "median": float(np.median(values_array)),
        "min": float(values_array.min()),
        "max": float(values_array.max()),
        "std": std,
        "mean_ci95_low": float(ci_low),
        "mean_ci95_high": float(ci_high),
        "mean_ci95_method": "deterministic percentile bootstrap, 10000 resamples",
    }


def _load_quality_mono(
    path: str | Path,
    target_rate: int,
    *,
    target_samples: int | None = None,
) -> torch.Tensor:
    audio, sample_rate = torchaudio.load(str(path))
    if int(audio.shape[0]) == 4:
        # Native FOA is stored as WYZX, so its mono content view is W.
        mono = audio[:1]
    elif int(audio.shape[0]) == 2:
        # Public stereo systems and natural OOD references use the frozen
        # arithmetic-mean downmix contract.  Do not select only one channel:
        # doing so would make paired quality metrics depend on channel order.
        mono = audio.mean(dim=0, keepdim=True)
    elif int(audio.shape[0]) == 1:
        mono = audio
    else:
        raise RuntimeError(
            "quality input must be mono, stereo, or native four-channel FOA: "
            f"{path} {audio.shape}"
        )
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


def _low_rank_frechet(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.to(torch.float64)
    right = right.to(torch.float64)
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1]:
        raise ValueError("FD inputs must be [N,D]")
    if left.shape[0] < 2 or right.shape[0] < 2:
        raise ValueError("FD requires at least two observations per side")
    left_centered = (left - left.mean(dim=0)) / (left.shape[0] - 1) ** 0.5
    right_centered = (right - right.mean(dim=0)) / (right.shape[0] - 1) ** 0.5
    mean_delta = (left.mean(dim=0) - right.mean(dim=0)).square().sum()
    cross = left_centered @ right_centered.transpose(0, 1)
    value = (
        mean_delta
        + left_centered.square().sum()
        + right_centered.square().sum()
        - 2.0 * torch.linalg.svdvals(cross).sum()
    )
    return float(value.clamp_min(0.0))


def _kl(reference: torch.Tensor, generated: torch.Tensor) -> float:
    reference = reference.to(torch.float64).clamp_min(1.0e-12)
    generated = generated.to(torch.float64).clamp_min(1.0e-12)
    return float((reference * (reference.log() - generated.log())).sum())


def _normalize_words(value: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", value.lower())


def _edit_distance(left: list[str], right: list[str]) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_value in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_value in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_value != right_value),
                )
            )
        previous = current
    return previous[-1]


def _error_rates(hypothesis: str, reference: str) -> tuple[float, float]:
    reference_words = _normalize_words(reference)
    hypothesis_words = _normalize_words(hypothesis)
    reference_chars = list("".join(reference_words))
    hypothesis_chars = list("".join(hypothesis_words))
    return (
        _edit_distance(reference_words, hypothesis_words) / max(len(reference_words), 1),
        _edit_distance(reference_chars, hypothesis_chars) / max(len(reference_chars), 1),
    )


def _transcribe(model: WhisperModel, waveform: np.ndarray) -> str:
    segments, _ = model.transcribe(
        waveform,
        language="en",
        beam_size=5,
        condition_on_previous_text=False,
        vad_filter=False,
    )
    return " ".join(segment.text.strip() for segment in segments).strip()


def _load_utmos(device: torch.device):
    try:
        model = torch.hub.load(
            "tarepan/SpeechMOS:v1.2.0",
            "utmos22_strong",
            trust_repo=True,
            verbose=False,
        )
        return model.to(device).eval(), None
    except Exception as error:
        return None, f"{type(error).__name__}: {error}"


@torch.inference_mode()
def _utmos_score(model, waveform: np.ndarray, device: torch.device) -> float:
    value = torch.from_numpy(waveform).float().to(device).view(1, -1)
    return float(torch.as_tensor(model(value, 16_000)).float().mean().cpu())


def _quality_sources(
    benchmark: Path, source: Path
) -> tuple[
    list[dict[str, Any]],
    dict[str, dict[str, dict[str, str]]],
    list[str],
    dict[str, str],
]:
    contract = json.loads(
        (benchmark / "BENCHMARK_CONTRACT.json").read_text(encoding="utf-8")
    )
    panel_path = Path(contract["source_panel_path"]).expanduser().resolve(strict=True)
    panel = _read_jsonl(panel_path)
    requests = _read_jsonl(benchmark / "generation_requests.jsonl")
    checkpoints = contract.get("ours_checkpoints")
    if checkpoints is None:
        checkpoints = [{"step": int(contract["ours_checkpoint_step"])}]
    steps = [int(row["step"]) for row in checkpoints]
    ours_ids = [_ours_id(step) for step in steps]
    display_names = dict(DISPLAY_NAMES)
    display_names.update(
        {_ours_id(step): f"Ours ScenePlan-FOA ({step // 1000}k)" for step in steps}
    )
    display_names.update(
        {
            row["id"]: row["display_name"]
            for row in contract.get("baselines", [])
        }
    )
    sources: dict[str, dict[str, dict[str, str]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for row in panel:
        domain = row["domain"]
        panel_id = row["panel_id"]
        vae_metadata = json.loads(
            (source / "vae_reconstruction" / panel_id / "metadata.json").read_text(
                encoding="utf-8"
            )
        )
        for step in steps:
            sources[domain][_ours_id(step)][panel_id] = str(
                source
                / "outputs"
                / f"step_{step:06d}"
                / domain
                / panel_id
                / "generated_foa_float32.wav"
            )
        sources[domain][REFERENCE_ID][panel_id] = row["reference_foa_path"]
        sources[domain][VAE_ID][panel_id] = vae_metadata["reconstruction_foa_path"]
    for row in requests:
        quality_path = Path(row["quality_w_path"])
        if not quality_path.is_file():
            raise FileNotFoundError(f"missing baseline output: {quality_path}")
        sources[row["domain"]][row["baseline_id"]][row["panel_id"]] = str(
            quality_path
        )
    return panel, sources, ours_ids, display_names


@torch.inference_mode()
def _score_audio_domains(
    panel: list[dict[str, Any]],
    sources: dict[str, dict[str, dict[str, str]]],
    device: torch.device,
    display_names: dict[str, str],
) -> dict[str, Any]:
    rows = [row for row in panel if row["domain"] in {"music", "sound"}]
    reference_items = {
        row["panel_id"]: row["reference_foa_path"] for row in rows
    }
    all_items: list[tuple[str, str]] = [
        (f"reference:{panel_id}", path) for panel_id, path in reference_items.items()
    ]
    for domain in ("music", "sound"):
        for system_id, items in sources[domain].items():
            if system_id == REFERENCE_ID:
                continue
            all_items.extend(
                (f"system:{system_id}:{panel_id}", path)
                for panel_id, path in items.items()
            )

    clap = load_clap_model("630k-audioset-fusion-best.pt", device=str(device))
    clap_audio: dict[str, torch.Tensor] = {}
    for start in range(0, len(all_items), 8):
        batch = all_items[start : start + 8]
        waves = torch.cat(
            [
                _load_quality_mono(path, 48_000, target_samples=480_000).to(device)
                for _, path in batch
            ],
            dim=0,
        )
        embeddings = clap.get_audio_embedding_from_data(x=waves, use_tensor=True).float()
        embeddings = F.normalize(embeddings, dim=-1).cpu()
        for (key, _), embedding in zip(batch, embeddings):
            clap_audio[key] = embedding
        print(
            json.dumps(
                {"event": "cross_clap", "completed": min(start + len(batch), len(all_items)), "total": len(all_items)}
            ),
            flush=True,
        )
    captions = {row["panel_id"]: row["semantic_text"] for row in rows}
    caption_ids = list(captions)
    text_kwargs: dict[str, Any] = {"use_tensor": True}
    if callable(getattr(clap, "tokenize", None)):
        text_kwargs["tokenizer"] = lambda values: clap.tokenize(
            values,
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
    clap_text_tensor = clap.get_text_embedding(
        [captions[key] for key in caption_ids], **text_kwargs
    ).float()
    clap_text_tensor = F.normalize(clap_text_tensor, dim=-1).cpu()
    clap_text = dict(zip(caption_ids, clap_text_tensor))
    del clap
    gc.collect()
    torch.cuda.empty_cache()

    vgg_model, vgg_processor, backend = load_vggish_model(str(device))
    if backend != "torchaudio" or vgg_processor is None:
        raise RuntimeError("frozen benchmark requires torchaudio VGGish preprocessing")
    vgg: dict[str, torch.Tensor] = {}
    for index, (key, path) in enumerate(all_items, start=1):
        waveform = _load_quality_mono(path, 16_000)[0]
        examples = vgg_processor(waveform)
        chunks = []
        for start in range(0, int(examples.shape[0]), 32):
            chunks.append(vgg_model(examples[start : start + 32].to(device)).float().cpu())
        vgg[key] = torch.cat(chunks, dim=0)
        print(json.dumps({"event": "cross_vggish", "completed": index, "total": len(all_items)}), flush=True)
    del vgg_model
    gc.collect()
    torch.cuda.empty_cache()

    panns_model = load_panns_model(str(device))
    panns: dict[str, dict[str, torch.Tensor]] = {}
    for start in range(0, len(all_items), 8):
        batch = all_items[start : start + 8]
        waves = torch.cat(
            [
                _load_quality_mono(path, 32_000, target_samples=320_000).to(device)
                for _, path in batch
            ],
            dim=0,
        )
        embedding = panns_model(waves).float()
        probability = torch.softmax(panns_model.fc_audioset(embedding).float(), dim=-1)
        for (key, _), item_embedding, item_probability in zip(
            batch, embedding, probability
        ):
            panns[key] = {
                "embedding": item_embedding.cpu(),
                "probability": item_probability.cpu(),
            }
        print(
            json.dumps(
                {"event": "cross_panns", "completed": min(start + len(batch), len(all_items)), "total": len(all_items)}
            ),
            flush=True,
        )
    del panns_model
    gc.collect()
    torch.cuda.empty_cache()

    report: dict[str, Any] = {}
    for domain in ("music", "sound"):
        domain_rows = [row for row in rows if row["domain"] == domain]
        domain_ids = [row["panel_id"] for row in domain_rows]
        reference_clap = torch.stack(
            [clap_audio[f"reference:{panel_id}"] for panel_id in domain_ids]
        )
        reference_vgg = torch.cat(
            [vgg[f"reference:{panel_id}"] for panel_id in domain_ids], dim=0
        )
        reference_pann = torch.stack(
            [panns[f"reference:{panel_id}"]["embedding"] for panel_id in domain_ids]
        )
        report[domain] = {}
        for system_id, items in sources[domain].items():
            if set(items) != set(domain_ids):
                raise RuntimeError(f"incomplete {domain}/{system_id} output set")
            feature_keys = {
                panel_id: (
                    f"reference:{panel_id}"
                    if system_id == REFERENCE_ID
                    else f"system:{system_id}:{panel_id}"
                )
                for panel_id in domain_ids
            }
            system_clap = torch.stack(
                [clap_audio[feature_keys[panel_id]] for panel_id in domain_ids]
            )
            system_vgg = torch.cat(
                [vgg[feature_keys[panel_id]] for panel_id in domain_ids], dim=0
            )
            system_pann = torch.stack(
                [panns[feature_keys[panel_id]]["embedding"] for panel_id in domain_ids]
            )
            clap_scores = [
                float(
                    clap_audio[feature_keys[panel_id]] @ clap_text[panel_id]
                )
                for panel_id in domain_ids
            ]
            paired_clap = [
                float(
                    clap_audio[feature_keys[panel_id]]
                    @ clap_audio[f"reference:{panel_id}"]
                )
                for panel_id in domain_ids
            ]
            paired_kl = [
                _kl(
                    panns[f"reference:{panel_id}"]["probability"],
                    panns[feature_keys[panel_id]]["probability"],
                )
                for panel_id in domain_ids
            ]
            report[domain][system_id] = {
                "display_name": display_names[system_id],
                "rows": len(domain_ids),
                "clap_text_audio_cosine": _summary(clap_scores),
                "paired_generated_reference_clap_cosine": _summary(paired_clap),
                "fd_clap_diagnostic": _low_rank_frechet(system_clap, reference_clap),
                "fad_vggish_diagnostic": _low_rank_frechet(system_vgg, reference_vgg),
                "fd_pann_diagnostic": _low_rank_frechet(system_pann, reference_pann),
                "paired_kl_pann_softmax": _summary(paired_kl),
            }
    return report


def _score_speech(
    panel: list[dict[str, Any]],
    sources: dict[str, dict[str, dict[str, str]]],
    *,
    whisper_path: Path,
    device_index: int,
    display_names: dict[str, str],
) -> tuple[dict[str, Any], str | None]:
    rows = [row for row in panel if row["domain"] == "speech"]
    model = WhisperModel(
        str(whisper_path),
        device="cuda",
        device_index=device_index,
        compute_type="float16",
    )
    device = torch.device(f"cuda:{device_index}")
    utmos, utmos_error = _load_utmos(device)
    report: dict[str, Any] = {}
    for system_id, items in sources["speech"].items():
        scored = []
        for row in rows:
            panel_id = row["panel_id"]
            waveform = _load_quality_mono(items[panel_id], 16_000)[0].numpy().astype(np.float32)
            transcription = _transcribe(model, waveform)
            source_spec = row["scene_plan"]["sources"][0]
            transcript = source_spec["transcript"]
            reference_words = _normalize_words(transcript)
            hypothesis_words = _normalize_words(transcription)
            reference_chars = list("".join(reference_words))
            hypothesis_chars = list("".join(hypothesis_words))
            word_errors = _edit_distance(reference_words, hypothesis_words)
            char_errors = _edit_distance(reference_chars, hypothesis_chars)
            wer = word_errors / max(len(reference_words), 1)
            cer = char_errors / max(len(reference_chars), 1)
            scored.append(
                {
                    "panel_id": panel_id,
                    "speech_seen_speaker": bool(
                        row.get("speech_seen_speaker", False)
                    ),
                    "speech_speaker_key": row.get("speech_speaker_key"),
                    "length_bucket": row.get("length_bucket"),
                    "wer": wer,
                    "cer": cer,
                    "word_errors": word_errors,
                    "reference_words": len(reference_words),
                    "char_errors": char_errors,
                    "reference_chars": len(reference_chars),
                    "utmos": _utmos_score(utmos, waveform, device) if utmos is not None else None,
                    "asr": transcription,
                    "reference": transcript,
                }
            )
            print(
                json.dumps(
                    {"event": "cross_speech", "system": system_id, "panel_id": panel_id, "wer": wer}
                ),
                flush=True,
            )
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

        overall = aggregate(scored)
        report[system_id] = {
            "display_name": display_names[system_id],
            **overall,
            "subgroups": {
                "seen_speaker": aggregate(
                    [row for row in scored if row["speech_seen_speaker"]]
                ),
                "unseen_speaker": aggregate(
                    [row for row in scored if not row["speech_seen_speaker"]]
                ),
                "latent_le_432": aggregate(
                    [row for row in scored if row["length_bucket"] == 432]
                ),
                "latent_433_648": aggregate(
                    [row for row in scored if row["length_bucket"] == 648]
                ),
            },
            "per_output": scored,
        }
    return report, utmos_error


def _spatial_metrics(
    source: Path, ours_ids: list[str]
) -> dict[str, dict[str, dict[str, float]]]:
    core = json.loads((source / "metrics/CORE_SUMMARY.json").read_text(encoding="utf-8"))
    output: dict[str, dict[str, dict[str, float]]] = {}
    output[REFERENCE_ID] = {}
    for domain in ("music", "sound", "speech"):
        values = core["aggregates"][str(int(ours_ids[0].rsplit("_", 1)[-1]))][
            domain
        ]
        output[REFERENCE_ID][domain] = {
            "doa_spherical_error_deg": values.get(
                "reference_plan_spherical_error_mean_deg", {"mean": None}
            )["mean"],
            "azimuth_mae_deg": None,
            "elevation_mae_deg": None,
            "trajectory_extent_error_deg": values.get(
                "reference_plan_trajectory_extent_error_deg", {"mean": None}
            )["mean"],
            "activity_iou": values.get(
                "reference_activity_temporal_iou", {"mean": None}
            )["mean"],
        }
    for system_id in ours_ids:
        step = int(system_id.rsplit("_", 1)[-1])
        output[system_id] = {}
        for domain in ("music", "sound", "speech"):
            values = core["aggregates"][str(step)][domain]
            output[system_id][domain] = {
                "doa_spherical_error_deg": values["plan_spherical_error_mean_deg"]["mean"],
                "azimuth_mae_deg": values["plan_azimuth_circular_mae_deg"]["mean"],
                "elevation_mae_deg": values["plan_elevation_mae_deg"]["mean"],
                "trajectory_extent_error_deg": values.get(
                    "plan_trajectory_extent_error_deg", {"mean": None}
                )["mean"],
                "activity_iou": values["activity_temporal_iou"]["mean"],
            }
    return output


def _fmt(value: float | None, digits: int = 3) -> str:
    return "N/A" if value is None else f"{value:.{digits}f}"


def _write_tables(
    root: Path,
    audio: dict[str, Any],
    speech: dict[str, Any],
    spatial: dict[str, dict[str, dict[str, float]]],
    ours_ids: list[str],
    rows_per_domain: int,
) -> None:
    table_root = root / "tables"
    warnings = (
        f"> Frozen panel: {rows_per_domain} clips per domain. FAD/FD are matched "
        "diagnostic values, not "
        "publication-scale population estimates. Quality uses mono W-view; spatial scores "
        "use native WYZX/ACN/SN3D FOA only.\n\n"
    )
    for domain in ("music", "sound"):
        lines = [
            f"# {domain.title()} — matched P10 cross-system comparison",
            "",
            warnings.rstrip(),
            "",
            "| System | CLAP ↑ | Paired CLAP ↑ | FD-CLAP ↓ | FAD-VGGish ↓ | FD-PANN ↓ | KL-PANN ↓ | DoA err ↓ | Traj. extent err ↓ | Activity IoU ↑ |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        order = [REFERENCE_ID, VAE_ID] + ours_ids + [
            key
            for key in audio[domain]
            if key not in {*ours_ids, REFERENCE_ID, VAE_ID}
        ]
        for system_id in order:
            row = audio[domain][system_id]
            native = spatial.get(system_id, {}).get(domain)
            lines.append(
                "| {name} | {clap} | {paired} | {fd} | {fad} | {fd_pann} | {kl} | {doa} | {trajectory} | {iou} |".format(
                    name=row["display_name"],
                    clap=_fmt(row["clap_text_audio_cosine"]["mean"]),
                    paired=_fmt(row["paired_generated_reference_clap_cosine"]["mean"]),
                    fd=_fmt(row["fd_clap_diagnostic"]),
                    fad=_fmt(row["fad_vggish_diagnostic"]),
                    fd_pann=_fmt(row["fd_pann_diagnostic"]),
                    kl=_fmt(row["paired_kl_pann_softmax"]["mean"]),
                    doa=_fmt(None if native is None else native["doa_spherical_error_deg"], 2),
                    trajectory=_fmt(
                        None if native is None else native["trajectory_extent_error_deg"],
                        2,
                    ),
                    iou=_fmt(None if native is None else native["activity_iou"]),
                )
            )
        lines.extend(
            [
                "",
                "Non-spatial baselines are intentionally `N/A` in spatial columns; no channel duplication is counted as FOA.",
                "",
            ]
        )
        _atomic_text(table_root / f"{domain.upper()}_TABLE.md", "\n".join(lines))

    lines = [
        "# Speech — matched P10 cross-system comparison",
        "",
        warnings.rstrip(),
        "",
        "| System | WER ↓ | Seen-spk WER ↓ | Unseen-spk WER ↓ | CER ↓ | UTMOS ↑ | DoA err ↓ | Activity IoU ↑ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    order = [REFERENCE_ID, VAE_ID] + ours_ids + [
        key for key in speech if key not in {*ours_ids, REFERENCE_ID, VAE_ID}
    ]
    for system_id in order:
        row = speech[system_id]
        native = spatial.get(system_id, {}).get("speech")
        lines.append(
            "| {name} | {wer} | {seen_wer} | {unseen_wer} | {cer} | {utmos} | {doa} | {iou} |".format(
                name=row["display_name"],
                wer=_fmt(row["corpus_wer"]),
                seen_wer=_fmt(row["subgroups"]["seen_speaker"]["corpus_wer"]),
                unseen_wer=_fmt(row["subgroups"]["unseen_speaker"]["corpus_wer"]),
                cer=_fmt(row["corpus_cer"]),
                utmos=_fmt(row["utmos"]["mean"]),
                doa=_fmt(None if native is None else native["doa_spherical_error_deg"], 2),
                iou=_fmt(None if native is None else native["activity_iou"]),
            )
        )
    lines.extend(
        [
            "",
            "Qwen3-TTS is a speech-specialist baseline supplied with the same exact transcript and speaker description; it is not a spatial scene generator.",
            "",
        ]
    )
    _atomic_text(table_root / "SPEECH_TABLE.md", "\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--source-eval", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--whisper-model", type=Path, default=DEFAULT_WHISPER)
    parser.add_argument("--device-index", type=int, default=0)
    args = parser.parse_args()

    benchmark = args.benchmark_root.expanduser().resolve(strict=True)
    source = args.source_eval.expanduser().resolve(strict=True)
    whisper = args.whisper_model.expanduser().resolve(strict=True)
    device = torch.device(f"cuda:{args.device_index}")
    if not torch.cuda.is_available():
        raise RuntimeError("cross-system benchmark requires CUDA")
    panel, sources, ours_ids, display_names = _quality_sources(benchmark, source)
    domain_counts = {
        domain: sum(row["domain"] == domain for row in panel)
        for domain in ("music", "sound", "speech")
    }
    if len(set(domain_counts.values())) != 1:
        raise RuntimeError(f"cross-system panel is not domain-balanced: {domain_counts}")
    rows_per_domain = next(iter(domain_counts.values()))
    audio_report = _score_audio_domains(
        panel, sources, device, display_names
    )
    speech_report, utmos_error = _score_speech(
        panel,
        sources,
        whisper_path=whisper,
        device_index=args.device_index,
        display_names=display_names,
    )
    spatial = _spatial_metrics(source, ours_ids)
    report = {
        "schema": "sceneplan_foa.p10_matched_cross_system_metrics",
        "schema_version": 2,
        "status": "PASS",
        "panel_rows": len(panel),
        "rows_per_domain": rows_per_domain,
        "domain_counts": domain_counts,
        "ours_system_ids": ours_ids,
        "audio_domains": audio_report,
        "speech": speech_report,
        "native_foa_spatial": spatial,
        "utmos_error": utmos_error,
        "quality_protocol": "Ours/reference/VAE use W; mono unchanged; stereo arithmetic mean; peak normalized to -1 dBFS in metric preprocessing.",
        "spatial_protocol": "Only original WYZX/ACN/SN3D FOA receives spatial scores; all public mono/stereo baselines are N/A.",
        "small_sample_warning": f"{rows_per_domain} clips per domain; FAD/FD are matched diagnostic estimates, not the 1000-row publication estimates.",
    }
    _atomic_json(benchmark / "metrics/CROSS_SYSTEM_METRICS.json", report)
    _write_tables(
        benchmark,
        audio_report,
        speech_report,
        spatial,
        ours_ids,
        rows_per_domain,
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "metrics": str(benchmark / "metrics/CROSS_SYSTEM_METRICS.json"),
                "tables": str(benchmark / "tables"),
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
