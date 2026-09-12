"""Independent semantic and speech metrics for real-FOA Editing evaluation.

These scorers run only after AR and DiT inference.  Target audio, target/new
ScenePlan truth, and source/old ScenePlan truth are offline metric inputs and
never enter either model call.  LAION-CLAP is deliberately independent of the
M2D-CLAP feature used by Editing AR.
"""

from __future__ import annotations

from collections import Counter
import hashlib
from importlib import metadata as importlib_metadata
import math
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F
import torchaudio

from stable_audio_tools.data.model_sceneplan import MODEL_SAMPLE_RATE
from stable_audio_tools.data.sceneplan_transfusion_editing_index import sha256_file


REPO_ROOT = Path(__file__).resolve().parents[3]
INDEPENDENT_CONTENT_METRIC_CONTRACT = (
    "codec_domain_laion_clap_two_view_and_distil_whisper_source_target_v2"
)
DEFAULT_LAION_CLAP_CHECKPOINT = (
    REPO_ROOT / "load/clap_score/630k-audioset-fusion-best.pt"
)
LAION_CLAP_CHECKPOINT_SHA256 = (
    "fb171dd9b608aebdac3d89286cd7615c5100af4cc7dc37797c7fb8d3cc15e3a5"
)
DEFAULT_WHISPER_MODEL = Path(
    "/mnt/sdb/audio_dataset/sceneplan_v2_1p124m/models/"
    "faster-distil-whisper-large-v3"
)
WHISPER_EXPECTED_SMALL_FILES = {
    "config.json": "90c55f775cc4e0bb17293d0bf12f96557a486f20dea886fabd8e6075a3588b21",
    "preprocessor_config.json": (
        "7ccc62c6f2765af1f3b46c00c9b5894426835a05021c8b9c01eecb6dfb542711"
    ),
    "tokenizer.json": "6d8cbd7cd0d8d5815e478dac67b85a26bbe77c1f5e0c6d76d1ce2abc0e5f21ca",
    "vocabulary.json": "c69260f2ab26d659b7c398f9a2b2b48ed0df16c3b47d7326782fd9cba71690c1",
}
WHISPER_MODEL_BIN_SHA256 = (
    "b79368e19b6623813609431a6e5ee309a71506701ebc49fd7820e692dec7c5f5"
)
CLAP_SAMPLE_RATE = 48_000
CLAP_WINDOW_SAMPLES = 480_000
ASR_SAMPLE_RATE = 16_000


CONTENT_METRIC_NAMES = (
    "independent_clap_output_target_cosine",
    "independent_clap_source_target_cosine",
    "independent_clap_target_progress",
    "independent_clap_change_direction_cosine",
    "independent_clap_output_edit_text_cosine",
    "independent_clap_source_edit_text_cosine",
    "independent_clap_target_edit_text_cosine",
    "independent_clap_edit_text_progress",
    "speech_output_wer",
    "speech_source_wer",
    "speech_target_wer",
    "speech_preservation_excess_wer",
    "speech_addition_wer_progress",
    "speech_output_cer",
    "speech_source_cer",
    "speech_target_cer",
    "speech_preservation_excess_cer",
    "speech_addition_cer_progress",
    "speech_output_target_stoi",
    "speech_output_target_si_sdr_db",
    "removed_speech_output_token_recall",
    "removed_speech_source_token_recall",
    "removed_speech_target_token_recall",
    "removed_speech_recall_progress",
)


def verify_independent_content_metric_assets() -> dict[str, Any]:
    clap = DEFAULT_LAION_CLAP_CHECKPOINT.resolve(strict=True)
    clap_sha = sha256_file(clap)
    if clap_sha != LAION_CLAP_CHECKPOINT_SHA256:
        raise RuntimeError("independent LAION-CLAP checkpoint changed")
    whisper = DEFAULT_WHISPER_MODEL.resolve(strict=True)
    files = {}
    expected_whisper_files = {
        **WHISPER_EXPECTED_SMALL_FILES,
        "model.bin": WHISPER_MODEL_BIN_SHA256,
    }
    for name, expected in expected_whisper_files.items():
        path = (whisper / name).resolve(strict=True)
        observed = sha256_file(path)
        if observed != expected:
            raise RuntimeError(f"independent Whisper asset changed: {name}")
        files[name] = {
            "path": str(path),
            "sha256": observed,
            "bytes": path.stat().st_size,
        }
    if int(files["model.bin"]["bytes"]) != 1_512_927_867:
        raise RuntimeError("independent Whisper model size changed")

    def version(distribution: str) -> str:
        try:
            return importlib_metadata.version(distribution)
        except importlib_metadata.PackageNotFoundError:
            return "missing"

    return {
        "contract": INDEPENDENT_CONTENT_METRIC_CONTRACT,
        "laion_clap": {
            "path": str(clap),
            "sha256": clap_sha,
            "distribution_version": version("laion-clap"),
            "sample_rate": CLAP_SAMPLE_RATE,
            "window_samples": CLAP_WINDOW_SAMPLES,
            "full_duration_view_policy": "one_padded_or_first_and_last_10s_mean",
            "foa_channel": "W",
        },
        "speech_asr": {
            "path": str(whisper),
            "files": files,
            "faster_whisper_version": version("faster-whisper"),
            "ctranslate2_version": version("ctranslate2"),
            "sample_rate": ASR_SAMPLE_RATE,
            "language": "en",
            "beam_size": 5,
            "activity_guard_sec": 0.05,
        },
        "model_inputs": False,
        "offline_metric_only": True,
        "comparison_domain": "frozen_vae_decoded_source_output_target_latents",
    }


def _normalize_words(value: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", str(value).lower())


def _edit_distance(left: Sequence[Any], right: Sequence[Any]) -> int:
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


def _error_rates(hypothesis: str, reference: str) -> dict[str, float | int]:
    reference_words = _normalize_words(reference)
    hypothesis_words = _normalize_words(hypothesis)
    reference_chars = list("".join(reference_words))
    hypothesis_chars = list("".join(hypothesis_words))
    word_edits = _edit_distance(reference_words, hypothesis_words)
    char_edits = _edit_distance(reference_chars, hypothesis_chars)
    return {
        "wer": word_edits / max(len(reference_words), 1),
        "cer": char_edits / max(len(reference_chars), 1),
        "word_edits": word_edits,
        "reference_words": len(reference_words),
        "hypothesis_words": len(hypothesis_words),
        "char_edits": char_edits,
        "reference_chars": len(reference_chars),
    }


def _token_recall(hypothesis: str, reference: str) -> float:
    """Bag-of-words recall used only for speech-removal evidence."""

    reference_counts = Counter(_normalize_words(reference))
    hypothesis_counts = Counter(_normalize_words(hypothesis))
    total = sum(reference_counts.values())
    if total == 0:
        raise ValueError("removed speech transcript is empty")
    overlap = sum(
        min(count, hypothesis_counts.get(token, 0))
        for token, count in reference_counts.items()
    )
    return overlap / total


def _source_caption(source: Mapping[str, Any]) -> str:
    kind = str(source.get("kind") or "")
    if kind == "speech":
        speaker = str(source.get("speaker_description") or "a speaker").strip()
        transcript = str(source.get("transcript") or "").strip()
        if not transcript:
            raise ValueError("speech source has no transcript")
        return f'{speaker} saying "{transcript}"'
    description = str(source.get("description") or "").strip()
    if kind not in {"music", "sound"} or not description:
        raise ValueError("non-speech source has no semantic description")
    return description


def edited_source_caption(
    old_plan: Mapping[str, Any],
    new_plan: Mapping[str, Any],
    edited_source_ids: Sequence[str],
    operation: str,
) -> str:
    source_plan = old_plan if str(operation) == "event_removal" else new_plan
    by_id = {
        str(source["source_id"]): source for source in source_plan["sources"]
    }
    wanted = [str(value) for value in edited_source_ids]
    if not wanted or any(source_id not in by_id for source_id in wanted):
        raise ValueError("edited source IDs do not resolve in metric-side plan")
    return " and ".join(_source_caption(by_id[source_id]) for source_id in wanted)


def _normalize_waveform(waveform: Tensor) -> Tensor:
    waveform = waveform.float()
    waveform = waveform - waveform.mean()
    peak = waveform.abs().amax()
    if float(peak) > 1.0e-8:
        waveform = waveform / peak * (10.0 ** (-1.0 / 20.0))
    return waveform.clamp(-1.0, 1.0)


def clap_full_duration_views(audio: Tensor) -> Tensor:
    if audio.ndim != 2 or int(audio.shape[0]) != 4 or int(audio.shape[-1]) < 400:
        raise ValueError("independent CLAP audio must be FOA [4,N>=400]")
    waveform = _normalize_waveform(audio[0])
    waveform = torchaudio.functional.resample(
        waveform[None], MODEL_SAMPLE_RATE, CLAP_SAMPLE_RATE
    )[0]
    count = int(waveform.shape[-1])
    if count <= CLAP_WINDOW_SAMPLES:
        return F.pad(waveform, (0, CLAP_WINDOW_SAMPLES - count))[None]
    return torch.stack(
        (waveform[:CLAP_WINDOW_SAMPLES], waveform[-CLAP_WINDOW_SAMPLES:])
    )


def _cosine(left: Tensor, right: Tensor) -> float:
    return float(F.cosine_similarity(left[None], right[None], dim=-1)[0])


def _progress(output: float, source: float, target: float, *, minimum: float) -> float | None:
    desired = target - source
    if abs(desired) < float(minimum):
        return None
    return (output - source) / desired


def _speech_source(plan: Mapping[str, Any], source_id: str) -> Mapping[str, Any] | None:
    for source in plan["sources"]:
        if str(source["source_id"]) == str(source_id) and source["kind"] == "speech":
            return source
    return None


def speech_activity_waveform(audio: Tensor, source: Mapping[str, Any]) -> np.ndarray:
    interval = source["activity"]
    start = max(0, round((float(interval["onset_sec"]) - 0.05) * MODEL_SAMPLE_RATE))
    stop = min(
        int(audio.shape[-1]),
        round((float(interval["offset_sec"]) + 0.05) * MODEL_SAMPLE_RATE),
    )
    if stop - start < 400:
        raise ValueError("speech metric interval is too short")
    waveform = _normalize_waveform(audio[0, start:stop].cpu())
    waveform = torchaudio.functional.resample(
        waveform[None], MODEL_SAMPLE_RATE, ASR_SAMPLE_RATE
    )[0]
    return waveform.numpy().astype(np.float32, copy=False)


def _si_sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    count = min(reference.size, estimate.size)
    target = torch.from_numpy(reference[:count]).to(torch.float64)
    value = torch.from_numpy(estimate[:count]).to(torch.float64)
    projection = torch.dot(value, target) * target / target.square().sum().clamp_min(1e-12)
    residual = value - projection
    return float(
        10.0
        * torch.log10(
            projection.square().sum().clamp_min(1e-12)
            / residual.square().sum().clamp_min(1e-12)
        )
    )


class IndependentEditingContentEvaluator:
    """Frozen LAION-CLAP and Whisper post-inference evaluator."""

    def __init__(self, *, device: torch.device, device_index: int) -> None:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        from faster_whisper import WhisperModel
        from stable_audio_tools.training.metrics.fad_metrics import load_clap_model

        self.device = device
        self.clap = load_clap_model(
            str(DEFAULT_LAION_CLAP_CHECKPOINT), device=str(device)
        )
        self.clap.model.eval().requires_grad_(False)
        self.whisper = WhisperModel(
            str(DEFAULT_WHISPER_MODEL),
            device="cuda",
            device_index=int(device_index),
            compute_type="float16",
        )

    @torch.inference_mode()
    def _audio_embeddings(self, audios: Sequence[Tensor]) -> Tensor:
        views = []
        owners = []
        for owner, audio in enumerate(audios):
            value = clap_full_duration_views(audio.to(self.device))
            views.extend(value.unbind(0))
            owners.extend([owner] * len(value))
        embeddings = []
        for start in range(0, len(views), 12):
            batch = torch.stack(views[start : start + 12])
            with torch.autocast(device_type="cuda", enabled=False):
                value = self.clap.get_audio_embedding_from_data(
                    x=batch, use_tensor=True
                ).float()
            if value.ndim == 1:
                value = value[None]
            embeddings.append(F.normalize(value, dim=-1))
        encoded = torch.cat(embeddings)
        output = []
        for owner in range(len(audios)):
            indices = torch.tensor(
                [index for index, value in enumerate(owners) if value == owner],
                device=encoded.device,
            )
            output.append(F.normalize(encoded.index_select(0, indices).mean(0), dim=-1))
        return torch.stack(output)

    @torch.inference_mode()
    def _text_embeddings(self, captions: Sequence[str]) -> Tensor:
        values = list(captions)
        requested = values if len(values) > 1 else values * 2
        encoded = self.clap.get_text_embedding(requested, use_tensor=True).float()
        if encoded.ndim == 1:
            encoded = encoded[None]
        return F.normalize(encoded[: len(values)], dim=-1)

    def _transcribe(self, waveform: np.ndarray) -> str:
        segments, _ = self.whisper.transcribe(
            waveform,
            language="en",
            beam_size=5,
            condition_on_previous_text=False,
            vad_filter=False,
        )
        return " ".join(segment.text.strip() for segment in segments).strip()

    @torch.inference_mode()
    def _semantic_batch(
        self,
        *,
        outputs: Sequence[Tensor],
        sources: Sequence[Tensor],
        targets: Sequence[Tensor],
        captions: Sequence[str],
        operations: Sequence[str],
    ) -> list[dict[str, float | None]]:
        count = len(outputs)
        embeddings = self._audio_embeddings([*outputs, *sources, *targets])
        output_embeddings = embeddings[:count]
        source_embeddings = embeddings[count : 2 * count]
        target_embeddings = embeddings[2 * count :]
        text_embeddings = self._text_embeddings(captions)
        rows = []
        for output, source, target, text, operation in zip(
            output_embeddings,
            source_embeddings,
            target_embeddings,
            text_embeddings,
            operations,
        ):
            output_target = _cosine(output, target)
            source_target = _cosine(source, target)
            output_text = _cosine(output, text)
            source_text = _cosine(source, text)
            target_text = _cosine(target, text)
            change_direction = None
            source_to_target = target - source
            source_to_output = output - source
            if float(source_to_target.norm()) >= 1.0e-5 and float(
                source_to_output.norm()
            ) >= 1.0e-5:
                change_direction = _cosine(source_to_output, source_to_target)
            text_progress = None
            if operation == "event_addition":
                text_progress = _progress(
                    output_text, source_text, target_text, minimum=0.01
                )
            elif operation == "event_removal":
                text_progress = _progress(
                    -output_text, -source_text, -target_text, minimum=0.01
                )
            rows.append(
                {
                    "independent_clap_output_target_cosine": output_target,
                    "independent_clap_source_target_cosine": source_target,
                    "independent_clap_target_progress": _progress(
                        output_target, source_target, 1.0, minimum=0.01
                    ),
                    "independent_clap_change_direction_cosine": change_direction,
                    "independent_clap_output_edit_text_cosine": output_text,
                    "independent_clap_source_edit_text_cosine": source_text,
                    "independent_clap_target_edit_text_cosine": target_text,
                    "independent_clap_edit_text_progress": text_progress,
                }
            )
        return rows

    def _speech_metrics(
        self,
        *,
        output: Tensor,
        source: Tensor,
        target: Tensor,
        old_plan: Mapping[str, Any],
        new_plan: Mapping[str, Any],
        edited_source_ids: Sequence[str],
        unchanged_source_ids: Sequence[str],
        operation: str,
    ) -> tuple[dict[str, float | None], dict[str, Any]]:
        metrics = {name: None for name in CONTENT_METRIC_NAMES if name.startswith("speech_") or name.startswith("removed_speech_")}
        diagnostics: dict[str, Any] = {"eligible": False}
        edited_ids = {str(value) for value in edited_source_ids}
        unchanged_ids = {str(value) for value in unchanged_source_ids}

        removed = []
        if operation == "event_removal":
            removed = [
                value
                for value in edited_ids
                if _speech_source(old_plan, value) is not None
                and _speech_source(new_plan, value) is None
            ]
        if len(removed) == 1:
            source_id = removed[0]
            speech = _speech_source(old_plan, source_id)
            assert speech is not None
            transcript = str(speech["transcript"])
            source_wave = speech_activity_waveform(source, speech)
            output_wave = speech_activity_waveform(output, speech)
            target_wave = speech_activity_waveform(target, speech)
            hypotheses = {
                "source": self._transcribe(source_wave),
                "output": self._transcribe(output_wave),
                "target": self._transcribe(target_wave),
            }
            source_recall = _token_recall(hypotheses["source"], transcript)
            output_recall = _token_recall(hypotheses["output"], transcript)
            target_recall = _token_recall(hypotheses["target"], transcript)
            metrics.update(
                {
                    "removed_speech_output_token_recall": output_recall,
                    "removed_speech_source_token_recall": source_recall,
                    "removed_speech_target_token_recall": target_recall,
                    "removed_speech_recall_progress": _progress(
                        -output_recall,
                        -source_recall,
                        -target_recall,
                        minimum=0.05,
                    ),
                }
            )
            diagnostics = {
                "eligible": True,
                "kind": "removed_speech",
                "source_id": source_id,
                "reference_transcript_sha256": hashlib.sha256(
                    transcript.encode("utf-8")
                ).hexdigest(),
                "hypotheses": hypotheses,
            }
            return metrics, diagnostics

        target_speech = [
            source
            for source in new_plan["sources"]
            if source["kind"] == "speech"
            and str(source["source_id"]) in edited_ids | unchanged_ids
        ]
        if len(target_speech) != 1:
            return metrics, diagnostics
        speech = target_speech[0]
        source_id = str(speech["source_id"])
        transcript = str(speech["transcript"])
        waveforms = {
            "output": speech_activity_waveform(output, speech),
            "source": speech_activity_waveform(source, speech),
            "target": speech_activity_waveform(target, speech),
        }
        hypotheses = {
            key: self._transcribe(value) for key, value in waveforms.items()
        }
        errors = {
            key: _error_rates(value, transcript) for key, value in hypotheses.items()
        }
        metrics.update(
            {
                "speech_output_wer": float(errors["output"]["wer"]),
                "speech_source_wer": float(errors["source"]["wer"]),
                "speech_target_wer": float(errors["target"]["wer"]),
                "speech_output_cer": float(errors["output"]["cer"]),
                "speech_source_cer": float(errors["source"]["cer"]),
                "speech_target_cer": float(errors["target"]["cer"]),
                "speech_output_target_stoi": None,
                "speech_output_target_si_sdr_db": _si_sdr(
                    waveforms["target"], waveforms["output"]
                ),
            }
        )
        content_should_be_preserved = source_id in unchanged_ids or (
            operation not in {"event_addition", "event_removal"}
            and source_id in edited_ids
        )
        if content_should_be_preserved:
            metrics["speech_preservation_excess_wer"] = (
                float(errors["output"]["wer"]) - float(errors["target"]["wer"])
            )
            metrics["speech_preservation_excess_cer"] = (
                float(errors["output"]["cer"]) - float(errors["target"]["cer"])
            )
        if operation == "event_addition" and source_id in edited_ids:
            metrics["speech_addition_wer_progress"] = _progress(
                -float(errors["output"]["wer"]),
                -float(errors["source"]["wer"]),
                -float(errors["target"]["wer"]),
                minimum=0.05,
            )
            metrics["speech_addition_cer_progress"] = _progress(
                -float(errors["output"]["cer"]),
                -float(errors["source"]["cer"]),
                -float(errors["target"]["cer"]),
                minimum=0.05,
            )
        try:
            from pystoi import stoi

            count = min(waveforms["target"].size, waveforms["output"].size)
            metrics["speech_output_target_stoi"] = float(
                stoi(
                    waveforms["target"][:count],
                    waveforms["output"][:count],
                    ASR_SAMPLE_RATE,
                    extended=False,
                )
            )
        except (ValueError, RuntimeError):
            pass
        diagnostics = {
            "eligible": True,
            "kind": "target_speech",
            "source_id": source_id,
            "unchanged": source_id in unchanged_ids,
            "edited": source_id in edited_ids,
            "content_should_be_preserved": content_should_be_preserved,
            "reference_transcript_sha256": hashlib.sha256(
                transcript.encode("utf-8")
            ).hexdigest(),
            "hypotheses": hypotheses,
            "errors": errors,
        }
        return metrics, diagnostics

    def score_batch(
        self,
        *,
        outputs: Sequence[Tensor],
        sources: Sequence[Tensor],
        targets: Sequence[Tensor],
        old_plans: Sequence[Mapping[str, Any]],
        new_plans: Sequence[Mapping[str, Any]],
        edited_source_ids: Sequence[Sequence[str]],
        unchanged_source_ids: Sequence[Sequence[str]],
        operations: Sequence[str],
    ) -> list[dict[str, Any]]:
        sizes = {
            len(outputs),
            len(sources),
            len(targets),
            len(old_plans),
            len(new_plans),
            len(edited_source_ids),
            len(unchanged_source_ids),
            len(operations),
        }
        if sizes != {len(outputs)} or not outputs:
            raise ValueError("independent content metric batch is misaligned")
        captions = [
            edited_source_caption(old, new, ids, operation)
            for old, new, ids, operation in zip(
                old_plans, new_plans, edited_source_ids, operations
            )
        ]
        semantic = self._semantic_batch(
            outputs=outputs,
            sources=sources,
            targets=targets,
            captions=captions,
            operations=operations,
        )
        rows = []
        for index in range(len(outputs)):
            speech_metrics, speech_diagnostics = self._speech_metrics(
                output=outputs[index],
                source=sources[index],
                target=targets[index],
                old_plan=old_plans[index],
                new_plan=new_plans[index],
                edited_source_ids=edited_source_ids[index],
                unchanged_source_ids=unchanged_source_ids[index],
                operation=str(operations[index]),
            )
            metrics = {name: None for name in CONTENT_METRIC_NAMES}
            metrics.update(semantic[index])
            metrics.update(speech_metrics)
            for name, value in metrics.items():
                if value is not None and not math.isfinite(float(value)):
                    raise RuntimeError(f"independent content metric is non-finite: {name}")
            rows.append(
                {
                    "contract": INDEPENDENT_CONTENT_METRIC_CONTRACT,
                    "metrics": metrics,
                    "diagnostics": {
                        "edited_caption_sha256": hashlib.sha256(
                            captions[index].encode("utf-8")
                        ).hexdigest(),
                        "speech": speech_diagnostics,
                        "old_and_new_plans_are_offline_metric_truth_only": True,
                    },
                }
            )
        return rows


__all__ = [
    "CONTENT_METRIC_NAMES",
    "INDEPENDENT_CONTENT_METRIC_CONTRACT",
    "IndependentEditingContentEvaluator",
    "clap_full_duration_views",
    "edited_source_caption",
    "speech_activity_waveform",
    "verify_independent_content_metric_assets",
]
