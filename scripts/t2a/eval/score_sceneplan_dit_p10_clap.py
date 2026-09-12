#!/usr/bin/env python3
"""Score CLAP semantics and matched FD-CLAP on a frozen P10 panel."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F
import torchaudio

from scripts.t2a.eval.sceneplan_dit_p10_panel_common import (
    DEFAULT_EVAL_ROOT,
    atomic_json,
    checkpoint_steps,
    load_foa,
    load_output_rows,
    load_panel,
    summarize,
)
from stable_audio_tools.training.metrics.fad_metrics import load_clap_model


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def _prepare_w(path: str | Path, device: torch.device) -> torch.Tensor:
    audio, sample_rate = load_foa(path)
    mono = audio[:1].to(device)
    peak = mono.abs().max().clamp_min(1.0e-8)
    mono = mono / peak * (10.0 ** (-1.0 / 20.0))
    if sample_rate != 48_000:
        mono = torchaudio.functional.resample(mono, sample_rate, 48_000)
    target = 480_000
    if int(mono.shape[-1]) < target:
        mono = F.pad(mono, (0, target - int(mono.shape[-1])))
    else:
        mono = mono[:, :target]
    return mono.clamp(-1.0, 1.0)


@torch.inference_mode()
def _audio_embeddings(model, items: list[tuple[str, str]], device: torch.device) -> dict[str, torch.Tensor]:
    output: dict[str, torch.Tensor] = {}
    batch_size = 10
    for start in range(0, len(items), batch_size):
        batch = items[start : start + batch_size]
        waves = torch.cat([_prepare_w(path, device) for _, path in batch], dim=0)
        with torch.autocast(device_type="cuda", enabled=False):
            embeddings = model.get_audio_embedding_from_data(x=waves, use_tensor=True).float()
        embeddings = F.normalize(embeddings, dim=-1).cpu()
        for (key, _), embedding in zip(batch, embeddings):
            output[key] = embedding
        print(
            json.dumps(
                {"event": "clap_audio_batch", "completed": min(start + len(batch), len(items)), "total": len(items)}
            ),
            flush=True,
        )
    return output


@torch.inference_mode()
def _text_embeddings(model, captions: dict[str, str]) -> dict[str, torch.Tensor]:
    keys = list(captions)
    texts = [captions[key] for key in keys]
    kwargs: dict[str, Any] = {"use_tensor": True}
    tokenizer_backend = getattr(model, "tokenize", None)
    if callable(tokenizer_backend):
        def tokenize_batch(values):
            return tokenizer_backend(
                values,
                padding="max_length",
                truncation=True,
                max_length=77,
                return_tensors="pt",
            )

        kwargs["tokenizer"] = tokenize_batch
    embeddings = model.get_text_embedding(texts, **kwargs).float()
    embeddings = F.normalize(embeddings, dim=-1).cpu()
    return dict(zip(keys, embeddings))


def _low_rank_frechet(left: torch.Tensor, right: torch.Tensor) -> float:
    """Exact Gaussian FD using the small sample matrices instead of D×D sqrtm."""

    left = left.to(torch.float64)
    right = right.to(torch.float64)
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1]:
        raise ValueError("FD inputs must be [N,D] with equal D")
    if left.shape[0] < 2 or right.shape[0] < 2:
        raise ValueError("FD requires at least two samples per side")
    mean_delta = (left.mean(dim=0) - right.mean(dim=0)).square().sum()
    left_centered = (left - left.mean(dim=0)) / (left.shape[0] - 1) ** 0.5
    right_centered = (right - right.mean(dim=0)) / (right.shape[0] - 1) ** 0.5
    trace_left = left_centered.square().sum()
    trace_right = right_centered.square().sum()
    nuclear = torch.linalg.svdvals(left_centered @ right_centered.transpose(0, 1)).sum()
    return float((mean_delta + trace_left + trace_right - 2.0 * nuclear).clamp_min(0.0))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", type=Path, default=DEFAULT_EVAL_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--clap-model", default="630k-audioset-fusion-best.pt")
    args = parser.parse_args()
    root = args.eval_root.expanduser().resolve(strict=True)
    steps = checkpoint_steps(root)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CLAP benchmark requires CUDA")
    panel = [row for row in load_panel(root) if row["domain"] in {"music", "sound"}]
    outputs = [row for row in load_output_rows(root) if row["domain"] in {"music", "sound"}]
    domain_counts = {
        domain: sum(row["domain"] == domain for row in panel)
        for domain in ("music", "sound")
    }
    if min(domain_counts.values(), default=0) < 2:
        raise RuntimeError(f"CLAP phase needs at least two rows per domain: {domain_counts}")
    expected_outputs = len(panel) * len(steps)
    if len(outputs) != expected_outputs:
        raise RuntimeError(
            f"CLAP phase requires {len(panel)} references and {expected_outputs} checkpoint outputs"
        )

    model = load_clap_model(args.clap_model, device=str(device))
    audio_items = [
        (f"reference:{row['panel_id']}", row["reference_foa_path"]) for row in panel
    ] + [
        (f"generated:{row['checkpoint_step']}:{row['panel_id']}", row["generated_foa_path"])
        for row in outputs
    ]
    captions = {row["panel_id"]: row["semantic_text"] for row in panel}
    audio_embeddings = _audio_embeddings(model, audio_items, device)
    text_embeddings = _text_embeddings(model, captions)

    per_output: list[dict[str, Any]] = []
    for row in outputs:
        panel_id = row["panel_id"]
        generated = audio_embeddings[f"generated:{row['checkpoint_step']}:{panel_id}"]
        reference = audio_embeddings[f"reference:{panel_id}"]
        text_embedding = text_embeddings[panel_id]
        domain_candidates = [item for item in panel if item["domain"] == row["domain"]]
        candidate_ids = [item["panel_id"] for item in domain_candidates]
        candidate_text = torch.stack([text_embeddings[value] for value in candidate_ids])
        retrieval_scores = generated @ candidate_text.transpose(0, 1)
        predicted = candidate_ids[int(retrieval_scores.argmax())]
        per_output.append(
            {
                "checkpoint_step": int(row["checkpoint_step"]),
                "panel_id": panel_id,
                "domain": row["domain"],
                "sample_id": row["sample_id"],
                "generated_text_cosine": float(generated @ text_embedding),
                "reference_text_cosine": float(reference @ text_embedding),
                "semantic_deficit_vs_reference": float(reference @ text_embedding - generated @ text_embedding),
                "generated_reference_audio_cosine": float(generated @ reference),
                "within_domain_text_retrieval_prediction": predicted,
                "within_domain_text_retrieval_correct": predicted == panel_id,
                "within_domain_text_retrieval_scores": {
                    key: float(value) for key, value in zip(candidate_ids, retrieval_scores)
                },
                "generated_foa_path": row["generated_foa_path"],
                "reference_foa_path": row["reference_foa_path"],
            }
        )

    aggregates: dict[str, Any] = {}
    for step in steps:
        aggregates[str(step)] = {}
        for domain in ("music", "sound"):
            chosen = [
                row for row in per_output if row["checkpoint_step"] == step and row["domain"] == domain
            ]
            generated_matrix = torch.stack(
                [audio_embeddings[f"generated:{step}:{row['panel_id']}"] for row in chosen]
            )
            reference_matrix = torch.stack(
                [audio_embeddings[f"reference:{row['panel_id']}"] for row in chosen]
            )
            aggregates[str(step)][domain] = {
                "rows": len(chosen),
                "clap_text_audio_cosine": summarize(row["generated_text_cosine"] for row in chosen),
                "reference_clap_text_audio_cosine": summarize(
                    row["reference_text_cosine"] for row in chosen
                ),
                "clap_semantic_deficit": summarize(
                    row["semantic_deficit_vs_reference"] for row in chosen
                ),
                "paired_generated_reference_clap_cosine": summarize(
                    row["generated_reference_audio_cosine"] for row in chosen
                ),
                "within_domain_text_retrieval_top1": sum(
                    bool(row["within_domain_text_retrieval_correct"]) for row in chosen
                ) / len(chosen),
                "fd_clap": _low_rank_frechet(
                    generated_matrix, reference_matrix
                ),
            }

    metrics_root = root / "metrics"
    _atomic_jsonl(metrics_root / "clap_per_output.jsonl", per_output)
    report = {
        "schema": "stable_audio_tools.sceneplan_dit_p10_clap_metrics",
        "schema_version": 1,
        "status": "PASS",
        "evaluation_rows": len(panel),
        "domain_counts": domain_counts,
        "checkpoint_outputs": expected_outputs,
        "channel": "W",
        "sample_rate": 48_000,
        "window_seconds": 10.0,
        "clap_model": args.clap_model,
        "fd_warning": (
            f"FD-CLAP uses {min(domain_counts.values())} matched clips per domain; "
            "it is suitable for checkpoint selection but not the paper-scale population result."
        ),
        "aggregates": aggregates,
    }
    atomic_json(metrics_root / "CLAP_SUMMARY.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
