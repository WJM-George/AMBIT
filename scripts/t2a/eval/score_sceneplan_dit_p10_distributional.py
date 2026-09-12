#!/usr/bin/env python3
"""Score diagnostic FAD-VGGish and paired KL-PANN on the P10 panel."""

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
from stable_audio_tools.training.metrics.fad_metrics import (
    load_panns_model,
    load_vggish_model,
)


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def _prepare_w(path: str | Path, target_rate: int, device: torch.device) -> torch.Tensor:
    audio, sample_rate = load_foa(path)
    mono = audio[0:1].to(device)
    peak = mono.abs().max().clamp_min(1.0e-8)
    mono = mono / peak * (10.0 ** (-1.0 / 20.0))
    if sample_rate != target_rate:
        mono = torchaudio.functional.resample(mono, sample_rate, target_rate)
    return mono[0].clamp(-1.0, 1.0)


@torch.inference_mode()
def _vggish_embeddings(
    model,
    processor,
    items: list[tuple[str, str]],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    output = {}
    for index, (key, path) in enumerate(items, start=1):
        waveform = _prepare_w(path, 16_000, torch.device("cpu"))
        examples = processor(waveform)
        if examples.ndim != 4 or int(examples.shape[0]) < 1:
            raise RuntimeError(f"VGGish produced no examples for {path}")
        chunks = []
        for start in range(0, int(examples.shape[0]), 32):
            chunks.append(model(examples[start : start + 32].to(device)).float().cpu())
        output[key] = torch.cat(chunks, dim=0)
        print(
            json.dumps({"event": "vggish_clip", "completed": index, "total": len(items), "key": key}),
            flush=True,
        )
    return output


@torch.inference_mode()
def _panns_outputs(
    model,
    items: list[tuple[str, str]],
    device: torch.device,
) -> dict[str, dict[str, torch.Tensor]]:
    waves = []
    keys = []
    target = 320_000
    output: dict[str, dict[str, torch.Tensor]] = {}
    for key, path in items:
        waveform = _prepare_w(path, 32_000, device)
        if int(waveform.numel()) < target:
            waveform = F.pad(waveform, (0, target - int(waveform.numel())))
        else:
            waveform = waveform[:target]
        keys.append(key)
        waves.append(waveform)
        if len(waves) == 8 or key == items[-1][0]:
            batch = torch.stack(waves)
            embedding = model(batch).float()
            logits = model.fc_audioset(embedding).float()
            probabilities = torch.softmax(logits, dim=-1)
            for item_key, item_embedding, probability in zip(keys, embedding, probabilities):
                output[item_key] = {
                    "embedding": item_embedding.cpu(),
                    "probability": probability.cpu(),
                }
            print(
                json.dumps({"event": "panns_batch", "completed": len(output), "total": len(items)}),
                flush=True,
            )
            waves = []
            keys = []
    return output


def _low_rank_frechet(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.to(torch.float64)
    right = right.to(torch.float64)
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1]:
        raise ValueError("FD inputs must be [N,D]")
    if left.shape[0] < 2 or right.shape[0] < 2:
        raise ValueError("FD requires at least two embeddings per side")
    mean_delta = (left.mean(dim=0) - right.mean(dim=0)).square().sum()
    left_centered = (left - left.mean(dim=0)) / (left.shape[0] - 1) ** 0.5
    right_centered = (right - right.mean(dim=0)) / (right.shape[0] - 1) ** 0.5
    nuclear = torch.linalg.svdvals(left_centered @ right_centered.transpose(0, 1)).sum()
    value = (
        mean_delta
        + left_centered.square().sum()
        + right_centered.square().sum()
        - 2.0 * nuclear
    )
    return float(value.clamp_min(0.0))


def _kl(reference: torch.Tensor, generated: torch.Tensor) -> float:
    reference = reference.to(torch.float64).clamp_min(1.0e-12)
    generated = generated.to(torch.float64).clamp_min(1.0e-12)
    return float((reference * (reference.log() - generated.log())).sum())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", type=Path, default=DEFAULT_EVAL_ROOT)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    root = args.eval_root.expanduser().resolve(strict=True)
    steps = checkpoint_steps(root)
    device = torch.device(args.device)
    panel = [row for row in load_panel(root) if row["domain"] in {"music", "sound"}]
    outputs = [row for row in load_output_rows(root) if row["domain"] in {"music", "sound"}]
    domain_counts = {
        domain: sum(row["domain"] == domain for row in panel)
        for domain in ("music", "sound")
    }
    if min(domain_counts.values(), default=0) < 2:
        raise RuntimeError(
            f"distributional phase needs at least two rows per domain: {domain_counts}"
        )
    expected_outputs = len(panel) * len(steps)
    if len(outputs) != expected_outputs:
        raise RuntimeError(
            f"distributional phase requires {len(panel)} references and {expected_outputs} outputs"
        )
    items = [
        (f"reference:{row['panel_id']}", row["reference_foa_path"]) for row in panel
    ] + [
        (f"generated:{row['checkpoint_step']}:{row['panel_id']}", row["generated_foa_path"])
        for row in outputs
    ]

    vggish_model, vggish_processor, backend = load_vggish_model(str(device))
    if backend != "torchaudio" or vggish_processor is None:
        raise RuntimeError("the frozen benchmark requires torchaudio VGGish preprocessing")
    vggish = _vggish_embeddings(vggish_model, vggish_processor, items, device)
    del vggish_model
    gc.collect()
    torch.cuda.empty_cache()

    panns_model = load_panns_model(str(device))
    panns = _panns_outputs(panns_model, items, device)
    del panns_model
    gc.collect()
    torch.cuda.empty_cache()

    per_output: list[dict[str, Any]] = []
    for row in outputs:
        panel_id = row["panel_id"]
        generated_key = f"generated:{row['checkpoint_step']}:{panel_id}"
        reference_key = f"reference:{panel_id}"
        per_output.append(
            {
                "checkpoint_step": int(row["checkpoint_step"]),
                "panel_id": panel_id,
                "domain": row["domain"],
                "sample_id": row["sample_id"],
                "paired_kl_pann_softmax": _kl(
                    panns[reference_key]["probability"],
                    panns[generated_key]["probability"],
                ),
                "vggish_generated_windows": int(vggish[generated_key].shape[0]),
                "vggish_reference_windows": int(vggish[reference_key].shape[0]),
            }
        )

    aggregates: dict[str, Any] = {}
    for step in steps:
        aggregates[str(step)] = {}
        for domain in ("music", "sound"):
            chosen = [
                row for row in per_output if row["checkpoint_step"] == step and row["domain"] == domain
            ]
            generated_vggish = torch.cat(
                [vggish[f"generated:{step}:{row['panel_id']}"] for row in chosen], dim=0
            )
            reference_vggish = torch.cat(
                [vggish[f"reference:{row['panel_id']}"] for row in chosen], dim=0
            )
            generated_panns = torch.stack(
                [panns[f"generated:{step}:{row['panel_id']}"]["embedding"] for row in chosen]
            )
            reference_panns = torch.stack(
                [panns[f"reference:{row['panel_id']}"]["embedding"] for row in chosen]
            )
            aggregates[str(step)][domain] = {
                "rows": len(chosen),
                "paired_kl_pann_softmax": summarize(
                    row["paired_kl_pann_softmax"] for row in chosen
                ),
                "fad_vggish_diagnostic": _low_rank_frechet(
                    generated_vggish, reference_vggish
                ),
                "generated_vggish_windows": int(generated_vggish.shape[0]),
                "reference_vggish_windows": int(reference_vggish.shape[0]),
                "fd_pann": _low_rank_frechet(
                    generated_panns, reference_panns
                ),
            }

    metrics_root = root / "metrics"
    _atomic_jsonl(metrics_root / "distributional_per_output.jsonl", per_output)
    report = {
        "schema": "stable_audio_tools.sceneplan_dit_p10_distributional_metrics",
        "schema_version": 1,
        "status": "PASS",
        "evaluation_rows": len(panel),
        "domain_counts": domain_counts,
        "checkpoint_outputs": expected_outputs,
        "channel": "W",
        "vggish_backend": backend,
        "small_sample_warning": (
            f"FAD-VGGish and FD-PANN use {min(domain_counts.values())} matched clips "
            "per domain; they select checkpoints but do not replace the 1,000-row paper benchmark."
        ),
        "aggregates": aggregates,
    }
    atomic_json(metrics_root / "DISTRIBUTIONAL_SUMMARY.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
