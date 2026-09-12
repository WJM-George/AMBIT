#!/usr/bin/env python3
"""Add the current P10-v11 150k model to the frozen public-baseline tables."""

from __future__ import annotations

import argparse
import copy
import gc
import json
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

import score_p10_v11_stratified_3000_public_baselines as public


OURS_ID = "ours_sceneplan_foa_150k"
OURS_NAME = "Ours ScenePlan-FOA (150k candidate)"
DEFAULT_SOURCE_EVAL = public.DEFAULT_ROOT.parent


def _ours_paths(source_eval: Path) -> dict[str, str]:
    output_root = source_eval / "outputs/step_150000"
    paths: dict[str, str] = {}
    for path in output_root.glob("*/*/generated_foa_float32.wav"):
        panel_id = path.parent.name
        if panel_id in paths:
            raise RuntimeError(f"duplicate Ours output: {panel_id}")
        paths[panel_id] = str(path.resolve(strict=True))
    if len(paths) != 3000:
        raise RuntimeError(f"Ours output count changed: {len(paths)} != 3000")
    return paths


def _selected_audio_items(
    inputs: dict[str, Any], ours_paths: dict[str, str]
) -> dict[str, str]:
    items: dict[str, str] = {}
    for domain in ("music", "sound"):
        for panel_id, row in inputs["domain_meta"][domain].items():
            items.setdefault(f"reference:{panel_id}", row["reference_foa_path"])
            items.setdefault(f"ours:{panel_id}", ours_paths[panel_id])
    if len(items) != 2248:
        raise RuntimeError(f"matched Ours audio item count changed: {len(items)} != 2248")
    return items


def _ours_lineage(
    inputs: dict[str, Any], source_eval: Path, arm: str
) -> dict[str, Any]:
    metadata = next(
        (source_eval / "outputs/step_150000").glob("*/*/metadata.json")
    )
    row = json.loads(metadata.read_text(encoding="utf-8"))
    return {
        **public._lineage(inputs, f"ours_{arm}"),
        "system_id": OURS_ID,
        "display_name": OURS_NAME,
        "checkpoint_step": 150000,
        "checkpoint_sha256": row["checkpoint_sha256"],
    }


@torch.inference_mode()
def _score_clap(
    inputs: dict[str, Any], ours_paths: dict[str, str], source_eval: Path,
    device: torch.device,
) -> dict[str, Any]:
    items = sorted(_selected_audio_items(inputs, ours_paths).items())
    model = load_clap_model("630k-audioset-fusion-best.pt", device=str(device))
    audio_embeddings: dict[str, torch.Tensor] = {}
    for start in range(0, len(items), 8):
        batch = items[start : start + 8]
        waves = torch.cat(
            [
                public._load_quality_mono(
                    path, 48_000, target_samples=480_000
                ).to(device)
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
                    "event": "ours_clap_audio",
                    "completed": min(start + len(batch), len(items)),
                    "total": len(items),
                }
            ),
            flush=True,
        )

    captions: dict[str, str] = {}
    for domain in ("music", "sound"):
        for panel_id, row in inputs["domain_meta"][domain].items():
            captions.setdefault(panel_id, row["semantic_prompt"])
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
        generated = torch.stack(
            [audio_embeddings[f"ours:{panel_id}"] for panel_id in panel_ids]
        )
        domains[domain] = {
            "display_name": OURS_NAME,
            "rows": len(panel_ids),
            "clap_text_audio_cosine": public._summary(
                [
                    float(audio_embeddings[f"ours:{panel_id}"] @ text_embeddings[panel_id])
                    for panel_id in panel_ids
                ]
            ),
            "paired_generated_reference_clap_cosine": public._summary(
                [
                    float(
                        audio_embeddings[f"ours:{panel_id}"]
                        @ audio_embeddings[f"reference:{panel_id}"]
                    )
                    for panel_id in panel_ids
                ]
            ),
            "fd_clap": public._low_rank_frechet(generated, reference),
        }
    return {
        **_ours_lineage(inputs, source_eval, "clap"),
        "domains": domains,
        "protocol": "Identical to the frozen public CLAP arm.",
    }


@torch.inference_mode()
def _score_vggish(
    inputs: dict[str, Any], ours_paths: dict[str, str], source_eval: Path,
    device: torch.device,
) -> dict[str, Any]:
    items = sorted(_selected_audio_items(inputs, ours_paths).items())
    model, processor, backend = load_vggish_model(str(device))
    if backend != "torchaudio" or processor is None:
        raise RuntimeError("frozen benchmark requires torchaudio VGGish preprocessing")
    embeddings: dict[str, torch.Tensor] = {}
    for index, (key, path) in enumerate(items, start=1):
        waveform = public._load_quality_mono(path, 16_000)[0]
        examples = processor(waveform)
        embeddings[key] = torch.cat(
            [
                model(examples[start : start + 32].to(device)).float().cpu()
                for start in range(0, int(examples.shape[0]), 32)
            ],
            dim=0,
        )
        if index % 25 == 0 or index == len(items):
            print(
                json.dumps(
                    {"event": "ours_vggish", "completed": index, "total": len(items)}
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
        generated = torch.cat(
            [embeddings[f"ours:{panel_id}"] for panel_id in panel_ids], dim=0
        )
        domains[domain] = {
            "display_name": OURS_NAME,
            "rows": len(panel_ids),
            "fad_vggish": public._feature_space_frechet(generated, reference),
        }
    return {
        **_ours_lineage(inputs, source_eval, "vggish"),
        "domains": domains,
        "protocol": "Identical to the frozen public VGGish arm.",
    }


@torch.inference_mode()
def _score_panns(
    inputs: dict[str, Any], ours_paths: dict[str, str], source_eval: Path,
    device: torch.device,
) -> dict[str, Any]:
    items = sorted(_selected_audio_items(inputs, ours_paths).items())
    model = load_panns_model(str(device))
    features: dict[str, dict[str, torch.Tensor]] = {}
    for start in range(0, len(items), 8):
        batch = items[start : start + 8]
        waves = torch.cat(
            [
                public._load_quality_mono(
                    path, 32_000, target_samples=320_000
                ).to(device)
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
                    "event": "ours_panns",
                    "completed": min(start + len(batch), len(items)),
                    "total": len(items),
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
        generated = torch.stack(
            [features[f"ours:{panel_id}"]["embedding"] for panel_id in panel_ids]
        )
        domains[domain] = {
            "display_name": OURS_NAME,
            "rows": len(panel_ids),
            "fd_pann": public._low_rank_frechet(generated, reference),
            "paired_kl_pann_softmax": public._summary(
                [
                    public._kl(
                        features[f"reference:{panel_id}"]["probability"],
                        features[f"ours:{panel_id}"]["probability"],
                    )
                    for panel_id in panel_ids
                ]
            ),
        }
    return {
        **_ours_lineage(inputs, source_eval, "panns"),
        "domains": domains,
        "protocol": "Identical to the frozen public PANN arm.",
    }


def _score_speech_shard(
    inputs: dict[str, Any], ours_paths: dict[str, str], source_eval: Path,
    *, device_index: int, shard_index: int, num_shards: int,
) -> dict[str, Any]:
    rows = [
        row
        for index, row in enumerate(inputs["speech_rows"])
        if index % num_shards == shard_index
    ]
    model = WhisperModel(
        str(public.DEFAULT_WHISPER.resolve(strict=True)),
        device="cuda",
        device_index=device_index,
        compute_type="float16",
    )
    device = torch.device(f"cuda:{device_index}")
    utmos, utmos_error = public._load_utmos(device)
    outputs = []
    for index, row in enumerate(rows, start=1):
        outputs.append(
            public._speech_output(
                row, OURS_ID, ours_paths[row["panel_id"]], model, utmos, device
            )
        )
        print(
            json.dumps(
                {
                    "event": "ours_speech",
                    "shard": shard_index,
                    "completed_rows": index,
                    "total_rows": len(rows),
                }
            ),
            flush=True,
        )
    return {
        **_ours_lineage(inputs, source_eval, "speech"),
        "shard_index": shard_index,
        "num_shards": num_shards,
        "rows": len(rows),
        "outputs": outputs,
        "utmos_error": utmos_error,
    }


def _fmt(value: float | None, digits: int = 3) -> str:
    return "N/A" if value is None else f"{value:.{digits}f}"


def _merge(
    inputs: dict[str, Any], source_eval: Path, root: Path, num_speech_shards: int
) -> dict[str, Any]:
    frozen = json.loads(
        (root / "metrics/PUBLIC_BASELINE_METRICS.json").read_text(encoding="utf-8")
    )
    partial = root / "metrics/ours_150k_partials"
    clap = json.loads((partial / "CLAP.json").read_text(encoding="utf-8"))
    vggish = json.loads((partial / "VGGISH.json").read_text(encoding="utf-8"))
    panns = json.loads((partial / "PANNS.json").read_text(encoding="utf-8"))
    expected_contract = public._sha256(inputs["contract_path"])
    expected_manifest = public._sha256(inputs["manifest_path"])
    for report in (clap, vggish, panns):
        if report["contract_sha256"] != expected_contract:
            raise RuntimeError(f"Ours contract mismatch in {report['arm']}")
        if report["manifest_sha256"] != expected_manifest:
            raise RuntimeError(f"Ours manifest mismatch in {report['arm']}")

    report = copy.deepcopy(frozen)
    report["schema"] = "sceneplan_foa.p10_v11_model_and_public_baseline_metrics"
    report["schema_version"] = 1
    report["ours_system_id"] = OURS_ID
    report["ours_checkpoint_step"] = 150000
    report["ours_checkpoint_sha256"] = clap["checkpoint_sha256"]
    for domain in ("music", "sound"):
        report["audio_domains"][domain][OURS_ID] = {
            **clap["domains"][domain],
            **vggish["domains"][domain],
            **panns["domains"][domain],
        }

    speech_outputs = []
    utmos_errors = []
    for shard in range(num_speech_shards):
        shard_report = json.loads(
            (
                partial
                / f"SPEECH_SHARD_{shard:02d}_OF_{num_speech_shards:02d}.json"
            ).read_text(encoding="utf-8")
        )
        speech_outputs.extend(shard_report["outputs"])
        if shard_report.get("utmos_error"):
            utmos_errors.append(shard_report["utmos_error"])
    if len(speech_outputs) != 469 or len(
        {row["panel_id"] for row in speech_outputs}
    ) != 469:
        raise RuntimeError(f"incomplete Ours Speech scores: {len(speech_outputs)}")
    report["speech"][OURS_ID] = {
        "display_name": OURS_NAME,
        **public._aggregate_speech(speech_outputs),
    }
    report["utmos_errors"] = sorted(
        set(report.get("utmos_errors", []) + utmos_errors)
    )
    report["status"] = "PASS"
    public._atomic_json(root / "metrics/MODEL_AND_BASELINE_METRICS.json", report)

    for domain in ("music", "sound"):
        rows = report["audio_domains"][domain]
        order = [public.REFERENCE_ID, OURS_ID] + [
            system_id
            for system_id in rows
            if system_id not in {public.REFERENCE_ID, OURS_ID}
        ]
        lines = [
            f"# {domain.title()} — Ours and frozen public baselines",
            "",
            "| System | CLAP ↑ | Paired CLAP ↑ | FD-CLAP ↓ | FAD-VGGish ↓ | FD-PANN ↓ | KL-PANN ↓ |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for system_id in order:
            row = rows[system_id]
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
                "Ours and every listed public system use the identical compatible 745-item slice and frozen metric implementation.",
                "",
            ]
        )
        public._atomic_text(
            root / f"tables/{domain.upper()}_TABLE_WITH_OURS.md", "\n".join(lines)
        )

    speech = report["speech"]
    lines = [
        "# Speech — Ours and frozen public baseline",
        "",
        "| System | WER ↓ | Seen-spk WER ↓ | Unseen-spk WER ↓ | CER ↓ | UTMOS ↑ |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for system_id in (public.REFERENCE_ID, OURS_ID, "qwen3_tts_1p7b_voice_design"):
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
            "All systems use the same 469 single-source Speech rows. Qwen3-TTS is non-spatial; Ours generates native FOA.",
            "",
        ]
    )
    public._atomic_text(root / "tables/SPEECH_TABLE_WITH_OURS.md", "\n".join(lines))
    (root / "MODEL_AND_BASELINE_METRICS_COMPLETE").write_text(
        "PASS\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, default=public.DEFAULT_ROOT)
    parser.add_argument("--source-eval", type=Path, default=DEFAULT_SOURCE_EVAL)
    parser.add_argument(
        "--arm", choices=("clap", "vggish", "panns", "speech", "merge"), required=True
    )
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()

    root = args.benchmark_root.expanduser().resolve(strict=True)
    source_eval = args.source_eval.expanduser().resolve(strict=True)
    inputs = public._load_inputs(root)
    ours_paths = _ours_paths(source_eval)
    partial = root / "metrics/ours_150k_partials"
    partial.mkdir(parents=True, exist_ok=True)
    if args.arm == "merge":
        report = _merge(inputs, source_eval, root, args.num_shards)
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "metrics": str(root / "metrics/MODEL_AND_BASELINE_METRICS.json"),
                    "tables": str(root / "tables"),
                }
            ),
            flush=True,
        )
        return 0

    if not torch.cuda.is_available():
        raise RuntimeError("Ours metric scoring requires CUDA")
    device = torch.device(f"cuda:{args.device_index}")
    if args.arm == "clap":
        report = _score_clap(inputs, ours_paths, source_eval, device)
        output = partial / "CLAP.json"
    elif args.arm == "vggish":
        report = _score_vggish(inputs, ours_paths, source_eval, device)
        output = partial / "VGGISH.json"
    elif args.arm == "panns":
        report = _score_panns(inputs, ours_paths, source_eval, device)
        output = partial / "PANNS.json"
    else:
        report = _score_speech_shard(
            inputs,
            ours_paths,
            source_eval,
            device_index=args.device_index,
            shard_index=args.shard_index,
            num_shards=args.num_shards,
        )
        output = partial / (
            f"SPEECH_SHARD_{args.shard_index:02d}_OF_{args.num_shards:02d}.json"
        )
    public._atomic_json(output, report)
    print(json.dumps({"status": "PASS", "arm": args.arm, "output": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
