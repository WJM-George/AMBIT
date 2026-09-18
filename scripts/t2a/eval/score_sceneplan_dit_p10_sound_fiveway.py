#!/usr/bin/env python3
"""Score and package the simple-sound Reference/VAE/r5/r6 five-way panel."""

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

from scripts.t2a.eval.compare_sceneplan_dit_p10_r5_r6 import build_montage
from scripts.t2a.eval.generate_sceneplan_dit_p10_panel import (
    _atomic_wav,
    _qc,
    _virtual_stereo,
)
from scripts.t2a.eval.sceneplan_dit_p10_panel_common import (
    atomic_json,
    load_foa,
    read_jsonl,
    sha256_file,
    summarize,
)
from scripts.t2a.eval.score_sceneplan_dit_p10_clap import (
    _audio_embeddings,
    _low_rank_frechet as _clap_frechet,
    _text_embeddings,
)
from scripts.t2a.eval.score_sceneplan_dit_p10_core import (
    _activity_metrics,
    _doa_metrics,
)
from scripts.t2a.eval.score_sceneplan_dit_p10_distributional import (
    _kl,
    _low_rank_frechet as _distributional_frechet,
    _panns_outputs,
    _vggish_embeddings,
)
from stable_audio_tools.training.metrics.fad_metrics import (
    load_clap_model,
    load_panns_model,
    load_vggish_model,
)


DEFAULT_ROOT = Path(
    os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/archives/p10_eval/"
    "p10_sound_simple_5way_v1"
)
SYSTEMS = (
    "reference",
    "vae_1p35m_reconstruction",
    "r5_50k",
    "r6_10k",
    "r6_20k",
)
MODEL_STEPS = {"r5_50k": 50_000, "r6_10k": 10_000, "r6_20k": 20_000}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    temporary.replace(path)


def _output_metadata(root: Path, step: int, panel_id: str) -> dict[str, Any]:
    path = (
        root
        / "outputs"
        / f"step_{step:06d}"
        / "sound"
        / panel_id
        / "metadata.json"
    )
    value = _read_json(path)
    if not (
        value.get("status") == "PASS"
        and int(value["checkpoint_step"]) == step
        and value["panel_id"] == panel_id
    ):
        raise RuntimeError(f"invalid generated metadata: {path}")
    return value


def _collect_items(root: Path, panel: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for panel_row in panel:
        panel_id = str(panel_row["panel_id"])
        expected_samples = int(panel_row["model_num_samples"])
        reference_path = Path(panel_row["reference_foa_path"]).resolve(strict=True)
        reference, sample_rate = load_foa(
            reference_path, expected_samples=expected_samples
        )
        if sample_rate != 44_100:
            raise RuntimeError("reference sample rate changed")
        if sha256_file(reference_path) != panel_row["reference_foa_sha256"]:
            raise RuntimeError(f"reference hash mismatch: {reference_path}")
        reference_preview, reference_preview_info = _virtual_stereo(reference)
        reference_preview_path = (
            root / "references" / "sound" / panel_id / "reference_stereo.wav"
        )
        _atomic_wav(
            reference_preview_path,
            reference_preview,
            sample_rate,
            subtype="PCM_16",
        )
        rows.append(
            {
                "system": "reference",
                "panel_id": panel_id,
                "sample_id": panel_row["sample_id"],
                "raw_foa_path": str(reference_path),
                "raw_foa_sha256": panel_row["reference_foa_sha256"],
                "stereo_path": str(reference_preview_path.resolve()),
                "stereo_sha256": sha256_file(reference_preview_path),
                "qc": _qc(reference),
                "preview": reference_preview_info,
            }
        )

        vae_path = root / "vae_reconstruction" / panel_id / "metadata.json"
        vae = _read_json(vae_path)
        if not (
            vae.get("status") == "PASS"
            and vae["panel_id"] == panel_id
            and vae["sample_id"] == panel_row["sample_id"]
        ):
            raise RuntimeError(f"invalid VAE metadata: {vae_path}")
        vae_raw = Path(vae["reconstruction_foa_path"]).resolve(strict=True)
        vae_preview = Path(vae["reconstruction_stereo_path"]).resolve(strict=True)
        if sha256_file(vae_raw) != vae["reconstruction_foa_sha256"]:
            raise RuntimeError(f"VAE raw hash mismatch: {vae_raw}")
        if sha256_file(vae_preview) != vae["reconstruction_stereo_sha256"]:
            raise RuntimeError(f"VAE preview hash mismatch: {vae_preview}")
        rows.append(
            {
                "system": "vae_1p35m_reconstruction",
                "panel_id": panel_id,
                "sample_id": panel_row["sample_id"],
                "raw_foa_path": str(vae_raw),
                "raw_foa_sha256": vae["reconstruction_foa_sha256"],
                "stereo_path": str(vae_preview),
                "stereo_sha256": vae["reconstruction_stereo_sha256"],
                "qc": vae["qc"],
                "vae_w_channel_si_sdr_db": vae["w_channel_si_sdr_db"],
                "vae_all_channel_rmse": vae["all_channel_rmse"],
            }
        )

        for system, step in MODEL_STEPS.items():
            generated = _output_metadata(root, step, panel_id)
            if generated["sample_id"] != panel_row["sample_id"]:
                raise RuntimeError(f"sample mismatch for {system}/{panel_id}")
            raw_path = Path(generated["generated_foa_path"]).resolve(strict=True)
            preview_path = Path(generated["generated_stereo_path"]).resolve(strict=True)
            if sha256_file(raw_path) != generated["generated_foa_sha256"]:
                raise RuntimeError(f"generated raw hash mismatch: {raw_path}")
            if sha256_file(preview_path) != generated["generated_stereo_sha256"]:
                raise RuntimeError(f"generated preview hash mismatch: {preview_path}")
            rows.append(
                {
                    "system": system,
                    "checkpoint_step": step,
                    "panel_id": panel_id,
                    "sample_id": panel_row["sample_id"],
                    "raw_foa_path": str(raw_path),
                    "raw_foa_sha256": generated["generated_foa_sha256"],
                    "stereo_path": str(preview_path),
                    "stereo_sha256": generated["generated_stereo_sha256"],
                    "qc": generated["qc"],
                    "noise_seed": generated["noise_seed"],
                }
            )
    expected = len(panel) * len(SYSTEMS)
    if len(rows) != expected:
        raise RuntimeError(f"five-way item count changed: {len(rows)} != {expected}")
    return rows


def _metric_mean(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    return summarize(row["metrics"].get(key) for row in rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--clap-model", default="630k-audioset-fusion-best.pt")
    args = parser.parse_args()
    root = args.eval_root.expanduser().resolve(strict=True)
    panel = read_jsonl(root / "SOUND_SIMPLE_PANEL_5.jsonl")
    if len(panel) != 5 or any(row.get("domain") != "sound" for row in panel):
        raise RuntimeError("expected the frozen five-row sound panel")
    panel_by_id = {str(row["panel_id"]): row for row in panel}
    items = _collect_items(root, panel)
    item_by_key = {(row["system"], row["panel_id"]): row for row in items}

    # Core FOA/QC metrics use the untouched four-channel files.
    for index, row in enumerate(items, start=1):
        panel_row = panel_by_id[row["panel_id"]]
        audio, sample_rate = load_foa(
            row["raw_foa_path"],
            expected_samples=int(panel_row["model_num_samples"]),
        )
        if sample_rate != 44_100:
            raise RuntimeError("five-way sample rate changed")
        row["metrics"] = {
            "raw_peak": float(audio.abs().max()),
            "raw_rms": float(audio.square().mean().sqrt()),
            "raw_fraction_abs_ge_1": float(
                audio.abs().ge(1.0).to(torch.float32).mean()
            ),
        }
        row["doa"] = _doa_metrics(
            audio,
            panel_row["scene_plan"],
            model_num_samples=int(panel_row["model_num_samples"]),
            latent_frames=int(panel_row["latent_frames_valid"]),
        )
        row["activity"] = _activity_metrics(
            audio,
            panel_row["scene_plan"],
            model_num_samples=int(panel_row["model_num_samples"]),
            latent_frames=int(panel_row["latent_frames_valid"]),
        )
        row["metrics"].update(
            {
                "plan_spherical_error_mean_deg": row["doa"][
                    "spherical_error_mean_deg"
                ],
                "valid_direction_fraction": row["doa"][
                    "valid_direction_fraction"
                ],
                "activity_temporal_iou": row["activity"]["temporal_iou"],
            }
        )
        print(
            json.dumps(
                {
                    "event": "fiveway_core",
                    "index": index,
                    "count": len(items),
                    "system": row["system"],
                    "panel_id": row["panel_id"],
                }
            ),
            flush=True,
        )

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("five-way semantic metrics require CUDA")
    audio_items = [
        (f"{row['system']}:{row['panel_id']}", row["raw_foa_path"])
        for row in items
    ]
    captions = {
        str(row["panel_id"]): str(row["semantic_text"]) for row in panel
    }

    clap_model = load_clap_model(args.clap_model, device=str(device))
    clap_audio = _audio_embeddings(clap_model, audio_items, device)
    clap_text = _text_embeddings(clap_model, captions)
    for row in items:
        panel_id = row["panel_id"]
        embedding = clap_audio[f"{row['system']}:{panel_id}"]
        reference = clap_audio[f"reference:{panel_id}"]
        candidate_ids = [str(item["panel_id"]) for item in panel]
        candidate_text = torch.stack([clap_text[value] for value in candidate_ids])
        retrieval = embedding @ candidate_text.transpose(0, 1)
        predicted = candidate_ids[int(retrieval.argmax())]
        row["metrics"].update(
            {
                "clap_text_audio_cosine": float(embedding @ clap_text[panel_id]),
                "paired_reference_clap_cosine": float(embedding @ reference),
                "within_panel_text_retrieval_correct": predicted == panel_id,
            }
        )
        row["clap_retrieval_prediction"] = predicted
    del clap_model
    gc.collect()
    torch.cuda.empty_cache()

    vggish_model, vggish_processor, vggish_backend = load_vggish_model(str(device))
    if vggish_backend != "torchaudio" or vggish_processor is None:
        raise RuntimeError("the frozen panel requires torchaudio VGGish preprocessing")
    vggish = _vggish_embeddings(
        vggish_model, vggish_processor, audio_items, device
    )
    del vggish_model
    gc.collect()
    torch.cuda.empty_cache()
    panns_model = load_panns_model(str(device))
    panns = _panns_outputs(panns_model, audio_items, device)
    del panns_model
    gc.collect()
    torch.cuda.empty_cache()
    for row in items:
        key = f"{row['system']}:{row['panel_id']}"
        reference_key = f"reference:{row['panel_id']}"
        row["metrics"]["paired_kl_pann_softmax"] = _kl(
            panns[reference_key]["probability"], panns[key]["probability"]
        )

    aggregates: dict[str, Any] = {}
    for system in SYSTEMS:
        chosen = [row for row in items if row["system"] == system]
        generated_clap = torch.stack(
            [clap_audio[f"{system}:{row['panel_id']}"] for row in chosen]
        )
        reference_clap = torch.stack(
            [clap_audio[f"reference:{row['panel_id']}"] for row in chosen]
        )
        generated_vggish = torch.cat(
            [vggish[f"{system}:{row['panel_id']}"] for row in chosen], dim=0
        )
        reference_vggish = torch.cat(
            [vggish[f"reference:{row['panel_id']}"] for row in chosen], dim=0
        )
        generated_panns = torch.stack(
            [panns[f"{system}:{row['panel_id']}"]["embedding"] for row in chosen]
        )
        reference_panns = torch.stack(
            [panns[f"reference:{row['panel_id']}"]["embedding"] for row in chosen]
        )
        aggregates[system] = {
            "rows": len(chosen),
            "clap_text_audio_cosine": _metric_mean(
                chosen, "clap_text_audio_cosine"
            ),
            "paired_reference_clap_cosine": _metric_mean(
                chosen, "paired_reference_clap_cosine"
            ),
            "within_panel_text_retrieval_top1": sum(
                bool(row["metrics"]["within_panel_text_retrieval_correct"])
                for row in chosen
            )
            / len(chosen),
            "fd_clap_diagnostic_n5": _clap_frechet(
                generated_clap, reference_clap
            ),
            "fad_vggish_diagnostic_n5": _distributional_frechet(
                generated_vggish, reference_vggish
            ),
            "fd_pann_diagnostic_n5": _distributional_frechet(
                generated_panns, reference_panns
            ),
            "paired_kl_pann_softmax": _metric_mean(
                chosen, "paired_kl_pann_softmax"
            ),
            "plan_spherical_error_mean_deg": _metric_mean(
                chosen, "plan_spherical_error_mean_deg"
            ),
            "valid_direction_fraction": _metric_mean(
                chosen, "valid_direction_fraction"
            ),
            "activity_temporal_iou": _metric_mean(
                chosen, "activity_temporal_iou"
            ),
            "raw_peak": _metric_mean(chosen, "raw_peak"),
            "raw_rms": _metric_mean(chosen, "raw_rms"),
            "raw_fraction_abs_ge_1": _metric_mean(
                chosen, "raw_fraction_abs_ge_1"
            ),
        }

    metrics_root = root / "metrics"
    _atomic_jsonl(metrics_root / "FIVE_WAY_PER_SAMPLE.jsonl", items)
    summary = {
        "schema": "stable_audio_tools.sceneplan_dit_p10_sound_fiveway_metrics",
        "schema_version": 1,
        "status": "PASS",
        "panel_rows": len(panel),
        "systems": list(SYSTEMS),
        "audio_files": len(items),
        "channel_for_semantic_metrics": "W",
        "raw_spatial_metrics": "four-channel WYZX/ACN/SN3D",
        "clap_model": args.clap_model,
        "vggish_backend": vggish_backend,
        "small_sample_warning": (
            "N=5. FAD/FD and rank metrics are matched diagnostic signals only, "
            "not publication-grade population estimates."
        ),
        "aggregates": aggregates,
    }
    atomic_json(metrics_root / "FIVE_WAY_METRICS.json", summary)

    montage_root = root / "listening_montages"
    montage_index: dict[str, Any] = {
        "status": "PASS",
        "system_order": list(SYSTEMS),
        "panel_order": [row["panel_id"] for row in panel],
        "by_system": {},
        "by_case": {},
    }
    for system in SYSTEMS:
        paths = [
            Path(item_by_key[(system, row["panel_id"])]["stereo_path"])
            for row in panel
        ]
        montage_index["by_system"][system] = build_montage(
            paths, montage_root / "by_system" / f"{system}_sound_5.wav"
        )
    for panel_row in panel:
        panel_id = str(panel_row["panel_id"])
        paths = [
            Path(item_by_key[(system, panel_id)]["stereo_path"])
            for system in SYSTEMS
        ]
        entry = build_montage(
            paths, montage_root / "by_case" / f"{panel_id}_fiveway.wav"
        )
        entry["demo_name"] = panel_row["demo_name"]
        entry["semantic_text"] = panel_row["semantic_text"]
        entry["renderer_caption"] = panel_row["renderer_caption"]
        entry["system_order"] = list(SYSTEMS)
        montage_index["by_case"][panel_id] = entry
    atomic_json(montage_root / "MONTAGE_INDEX.json", montage_index)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
