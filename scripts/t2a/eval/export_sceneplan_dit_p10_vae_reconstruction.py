#!/usr/bin/env python3
"""Export the frozen 1.35M WDMix VAE codec ceiling for a P10 panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import soundfile as sf
import torch
import torch.nn.functional as F

from scripts.t2a.eval.generate_sceneplan_dit_p10_panel import (
    _atomic_wav,
    _qc,
    _virtual_stereo,
)
from scripts.t2a.eval.sceneplan_dit_p10_panel_common import load_panel
from stable_audio_tools.configuration import load_config
from stable_audio_tools.models import create_model_from_config
from stable_audio_tools.models.utils import copy_state_dict, load_ckpt_state_dict


DEFAULT_ROOT = Path(
    os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/archives/p10_eval/"
    "p10_ckpt_5k_10k_15k_sceneplan44_v1"
)
DEFAULT_CONFIG = REPO_ROOT / (
    "stable_audio_tools/configs/model_configs/autoencoders/"
    "stable_audio_4ch_vae_ds1024_z64_wdmix_scm.json"
)
DEFAULT_CHECKPOINT = Path(
    os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/compareVAE_ckpt/unwrapped_wdmix_1350000.ckpt"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _si_sdr(reference: torch.Tensor, estimate: torch.Tensor) -> float:
    reference = reference.to(torch.float64).flatten()
    estimate = estimate.to(torch.float64).flatten()
    reference = reference - reference.mean()
    estimate = estimate - estimate.mean()
    projection = (
        (estimate @ reference) / reference.square().sum().clamp_min(1.0e-12)
    ) * reference
    noise = estimate - projection
    return float(
        10.0
        * torch.log10(
            projection.square().sum().clamp_min(1.0e-12)
            / noise.square().sum().clamp_min(1.0e-12)
        )
    )


def _summarize_existing(
    root: Path,
    panel: list[dict[str, Any]],
    config_path: Path,
    checkpoint_path: Path,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for panel_row in panel:
        metadata_path = (
            root / "vae_reconstruction" / str(panel_row["panel_id"]) / "metadata.json"
        )
        if not metadata_path.is_file():
            raise FileNotFoundError(f"missing VAE reconstruction metadata: {metadata_path}")
        value = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            value.get("status") != "PASS"
            or value.get("panel_id") != panel_row["panel_id"]
            or value.get("sample_id") != panel_row["sample_id"]
            or not Path(value["reconstruction_foa_path"]).is_file()
        ):
            raise RuntimeError(f"invalid VAE reconstruction metadata: {metadata_path}")
        rows.append(value)
    summary = {
        "schema": "stable_audio_tools.sceneplan_dit_p10_vae_reconstruction_summary",
        "schema_version": 2,
        "status": "PASS",
        "rows": len(rows),
        "domain_counts": {
            domain: sum(row.get("domain") == domain for row in panel)
            for domain in ("music", "sound", "speech")
        },
        "vae_config": str(config_path),
        "vae_config_sha256": _sha256_file(config_path),
        "vae_checkpoint": str(checkpoint_path),
        "vae_checkpoint_sha256": _sha256_file(checkpoint_path),
        "mean_w_channel_si_sdr_db": sum(
            float(row["w_channel_si_sdr_db"]) for row in rows
        )
        / len(rows),
        "mean_all_channel_rmse": sum(
            float(row["all_channel_rmse"]) for row in rows
        )
        / len(rows),
        "outputs": [row["reconstruction_foa_path"] for row in rows],
    }
    _atomic_json(root / "vae_reconstruction" / "SUMMARY.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--panel", type=Path)
    parser.add_argument("--vae-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--vae-checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()

    root = args.eval_root.expanduser().resolve(strict=True)
    config_path = args.vae_config.expanduser().resolve(strict=True)
    checkpoint_path = args.vae_checkpoint.expanduser().resolve(strict=True)
    panel = (
        _read_jsonl(args.panel.expanduser().resolve(strict=True))
        if args.panel is not None
        else load_panel(root)
    )
    if not panel or any(row.get("domain") not in {"music", "sound", "speech"} for row in panel):
        raise RuntimeError("VAE comparison requires a non-empty P10 listening panel")
    if len({str(row["panel_id"]) for row in panel}) != len(panel):
        raise RuntimeError("VAE comparison panel IDs must be unique")
    if args.summarize_only:
        summary = _summarize_existing(root, panel, config_path, checkpoint_path)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return 0
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError(
            f"invalid VAE shard {args.shard_index}/{args.num_shards}"
        )
    selected_panel = [
        row for index, row in enumerate(panel) if index % args.num_shards == args.shard_index
    ]
    if not selected_panel:
        raise RuntimeError(
            f"VAE shard {args.shard_index}/{args.num_shards} selected no rows"
        )

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("the frozen VAE comparison requires CUDA")
    config = load_config(config_path)
    if int(config["sample_rate"]) != 44_100 or int(config["audio_channels"]) != 4:
        raise RuntimeError("the frozen WDMix VAE must be 44.1 kHz native FOA")
    model = create_model_from_config(config)
    copy_state_dict(model, load_ckpt_state_dict(str(checkpoint_path)))
    model.eval().requires_grad_(False).to(device)
    downsampling_ratio = int(getattr(model, "downsampling_ratio", 1024))
    if downsampling_ratio != 1024:
        raise RuntimeError(f"VAE downsampling ratio changed: {downsampling_ratio}")

    rows: list[dict[str, Any]] = []
    for index, row in enumerate(selected_panel, start=1):
        started = time.monotonic()
        sample_root = root / "vae_reconstruction" / str(row["panel_id"])
        metadata_path = sample_root / "metadata.json"
        if metadata_path.is_file() and not args.force:
            existing = json.loads(metadata_path.read_text(encoding="utf-8"))
            if (
                existing.get("status") == "PASS"
                and existing.get("panel_id") == row["panel_id"]
                and existing.get("sample_id") == row["sample_id"]
                and Path(existing["reconstruction_foa_path"]).is_file()
            ):
                rows.append(existing)
                print(
                    json.dumps(
                        {
                            "event": "vae_reconstruction_skip_valid",
                            "index": index,
                            "count": len(selected_panel),
                            "panel_id": row["panel_id"],
                        }
                    ),
                    flush=True,
                )
                continue
        reference_path = Path(row["reference_foa_path"]).resolve(strict=True)
        reference_np, sample_rate = sf.read(
            reference_path, dtype="float32", always_2d=True
        )
        reference = torch.from_numpy(reference_np.T.copy())
        expected_samples = int(row["model_num_samples"])
        if sample_rate != 44_100 or tuple(reference.shape) != (4, expected_samples):
            raise RuntimeError(
                f"reference geometry changed for {row['panel_id']}: "
                f"rate={sample_rate}, shape={tuple(reference.shape)}"
            )
        latent_frames = int(row["latent_frames_valid"])
        padded_samples = latent_frames * downsampling_ratio
        if padded_samples < expected_samples:
            raise RuntimeError(f"negative VAE padding for {row['panel_id']}")
        padded = F.pad(reference, (0, padded_samples - expected_samples)).unsqueeze(0)
        with torch.inference_mode():
            latent = model.encode(padded.to(device))
            reconstruction = model.decode(latent)[0, :, :expected_samples]
        reconstruction = reconstruction.to(torch.float32).cpu().contiguous()
        if tuple(latent.shape) != (1, 64, latent_frames):
            raise RuntimeError(
                f"latent geometry changed for {row['panel_id']}: {tuple(latent.shape)}"
            )
        qc = _qc(reconstruction)
        if not qc.get("finite") or qc.get("channels") != 4:
            raise RuntimeError(f"invalid VAE reconstruction for {row['panel_id']}: {qc}")
        preview, preview_info = _virtual_stereo(reconstruction)
        raw_path = sample_root / "reconstruction_foa_float32.wav"
        preview_path = sample_root / "reconstruction_stereo.wav"
        _atomic_wav(raw_path, reconstruction, sample_rate, subtype="FLOAT")
        _atomic_wav(preview_path, preview, sample_rate, subtype="PCM_16")
        result = {
            "schema": "stable_audio_tools.sceneplan_dit_p10_vae_reconstruction",
            "schema_version": 1,
            "status": "PASS",
            "system_label": "vae_1p35m_reconstruction",
            "panel_id": row["panel_id"],
            "sample_id": row["sample_id"],
            "demo_name": row.get("demo_name"),
            "semantic_text": row["semantic_text"],
            "renderer_caption": row["renderer_caption"],
            "scene_plan": row["scene_plan"],
            "sample_rate": sample_rate,
            "model_num_samples": expected_samples,
            "latent_frames": latent_frames,
            "padded_num_samples": padded_samples,
            "latent_channels": int(latent.shape[1]),
            "reference_foa_path": str(reference_path),
            "reference_foa_sha256": row["reference_foa_sha256"],
            "reconstruction_foa_path": str(raw_path.resolve()),
            "reconstruction_foa_sha256": _sha256_file(raw_path),
            "reconstruction_stereo_path": str(preview_path.resolve()),
            "reconstruction_stereo_sha256": _sha256_file(preview_path),
            "w_channel_si_sdr_db": _si_sdr(reference[0], reconstruction[0]),
            "all_channel_rmse": float(
                (reference - reconstruction).square().mean().sqrt()
            ),
            "qc": qc,
            "seconds": round(time.monotonic() - started, 3),
            **preview_info,
        }
        _atomic_json(metadata_path, result)
        rows.append(result)
        print(
            json.dumps(
                {
                    "event": "vae_reconstruction_complete",
                    "index": index,
                    "count": len(selected_panel),
                    "panel_id": row["panel_id"],
                    "w_si_sdr_db": result["w_channel_si_sdr_db"],
                    "seconds": result["seconds"],
                }
            ),
            flush=True,
        )
        del latent

    shard_summary = {
        "schema": "stable_audio_tools.sceneplan_dit_p10_vae_reconstruction_shard",
        "schema_version": 1,
        "status": "PASS",
        "shard_index": int(args.shard_index),
        "num_shards": int(args.num_shards),
        "rows": len(rows),
        "mean_w_channel_si_sdr_db": sum(
            float(row["w_channel_si_sdr_db"]) for row in rows
        ) / len(rows),
        "mean_all_channel_rmse": sum(
            float(row["all_channel_rmse"]) for row in rows
        ) / len(rows),
    }
    _atomic_json(
        root
        / "vae_reconstruction"
        / "shards"
        / f"shard_{args.shard_index:02d}_of_{args.num_shards:02d}.json",
        shard_summary,
    )
    if args.num_shards == 1:
        summary = _summarize_existing(root, panel, config_path, checkpoint_path)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    else:
        print(json.dumps(shard_summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
