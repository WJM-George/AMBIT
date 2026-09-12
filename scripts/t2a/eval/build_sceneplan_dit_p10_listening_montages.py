#!/usr/bin/env python3
"""Build explicit Reference/VAE/checkpoint listening montages."""

from __future__ import annotations
import os

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import soundfile as sf
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.t2a.eval.sceneplan_44_eval_common import atomic_wav
from scripts.t2a.eval.sceneplan_dit_p10_panel_common import (
    atomic_json,
    checkpoint_steps,
    sha256_file,
)


DEFAULT_EVAL_ROOT = Path(
    os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/archives//p10_pre_v11_20260831/sceneplan_dit_fail/"
    "sceneplan_dit_v4_r8_300m/evaluation/"
    "p10_ckpt_5k_10k_15k_sceneplan44_v1"
)
SILENCE_SECONDS = 0.75


def _load_stereo(path: str | Path) -> torch.Tensor:
    resolved = Path(path).expanduser().resolve(strict=True)
    value, sample_rate = sf.read(resolved, dtype="float32", always_2d=True)
    if sample_rate != 44_100 or value.shape[1] != 2:
        raise RuntimeError(
            f"listening preview must be 44.1 kHz stereo: {resolved}, "
            f"rate={sample_rate}, shape={value.shape}"
        )
    tensor = torch.from_numpy(value.T.copy())
    if not bool(torch.isfinite(tensor).all()):
        raise RuntimeError(f"non-finite listening preview: {resolved}")
    return tensor


def _montage(paths: list[str | Path], destination: Path) -> dict[str, Any]:
    clips = [_load_stereo(path) for path in paths]
    silence = torch.zeros(2, int(round(SILENCE_SECONDS * 44_100)))
    pieces = []
    for index, clip in enumerate(clips):
        if index:
            pieces.append(silence)
        pieces.append(clip)
    audio = torch.cat(pieces, dim=-1)
    atomic_wav(destination, audio, 44_100, subtype="PCM_16")
    return {
        "path": str(destination.resolve()),
        "sha256": sha256_file(destination),
        "clips": len(clips),
        "duration_sec": float(audio.shape[-1] / 44_100.0),
        "separator_sec": SILENCE_SECONDS,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", type=Path, default=DEFAULT_EVAL_ROOT)
    parser.add_argument(
        "--max-cases-per-domain",
        type=int,
        default=0,
        help="Limit listening montages per domain; zero keeps the entire panel.",
    )
    args = parser.parse_args()
    root = args.eval_root.expanduser().resolve(strict=True)
    listening = json.loads((root / "metrics/LISTENING_INDEX.json").read_text(encoding="utf-8"))
    steps = checkpoint_steps(root)
    step_labels = [f"{step // 1000}k" if step % 1000 == 0 else str(step) for step in steps]
    step_suffix = "_".join(step_labels)
    montage_root = root / "listening_montages"
    result: dict[str, Any] = {
        "schema": "stable_audio_tools.sceneplan_dit_p10_listening_montages",
        "schema_version": 1,
        "status": "PASS",
        "case_order": [
            "raw_reference",
            "vae_codec_ceiling",
            *(f"step_{step}" for step in steps),
        ],
        "cases": [],
        "domains": {},
    }

    panel_by_domain: dict[str, list[dict[str, Any]]] = {
        "music": [], "sound": [], "speech": []
    }
    chosen_panels = []
    kept_by_domain = {"music": 0, "sound": 0, "speech": 0}
    for panel in listening["panels"]:
        domain = panel["domain"]
        if (
            args.max_cases_per_domain > 0
            and kept_by_domain[domain] >= args.max_cases_per_domain
        ):
            continue
        kept_by_domain[domain] += 1
        chosen_panels.append(panel)

    for panel in chosen_panels:
        panel_by_domain[panel["domain"]].append(panel)
        generated = {int(row["step"]): row["generated_stereo_path"] for row in panel["checkpoints"]}
        if set(generated) != set(steps):
            raise RuntimeError(f"checkpoint set changed for {panel['panel_id']}")
        vae_metadata_path = root / "vae_reconstruction" / panel["panel_id"] / "metadata.json"
        vae = json.loads(vae_metadata_path.read_text(encoding="utf-8"))
        paths = [
            panel["reference_stereo_path"],
            vae["reconstruction_stereo_path"],
            *(generated[step] for step in steps),
        ]
        entry = _montage(
            paths,
            montage_root / "by_case" / f"{panel['panel_id']}_reference_vae_{step_suffix}.wav",
        )
        entry.update(
            {
                "panel_id": panel["panel_id"],
                "domain": panel["domain"],
                "sample_id": panel["sample_id"],
                "semantic_text": panel["semantic_text"],
                "renderer_caption": panel["renderer_caption"],
                "source_paths": [str(Path(path).resolve()) for path in paths],
            }
        )
        result["cases"].append(entry)

    for domain, panels in panel_by_domain.items():
        panels.sort(key=lambda row: row["panel_id"])
        systems: dict[str, list[str]] = {
            "raw_reference": [row["reference_stereo_path"] for row in panels],
            "vae_codec_ceiling": [],
            **{f"step_{step}": [] for step in steps},
        }
        for panel in panels:
            vae = json.loads(
                (root / "vae_reconstruction" / panel["panel_id"] / "metadata.json").read_text(encoding="utf-8")
            )
            systems["vae_codec_ceiling"].append(vae["reconstruction_stereo_path"])
            generated = {int(row["step"]): row["generated_stereo_path"] for row in panel["checkpoints"]}
            for step in steps:
                systems[f"step_{step}"].append(generated[step])
        result["domains"][domain] = {
            system: _montage(
                paths,
                montage_root / "by_domain" / f"{domain}_{system}.wav",
            )
            for system, paths in systems.items()
        }

    atomic_json(montage_root / "MONTAGE_INDEX.json", result)
    lines = [
        "# P10 listening montages",
        "",
        (
            "Every per-case montage uses this fixed order: raw reference → "
            "VAE codec ceiling → " + " → ".join(step_labels)
            + ", with 0.75 s silence between clips."
        ),
        "",
    ]
    for row in result["cases"]:
        lines.extend(
            [
                f"## {row['panel_id']}",
                "",
                row["semantic_text"],
                "",
                f"[Listen]({row['path']})",
                "",
            ]
        )
    (montage_root / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
