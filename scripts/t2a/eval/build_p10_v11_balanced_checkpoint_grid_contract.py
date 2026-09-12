#!/usr/bin/env python3
"""Freeze the 110k--150k semantic-v2 P10 checkpoint benchmark.

The panel is copied byte-for-byte from the established balanced 400x3
benchmark.  Public-baseline generations and the VAE ceiling are reused only
after their panel digest has been verified.  The only prompt-protocol change is
the explicitly frozen semantic-caption compiler v2 used by the continuation
run (``speaker says: transcript``).
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE = Path(
    os.environ.get("AMBIT_DATA_ROOT", "data") + "/sceneplan_v2_1p124m/revisions/"
    "speech_expansion_noalign_15s_v1/evaluation/"
    "p10_v9_balanced_1200_ckpt20k_100k_v1"
)
DEFAULT_OUTPUT = Path(
    os.environ.get("AMBIT_DATA_ROOT", "data") + "/sceneplan_v2_1p124m/revisions/"
    "speech_expansion_noalign_15s_v1/evaluation/"
    "p10_v11_balanced_1200_ckpt110k_150k_semantic_v2"
)
DEFAULT_MODEL_CONFIG = REPO_ROOT / (
    "stable_audio_tools/configs/model_configs/txt2audio/t2a/dit/"
    "qwen35_0p8b_300m_model_sceneplan_44_"
    "soundexp_noalign_15s_resume_cosine_40k.json"
)
CHECKPOINTS = (
    (
        110_000,
        Path(
            os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/dit/"
            "sceneplan_dit_v10_semantic_v2_protected_resume_110k/"
            "checkpoints/epoch=35-step=110000.ckpt"
        ),
    ),
    *(
        (
            step,
            Path(
                os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/dit/"
                "sceneplan_dit_v11_semantic_v2_protected_resume_150k/"
                f"checkpoints/{name}"
            ),
        )
        for step, name in (
            (120_000, "epoch=38-step=120000.ckpt"),
            (130_000, "epoch=41-step=130000.ckpt"),
            (140_000, "epoch=44-step=140000.ckpt"),
            (150_000, "epoch=48-step=150000.ckpt"),
        )
    ),
)


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _checkpoint(step: int, path: Path) -> dict[str, object]:
    resolved = path.expanduser().resolve(strict=True)
    return {
        "step": int(step),
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _safe_symlink(link: Path, target: Path) -> None:
    target = target.expanduser().resolve(strict=True)
    if link.is_symlink():
        if link.resolve(strict=True) != target:
            raise RuntimeError(f"existing symlink points elsewhere: {link}")
        return
    if link.exists():
        raise RuntimeError(f"refusing to replace existing path: {link}")
    link.symlink_to(target, target_is_directory=target.is_dir())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    args = parser.parse_args()

    source = args.source_root.expanduser().resolve(strict=True)
    output = args.output_root.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_contract_path = source / "EVAL_CONTRACT.json"
    source_contract = json.loads(source_contract_path.read_text(encoding="utf-8"))
    source_panel = source / source_contract["test_set"]["panel_filename"]
    source_panel_digest = _sha256(source_panel)
    if source_panel_digest != source_contract["test_set"]["panel_sha256"]:
        raise RuntimeError("source balanced panel digest changed")
    if source_contract["test_set"]["domain_counts"] != {
        "music": 400,
        "sound": 400,
        "speech": 400,
    }:
        raise RuntimeError("source panel is not the frozen balanced 400x3 panel")

    panel_path = output / "balanced_test_1200.jsonl"
    _atomic_text(panel_path, source_panel.read_text(encoding="utf-8"))
    if _sha256(panel_path) != source_panel_digest:
        raise RuntimeError("byte-preserving panel copy failed")

    model_config = args.model_config.expanduser().resolve(strict=True)
    checkpoints = [_checkpoint(step, path) for step, path in CHECKPOINTS]
    contract = copy.deepcopy(source_contract)
    contract.update(
        {
            "schema": "stable_audio_tools.p10_v11_balanced_checkpoint_grid_contract",
            "schema_version": 1,
            "status": "FROZEN_110K_150K_SEMANTIC_V2",
            "purpose": (
                "matched 110k/120k/130k/140k/150k checkpoint selection and "
                "frozen-public-baseline comparison"
            ),
            "checkpoints": checkpoints,
            "source_balanced_contract": str(source_contract_path),
            "source_balanced_contract_sha256": _sha256(source_contract_path),
            "protocol_delta": {
                "only_intended_delta": "semantic caption compiler v2 for formal Speech",
                "speech_template": "<speaker description> says: <exact transcript>",
                "music_sound_semantic_text_unchanged": True,
                "sceneplans_noise_sampling_panel_unchanged": True,
            },
        }
    )
    contract["sampling"].update(
        {
            "model_config": str(model_config),
            "model_config_sha256": _sha256(model_config),
            "semantic_caption_compiler_version": 2,
            "inference_batch_size": 1,
        }
    )
    contract["test_set"].update(
        {
            "panel_filename": panel_path.name,
            "panel_sha256": source_panel_digest,
        }
    )
    contract_path = output / "EVAL_CONTRACT.json"
    _atomic_text(
        contract_path,
        json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )

    # The VAE ceiling is a deterministic function of the same reference audio;
    # retaining one checksum-audited copy avoids another 8.5 GiB duplication.
    _safe_symlink(output / "vae_reconstruction", source / "vae_reconstruction")

    old_benchmark = source / "cross_system_baselines"
    benchmark = output / "cross_system_baselines"
    benchmark.mkdir(parents=True, exist_ok=True)
    old_baseline_contract_path = old_benchmark / "BENCHMARK_CONTRACT.json"
    old_baseline_contract = json.loads(
        old_baseline_contract_path.read_text(encoding="utf-8")
    )
    if old_baseline_contract["source_panel_sha256"] != source_panel_digest:
        raise RuntimeError("public-baseline contract used a different panel")
    baseline_contract = copy.deepcopy(old_baseline_contract)
    baseline_contract.update(
        {
            "schema_version": 4,
            "status": "FROZEN_REUSED_OUTPUTS_NEW_OURS_110K_150K",
            "source_eval_root": str(output),
            "source_eval_contract": str(contract_path),
            "source_eval_contract_sha256": _sha256(contract_path),
            "source_panel_path": str(panel_path),
            "source_panel_sha256": source_panel_digest,
            "ours_checkpoints": [
                {"step": row["step"], "path": row["path"]}
                for row in checkpoints
            ],
            "reuse_provenance": {
                "source_benchmark_contract": str(old_baseline_contract_path),
                "source_benchmark_contract_sha256": _sha256(
                    old_baseline_contract_path
                ),
                "public_generation_outputs_reused": True,
                "reason": "identical frozen panel, semantic requests, and durations",
            },
        }
    )
    _atomic_text(
        benchmark / "BENCHMARK_CONTRACT.json",
        json.dumps(baseline_contract, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
    )
    _safe_symlink(
        benchmark / "generation_requests.jsonl",
        old_benchmark / "generation_requests.jsonl",
    )

    summary = {
        "status": "PASS",
        "evaluation_rows": 1200,
        "domain_counts": contract["test_set"]["domain_counts"],
        "checkpoint_steps": [row["step"] for row in checkpoints],
        "panel_sha256": source_panel_digest,
        "semantic_caption_compiler_version": 2,
        "sampling_steps": contract["sampling"]["steps"],
        "cfg_scale": contract["sampling"]["cfg_scale"],
        "same_noise_per_sample_across_checkpoints": True,
        "frozen_public_baseline_metrics": str(
            old_benchmark / "metrics/CROSS_SYSTEM_METRICS.json"
        ),
        "eval_contract": str(contract_path),
    }
    _atomic_text(
        output / "BUILD_SUMMARY.json",
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
