#!/usr/bin/env python3
"""Freeze a matched balanced-1200 evaluation contract for one P10-v12 arm.

The input checkpoint labels are *local upcycle steps* (for example, 2500),
not the 150k Dense source step.  Every candidate is evaluated with the exact
P10-v11 semantic-v2 panel, sample noise, sampler, CFG, and VAE contract.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE_ROOT = Path(
    os.environ.get("AMBIT_DATA_ROOT", "data") + "/sceneplan_v2_1p124m/revisions/"
    "speech_expansion_noalign_15s_v1/evaluation/"
    "p10_v11_balanced_1200_ckpt110k_150k_semantic_v2"
)
EXPECTED_PANEL_SHA256 = (
    "9327379bf74101c5efc5c300a311956ee7a784ec91d99792af62bf38340353fa"
)
EXPECTED_DENSE_SOURCE_SHA256 = (
    "be8c90cd1434bd71f73951531175c3674ff0f3173d5db591e2e1c476152ff59e"
)
ARM_FLAGS = {
    "dense": (False, False),
    "moe": (True, False),
    "attention": (False, True),
    "combined": (True, True),
}


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def safe_symlink(link: Path, target: Path) -> None:
    resolved_target = target.expanduser().resolve(strict=True)
    if link.is_symlink():
        if link.resolve(strict=True) != resolved_target:
            raise RuntimeError(f"existing symlink points elsewhere: {link}")
        return
    if link.exists():
        raise RuntimeError(f"refusing to replace existing path: {link}")
    link.symlink_to(resolved_target, target_is_directory=resolved_target.is_dir())


def module_flags(config: dict[str, Any]) -> tuple[bool, bool]:
    dit = config["model"]["diffusion"]["config"]
    moe = bool((dit.get("sceneplan_chunk_moe") or {}).get("enabled", False))
    attention = bool(
        (dit.get("sceneplan_soft_block_attention") or {}).get("enabled", False)
    )
    return moe, attention


def inference_model_spec(config: dict[str, Any]) -> dict[str, Any]:
    """Return every config field that can alter checkpoint inference."""

    keys = (
        "task",
        "model_type",
        "sample_rate",
        "sample_size",
        "audio_channels",
        "model",
    )
    return {key: config[key] for key in keys}


def json_sha256(value: Any) -> str:
    import hashlib

    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def parse_checkpoint_spec(value: str) -> tuple[int, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not label or not raw_path:
        raise argparse.ArgumentTypeError(
            "checkpoint must use LOCAL_STEP=/absolute/path.ckpt"
        )
    try:
        step = int(label)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"invalid local checkpoint step: {label!r}"
        ) from error
    if step <= 0:
        raise argparse.ArgumentTypeError("local checkpoint step must be positive")
    return step, Path(raw_path)


def checkpoint_record(
    spec: tuple[int, Path],
    *,
    expected_flags: tuple[bool, bool],
    expected_inference_spec: dict[str, Any],
) -> dict[str, Any]:
    local_step, unresolved_path = spec
    path = unresolved_path.expanduser().resolve(strict=True)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    observed_step = int(checkpoint.get("global_step", -1))
    if observed_step != local_step:
        raise RuntimeError(
            f"checkpoint global_step mismatch: label={local_step}, "
            f"embedded={observed_step}, path={path}"
        )
    embedded_config = checkpoint.get("model_config")
    if not isinstance(embedded_config, dict):
        raise RuntimeError(f"checkpoint has no embedded model_config: {path}")
    if module_flags(embedded_config) != expected_flags:
        raise RuntimeError(
            f"checkpoint arm mismatch: expected={expected_flags}, "
            f"observed={module_flags(embedded_config)}, path={path}"
        )
    embedded_inference_spec = inference_model_spec(embedded_config)
    if embedded_inference_spec != expected_inference_spec:
        raise RuntimeError(
            "checkpoint embedded inference config does not match --model-config: "
            f"checkpoint={json_sha256(embedded_inference_spec)}, "
            f"requested={json_sha256(expected_inference_spec)}, path={path}"
        )
    if "state_dict" not in checkpoint:
        raise RuntimeError(f"checkpoint has no state_dict: {path}")
    return {
        "step": local_step,
        "fine_tune_step": local_step,
        "dense_source_step": 150_000,
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "embedded_inference_config_sha256": json_sha256(
            embedded_inference_spec
        ),
    }


def contract_signature(contract: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": contract.get("schema"),
        "arm": contract.get("upgrade_arm"),
        "panel_sha256": contract.get("test_set", {}).get("panel_sha256"),
        "model_config_sha256": contract.get("sampling", {}).get(
            "model_config_sha256"
        ),
        "checkpoints": [
            {
                "step": row.get("step"),
                "path": row.get("path"),
                "bytes": row.get("bytes"),
                "sha256": row.get("sha256"),
            }
            for row in contract.get("checkpoints", [])
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--arm", choices=tuple(ARM_FLAGS), required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        action="append",
        type=parse_checkpoint_spec,
        required=True,
        metavar="LOCAL_STEP=PATH",
    )
    parser.add_argument("--training-launch-contract", type=Path)
    args = parser.parse_args()

    source = args.source_root.expanduser().resolve(strict=True)
    source_contract_path = source / "EVAL_CONTRACT.json"
    source_contract = json.loads(source_contract_path.read_text(encoding="utf-8"))
    source_panel = source / source_contract["test_set"]["panel_filename"]
    panel_sha256 = sha256_file(source_panel)
    if not (
        panel_sha256 == EXPECTED_PANEL_SHA256
        and source_contract["test_set"]["panel_sha256"] == panel_sha256
        and source_contract["test_set"]["domain_counts"]
        == {"music": 400, "sound": 400, "speech": 400}
        and int(source_contract["test_set"]["evaluation_rows"]) == 1200
    ):
        raise RuntimeError("source is not the frozen balanced 400x3 panel")
    panel_rows = [
        json.loads(line)
        for line in source_panel.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    domains = Counter(str(row["domain"]) for row in panel_rows)
    if domains != {"music": 400, "sound": 400, "speech": 400}:
        raise RuntimeError(f"panel domain audit failed: {dict(domains)}")
    if len({str(row["panel_id"]) for row in panel_rows}) != 1200:
        raise RuntimeError("panel IDs are not unique")

    expected_flags = ARM_FLAGS[args.arm]
    model_config = args.model_config.expanduser().resolve(strict=True)
    from stable_audio_tools.configuration import load_config

    loaded_model_config = load_config(model_config)
    if module_flags(loaded_model_config) != expected_flags:
        raise RuntimeError(
            f"model config arm mismatch: arm={args.arm}, "
            f"flags={module_flags(loaded_model_config)}"
        )
    model_config_sha256 = sha256_file(model_config)
    expected_inference_spec = inference_model_spec(loaded_model_config)

    if len({step for step, _ in args.checkpoint}) != len(args.checkpoint):
        raise RuntimeError("checkpoint local steps are not unique")
    checkpoints = sorted(
        (
            checkpoint_record(
                spec,
                expected_flags=expected_flags,
                expected_inference_spec=expected_inference_spec,
            )
            for spec in args.checkpoint
        ),
        key=lambda row: int(row["step"]),
    )

    dense_rows = [
        row
        for row in source_contract["checkpoints"]
        if int(row["step"]) == 150_000
    ]
    if len(dense_rows) != 1 or dense_rows[0]["sha256"] != EXPECTED_DENSE_SOURCE_SHA256:
        raise RuntimeError("source contract does not freeze the canonical Dense 150k baseline")

    launch_provenance = None
    if args.training_launch_contract is not None:
        launch_path = args.training_launch_contract.expanduser().resolve(strict=True)
        launch = json.loads(launch_path.read_text(encoding="utf-8"))
        if not (
            launch.get("schema") == "stable_audio_tools.p10_v12_upgrade_launch"
            and launch.get("status") == "PASS"
            and launch.get("arm") == args.arm
            and launch.get("selection_panel_sha256") == panel_sha256
            and Path(launch["model_config"]).resolve(strict=True) == model_config
        ):
            raise RuntimeError("training launch contract does not match this evaluation")
        launch_provenance = {
            "path": str(launch_path),
            "sha256": sha256_file(launch_path),
            "source_checkpoint": launch["source_checkpoint"],
            "source_checkpoint_sha256": launch["source_checkpoint_sha256"],
            "source_global_step": int(launch["source_global_step"]),
        }

    output = args.output_root.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    panel_path = output / "balanced_test_1200.jsonl"
    if panel_path.exists() and sha256_file(panel_path) != panel_sha256:
        raise RuntimeError(f"refusing to replace a different panel: {panel_path}")
    if not panel_path.exists():
        atomic_text(panel_path, source_panel.read_text(encoding="utf-8"))
    if sha256_file(panel_path) != panel_sha256:
        raise RuntimeError("byte-preserving panel copy failed")

    contract = copy.deepcopy(source_contract)
    contract.update(
        {
            "schema": "stable_audio_tools.p10_v12_upgrade_eval_contract",
            "schema_version": 1,
            "status": "FROZEN_BALANCED_1200",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "purpose": (
                f"P10-v12 {args.arm} checkpoint selection against the frozen "
                "P10-v11 Dense 150k baseline"
            ),
            "upgrade_arm": args.arm,
            "upgrade_flags": {
                "moe": expected_flags[0],
                "attention": expected_flags[1],
            },
            "checkpoints": checkpoints,
            "training_launch": launch_provenance,
            "dense_150k_baseline": {
                **dense_rows[0],
                "eval_root": str(source),
                "eval_contract": str(source_contract_path),
                "eval_contract_sha256": sha256_file(source_contract_path),
                "metric_root": str((source / "metrics").resolve(strict=True)),
            },
            "comparison_contract": {
                "same_panel": True,
                "same_noise_per_sample": True,
                "same_semantic_caption_compiler": True,
                "same_sampler_cfg_and_vae": True,
                "baseline_step": 150_000,
                "candidate_steps_are_local_upcycle_steps": True,
            },
        }
    )
    contract["sampling"].update(
        {
            "model_config": str(model_config),
            "model_config_sha256": model_config_sha256,
            "semantic_caption_compiler_version": 2,
            "inference_batch_size": 1,
        }
    )
    contract["test_set"].update(
        {
            "panel_filename": panel_path.name,
            "panel_sha256": panel_sha256,
        }
    )

    contract_path = output / "EVAL_CONTRACT.json"
    if contract_path.exists():
        previous = json.loads(contract_path.read_text(encoding="utf-8"))
        if contract_signature(previous) != contract_signature(contract):
            raise RuntimeError(
                f"refusing to mutate an existing evaluation contract: {contract_path}"
            )
        contract["created_at_utc"] = previous.get(
            "created_at_utc", contract["created_at_utc"]
        )
    atomic_text(
        contract_path,
        json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    safe_symlink(output / "vae_reconstruction", source / "vae_reconstruction")

    summary = {
        "schema": contract["schema"],
        "status": "PASS",
        "arm": args.arm,
        "checkpoint_steps": [int(row["step"]) for row in checkpoints],
        "checkpoint_sha256": [str(row["sha256"]) for row in checkpoints],
        "evaluation_rows": 1200,
        "domain_counts": dict(domains),
        "panel_sha256": panel_sha256,
        "dense_baseline_step": 150_000,
        "same_panel_noise_sampling": True,
        "eval_contract": str(contract_path),
    }
    atomic_text(
        output / "BUILD_SUMMARY.json",
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
