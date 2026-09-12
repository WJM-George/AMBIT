#!/usr/bin/env python3
"""Audit the immutable P10 release and the mutable P11 latest pointer.

The quick audit verifies release linkage, small-file hashes, source constants,
checkpoint paths/sizes, and P11 code fingerprints.  Promotion audits can add
``--verify-large-checkpoints`` to stream and verify every checkpoint hash.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stable_audio_tools.configuration import load_config  # noqa: E402
from stable_audio_tools.data.sceneplan_p11_single_turn import (  # noqa: E402
    P10_CANONICAL_CHECKPOINT,
    P10_CANONICAL_CHECKPOINT_SHA256,
    P10_CANONICAL_CHECKPOINT_STEP,
    P10_CANONICAL_EXECUTOR_FAMILY,
    P10_CANONICAL_MODEL_CONFIG,
    P10_CANONICAL_MODEL_CONFIG_SHA256,
    P10_MAX_LATENT_FRAMES,
    P11_CONTRACT_VERSION,
    P11_MAX_LATENT_FRAMES,
)


DEFAULT_P10_RELEASE = (
    REPO_ROOT / "artifacts/releases/P10_SCENEPLAN_DIT_V11_150K_RELEASE.json"
)
DEFAULT_P11_LATEST = REPO_ROOT / "artifacts/sceneplan_p11/P11_LATEST.json"


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AssertionError(f"expected a JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _assert_hash(path: Path, expected: str) -> None:
    actual = _sha256(path)
    if actual != expected:
        raise AssertionError(
            f"sha256 mismatch for {path}: expected {expected}, got {actual}"
        )


def _canonical_json_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p10-release", type=Path, default=DEFAULT_P10_RELEASE)
    parser.add_argument("--p11-latest", type=Path, default=DEFAULT_P11_LATEST)
    parser.add_argument("--verify-large-checkpoints", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    p10_path = args.p10_release.expanduser().resolve(strict=True)
    latest_path = args.p11_latest.expanduser().resolve(strict=True)
    p10 = _load_json(p10_path)
    latest = _load_json(latest_path)
    revision_path = Path(latest["latest_revision_manifest"]).resolve(strict=True)
    revision = _load_json(revision_path)

    assert p10["schema"] == "stable_audio_tools.p10_release_manifest"
    assert int(p10["schema_version"]) >= 2
    assert p10["status"] == "frozen_canonical"
    assert p10["immutable"] is True
    assert latest["schema"] == "stable_audio_tools.p11_latest_pointer"
    assert revision["schema"] == "stable_audio_tools.p11_revision_manifest"
    _assert_hash(revision_path, latest["latest_revision_manifest_sha256"])
    assert latest["latest_revision"] == revision["revision_id"]
    assert latest["status"] == revision["status"]
    assert latest["canonical_executor_release"] == p10["release_id"]
    assert revision["canonical_executor_release"] == p10["release_id"]
    assert Path(latest["canonical_executor_manifest"]).resolve() == p10_path
    p10_manifest_sha256 = _sha256(p10_path)
    assert latest["canonical_executor_manifest_sha256"] == p10_manifest_sha256
    assert revision["canonical_executor_manifest_sha256"] == p10_manifest_sha256

    revision_architecture = revision.get("architecture") or revision.get(
        "canonical_architecture"
    )
    if not isinstance(revision_architecture, dict):
        raise AssertionError("P11 revision lacks a canonical architecture object")
    p11_model_config_path = Path(latest["canonical_model_config"]).resolve(
        strict=True
    )
    _assert_hash(p11_model_config_path, latest["canonical_model_config_sha256"])
    p11_resolved_sha256 = _canonical_json_hash(load_config(p11_model_config_path))
    assert (
        latest["canonical_model_config_resolved_sha256"] == p11_resolved_sha256
    )
    assert Path(revision_architecture["model_config"]).resolve() == p11_model_config_path
    assert (
        revision_architecture["model_config_sha256"]
        == latest["canonical_model_config_sha256"]
    )
    assert (
        revision_architecture["model_config_resolved_sha256"]
        == p11_resolved_sha256
    )

    checkpoint = p10["checkpoint"]
    checkpoint_path = Path(checkpoint["path"]).resolve(strict=True)
    assert checkpoint_path.stat().st_size == int(checkpoint["bytes"])
    assert checkpoint["path"] == P10_CANONICAL_CHECKPOINT
    assert int(checkpoint["step"]) == P10_CANONICAL_CHECKPOINT_STEP
    assert checkpoint["sha256"] == P10_CANONICAL_CHECKPOINT_SHA256
    assert p10["executor_family"] == P10_CANONICAL_EXECUTOR_FAMILY

    model_config = p10["model_config"]
    model_config_path = Path(model_config["path"]).resolve(strict=True)
    assert model_config["path"] == P10_CANONICAL_MODEL_CONFIG
    assert model_config["file_sha256"] == P10_CANONICAL_MODEL_CONFIG_SHA256
    _assert_hash(model_config_path, model_config["file_sha256"])
    assert _canonical_json_hash(load_config(model_config_path)) == model_config[
        "resolved_sha256"
    ]

    pretransform = p10["pretransform"]
    pretransform_config = pretransform["model_config"]
    _assert_hash(
        Path(pretransform_config["path"]).resolve(strict=True),
        pretransform_config["sha256"],
    )
    pretransform_checkpoint = pretransform["checkpoint"]
    pretransform_checkpoint_path = Path(
        pretransform_checkpoint["path"]
    ).resolve(strict=True)
    assert pretransform_checkpoint_path.stat().st_size == int(
        pretransform_checkpoint["bytes"]
    )

    capability = p10["capability_contract"]
    capability_path = Path(capability["path"]).resolve(strict=True)
    _assert_hash(capability_path, capability["sha256"])

    data_release = p10["training_data_release"]
    _assert_hash(
        Path(data_release["freeze_manifest"]).resolve(strict=True),
        data_release["freeze_manifest_sha256"],
    )
    evidence = p10["promotion_evidence"]
    evaluation_contract_path = Path(evidence["evaluation_contract"]).resolve(
        strict=True
    )
    _assert_hash(evaluation_contract_path, evidence["evaluation_contract_sha256"])
    _assert_hash(
        Path(evidence["evaluation_summary"]).resolve(strict=True),
        evidence["evaluation_summary_sha256"],
    )
    evaluation_contract = _load_json(evaluation_contract_path)
    sampling = evaluation_contract["sampling"]
    canonical_inference = p10["canonical_inference"]
    for key in (
        "weights",
        "sampler",
        "steps",
        "cfg_scale",
        "cfg_rescale_phi",
        "rescale_cfg",
        "apg_scale",
        "negative_condition",
        "raw_output",
    ):
        assert canonical_inference[key] == sampling[key], key
    assert sampling["vae_checkpoint"] == pretransform_checkpoint["path"]

    envelope = p10["runtime_envelope"]
    assert int(envelope["max_latent_frames"]) == P10_MAX_LATENT_FRAMES
    assert P10_MAX_LATENT_FRAMES == P11_MAX_LATENT_FRAMES
    assert int(revision["route_contract"]["p11_contract_version"]) == P11_CONTRACT_VERSION

    code_hashes: dict[str, str] = {}
    for relative, expected in revision["code_fingerprints"].items():
        path = (REPO_ROOT / relative).resolve(strict=True)
        _assert_hash(path, expected)
        code_hashes[relative] = expected

    candidate_audit: dict[str, Any] | None = None
    candidate = revision.get("active_candidate")
    if candidate is not None:
        if not isinstance(candidate, dict):
            raise AssertionError("active_candidate must be an object")
        run_root = Path(candidate["run_root"]).resolve(strict=True)
        launch_spec = candidate["training_launch_contract"]
        launch_path = Path(launch_spec["path"]).resolve(strict=True)
        _assert_hash(launch_path, launch_spec["file_sha256"])
        launch = _load_json(launch_path)
        launch_unhashed = dict(launch)
        launch_claimed = launch_unhashed.pop("report_sha256_without_self")
        assert launch_claimed == _canonical_json_hash(launch_unhashed)
        assert launch_claimed == launch_spec["self_sha256"]
        assert launch["run_name"] == candidate["run_name"]
        assert Path(launch_path).parent == run_root
        assert int(launch["training"]["world_size"]) == int(
            candidate["world_size"]
        )
        assert int(launch["training"]["batch_size_per_gpu"]) == int(
            candidate["batch_size_per_gpu"]
        )
        assert int(launch["training"]["seed"]) == int(candidate["seed"])
        candidate_checkpoint = candidate.get("checkpoint")
        if candidate_checkpoint is not None:
            Path(candidate_checkpoint).resolve(strict=True)
        candidate_audit = {
            "run_name": candidate["run_name"],
            "status": candidate["status"],
            "checkpoint": candidate_checkpoint,
            "launch_contract_sha256": launch_spec["file_sha256"],
            "launch_contract_self_sha256": launch_claimed,
            "promotion_authorized": bool(candidate.get("promotion_authorized")),
        }

    protocol_audit: dict[str, Any] | None = None
    posttrain = revision.get("posttrain_evaluation")
    if posttrain is not None:
        if not isinstance(posttrain, dict):
            raise AssertionError("posttrain_evaluation must be an object")
        protocol_path = Path(posttrain["protocol"]).resolve(strict=True)
        _assert_hash(protocol_path, posttrain["protocol_file_sha256"])
        protocol = _load_json(protocol_path)
        protocol_unhashed = dict(protocol)
        protocol_claimed = protocol_unhashed.pop("report_sha256_without_self")
        assert protocol_claimed == _canonical_json_hash(protocol_unhashed)
        assert protocol_claimed == posttrain["protocol_self_sha256"]
        assert protocol["status"] == "FROZEN_BEFORE_CANDIDATE_CHECKPOINT"
        assert protocol["candidate_run"]["run_name"] == candidate["run_name"]
        assert protocol["scientific_scope"]["selection_target"] == "P11_planner_only"
        assert protocol["evaluation"]["gpu_count"] == 8
        assert len(protocol["evaluation"]["gpu_assignments"]) == 8
        protocol_audit = {
            "path": str(protocol_path),
            "file_sha256": posttrain["protocol_file_sha256"],
            "self_sha256": protocol_claimed,
            "status": protocol["status"],
            "gpu_assignment_count": len(
                protocol["evaluation"]["gpu_assignments"]
            ),
            "selection_target": protocol["scientific_scope"][
                "selection_target"
            ],
        }

    arm_checkpoints: dict[str, dict[str, Any]] = {}
    for arm, spec in revision["arms"].items():
        if spec.get("checkpoint") is None:
            if revision["status"] not in {
                "mechanism_validated_pilot_pending",
                "preflight_running",
                "pilot_running",
            }:
                raise AssertionError(
                    f"{arm} has no checkpoint in revision status {revision['status']}"
                )
            arm_checkpoints[arm] = {
                "path": None,
                "bytes": None,
                "manifest_sha256": None,
                "content_hash_verified": False,
                "state": "pending_training",
            }
            continue
        path = Path(spec["checkpoint"]).resolve(strict=True)
        item: dict[str, Any] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "manifest_sha256": spec["sha256"],
            "content_hash_verified": False,
        }
        if args.verify_large_checkpoints:
            _assert_hash(path, spec["sha256"])
            item["content_hash_verified"] = True
        arm_checkpoints[arm] = item

    if args.verify_large_checkpoints:
        _assert_hash(checkpoint_path, checkpoint["sha256"])
        _assert_hash(
            pretransform_checkpoint_path, pretransform_checkpoint["sha256"]
        )

    report = {
        "schema": "stable_audio_tools.p10_p11_release_audit",
        "schema_version": 1,
        "status": "PASS",
        "p10_release_id": p10["release_id"],
        "p10_checkpoint_content_hash_verified": args.verify_large_checkpoints,
        "p10_pretransform_content_hash_verified": args.verify_large_checkpoints,
        "p11_revision_id": revision["revision_id"],
        "p11_status": revision["status"],
        "p11_revision_manifest_sha256": latest[
            "latest_revision_manifest_sha256"
        ],
        "p11_model_config_resolved_sha256": p11_resolved_sha256,
        "large_checkpoint_hashes_verified": args.verify_large_checkpoints,
        "code_fingerprints": code_hashes,
        "arm_checkpoints": arm_checkpoints,
        "active_candidate": candidate_audit,
        "posttrain_evaluation_protocol": protocol_audit,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
