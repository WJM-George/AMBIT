#!/usr/bin/env python3
"""Create a fail-closed run contract for the replacement ScenePlan 4+4 DiT."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import subprocess
import tempfile
import time
import zlib
from pathlib import Path

from stable_audio_tools.configuration import load_config, validate_training_configs


REPO = Path(__file__).resolve().parents[3]
DATASET_ROOT = Path(
    os.environ.get("AMBIT_DATA_ROOT", "data") + "/sceneplan_v2_1p124m/revisions/sound_expansion_v1"
)
DEFAULT_MODEL_CONFIG = REPO / (
    "stable_audio_tools/configs/model_configs/txt2audio/t2a/dit/"
    "qwen35_0p8b_300m_model_sceneplan_44_preflight_candidate.json"
)
FULL_DATASET_CONFIG = DATASET_ROOT / (
    "p10_configs/sceneplan_v2_sound_expansion_v1_train.json"
)
FULL_VALIDATION_DATASET_CONFIG = DATASET_ROOT / (
    "p10_configs/sceneplan_v2_sound_expansion_v1_validation.json"
)
OVERFIT_DATASET_CONFIG = REPO / (
    "stable_audio_tools/configs/dataset_configs/sceneplan_44_overfit10.json"
)
DEFAULT_OVERFIT_GATE = Path(
    os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/archives/pilots/"
    "overfit/"
    "OVERFIT_GATE.json"
)
PRETRANSFORM_CKPT = Path(
    os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/compareVAE_ckpt/unwrapped_wdmix_1350000.ckpt"
)
# These are the active P10 implementation surfaces.  The contract records each
# digest so a full run cannot silently start after a code/configuration edit.
# The frozen legacy renderer compiler is intentionally absent: it is retained
# only for P0--P6 artifact reproduction and is not imported by this model path.
IMPLEMENTATION_PATHS = (
    "pyproject.toml",
    "uv.lock",
    "train.py",
    "stable_audio_tools/configuration.py",
    "stable_audio_tools/data/dataset.py",
    "stable_audio_tools/data/model_sceneplan.py",
    "stable_audio_tools/data/resumable_dataloader.py",
    "stable_audio_tools/data/sceneplan_v2_dataset.py",
    "stable_audio_tools/models/sceneplan_conditioning.py",
    "stable_audio_tools/models/conditioners.py",
    "stable_audio_tools/models/dit.py",
    "stable_audio_tools/models/diffusion.py",
    "stable_audio_tools/models/factory.py",
    "stable_audio_tools/models/transformer.py",
    "stable_audio_tools/models/utils.py",
    "stable_audio_tools/training/diffusion.py",
    "stable_audio_tools/training/ema.py",
    "stable_audio_tools/training/factory.py",
    "stable_audio_tools/configs/dataset_configs/sceneplan_44_overfit10.json",
    "scripts/t2a/data/audit_sceneplan_44_contract.py",
    "scripts/t2a/eval/sceneplan_44_eval_common.py",
    "scripts/t2a/eval/evaluate_sceneplan_44_overfit10.py",
    "scripts/t2a/train/test_sceneplan_dit_p10_44.py",
    "scripts/t2a/train/prepare_sceneplan_dit_p10_44.py",
    "scripts/t2a/train/run_sceneplan_dit_p10_44_overfit10.sh",
    "scripts/t2a/train/run_sceneplan_dit_p10_44_8gpu.sh",
    "scripts/t2a/train/run_sceneplan_dit_p10_44_benchmark.sh",
    "scripts/t2a/train/run_sceneplan_dit_p10_44_resume_preflight.sh",
    "scripts/t2a/train/verify_sceneplan_dit_p10_44_resume.py",
    "scripts/t2a/train/run_t2a_common_8gpu.sh",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object: {path}")
    return value


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def config_dependency_paths(config_path: Path) -> list[Path]:
    """Return the selected config and every recursively inherited parent."""

    ordered: list[Path] = []
    visited: set[Path] = set()

    def visit(path: Path) -> None:
        path = path.expanduser().resolve(strict=True)
        if path in visited:
            return
        try:
            path.relative_to(REPO.resolve(strict=True))
        except ValueError as error:
            raise RuntimeError(
                f"model config escaped the checked-out repository: {path}"
            ) from error
        value = read_json(path)
        parents = value.get("extends")
        if parents is None:
            parents = []
        elif isinstance(parents, str):
            parents = [parents]
        elif not isinstance(parents, list) or not all(
            isinstance(parent, str) for parent in parents
        ):
            raise RuntimeError(f"invalid config inheritance in {path}")
        for parent in parents:
            parent_path = Path(parent).expanduser()
            if not parent_path.is_absolute():
                parent_path = path.parent / parent_path
            visit(parent_path)
        visited.add(path)
        ordered.append(path)

    visit(config_path)
    return ordered


def implementation_inventory(model_config: Path) -> list[dict[str, str]]:
    inventory = []
    paths = [REPO / relative for relative in IMPLEMENTATION_PATHS]
    paths.extend(config_dependency_paths(model_config))
    deduplicated = list(dict.fromkeys(path.resolve() for path in paths))
    for path in deduplicated:
        if not path.is_file():
            raise RuntimeError(f"missing active P10 implementation file: {path}")
        inventory.append(
            {"path": str(path.resolve()), "sha256": sha256(path)}
        )
    return inventory


def validate_overfit_panel(dataset_config_path: Path) -> dict[str, object]:
    """Prove that the ten-row gate covers new Sound and Qwen timing data."""

    config = read_json(dataset_config_path)
    expected_contract = {
        "music": 3,
        "new_sound_replacements": 3,
        "speech": 4,
        "single_source_only": True,
    }
    ordinals = config.get("sample_ordinals")
    datasets = config.get("datasets")
    if not (
        config.get("expected_num_samples") == 10
        and config.get("index_num_samples") == 1_100_000
        and isinstance(ordinals, list)
        and len(ordinals) == len(set(ordinals)) == 10
        and config.get("sample_contract") == expected_contract
        and isinstance(datasets, list)
        and len(datasets) == 1
    ):
        raise RuntimeError("invalid integrated ten-row overfit panel contract")

    training_index = Path(datasets[0]["path"]).expanduser().resolve(strict=True)
    rows: list[dict[str, object]] = []
    with sqlite3.connect(f"file:{training_index}?mode=ro", uri=True) as connection:
        for ordinal in ordinals:
            row = connection.execute(
                "SELECT sample_id, scene_plan_zlib FROM samples WHERE ordinal=?",
                (int(ordinal),),
            ).fetchone()
            if row is None:
                raise RuntimeError(f"overfit ordinal is absent: {ordinal}")
            sceneplan = json.loads(zlib.decompress(row[1]))
            sources = sceneplan.get("sources")
            if not isinstance(sources, list) or len(sources) != 1:
                raise RuntimeError(
                    f"overfit row must have exactly one source: {row[0]}"
                )
            kind = sources[0].get("kind")
            if kind not in {"music", "sound", "speech"}:
                raise RuntimeError(f"invalid overfit source kind: {kind!r}")
            rows.append(
                {
                    "ordinal": int(ordinal),
                    "sample_id": str(row[0]),
                    "kind": str(kind),
                }
            )

    counts = {
        kind: sum(row["kind"] == kind for row in rows)
        for kind in ("music", "sound", "speech")
    }
    if counts != {"music": 3, "sound": 3, "speech": 4}:
        raise RuntimeError(f"overfit domain counts changed: {counts}")

    replacement_map = DATASET_ROOT / "sceneplans_model_v1/replacement_map.jsonl"
    replacement_ids = {
        json.loads(line)["sample_id"]
        for line in replacement_map.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    sound_ids = {str(row["sample_id"]) for row in rows if row["kind"] == "sound"}
    if not sound_ids <= replacement_ids:
        raise RuntimeError(
            "overfit Sound rows are not all new expansion replacements: "
            f"{sorted(sound_ids - replacement_ids)}"
        )

    timing_path = Path(config.get("speech_timing_index_path", "")).expanduser()
    timing_sha = config.get("speech_timing_index_sha256")
    expected_timing_rows = config.get("expected_speech_timing_rows")
    if not (
        config.get("require_speech_timing") is True
        and timing_path.is_file()
        and isinstance(timing_sha, str)
        and sha256(timing_path) == timing_sha
        and expected_timing_rows == 500_000
    ):
        raise RuntimeError("overfit Qwen/forced-alignment sidecar contract changed")
    with sqlite3.connect(f"file:{timing_path.resolve()}?mode=ro", uri=True) as connection:
        timing_rows = int(
            connection.execute("SELECT COUNT(*) FROM speech_timing").fetchone()[0]
        )
    if timing_rows != expected_timing_rows:
        raise RuntimeError(
            f"speech timing row count changed: {timing_rows} != {expected_timing_rows}"
        )
    return {
        "domain_counts": counts,
        "new_sound_sample_ids": sorted(sound_ids),
        "speech_timing_rows": timing_rows,
        "speech_timing_sha256": timing_sha,
        "rows": rows,
    }


def validate_full_training_timing_sidecar(
    dataset_config: dict,
) -> dict[str, object]:
    """Verify the immutable train-only speech timing teacher end to end."""

    timing_value = dataset_config.get("speech_timing_index_path")
    timing_sha = dataset_config.get("speech_timing_index_sha256")
    expected_rows = dataset_config.get("expected_speech_timing_rows")
    if not (
        dataset_config.get("require_speech_timing") is True
        and isinstance(timing_value, str)
        and timing_value
        and isinstance(timing_sha, str)
        and len(timing_sha) == 64
        and expected_rows == 500_000
    ):
        raise RuntimeError(
            "full P10 training requires the frozen 500k speech timing sidecar"
        )
    timing_path = Path(timing_value).expanduser().resolve(strict=True)
    observed_sha = sha256(timing_path)
    if observed_sha != timing_sha:
        raise RuntimeError(
            "full P10 speech timing sidecar SHA256 changed: "
            f"{observed_sha} != {timing_sha}"
        )
    with sqlite3.connect(f"file:{timing_path}?mode=ro", uri=True) as connection:
        observed_rows = int(
            connection.execute("SELECT COUNT(*) FROM speech_timing").fetchone()[0]
        )
    if observed_rows != expected_rows:
        raise RuntimeError(
            "full P10 speech timing row count changed: "
            f"{observed_rows} != {expected_rows}"
        )
    return {
        "path": str(timing_path),
        "sha256": observed_sha,
        "rows": observed_rows,
    }


def selected_throughput_benchmark(
    benchmark_log: Path,
) -> dict[str, object]:
    benchmark_log = benchmark_log.expanduser().resolve(strict=False)
    if not benchmark_log.is_file():
        raise RuntimeError(
            f"selected throughput benchmark is missing: {benchmark_log}"
        )
    raw = benchmark_log.read_text(encoding="utf-8", errors="replace")
    launch_match = re.search(
        r"batch/GPU=(\d+) global_batch=(\d+) workers/rank=(\d+)", raw
    )
    compile_match = re.search(r"torch_compile=([01])", raw)
    ddp_match = re.search(
        r"strategy=ddp_static ddp_bucket_mb=(\d+) "
        r"ddp_comm_hook=(none|bf16)",
        raw,
    )
    required_headers = (
        "training_gate=1 gate_window=100 gate_max_loss_ratio=-1.0 "
        "gate_gradient_every=50",
    )
    if (
        launch_match is None
        or compile_match is None
        or ddp_match is None
        or any(header not in raw for header in required_headers)
    ):
        raise RuntimeError("selected benchmark launch configuration is not canonical")
    batch_size, global_batch_size, num_workers = [
        int(value) for value in launch_match.groups()
    ]
    if global_batch_size != batch_size * 8:
        raise RuntimeError("selected benchmark global batch is inconsistent")
    torch_compile = compile_match.group(1) == "1"
    ddp_bucket_cap_mb = int(ddp_match.group(1))
    ddp_comm_hook = ddp_match.group(2)
    if ddp_bucket_cap_mb <= 0:
        raise RuntimeError("selected DDP bucket size must be positive")
    matches = re.findall(r"SAT_BENCHMARK_RESULT=(\{[^\r\n]+\})", raw)
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one benchmark result, found {len(matches)}"
        )
    result = json.loads(matches[0])
    fast_path_matches = re.findall(
        r"SAT_QWEN35_FAST_PATH=(\{[^\r\n]+\})", raw
    )
    if len(fast_path_matches) != 1:
        raise RuntimeError(
            "selected benchmark must contain exactly one Qwen3.5 fast-path receipt"
        )
    fast_path = json.loads(fast_path_matches[0])
    if fast_path != {
        "causal_conv1d_version": "1.7.0",
        "fast_path_available": True,
        "flash_linear_attention_version": "0.5.2",
    }:
        raise RuntimeError(
            f"selected benchmark did not use the required Qwen3.5 fast path: {fast_path}"
        )
    gate_matches = re.findall(
        r"SAT_TRAINING_GATE_RESULT=(\{[^\r\n]+\})", raw
    )
    if len(gate_matches) != 1:
        raise RuntimeError(
            f"expected exactly one training-gate result, found {len(gate_matches)}"
        )
    health_gate = json.loads(gate_matches[0])
    scalar_metrics = result.get("final_scalar_metrics", {})
    caption_unknown = scalar_metrics.get("train/cfg_caption_unknown_fraction")
    structured_unknown = scalar_metrics.get(
        "train/cfg_structured_unknown_fraction"
    )
    joint_unknown = scalar_metrics.get("train/cfg_joint_unknown_fraction")
    full_condition = scalar_metrics.get("train/cfg_full_condition_fraction")
    if not (
        result.get("world_size") == 8
        and result.get("batch_size_per_gpu") == batch_size
        and result.get("measured_optimizer_steps", 0) >= 18
        and result.get("global_samples_per_second", 0.0) >= 280.0
        and result.get("peak_reserved_gib", 100.0) < 46.0
        and caption_unknown is not None
        and 0.03 <= float(caption_unknown) <= 0.35
        and structured_unknown is not None
        and 0.03 <= float(structured_unknown) <= 0.35
        and joint_unknown is not None
        and 0.0 <= float(joint_unknown) <= 0.15
        and full_condition is not None
        and 0.45 <= float(full_condition) <= 0.95
        and health_gate.get("status") == "PASS"
        and health_gate.get("optimizer_state_step_max", 0) >= 20
        and health_gate.get("ema_final", {}).get("diffusion_ema", 0) >= 20
        and health_gate.get("ema_final", {}).get("conditioner_ema", 0) >= 20
    ):
        raise RuntimeError(f"selected throughput benchmark failed: {result}")
    return {
        "schema": "stable_audio_tools.sceneplan_44_throughput_selection",
        "schema_version": 1,
        "selected_config": {
            "world_size": 8,
            "batch_size_per_gpu": batch_size,
            "global_batch_size": global_batch_size,
            "num_workers_per_rank": num_workers,
            "strategy": "ddp_static",
            "ddp_bucket_cap_mb": ddp_bucket_cap_mb,
            "ddp_comm_hook": ddp_comm_hook,
            "precision": "bf16-mixed",
            "qwen35_fast_path": True,
            "flash_linear_attention_version": "0.5.2",
            "causal_conv1d_version": "1.7.0",
            "torch_compile": torch_compile,
            "training_gate": True,
            "training_gate_window": 100,
            "training_gate_max_loss_ratio": -1.0,
            "training_gate_gradient_every": 50,
        },
        "benchmark_log": str(benchmark_log),
        "benchmark_log_sha256": sha256(benchmark_log),
        "result": result,
        "qwen35_fast_path": fast_path,
        "training_gate_result": health_gate,
    }


def gpu_inventory() -> list[dict[str, object]]:
    output = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    result = []
    for line in output.splitlines():
        index, name, total, used = [value.strip() for value in line.split(",", 3)]
        result.append(
            {
                "index": int(index),
                "name": name,
                "memory_total_mib": int(total),
                "memory_used_mib": int(used),
            }
        )
    if len(result) != 8 or any(
        row["name"] != "NVIDIA GeForce RTX 4090" for row in result
    ):
        raise RuntimeError(f"expected eight RTX 4090 GPUs: {result}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("overfit10", "benchmark", "resume", "full"),
        required=True,
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--approval-note", required=True)
    parser.add_argument(
        "--model-config",
        type=Path,
        default=DEFAULT_MODEL_CONFIG,
        help="exact inherited model config to freeze into the run contract",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DATASET_ROOT,
        help="frozen base dataset or immutable revision root",
    )
    parser.add_argument(
        "--dataset-config",
        type=Path,
        help="train dataset config; defaults to the base ScenePlan-v2 config",
    )
    parser.add_argument(
        "--validation-dataset-config",
        type=Path,
        help="validation dataset config paired with --dataset-config",
    )
    parser.add_argument("--conditioning-audit", type=Path)
    parser.add_argument("--validation-conditioning-audit", type=Path)
    parser.add_argument("--test-conditioning-audit", type=Path)
    parser.add_argument(
        "--overfit-gate",
        type=Path,
        default=DEFAULT_OVERFIT_GATE,
        help="ten-sample gate produced with the same resolved model config",
    )
    parser.add_argument(
        "--throughput-benchmark-log",
        type=Path,
        help="required in full mode; exact benchmark log selected by measured throughput",
    )
    parser.add_argument("--max-steps", type=int, default=15_000)
    parser.add_argument("--checkpoint-every", type=int, default=5_000)
    parser.add_argument("--validation-every", type=int, default=5_000)
    parser.add_argument("--validation-batches", type=int, default=64)
    args = parser.parse_args()
    if args.max_steps <= 0:
        raise RuntimeError("max steps must be positive")
    if args.checkpoint_every <= 0 or args.max_steps % args.checkpoint_every:
        raise RuntimeError(
            "checkpoint cadence must be positive and divide max steps exactly"
        )
    if args.validation_every <= 0 or args.max_steps % args.validation_every:
        raise RuntimeError(
            "validation cadence must be positive and divide max steps exactly"
        )
    if args.validation_batches <= 0:
        raise RuntimeError("validation batches must be positive")
    model_config = args.model_config.expanduser().resolve(strict=True)
    overfit_gate = args.overfit_gate.expanduser().resolve(strict=False)
    dataset_root = args.dataset_root.expanduser().resolve(strict=True)
    audit_root = dataset_root / "audit/conditioning_v3"
    conditioning_audit = (
        args.conditioning_audit or audit_root / "sceneplan_44_train_audit.json"
    ).expanduser().resolve(strict=True)
    validation_audit = (
        args.validation_conditioning_audit
        or audit_root / "sceneplan_44_validation_audit.json"
    ).expanduser().resolve(strict=True)
    test_audit = (
        args.test_conditioning_audit
        or audit_root / "sceneplan_44_test_audit.json"
    ).expanduser().resolve(strict=True)

    marker_path = dataset_root / "FROZEN_P9.json"
    marker = read_json(marker_path)
    if not (
        marker.get("schema") == "stable_audio_tools.sceneplan_v2_p9_marker"
        and marker.get("state")
        == "P9_complete_frozen_waiting_for_user_acceptance"
        and marker.get("p11_training_started") is False
    ):
        raise RuntimeError("frozen P9 marker is not valid")
    audit = read_json(conditioning_audit)
    if not (
        audit.get("ok") is True
        and int(audit.get("rows_checked", 0)) >= 20_000
        and audit.get("observed_max_tokens", 513) <= 512
        and audit.get("local_condition", {}).get("gain_db_used") is False
        and audit.get("local_condition", {}).get("trajectory_tracks") == 4
    ):
        raise RuntimeError("ScenePlan 4+4 conditioning audit is not an all-pass gate")
    split_audits = []
    for split, path, expected_rows in (
        ("validation", validation_audit, 20_000),
        ("test", test_audit, 4_000),
    ):
        value = read_json(path)
        if not (
            value.get("ok") is True
            and int(value.get("index_rows", -1)) == expected_rows
            and int(value.get("rows_checked", -1)) == expected_rows
            and value.get("selection") == "all"
            and value.get("observed_max_tokens", 513) <= 512
            and value.get("local_condition", {}).get("gain_db_used") is False
        ):
            raise RuntimeError(f"ScenePlan 4+4 {split} audit is not all-pass")
        split_audits.append(
            {"split": split, "path": str(path), "sha256": sha256(path)}
        )

    if args.mode == "overfit10":
        if args.dataset_config is not None:
            raise RuntimeError("overfit10 uses its frozen ten-row dataset config")
        dataset_config_path = OVERFIT_DATASET_CONFIG
    else:
        dataset_config_path = (
            args.dataset_config or FULL_DATASET_CONFIG
        ).expanduser().resolve(strict=True)
    validation_dataset_config = (
        args.validation_dataset_config or FULL_VALIDATION_DATASET_CONFIG
    ).expanduser().resolve(strict=True)
    model = load_config(model_config)
    dataset = load_config(dataset_config_path)
    validation = validate_training_configs(model, dataset)
    full_training_timing = (
        validate_full_training_timing_sidecar(dataset)
        if args.mode in {"benchmark", "resume", "full"}
        else None
    )
    validation_dataset = load_config(validation_dataset_config)
    validate_training_configs(
        model,
        validation_dataset,
        allow_missing_speech_timing=True,
    )
    if validation.get("latent_length") != 432 or validation.get("latent_channels") != 64:
        raise RuntimeError("resolved DiT/VAE geometry changed")

    unit = subprocess.run(
        [
            str(REPO / ".venv/bin/python"),
            str(REPO / "scripts/t2a/train/test_sceneplan_dit_p10_44.py"),
        ],
        cwd=REPO,
        check=False,
        capture_output=True,
        text=True,
    )
    if unit.returncode != 0 or "OK" not in unit.stderr + unit.stdout:
        raise RuntimeError(
            "ScenePlan 4+4 regression tests failed:\n"
            f"{unit.stdout}\n{unit.stderr}"
        )

    overfit_panel = (
        validate_overfit_panel(dataset_config_path)
        if args.mode == "overfit10"
        else None
    )
    gate = None
    throughput = None
    if args.mode in {"benchmark", "resume", "full"}:
        gate = read_json(overfit_gate)
        if not (
            gate.get("ok") is True
            and int(gate.get("schema_version", 0)) >= 5
            and gate.get("architecture") == "semantic_cross_attention_plus_4+4"
            and gate.get("initialization") == "random_seeded_sceneplan_44_from_scratch"
            and gate.get("warm_start") is False
            and gate.get("semantic_gate") is True
            and gate.get("spatial_gate") is True
            and gate.get("audio_gate") is True
            and gate.get("cfg_dropout_gate") is True
            and gate.get("alignment_gate") is True
            and gate.get("sound_temporal_gate") is True
            and gate.get("model_config_sha256") == sha256(model_config)
            and gate.get("cfg_dropout")
            == {
                "mode": "independent",
                "caption_unknown_prob": 0.15,
                "structured_unknown_prob": 0.15,
                "expected_joint_unknown_prob": 0.0225,
                "expected_full_condition_prob": 0.7225,
            }
            and gate.get("thresholds", {}).get(
                "paired_clap_audio_cosine_min"
            ) == 0.80
            and gate.get("thresholds", {}).get("mean_w_si_sdr_db_min") == 10.0
            and gate.get("thresholds", {}).get(
                "valid_direction_fraction_min"
            ) == 0.70
            and gate.get("thresholds", {}).get(
                "duration_last_over_first_max"
            ) == 0.80
            and gate.get("thresholds", {}).get(
                "duration_last_over_uniform_max"
            ) == 0.80
            and gate.get("thresholds", {}).get(
                "sound_temporal_last_over_first_max"
            ) == 0.80
        ):
            raise RuntimeError("full training is blocked by the ten-sample overfit gate")
        if args.mode == "full":
            if args.throughput_benchmark_log is None:
                raise RuntimeError(
                    "full mode requires --throughput-benchmark-log"
                )
            throughput = selected_throughput_benchmark(
                args.throughput_benchmark_log
            )

    run_root = args.run_root.expanduser().resolve(strict=False)
    dit_root = Path(os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/dit").resolve(strict=True)
    try:
        run_root.relative_to(dit_root)
    except ValueError as error:
        raise RuntimeError("run root must be below ${AMBIT_CKPT_ROOT}/dit") from error
    run_root.mkdir(parents=True, exist_ok=True)
    for child in (
        "checkpoints",
        "logs",
        "wandb",
        "wandb-cache",
        "wandb-artifacts",
        "tmp",
        "torchinductor-cache",
        "triton-cache",
        "evaluation",
    ):
        (run_root / child).mkdir(exist_ok=True)

    receipt = {
        "schema": "stable_audio_tools.sceneplan_44_dit_run_contract",
        "schema_version": 7,
        "state": "ready_to_train",
        "mode": args.mode,
        "architecture": "semantic_cross_attention_plus_4+4",
        "initialization": "random_seeded_sceneplan_44_from_scratch",
        "random_seed": 42,
        "warm_start": False,
        "pretrained_checkpoint": None,
        "pretrained_checkpoint_sha256": None,
        "pretrained_route_weights": None,
        "pretrained_route_expectation": None,
        "optimizer_state": "fresh",
        "scheduler_state": "fresh_from_selected_model_config",
        "ema_state": "fresh_from_random_model",
        "audio_function_at_step_zero": "random; no inherited DiT audio function",
        "resume_policy": "same_run_root_checkpoints_only_after_first_scratch_step",
        "run_root": str(run_root),
        "model_config": str(model_config),
        "model_config_sha256": sha256(model_config),
        "model_config_dependency_chain": [
            {"path": str(path), "sha256": sha256(path)}
            for path in config_dependency_paths(model_config)
        ],
        "training_recipe": {
            "timestep_sampler": model["training"]["timestep_sampler"],
            "optimizer": model["training"]["optimizer_configs"]["diffusion"],
        },
        "training_schedule": {
            "max_steps": args.max_steps,
            "checkpoint_every": args.checkpoint_every,
            "validation_every": args.validation_every,
            "validation_batches": args.validation_batches,
            "milestone_steps": list(
                range(args.checkpoint_every, args.max_steps + 1, args.checkpoint_every)
            ),
        },
        "dataset_config": str(dataset_config_path),
        "dataset_config_sha256": sha256(dataset_config_path),
        "validation_dataset_config": str(validation_dataset_config),
        "validation_dataset_config_sha256": sha256(validation_dataset_config),
        "full_training_speech_timing": full_training_timing,
        "pretransform_checkpoint": str(PRETRANSFORM_CKPT),
        "pretransform_checkpoint_sha256": sha256(PRETRANSFORM_CKPT),
        "pretransform_usage": (
            "frozen offline FOA latent codec only; never loaded into the DiT, "
            "conditioner, optimizer, scheduler, or EMA"
        ),
        "implementation_files": implementation_inventory(model_config),
        "conditioning_audit": str(conditioning_audit),
        "conditioning_audit_sha256": sha256(conditioning_audit),
        "complete_split_audits": split_audits,
        "frozen_p9_marker": str(marker_path),
        "frozen_p9_marker_sha256": sha256(marker_path),
        "caption_path": "Qwen cross-attention semantic text only",
        "caption_roles": "event_source_ids plus speech_source_ids in {-1,0,1,2,3,4}",
        "local_path": "four event tracks plus four geometric trajectory tracks",
        "activity_path": "encoded by framewise event ids",
        "gain_db_in_sceneplan": True,
        "gain_db_used_by_model": False,
        "explicit_cfg_unknown_id": -1,
        "known_inactive_id": 0,
        "cfg_dropout_contract": {
            "mode": "independent",
            "caption_unknown_prob": 0.15,
            "structured_unknown_prob": 0.15,
            "expected_joint_unknown_prob": 0.0225,
            "expected_full_condition_prob": 0.7225,
        },
        "approval_note": args.approval_note,
        "created_unix": time.time(),
        "gpus": gpu_inventory(),
        "unit_tests": "PASS",
        "p11_training_started": False,
    }
    if overfit_panel is not None:
        receipt["overfit_panel"] = overfit_panel
    if gate is not None:
        receipt["overfit_gate"] = str(overfit_gate)
        receipt["overfit_gate_sha256"] = sha256(overfit_gate)
        receipt["overfit_checkpoint"] = gate["checkpoint"]
        receipt["overfit_checkpoint_usage"] = "gate_evidence_only_never_loaded"
        receipt["overfit_gate_aggregates"] = gate["aggregates"]
    if throughput is not None:
        throughput_path = run_root / "THROUGHPUT_SELECTION.json"
        atomic_json(throughput_path, throughput)
        receipt["throughput_selection"] = str(throughput_path)
        receipt["throughput_selection_sha256"] = sha256(throughput_path)
        receipt["runtime_selection"] = throughput["selected_config"]
    path = run_root / "SCENEPLAN_44_RUN_CONTRACT.json"
    atomic_json(path, receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
