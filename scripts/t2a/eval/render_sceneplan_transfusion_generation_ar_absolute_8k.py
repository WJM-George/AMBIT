#!/usr/bin/env python3
"""Render the Generation-AR lane for absolute GT scoring.

The existing Generation-AR P10 closure compares two P10 renders with each
other and intentionally discards waveforms.  This evaluator serves a
different purpose: it persists the native Generation-AR FOA lane so the
frozen 8K cross-system benchmark can compare it with the real rendered GT FOA.

For every row it renders the freely decoded Generation-AR ScenePlan with the
same frozen P10-v11 and the benchmark's original noise seed.  The historical
high-precision GT-ScenePlan P10 lane is already complete on this exact panel;
its frozen metrics are reused by the comparison stage instead of rerendering
an oracle.

The generated output is tail-cropped or right-zero-padded to the reference panel's
sample count.  This preserves duration mistakes as end-to-end errors instead
of silently repairing the predicted ScenePlan.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping


# Must be set before importing torch/CUDA.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.t2a.eval.sceneplan_44_eval_common import atomic_wav, audio_qc
from scripts.t2a.eval.sceneplan_dit_p10_panel_common import (
    load_panel,
    sha256_file,
)
from scripts.t2a.test import (
    evaluate_sceneplan_transfusion_generation_ar_p10_audio_8k as closure,
)
from stable_audio_tools.data.model_sceneplan_codec_v4 import ModelScenePlanCodecV4
from stable_audio_tools.data.sceneplan_p11_single_turn import P11Task
from stable_audio_tools.data.sceneplan_p11_single_turn import (
    finalize_sceneplan_for_p10,
)


CONTRACT = "p10v11_generation_ar_absolute_gt_8k_render_ar_only_v1"
SCHEMA = (
    "stable_audio_tools.sceneplan_transfusion_generation_ar_absolute_gt_ar_only_render"
)
EXPECTED_ROWS = 8_000
SAMPLE_RATE = 44_100
DEFAULT_BASELINE_ROOT = Path(
    "/mnt/sdb/audio_dataset/sceneplan_v2_1p124m/revisions/"
    "speech_expansion_noalign_15s_v1/evaluation/"
    "p10_v11_150k_full_test_8000_semantic_v2"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-evaluation-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--test-manifest", type=Path, default=closure.TEST_MANIFEST
    )
    parser.add_argument(
        "--baseline-root", type=Path, default=DEFAULT_BASELINE_ROOT
    )
    parser.add_argument("--p10-release", type=Path, default=closure.P10_RELEASE)
    parser.add_argument("--codec-path", type=Path, default=closure.CODEC_PATH)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=3)
    parser.add_argument("--log-every", type=int, default=10)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--prepare-only", action="store_true")
    action.add_argument("--finalize-only", action="store_true")
    action.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(_canonical_bytes(row).decode("utf-8") + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _source_identity() -> dict[str, dict[str, Any]]:
    paths = {
        "renderer": Path(__file__).resolve(strict=True),
        "p10_closure": Path(closure.__file__).resolve(strict=True),
        "panel_common": (
            REPO_ROOT / "scripts/t2a/eval/sceneplan_dit_p10_panel_common.py"
        ).resolve(strict=True),
        "audio_io": (
            REPO_ROOT / "scripts/t2a/eval/sceneplan_44_eval_common.py"
        ).resolve(strict=True),
        "p10_executor": (
            REPO_ROOT / "stable_audio_tools/inference/sceneplan_cot.py"
        ).resolve(strict=True),
        "p10_finalize": (
            REPO_ROOT
            / "stable_audio_tools/data/sceneplan_p11_single_turn.py"
        ).resolve(strict=True),
    }
    return {
        key: {
            "path": str(path),
            "bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
        }
        for key, path in sorted(paths.items())
    }


def _load_context(args: argparse.Namespace):
    plan_rows, plan_identity = closure._load_verified_plan_rows(
        args.plan_evaluation_dir,
        args.test_manifest,
        expected_rows=EXPECTED_ROWS,
        # The plan evaluator already binds its immutable source snapshot.  This
        # renderer records and rechecks its own executable source separately.
        verify_source_hashes=False,
    )
    baseline_root = args.baseline_root.expanduser().resolve(strict=True)
    panel = load_panel(baseline_root)
    if len(panel) != EXPECTED_ROWS:
        raise RuntimeError("absolute benchmark panel is not exactly 8K")
    for row, panel_row in zip(plan_rows, panel):
        if (
            row.ordinal != int(panel_row["ordinal"])
            or row.sample_id != str(panel_row["sample_id"])
            or row.source_count != int(panel_row["source_count"])
            or row.manifest_latent_frames != int(panel_row["latent_frames_valid"])
        ):
            raise RuntimeError(
                f"Generation-AR/test panel identity mismatch at {row.ordinal}"
            )
        reference = Path(str(panel_row["reference_foa_path"])).resolve(strict=True)
        if not reference.is_file():
            raise RuntimeError(f"missing real GT FOA: {reference}")
    p10_identity = closure._build_p10_identity(
        args.p10_release, args.codec_path
    )
    eval_contract = (baseline_root / "EVAL_CONTRACT.json").resolve(strict=True)
    panel_path = (
        baseline_root
        / json.loads(eval_contract.read_text(encoding="utf-8"))["test_set"][
            "panel_filename"
        ]
    ).resolve(strict=True)
    benchmark_contract = (
        baseline_root / "cross_system_baselines_final_8k/BENCHMARK_CONTRACT.json"
    ).resolve(strict=True)
    contract = {
        "schema": SCHEMA + "_run",
        "schema_version": 1,
        "contract": CONTRACT,
        "rows": EXPECTED_ROWS,
        "sample_rate": SAMPLE_RATE,
        "fit_policy": "tail_crop_or_right_zero_pad_to_real_gt_sample_count_v1",
        "noise_policy": "frozen_baseline_panel_noise_seed_per_row_v1",
        "lanes": {
            "generation_ar": (
                "raw_user_input -> Generation_AR_ScenePlan -> frozen_P10_v11"
            ),
            "reference": "existing_real_materialized_GT_FOA",
            "historical_oracle": (
                "reuse_existing_high_precision_GT_ScenePlan_P10_metrics"
            ),
        },
        "plan_evaluation": plan_identity,
        "p10": p10_identity,
        "test_manifest": {
            "path": str(args.test_manifest.expanduser().resolve(strict=True)),
            "sha256": sha256_file(args.test_manifest.expanduser().resolve(strict=True)),
        },
        "baseline_panel": {
            "root": str(baseline_root),
            "eval_contract": str(eval_contract),
            "eval_contract_sha256": sha256_file(eval_contract),
            "benchmark_contract": str(benchmark_contract),
            "benchmark_contract_sha256": sha256_file(benchmark_contract),
            "panel": str(panel_path),
            "panel_sha256": sha256_file(panel_path),
        },
        "source": _source_identity(),
    }
    return plan_rows, panel, contract


def _ensure_run_contract(output_dir: Path, expected: Mapping[str, Any]) -> str:
    path = output_dir / "RUN_CONTRACT.json"
    if path.exists():
        actual = json.loads(path.read_text(encoding="utf-8"))
        if actual != expected:
            raise RuntimeError("absolute render output directory contract mismatch")
    else:
        _atomic_json(path, expected)
    return _canonical_sha256(expected)


def _metadata_path(output_dir: Path, panel_id: str) -> Path:
    return output_dir / "outputs" / panel_id / "metadata.json"


def _valid_complete(
    metadata_path: Path,
    *,
    row,
    panel_row: Mapping[str, Any],
    run_contract_sha256: str,
) -> dict[str, Any] | None:
    if not metadata_path.is_file():
        return None
    try:
        value = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not (
            value.get("schema") == SCHEMA + "_output"
            and int(value.get("schema_version", -1)) == 1
            and value.get("status") == "PASS"
            and value.get("run_contract_canonical_sha256")
            == run_contract_sha256
            and int(value.get("ordinal", -1)) == row.ordinal
            and value.get("panel_id") == panel_row["panel_id"]
            and value.get("sample_id") == row.sample_id
            and value.get("target_sceneplan_sha256")
            == row.target_sceneplan_sha256
            and value.get("prediction_sceneplan_sha256")
            == row.prediction_sceneplan_sha256
            and value.get("reference_foa_sha256")
            == panel_row["reference_foa_sha256"]
        ):
            return None
        audio_path = Path(value["generation_ar_foa_path"]).resolve(strict=True)
        if sha256_file(audio_path) != value["generation_ar_foa_sha256"]:
            return None
        return value
    except Exception:
        return None


def _finalize_bundle(row, *, codec, tokenizer):
    # Keep this local and explicit so the task label and sample identity are
    # visible in this evaluator's contract rather than inferred from a panel.
    tokens = row.prediction_token_ids
    bundle = finalize_sceneplan_for_p10(
        codec,
        tokens,
        tokenizer=tokenizer,
        task=P11Task.GENERATION,
        sample_id=row.sample_id,
    )
    expected = row.prediction_sceneplan_bytes
    if closure._canonical_json_bytes(bundle.sceneplan) != expected:
        raise RuntimeError(f"P10 finalize changed ScenePlan at ordinal {row.ordinal}")
    bundle.assert_external_p10_boundary()
    return bundle


def _render_worker(
    args: argparse.Namespace,
    rows,
    panel,
    contract: Mapping[str, Any],
    run_contract_sha256: str,
) -> int:
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be in [0,num-shards)")
    if args.log_every < 1:
        raise ValueError("log-every must be positive")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("absolute P10 rendering requires CUDA")
    torch.cuda.set_device(device)
    closure._configure_determinism(closure.CANONICAL_ROOT_SEED, rank=args.shard_index)

    codec = ModelScenePlanCodecV4(args.codec_path.expanduser().resolve(strict=True))
    if codec.fingerprint != contract["p10"]["codec"]["fingerprint"]:
        raise RuntimeError("codec fingerprint changed after render contract creation")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        contract["p10"]["qwen"]["path"],
        local_files_only=True,
        use_fast=True,
    )
    executor = closure._executor_from_identity(contract["p10"], device=device)

    output_dir = args.output_dir.expanduser().resolve()
    assigned = [row for row in rows if row.ordinal % args.num_shards == args.shard_index]
    panel_by_ordinal = {int(value["ordinal"]): value for value in panel}
    completed = 0
    rendered = 0
    started = time.perf_counter()
    for row in assigned:
        panel_row = panel_by_ordinal[row.ordinal]
        metadata_path = _metadata_path(output_dir, str(panel_row["panel_id"]))
        if _valid_complete(
            metadata_path,
            row=row,
            panel_row=panel_row,
            run_contract_sha256=run_contract_sha256,
        ) is not None:
            completed += 1
            continue

        prediction_bundle = _finalize_bundle(
            row, codec=codec, tokenizer=tokenizer
        )
        seed = int(panel_row["noise_seed"])
        reference_samples = int(panel_row["model_num_samples"])
        row_root = metadata_path.parent
        render_started = time.perf_counter()
        prediction_native = executor.render(prediction_bundle, seed=seed)
        prediction_native_samples = int(prediction_native.shape[-1])
        prediction_audio = closure._fit_length(
            prediction_native, reference_samples
        ).contiguous()
        prediction_path = row_root / "generation_ar_foa_float32.wav"
        atomic_wav(prediction_path, prediction_audio, SAMPLE_RATE, subtype="FLOAT")

        prediction_qc = audio_qc(prediction_audio)
        if not (
            prediction_qc.get("finite") is True
            and prediction_qc.get("channels") == 4
            and prediction_qc.get("samples") == reference_samples
        ):
            raise RuntimeError(
                f"invalid Generation-AR scored FOA at ordinal {row.ordinal}"
            )
        prediction_sha = sha256_file(prediction_path)
        value = {
            "schema": SCHEMA + "_output",
            "schema_version": 1,
            "status": "PASS",
            "run_contract_canonical_sha256": run_contract_sha256,
            "ordinal": row.ordinal,
            "panel_id": str(panel_row["panel_id"]),
            "sample_id": row.sample_id,
            "domain": str(panel_row["domain"]),
            "source_count": row.source_count,
            "source_kinds": panel_row["source_kinds"],
            "noise_seed": seed,
            "plan_exact": row.plan_exact,
            "target_sceneplan_sha256": row.target_sceneplan_sha256,
            "prediction_sceneplan_sha256": row.prediction_sceneplan_sha256,
            "reference_sceneplan_sha256": hashlib.sha256(
                _canonical_bytes(panel_row["scene_plan"])
            ).hexdigest(),
            "reference_foa_path": str(
                Path(panel_row["reference_foa_path"]).resolve(strict=True)
            ),
            "reference_foa_sha256": str(panel_row["reference_foa_sha256"]),
            "reference_samples": reference_samples,
            "reference_latent_frames": int(panel_row["latent_frames_valid"]),
            "generation_ar_native_samples": prediction_native_samples,
            "generation_ar_duration_fitted": prediction_native_samples != reference_samples,
            "generation_ar_foa_path": str(prediction_path.resolve(strict=True)),
            "generation_ar_foa_sha256": prediction_sha,
            "generation_ar_qc": prediction_qc,
            "render_seconds": float(time.perf_counter() - render_started),
        }
        _atomic_json(metadata_path, value)
        del prediction_native, prediction_audio
        completed += 1
        rendered += 1
        if rendered % args.log_every == 0 or completed == len(assigned):
            print(
                json.dumps(
                    {
                        "event": "absolute_gt_render_progress",
                        "shard": args.shard_index,
                        "completed": completed,
                        "assigned": len(assigned),
                        "newly_rendered": rendered,
                        "elapsed_sec": round(time.perf_counter() - started, 3),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    del executor, tokenizer, codec
    gc.collect()
    torch.cuda.empty_cache()
    status = {
        "status": "PASS",
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "assigned_rows": len(assigned),
        "newly_rendered_rows": rendered,
        "run_contract_canonical_sha256": run_contract_sha256,
    }
    _atomic_json(output_dir / "logs" / f"worker_{args.shard_index:02d}.json", status)
    return 0


def _finalize(
    args: argparse.Namespace,
    rows,
    panel,
    run_contract_sha256: str,
    *,
    write: bool,
) -> dict[str, Any]:
    output_dir = args.output_dir.expanduser().resolve()
    manifest: list[dict[str, Any]] = []
    exact = 0
    prediction_fitted = 0
    render_seconds = 0.0
    for row, panel_row in zip(rows, panel):
        value = _valid_complete(
            _metadata_path(output_dir, str(panel_row["panel_id"])),
            row=row,
            panel_row=panel_row,
            run_contract_sha256=run_contract_sha256,
        )
        if value is None:
            raise RuntimeError(f"incomplete absolute render row {row.ordinal}")
        exact += int(bool(value["plan_exact"]))
        prediction_fitted += int(bool(value["generation_ar_duration_fitted"]))
        render_seconds += float(value["render_seconds"])
        manifest.append(
            {
                key: value[key]
                for key in (
                    "ordinal",
                    "panel_id",
                    "sample_id",
                    "domain",
                    "source_count",
                    "source_kinds",
                    "noise_seed",
                    "plan_exact",
                    "target_sceneplan_sha256",
                    "prediction_sceneplan_sha256",
                    "reference_sceneplan_sha256",
                    "reference_foa_path",
                    "reference_foa_sha256",
                    "reference_samples",
                    "reference_latent_frames",
                    "generation_ar_native_samples",
                    "generation_ar_duration_fitted",
                    "generation_ar_foa_path",
                    "generation_ar_foa_sha256",
                )
            }
        )
    if len(manifest) != EXPECTED_ROWS:
        raise RuntimeError("absolute render manifest does not cover exact 8K")
    manifest_path = output_dir / "OUTPUT_MANIFEST.jsonl"
    if write:
        _atomic_jsonl(manifest_path, manifest)
    elif not manifest_path.is_file():
        raise RuntimeError("completed render is missing OUTPUT_MANIFEST.jsonl")
    manifest_sha = (
        sha256_file(manifest_path)
        if manifest_path.is_file()
        else hashlib.sha256(
            b"".join(_canonical_bytes(row) + b"\n" for row in manifest)
        ).hexdigest()
    )
    summary = {
        "schema": SCHEMA + "_summary",
        "schema_version": 1,
        "status": "PASS",
        "contract": CONTRACT,
        "rows": EXPECTED_ROWS,
        "plan_exact_rows": exact,
        "generation_ar_duration_fitted_rows": prediction_fitted,
        "total_render_seconds": render_seconds,
        "run_contract": str(output_dir / "RUN_CONTRACT.json"),
        "run_contract_canonical_sha256": run_contract_sha256,
        "output_manifest": str(manifest_path),
        "output_manifest_sha256": manifest_sha,
    }
    if write:
        _atomic_json(output_dir / "SUMMARY.json", summary)
        _atomic_text(output_dir / "GENERATION_COMPLETE", "PASS\n")
    else:
        stored = json.loads((output_dir / "SUMMARY.json").read_text(encoding="utf-8"))
        if stored != summary:
            raise RuntimeError("absolute render summary changed on verification")
        if (output_dir / "GENERATION_COMPLETE").read_text(encoding="utf-8") != "PASS\n":
            raise RuntimeError("absolute render completion marker changed")
    return summary


def main() -> int:
    args = _parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, panel, contract = _load_context(args)
    run_contract_sha256 = _ensure_run_contract(output_dir, contract)
    if args.prepare_only:
        print(
            json.dumps(
                {
                    "event": "absolute_gt_render_prepared",
                    "rows": len(rows),
                    "run_contract_canonical_sha256": run_contract_sha256,
                },
                sort_keys=True,
            )
        )
        return 0
    if args.finalize_only or args.verify_only:
        summary = _finalize(
            args,
            rows,
            panel,
            run_contract_sha256,
            write=not args.verify_only,
        )
        print(json.dumps(summary, sort_keys=True))
        return 0
    return _render_worker(args, rows, panel, contract, run_contract_sha256)


if __name__ == "__main__":
    raise SystemExit(main())
