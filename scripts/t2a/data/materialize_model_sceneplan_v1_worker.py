#!/usr/bin/env python3
"""Persistent single-GPU P8 worker for revision-5 model ScenePlans."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing as mp
import time
from pathlib import Path

import pyarrow.parquet as pq
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
import sys

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from materialize_model_sceneplan_v1_shard import (  # noqa: E402
    encode_results,
    load_shard_rows,
    load_vae,
    render_one,
    sha256_file,
    shard_number,
)
from sceneplan_v2_common import (  # noqa: E402
    DATASET_ROOT,
    atomic_write_json,
    require_dataset_not_frozen,
)


def assigned_shards(
    root: Path, splits: list[str], worker_index: int, worker_count: int
) -> list[Path]:
    paths: list[Path] = []
    for split in splits:
        candidates = sorted(
            (root / split).glob(f"model-sceneplans-{split}-*.jsonl")
        )
        paths.extend(
            path
            for path in candidates
            if shard_number(path) % int(worker_count) == int(worker_index)
        )
    return paths


def completed_shard_is_valid(
    done: Path,
    model_shard: Path,
    rows: list[dict],
    contract_revision: int,
) -> bool:
    if not done.is_file():
        return False
    try:
        summary = json.loads(done.read_text(encoding="utf-8"))
        if (
            Path(summary["model_sceneplan_shard"]).resolve(strict=True) != model_shard
            or int(summary["rows"]) != len(rows)
            or int(summary.get("dataset_contract_revision", -1))
            != int(contract_revision)
        ):
            return False
        manifest = Path(summary["materialized_manifest"])
        if not manifest.is_file() or pq.read_metadata(manifest).num_rows != len(rows):
            return False
        table = pq.read_table(
            manifest,
            columns=[
                "sample_id",
                "planned_bundle_sha256",
                "latent_ref",
                "latent_shard_sha256",
            ],
        )
        expected_bundles = [
            (str(row["sample_id"]), str(row["planned_bundle_sha256"]))
            for row in rows
        ]
        materialized_bundles = list(
            zip(
                map(str, table["sample_id"].to_pylist()),
                map(str, table["planned_bundle_sha256"].to_pylist()),
            )
        )
        if materialized_bundles != expected_bundles:
            return False
        refs = {str(value).split("#", 1)[0] for value in table["latent_ref"].to_pylist()}
        hashes = set(map(str, table["latent_shard_sha256"].to_pylist()))
        if len(refs) != 1 or len(hashes) != 1:
            return False
        latent_path = Path(refs.pop())
        return latent_path.is_file() and sha256_file(latent_path) == hashes.pop()
    except Exception:  # noqa: BLE001
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sceneplan-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "validation", "test"),
        default=["validation", "test", "train"],
    )
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--worker-index", type=int, required=True)
    parser.add_argument("--worker-count", type=int, required=True)
    parser.add_argument("--jobs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--contract-revision", type=int, choices=(5, 6), default=5)
    parser.add_argument("--train-render-root", type=Path)
    parser.add_argument(
        "--supplement-mode",
        action="store_true",
        help="Permit an isolated delta under DATASET_ROOT/supplements after base P9.",
    )
    args = parser.parse_args()
    sceneplan_root = args.sceneplan_root.expanduser().resolve(strict=True)
    output_root = args.output_root.expanduser().resolve(strict=False)
    try:
        sceneplan_root.relative_to("/mnt/sdb")
        output_root.relative_to("/mnt/sdb")
    except ValueError as error:
        raise ValueError("P8 inputs and outputs must remain on SDB") from error
    if args.supplement_mode:
        allowed_roots = [
            (DATASET_ROOT / "supplements").resolve(strict=True),
            (DATASET_ROOT / "revisions").resolve(strict=True),
        ]
        for name, path in (
            ("sceneplan-root", sceneplan_root),
            ("output-root", output_root),
        ):
            if not any(
                path == root or root in path.parents for root in allowed_roots
            ):
                raise ValueError(
                    f"supplement {name} must remain below one of {allowed_roots}: {path}"
                )
        if not (DATASET_ROOT / "FROZEN_P9.json").is_file():
            raise RuntimeError("supplement mode requires an already-frozen base P9")
    else:
        require_dataset_not_frozen()
    if not 0 <= args.worker_index < args.worker_count:
        raise ValueError("worker-index must be in [0, worker-count)")
    train_render_root = None
    if args.train_render_root is not None:
        train_render_root = args.train_render_root.expanduser().resolve(strict=False)
        try:
            train_render_root.relative_to("/dev/shm")
        except ValueError as error:
            raise ValueError("transient train renders must remain under /dev/shm") from error
        train_render_root.mkdir(parents=True, exist_ok=True)
    paths = assigned_shards(
        sceneplan_root, list(args.splits), args.worker_index, args.worker_count
    )
    if not paths:
        raise RuntimeError("worker has no assigned model ScenePlan shards")
    output_root.mkdir(parents=True, exist_ok=True)
    worker_root = output_root / "workers"
    worker_root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    context = mp.get_context("spawn")
    completed = 0
    skipped = 0
    rows_done = 0
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.jobs, mp_context=context
    ) as pool:
        model = load_vae(device)
        for path_index, model_shard in enumerate(paths, start=1):
            rows = load_shard_rows(model_shard)
            revisions = {
                int(
                    json.loads(row["render_recipe_json"]).get(
                        "dataset_contract_revision", 5
                    )
                )
                for row in rows
            }
            if revisions != {int(args.contract_revision)}:
                raise RuntimeError(
                    f"{model_shard}: recipe contract revisions {revisions} do not "
                    f"match --contract-revision={args.contract_revision}"
                )
            shard = int(rows[0]["work_shard"])
            split = str(rows[0]["split"])
            done = output_root / "work_done" / split / f"work-{shard:05d}.json"
            if completed_shard_is_valid(
                done, model_shard, rows, args.contract_revision
            ):
                skipped += 1
                rows_done += len(rows)
                print(
                    json.dumps(
                        {
                            "worker": args.worker_index,
                            "state": "skip_verified",
                            "split": split,
                            "work_shard": shard,
                            "assigned_progress": f"{path_index}/{len(paths)}",
                        }
                    ),
                    flush=True,
                )
                continue
            retain_stems = split != "train"
            cleanup_foa = split == "train"
            render_root = (
                train_render_root
                if cleanup_foa and train_render_root is not None
                else output_root / "renders"
            )
            shard_started = time.time()
            futures = {
                pool.submit(
                    render_one, row, str(render_root), retain_stems
                ): str(row["sample_id"])
                for row in rows
            }
            results: list[dict] = []
            for result_index, future in enumerate(
                concurrent.futures.as_completed(futures), start=1
            ):
                results.append(future.result())
                if result_index % 100 == 0 or result_index == len(rows):
                    print(
                        json.dumps(
                            {
                                "worker": args.worker_index,
                                "state": "render",
                                "split": split,
                                "work_shard": shard,
                                "rendered": result_index,
                                "total": len(rows),
                                "errors": sum(
                                    result.get("status") != "ok" for result in results
                                ),
                                "elapsed_sec": round(time.time() - shard_started, 1),
                            }
                        ),
                        flush=True,
                    )
            results.sort(key=lambda result: str(result["sample_id"]))
            failures = [result for result in results if result.get("status") != "ok"]
            if failures:
                quarantine = (
                    output_root / "quarantine" / split / f"work-{shard:05d}.json"
                )
                atomic_write_json(
                    quarantine, {"errors": failures, "error_count": len(failures)}
                )
                raise RuntimeError(
                    f"{split} work shard {shard} has {len(failures)} quarantines"
                )
            materialized, manifest = encode_results(
                rows,
                results,
                output_root=output_root,
                device=device,
                batch_size=args.batch_size,
                cleanup_foa=cleanup_foa,
                retain_stems=retain_stems,
                model=model,
                logical_render_root=output_root / "renders",
            )
            summary = {
                "schema": "stable_audio_tools.model_sceneplan_materialized_work_shard",
                "schema_version": 1,
                "dataset_contract_revision": args.contract_revision,
                "split": split,
                "work_shard": shard,
                "rows": len(materialized),
                "model_sceneplan_shard": str(model_shard),
                "materialized_manifest": str(manifest),
                "cleanup_foa": cleanup_foa,
                "retain_stems": retain_stems,
                "max_model_num_samples": max(
                    int(row["model_num_samples"]) for row in materialized
                ),
                "max_latent_frames_valid": max(
                    int(row["latent_frames_valid"]) for row in materialized
                ),
                "elapsed_sec": round(time.time() - shard_started, 3),
                "worker_index": args.worker_index,
                "gpu": args.gpu,
            }
            (output_root / "quarantine" / split / f"work-{shard:05d}.json").unlink(
                missing_ok=True
            )
            atomic_write_json(done, summary)
            completed += 1
            rows_done += len(materialized)
            print(json.dumps(summary, ensure_ascii=False), flush=True)
    final = {
        "schema": "stable_audio_tools.model_sceneplan_materialization_worker_summary",
        "schema_version": 1,
        "dataset_contract_revision": args.contract_revision,
        "worker_index": args.worker_index,
        "worker_count": args.worker_count,
        "gpu": args.gpu,
        "assigned_shards": len(paths),
        "completed_shards": completed,
        "skipped_verified_shards": skipped,
        "rows": rows_done,
        "elapsed_sec": round(time.time() - started, 3),
    }
    atomic_write_json(worker_root / f"worker-{args.worker_index:02d}.json", final)
    print(json.dumps(final, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
