#!/usr/bin/env python3
"""Persistently materialize multiple Editing target shards on one GPU.

One VAE instance and one CPU render pool are reused across shards.  This is
the full-scale worker for the 1M/20K/5K inventory; the single-shard program
remains the diagnostic entry point.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import sys
import time


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.t2a.data.materialize_sceneplan_transfusion_editing_targets import (  # noqa: E402
    MATERIALIZATION_CONTRACT,
    _encode,
    _load_rows,
    _render_one,
    _validate_physical_gpu,
    load_vae,
)
from stable_audio_tools.data.sceneplan_transfusion_editing import (  # noqa: E402
    canonical_json,
)
from stable_audio_tools.data.sceneplan_transfusion_editing_index import (  # noqa: E402
    sha256_file,
)


EDITING_AR_INPUT_CONTRACT = "source_foa_latent_plus_raw_edit_request_v2"


def _prefetch_enabled(args) -> bool:
    if args.prefetch_control is None:
        return bool(args.prefetch_render)
    value = json.loads(args.prefetch_control.read_text(encoding="utf-8"))
    gpus = value.get("prefetch_gpus")
    if not isinstance(gpus, list) or any(type(gpu) is not int or gpu not in range(3, 8) for gpu in gpus):
        raise RuntimeError("prefetch control must list only physical GPUs 3-7")
    return int(args.gpu) in gpus


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _completed(done: Path, pair_index_sha256: str) -> bool:
    if not done.is_file():
        return False
    try:
        record = json.loads(done.read_text(encoding="utf-8"))
        manifest = Path(record["materialized_manifest"])
        return bool(
            record.get("status") == "ok"
            and record.get("pair_index_sha256") == pair_index_sha256
            and manifest.is_file()
            and sha256_file(manifest)
            == record.get("materialized_manifest_sha256")
        )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-index", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--shard-start", type=int, required=True)
    parser.add_argument("--shard-stop", type=int, required=True)
    parser.add_argument("--shard-stride", type=int, default=1)
    parser.add_argument("--jobs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--source-parity", action="store_true")
    parser.add_argument("--retain-stems", action="store_true")
    parser.add_argument("--cleanup-foa", action="store_true")
    parser.add_argument("--cleanup-render-work", action="store_true")
    parser.add_argument(
        "--prefetch-render",
        action="store_true",
        help="render one following shard in the existing CPU pool during VAE encoding",
    )
    parser.add_argument(
        "--prefetch-control", type=Path,
        help="optional JSON prefetch_gpus list, reread at each shard boundary",
    )
    args = parser.parse_args()
    if (
        args.jobs <= 0
        or args.batch_size <= 0
        or args.shard_start < 0
        or args.shard_stop <= args.shard_start
        or args.shard_stride <= 0
    ):
        raise ValueError("invalid multi-shard worker bounds/settings")
    if args.retain_stems and args.cleanup_render_work:
        raise ValueError("cannot retain stems while cleaning render work")

    pair_index = args.pair_index.expanduser().resolve(strict=True)
    pair_index_sha = sha256_file(pair_index)
    device = _validate_physical_gpu(args.gpu)
    first_metadata, first_rows = _load_rows(pair_index, args.shard_start)
    if (
        first_metadata.get("editing_ar_input_contract")
        != EDITING_AR_INPUT_CONTRACT
        or first_metadata.get("editing_ar_old_sceneplan_input") != "false"
    ):
        raise RuntimeError("pair index predates the audio-reference Editing-AR contract")
    split = str(first_rows[0]["split"])
    output_root = Path(first_metadata["target_root"]).resolve()
    try:
        output_root.relative_to("/mnt/sdb")
    except ValueError as error:
        raise RuntimeError("full Editing output root must remain on /mnt/sdb") from error

    model = load_vae(device)
    context = mp.get_context("spawn")
    worker_started = time.time()
    completed_shards = 0
    completed_rows = 0
    def pending_work():
        nonlocal completed_shards, completed_rows
        for work_shard in range(
            int(args.shard_start), int(args.shard_stop), int(args.shard_stride)
        ):
            metadata, rows = _load_rows(pair_index, work_shard)
            if metadata != first_metadata or str(rows[0]["split"]) != split:
                raise RuntimeError("pair-index metadata changed between work shards")
            done = (
                output_root
                / "materialized/work_done"
                / split
                / f"work-{work_shard:05d}.json"
            )
            if _completed(done, pair_index_sha):
                record = json.loads(done.read_text(encoding="utf-8"))
                completed_shards += 1
                completed_rows += int(record["rows"])
                print(
                    canonical_json(
                        {
                            "stage": "skip_complete",
                            "gpu": int(args.gpu),
                            "work_shard": work_shard,
                            "rows": int(record["rows"]),
                        }
                    ),
                    flush=True,
                )
                continue
            yield work_shard, rows, done

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=int(args.jobs), mp_context=context
    ) as pool:
        work = iter(pending_work())

        def submit_next():
            item = next(work, None)
            if item is None:
                return None
            work_shard, rows, done = item
            started = time.time()
            futures = {
                pool.submit(
                    _render_one,
                    row,
                    str(output_root),
                    bool(args.retain_stems),
                    bool(args.source_parity),
                ): str(row["pair_id"])
                for row in rows
            }
            print(
                canonical_json({
                    "stage": "render_submit",
                    "gpu": int(args.gpu),
                    "work_shard": work_shard,
                    "rows": len(rows),
                    "prefetch_render": bool(args.prefetch_render),
                    "at_unix": started,
                }),
                flush=True,
            )
            return work_shard, rows, done, started, futures

        active = submit_next()
        while active is not None:
            work_shard, rows, done, shard_started, futures = active
            rendered = []
            for count, future in enumerate(
                concurrent.futures.as_completed(futures), start=1
            ):
                rendered.append(future.result())
                if count % 64 == 0 or count == len(rows):
                    print(
                        canonical_json(
                            {
                                "stage": "target_render",
                                "gpu": int(args.gpu),
                                "work_shard": work_shard,
                                "completed": count,
                                "total": len(rows),
                                "errors": sum(
                                    item.get("status") != "ok"
                                    for item in rendered
                                ),
                                "elapsed_sec": round(time.time() - shard_started, 1),
                            }
                        ),
                        flush=True,
                    )
            by_id = {str(item["pair_id"]): item for item in rendered}
            rendered = [by_id[str(row["pair_id"])] for row in rows]
            failures = [item for item in rendered if item.get("status") != "ok"]
            if failures:
                quarantine = (
                    output_root
                    / "materialized/quarantine"
                    / split
                    / f"work-{work_shard:05d}.json"
                )
                _atomic_json(quarantine, {"failures": failures})
                raise RuntimeError(
                    f"work shard {work_shard} has {len(failures)} render failures"
                )
            render_finished = time.time()
            # Each worker retains its original disjoint shard stride and one
            # VAE. The same CPU pool can prepare only one following shard while
            # the current shard is encoded. No target ordering, RNG seed,
            # encoding batch, or publication rule changes.
            prefetch_render = _prefetch_enabled(args)
            following = submit_next() if prefetch_render else None
            encode_started = time.time()
            print(
                canonical_json({
                    "stage": "encode_start",
                    "gpu": int(args.gpu),
                    "work_shard": work_shard,
                    "prefetched_work_shard": following[0] if following else None,
                    "at_unix": encode_started,
                }),
                flush=True,
            )
            manifest = _encode(
                rows,
                rendered,
                output_root=output_root,
                device=device,
                batch_size=int(args.batch_size),
                cleanup_foa=bool(args.cleanup_foa),
                model=model,
            )
            summary = {
                "schema": MATERIALIZATION_CONTRACT,
                "schema_version": 2,
                "status": "ok",
                "split": split,
                "work_shard": work_shard,
                "rows": len(rows),
                "physical_gpu": int(args.gpu),
                "pair_index": str(pair_index),
                "pair_index_sha256": pair_index_sha,
                "editing_ar_input_contract": EDITING_AR_INPUT_CONTRACT,
                "editing_ar_old_sceneplan_input": False,
                "source_parity_required": bool(args.source_parity),
                "source_parity_verified_rows": sum(
                    bool(item["source_parity_verified"]) for item in rendered
                ),
                "cleanup_foa": bool(args.cleanup_foa),
                "retain_stems": bool(args.retain_stems),
                "cleanup_render_work": bool(args.cleanup_render_work),
                "materialized_manifest": str(manifest),
                "materialized_manifest_sha256": sha256_file(manifest),
                "elapsed_sec": round(time.time() - shard_started, 2),
                "render_sec": round(render_finished - shard_started, 2),
                "encode_publish_sec": round(time.time() - encode_started, 2),
                "prefetch_render": prefetch_render,
                "completed_at_unix": time.time(),
            }
            _atomic_json(done, summary)
            if args.cleanup_render_work:
                render_work = (
                    output_root
                    / "materialized/renders"
                    / split
                    / f"work-{work_shard:05d}"
                )
                expected_parent = output_root / "materialized/renders" / split
                if render_work.parent != expected_parent or not render_work.name.startswith(
                    "work-"
                ):
                    raise RuntimeError("refusing unsafe render-work cleanup")
                shutil.rmtree(render_work, ignore_errors=False)
            completed_shards += 1
            completed_rows += len(rows)
            print(canonical_json(summary), flush=True)
            active = following if following is not None else submit_next()

    print(
        json.dumps(
            {
                "ok": True,
                "physical_gpu": int(args.gpu),
                "split": split,
                "completed_shards": completed_shards,
                "completed_rows": completed_rows,
                "elapsed_sec": round(time.time() - worker_started, 2),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
