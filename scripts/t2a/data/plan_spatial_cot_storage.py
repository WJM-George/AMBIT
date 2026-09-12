#!/usr/bin/env python3
"""Create a deterministic shard-to-volume plan without writing audio data.

Large rendered edit families should not be scattered file-by-file according to
momentary free space. This planner snapshots explicitly selected volumes,
reserves a safety margin, and assigns complete shards to one root. By default,
all newly generated Spatial-CoT data is kept on /mnt/sdb; other roots are only
used when the caller opts in with --root. The resulting small JSON is the
durable routing contract used by later render workers.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any


DEFAULT_ROOTS = (
    "/mnt/sdb/audio_dataset/spatial_cot_v1",
)
GIB = 1024**3


def _volume(root: Path, reserve_gib: float) -> dict[str, Any]:
    probe = root
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    if not probe.exists():
        raise FileNotFoundError(f"no mounted parent exists for {root}")
    stat = os.stat(probe)
    usage = shutil.disk_usage(probe)
    free_gib = usage.free / GIB
    return {
        "root": str(root),
        "mount_probe": str(probe),
        "device_id": int(stat.st_dev),
        "capacity_gib": usage.total / GIB,
        "free_gib_snapshot": free_gib,
        "reserve_gib": reserve_gib,
        "usable_gib": max(0.0, free_gib - reserve_gib),
    }


def build_plan(
    roots: list[Path],
    *,
    reserve_gib: float,
    num_shards: int,
    estimated_total_gib: float | None,
) -> dict[str, Any]:
    volumes = [_volume(root.expanduser().resolve(), reserve_gib) for root in roots]
    devices = [volume["device_id"] for volume in volumes]
    if len(devices) != len(set(devices)):
        raise ValueError("storage roots must resolve to distinct mounted devices")
    usable_total = sum(volume["usable_gib"] for volume in volumes)
    if usable_total <= 0:
        raise ValueError("no usable capacity remains after safety reserves")
    if estimated_total_gib is not None and estimated_total_gib > usable_total:
        raise ValueError(
            f"estimate {estimated_total_gib:.1f} GiB exceeds usable "
            f"capacity {usable_total:.1f} GiB"
        )

    shard_gib = (
        estimated_total_gib / num_shards
        if estimated_total_gib is not None
        else None
    )
    remaining = [float(volume["usable_gib"]) for volume in volumes]
    assignments = []
    for shard_id in range(num_shards):
        # Greedy largest-relative-headroom assignment keeps whole shards on one
        # disk and remains deterministic for a fixed capacity snapshot.
        scores = [
            remaining[index] / max(volume["usable_gib"], 1e-9)
            for index, volume in enumerate(volumes)
        ]
        chosen = max(range(len(volumes)), key=lambda index: (scores[index], remaining[index], -index))
        if shard_gib is not None and remaining[chosen] + 1e-9 < shard_gib:
            raise ValueError(f"no volume can fit planned shard {shard_id}")
        assignments.append(
            {
                "shard": shard_id,
                "root": volumes[chosen]["root"],
                "relative_output": f"render_shards/shard-{shard_id:05d}",
            }
        )
        decrement = shard_gib if shard_gib is not None else (
            volumes[chosen]["usable_gib"] / num_shards
        )
        remaining[chosen] -= decrement

    for index, volume in enumerate(volumes):
        volume["assigned_shards"] = sum(
            assignment["root"] == volume["root"] for assignment in assignments
        )
        volume["planned_gib"] = (
            volume["assigned_shards"] * shard_gib
            if shard_gib is not None
            else None
        )
    return {
        "schema": "stable_audio_tools.spatial_cot_storage_plan",
        "schema_version": 1,
        "policy": "whole_shard_greedy_relative_headroom",
        "num_shards": num_shards,
        "estimated_total_gib": estimated_total_gib,
        "estimated_shard_gib": shard_gib,
        "usable_total_gib": usable_total,
        "volumes": volumes,
        "assignments": assignments,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", default=None)
    parser.add_argument("--reserve-gib", type=float, default=100.0)
    parser.add_argument("--num-shards", type=int, default=64)
    parser.add_argument("--estimated-total-gib", type=float, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.reserve_gib < 0 or args.num_shards <= 0:
        raise SystemExit("reserve-gib must be nonnegative and num-shards positive")
    if args.estimated_total_gib is not None and args.estimated_total_gib <= 0:
        raise SystemExit("estimated-total-gib must be positive")
    try:
        plan = build_plan(
            [Path(value) for value in (args.root or DEFAULT_ROOTS)],
            reserve_gib=args.reserve_gib,
            num_shards=args.num_shards,
            estimated_total_gib=args.estimated_total_gib,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    payload = json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(payload, end="")
        return 0
    output = args.output.expanduser().resolve()
    if output.exists() and not args.overwrite:
        raise SystemExit(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, output)
    print(f"[spatial-cot-storage] wrote plan only: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
