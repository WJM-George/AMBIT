#!/usr/bin/env python3
"""Plan or apply safe pruning of step-numbered Lightning checkpoints.

Dry-run is the default. ``last.ckpt``, the highest step, recent checkpoints, and
regular milestones are always retained. Deletion requires the explicit
``--apply`` switch.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

STEP_RE = re.compile(r"(?:^|[-_])step[=_-]?(\d+)(?:\.ckpt)?$")


@dataclass(frozen=True)
class Checkpoint:
    path: Path
    step: int
    size: int


def parse_step(path: Path) -> int | None:
    match = STEP_RE.search(path.stem)
    return int(match.group(1)) if match else None


def discover(roots: list[Path]) -> list[Checkpoint]:
    by_path: dict[Path, Checkpoint] = {}
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.ckpt"):
            if path.is_symlink() or path.name == "last.ckpt":
                continue
            step = parse_step(path)
            if step is None:
                continue
            resolved = path.resolve()
            by_path[resolved] = Checkpoint(resolved, step, path.stat().st_size)
    return sorted(by_path.values(), key=lambda item: (item.step, str(item.path)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--keep-recent", type=int, default=3)
    parser.add_argument("--keep-every", type=int, default=50_000)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if args.keep_recent < 1:
        raise SystemExit("--keep-recent must be >= 1")
    if args.keep_every < 0:
        raise SystemExit("--keep-every must be >= 0")

    checkpoints = discover(args.roots)
    if not checkpoints:
        print("No step-numbered checkpoints found.")
        return

    recent = set(checkpoints[-args.keep_recent :])
    highest_step = checkpoints[-1].step
    keep: set[Checkpoint] = set(recent)
    for checkpoint in checkpoints:
        if checkpoint.step == highest_step:
            keep.add(checkpoint)
        if args.keep_every and checkpoint.step % args.keep_every == 0:
            keep.add(checkpoint)

    prune = [checkpoint for checkpoint in checkpoints if checkpoint not in keep]
    reclaim = sum(checkpoint.size for checkpoint in prune)
    print(
        f"found={len(checkpoints)} keep={len(keep)} prune={len(prune)} "
        f"reclaim_gib={reclaim / 2**30:.2f} mode={'APPLY' if args.apply else 'DRY-RUN'}"
    )
    for checkpoint in prune:
        print(f"{'DELETE' if args.apply else 'would delete'} step={checkpoint.step} {checkpoint.path}")
        if args.apply:
            checkpoint.path.unlink()


if __name__ == "__main__":
    main()
