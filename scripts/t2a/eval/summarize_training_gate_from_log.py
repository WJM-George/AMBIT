#!/usr/bin/env python3
"""Extract the final machine-readable training gate from a retained log."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


PREFIX = "SAT_TRAINING_GATE_RESULT="


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.log.is_file():
        raise FileNotFoundError(args.log)
    matches = []
    text = args.log.read_text(encoding="utf-8", errors="replace").replace("\r", "\n")
    for line in text.splitlines():
        position = line.find(PREFIX)
        if position >= 0:
            matches.append(json.loads(line[position + len(PREFIX) :]))
    if len(matches) != 1:
        raise RuntimeError(f"expected one training gate in {args.log}, found {len(matches)}")
    output = {
        "schema": "stable_audio_tools.retained_training_gate",
        "schema_version": 1,
        "source_log": str(args.log.expanduser().resolve()),
        "gate": matches[0],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(args.output.expanduser().resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
