#!/usr/bin/env python3
"""Require complete PASS markers for both retained evaluation splits."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from stable_audio_tools.data.spatial_edit_recipe import RECIPE_VERSION


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-spec", type=Path, required=True)
    args = parser.parse_args()
    build_spec = args.build_spec.expanduser().resolve()
    spec = json.loads(build_spec.read_text(encoding="utf-8"))
    build_spec_sha256 = _sha256(build_spec)
    root = Path(spec["storage"]["retained_eval_render_root"]).expanduser().resolve()
    accepted = {}
    for split in ("test", "validation"):
        expected = int(spec["splits"][split]["families"])
        path = root / split / "QUALITY.json"
        if not path.is_file():
            raise SystemExit(
                f"refusing train start: retained {split} quality gate is missing: {path}"
            )
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise SystemExit(f"invalid retained {split} quality gate: {path}: {error}")
        if (
            report.get("schema") != "stable_audio_tools.spatial_cot_eval_quality"
            or int(report.get("schema_version", -1)) != 2
            or report.get("status") != "PASS"
            or report.get("split") != split
            or int(report.get("families", -1)) != expected
            or report.get("recipe_version") != RECIPE_VERSION
            or report.get("build_spec_sha256") != build_spec_sha256
            or int(report.get("all_state_signal_stats_verified", -1))
            != expected * int(spec["splits"][split]["states_per_family"])
            or float(report.get("rms_min", -1.0))
            < float(spec["quality_control"]["minimum_rms"])
            or float(report.get("active_100ms_fraction_min", -1.0))
            < float(
                spec["quality_control"]["minimum_active_100ms_fraction"]
            )
        ):
            raise SystemExit(f"retained {split} quality gate did not pass: {path}")
        accepted[split] = str(path)
    print(json.dumps({"status": "PASS", "quality_gates": accepted}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
