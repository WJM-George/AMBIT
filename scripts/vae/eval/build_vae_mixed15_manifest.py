#!/usr/bin/env python3
"""Build a fixed held-out 5 SLS + 5 music + 5 sound FOA manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SLS_ROOT = Path("/mnt/sdb/audio_dataset/datasets/spatial_librispeech/ambisonics")
SYNTHETIC_MANIFEST = Path(
    "/mnt/sdd/audio_dataset/spatial_foa_v2/manifest_test_synthetic.jsonl"
)

SLS_IDS = ("027352", "199154", "026810", "053407", "189938")
MUSIC_IDS = (
    "spv2_test_0004546",  # heavy metal
    "spv2_test_0003559",  # general music
    "spv2_test_0004177",  # wind instrument
    "spv2_test_0000514",  # zither/country
    "spv2_test_0004310",  # disco/opera/pop
)
SOUND_IDS = (
    "spv2_test_0000411",  # glass shatter
    "spv2_test_0000229",  # keyboard typing
    "spv2_test_0000030",  # bird chirping
    "spv2_test_0000298",  # chainsaw
    "spv2_test_0000201",  # rain and thunder
)


def load_synthetic_rows() -> dict[str, dict]:
    rows = {}
    with SYNTHETIC_MANIFEST.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows[row["id"]] = row
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    synthetic = load_synthetic_rows()
    selected = []
    for sample_id in SLS_IDS:
        path = SLS_ROOT / f"{sample_id}.flac"
        selected.append(
            {
                "id": f"sls:{sample_id}",
                "sample_id": sample_id,
                "path": str(path),
                "source_group": "sls",
                "source_family": "speech",
                "caption": "Held-out Spatial LibriSpeech FOA",
                "native_sample_rate": 16_000,
                "high_band_valid": False,
            }
        )

    for group, ids in (("music", MUSIC_IDS), ("sound", SOUND_IDS)):
        expected_category = "music" if group == "music" else "audio"
        for sample_id in ids:
            row = synthetic.get(sample_id)
            if row is None:
                raise KeyError(f"Missing {sample_id} in {SYNTHETIC_MANIFEST}")
            if row.get("status") != "ok" or row.get("category") != expected_category:
                raise ValueError(f"Unexpected category/status for {sample_id}: {row}")
            selected.append(
                {
                    "id": f"{group}:{sample_id}",
                    "sample_id": sample_id,
                    "path": row["foa_path"],
                    "source_group": group,
                    "source_family": "non_speech",
                    "caption": row["spatial_caption"],
                    "labels": [source.get("label", "") for source in row["sources"]],
                    "native_sample_rate": int(row["sample_rate"]),
                    "high_band_valid": True,
                }
            )

    if len(selected) != 15 or len({row["id"] for row in selected}) != 15:
        raise RuntimeError("Expected exactly 15 unique mixed evaluation items")
    for row in selected:
        if not Path(row["path"]).is_file():
            raise FileNotFoundError(row["path"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    summary = {
        "manifest": str(args.out),
        "count": len(selected),
        "groups": {group: sum(row["source_group"] == group for row in selected) for group in ("sls", "music", "sound")},
        "selection": {
            "sls": list(SLS_IDS),
            "music": list(MUSIC_IDS),
            "sound": list(SOUND_IDS),
        },
        "notes": {
            "sls": "Same held-out SLS clips used in earlier VAE comparisons",
            "music_sound": "Held-out Spatial FOA v2 synthetic test split, absent from training",
        },
    }
    args.out.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
