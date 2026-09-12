#!/usr/bin/env python3
"""Build ``spatial_plan_codec_v2`` from canonical ScenePlan string fields.

Only the compact BPE is learned from data. Grammar and quantization ids are
versioned in code, so two builds with the same corpus and arguments have the
same token layout. The source JSONL remains the authoritative plan index;
crop-aware plan tokens are produced by the dataset worker at training time.
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import chain
from pathlib import Path
from typing import Iterator, Optional

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stable_audio_tools.data.spatial_plan_codec import (  # noqa: E402
    DEFAULT_TEXT_CORPUS,
    SpatialPlanCodec,
    create_codec_artifact,
    iter_plan_text_fields,
)


def _iter_scene_plan_texts(root: Path, limit: Optional[int]) -> Iterator[str]:
    count = 0
    shards = sorted((root / "shards").glob("*.jsonl"))
    if not shards:
        raise FileNotFoundError(f"no ScenePlan JSONL shards under {root}")
    for shard in shards:
        with shard.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                plan = json.loads(line)
                yield from iter_plan_text_fields(plan)
                count += 1
                if limit is not None and count >= limit:
                    return


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scene-plan-root", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None, help="plan cap for smoke builds")
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--text-vocab-size", type=int, default=2048)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive")
    if args.scene_plan_root is None:
        texts = iter(DEFAULT_TEXT_CORPUS)
        source = "built-in bootstrap corpus"
    else:
        scene_plan_root = args.scene_plan_root.expanduser().resolve()
        # Always retain the control-domain bootstrap words, even in a tiny smoke
        # subset that happens not to contain a particular direction or quality.
        texts = chain(DEFAULT_TEXT_CORPUS, _iter_scene_plan_texts(scene_plan_root, args.limit))
        source = str(scene_plan_root)

    output = create_codec_artifact(
        args.output_root,
        texts,
        vocab_size=args.vocab_size,
        text_vocab_size=args.text_vocab_size,
        overwrite=args.overwrite,
    )
    codec = SpatialPlanCodec(output)
    print(
        json.dumps(
            {
                "status": "READY",
                "output_root": str(output),
                "source": source,
                "codec_fingerprint": codec.fingerprint,
                "vocab_size": codec.vocab_size,
                "used_vocab_size": codec.details["used_vocab_size"],
                "text_vocab_size": codec.text_vocab_size,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
