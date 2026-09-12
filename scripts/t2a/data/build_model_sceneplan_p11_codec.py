#!/usr/bin/env python3
"""Build the canonical frame-aligned P11 ScenePlan codec from frozen source text."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import zlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from stable_audio_tools.data.model_sceneplan_codec import (  # noqa: E402
    iter_model_sceneplan_text_fields,
)
from stable_audio_tools.data.model_sceneplan_codec_v4 import (  # noqa: E402
    ModelScenePlanCodecV4,
    create_model_sceneplan_codec_v4_artifact,
)


def iter_text(index_path: Path, *, max_rows: int | None):
    uri = f"file:{index_path.resolve()}?mode=ro&immutable=1"
    connection = sqlite3.connect(uri, uri=True)
    try:
        query = "SELECT scene_plan_zlib FROM samples ORDER BY ordinal"
        parameters = ()
        if max_rows is not None:
            query += " LIMIT ?"
            parameters = (int(max_rows),)
        for (compressed,) in connection.execute(query, parameters):
            plan = json.loads(zlib.decompress(compressed))
            yield from iter_model_sceneplan_text_fields(plan)
    finally:
        connection.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-rows", type=int, default=100_000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.max_rows is not None and args.max_rows <= 0:
        raise ValueError("--max-rows must be positive")
    output = create_model_sceneplan_codec_v4_artifact(
        args.output,
        iter_text(args.index, max_rows=args.max_rows),
        vocab_size=4096,
        text_vocab_size=1536,
        overwrite=bool(args.overwrite),
    )
    codec = ModelScenePlanCodecV4(output)
    print(
        json.dumps(
            {
                "output": str(output),
                "codec_fingerprint": codec.fingerprint,
                "vocab_size": codec.vocab_size,
                "used_vocab_size": codec.details["used_vocab_size"],
                "projection_contract": codec.details["projection_contract"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
