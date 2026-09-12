#!/usr/bin/env python3
"""Generate the frozen Speech panel with Qwen3-TTS VoiceDesign."""

from __future__ import annotations
import os

import argparse
from pathlib import Path

import torch
from qwen_tts import Qwen3TTSModel

from baseline_common import (
    add_common_arguments,
    load_requests,
    pending_rows,
    save_result,
    seed_process,
)


DEFAULT_MODEL = Path(
    os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/baselines/p10_60k_15row_v1/models/"
    "Qwen3-TTS-12Hz-1.7B-VoiceDesign"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser, "qwen3_tts_1p7b_voice_design")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args()

    rows = pending_rows(
        load_requests(
            args.manifest,
            args.baseline_id,
            shard_index=args.shard_index,
            num_shards=args.num_shards,
        ),
        force=args.force,
    )
    if not rows:
        print('{"status":"SKIP","reason":"all outputs valid"}')
        return 0
    model_dir = args.model_dir.expanduser().resolve(strict=True)
    if not torch.cuda.is_available():
        raise RuntimeError("Qwen3-TTS benchmark requires CUDA")
    tts = Qwen3TTSModel.from_pretrained(
        str(model_dir),
        device_map=args.device,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )

    for row in rows:
        seed_process(int(row["seed"]))
        wavs, sample_rate = tts.generate_voice_design(
            text=row["transcript"],
            language="English",
            instruct=row["speaker_description"],
            max_new_tokens=2048,
        )
        save_result(
            row,
            wavs[0],
            int(sample_rate),
            backend_metadata={
                "model": "Qwen3-TTS-12Hz-1.7B-VoiceDesign",
                "language": "English",
                "voice_condition": "speaker_description",
                "attention": "sdpa",
            },
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
