#!/usr/bin/env python3
"""Tests for deterministic sharding and crash-tail recovery."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from caption_transformers import load_done, load_input, output_for_shard
from finalize_source_registry import iter_jsonl


class ResumeContractTest(unittest.TestCase):
    def test_worker_loads_only_missing_rows_from_its_modulo_shard(self) -> None:
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            audio = root / "audio.wav"
            audio.write_bytes(b"fixture")
            manifest = root / "input.jsonl"
            with manifest.open("w", encoding="utf-8") as sink:
                for ordinal in range(12):
                    sink.write(
                        json.dumps(
                            {
                                "id": f"id-{ordinal}",
                                "audio_path": str(audio),
                                "kind": "music",
                            }
                        )
                        + "\n"
                    )
            rows, scanned, assigned, complete = load_input(
                manifest,
                shard=1,
                num_shards=4,
                done={"id-1"},
                limit=2,
            )
            self.assertEqual(["id-5", "id-9"], [row["id"] for row in rows])
            self.assertEqual(10, scanned)
            self.assertEqual(3, assigned)
            self.assertFalse(complete)

    def test_runner_ignores_only_an_incomplete_final_line(self) -> None:
        with tempfile.TemporaryDirectory() as value:
            path = Path(value) / "output.jsonl"
            path.write_text('{"id":"done"}\n{"id":', encoding="utf-8")
            self.assertEqual({"done"}, load_done(path))

    def test_finalizer_ignores_only_an_incomplete_final_line(self) -> None:
        with tempfile.TemporaryDirectory() as value:
            path = Path(value) / "output.jsonl"
            path.write_text('{"id":"done"}\n{"id":', encoding="utf-8")
            rows = list(iter_jsonl(path, tolerate_incomplete_tail=True))
            self.assertEqual([{"id": "done"}], rows)
            with self.assertRaises(ValueError):
                list(iter_jsonl(path, tolerate_incomplete_tail=False))

    def test_shard_filename_is_stable(self) -> None:
        path = output_for_shard(Path("/tmp/output.jsonl"), 3, 4)
        self.assertEqual("output.shard003-of-004.jsonl", path.name)


if __name__ == "__main__":
    unittest.main()
