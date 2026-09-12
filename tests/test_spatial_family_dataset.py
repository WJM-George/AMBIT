from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

from stable_audio_tools.data.spatial_family_dataset import SpatialFamilyDataset


def _create_store(root: Path, family_count: int = 3) -> None:
    (root / "shards").mkdir(parents=True)
    tensors = {}
    metadata = bytearray()
    rows = []
    for rank in range(family_count):
        family_id = f"family_{rank}"
        tensor_key = f"tensor_{rank}"
        tensors[tensor_key] = torch.full((1, 2, 3), float(rank))
        payload = json.dumps(
            {"family_id": family_id, "turns": [{"turn_id": f"turn_{rank}"}]},
            separators=(",", ":"),
        ).encode("utf-8")
        offset = len(metadata)
        metadata.extend(payload + b"\n")
        rows.append(
            (
                rank,
                family_id,
                "shards/tensors.safetensors",
                tensor_key,
                "shards/metadata.jsonl",
                offset,
                len(payload),
                1,
                2,
                3,
                "float32",
            )
        )

    save_file(tensors, str(root / "shards" / "tensors.safetensors"))
    (root / "shards" / "metadata.jsonl").write_bytes(metadata)
    with sqlite3.connect(root / "index.sqlite") as connection:
        connection.execute(
            "CREATE TABLE families ("
            "family_rank INTEGER PRIMARY KEY, family_id TEXT NOT NULL UNIQUE, "
            "tensor_shard TEXT NOT NULL, tensor_key TEXT NOT NULL, "
            "metadata_shard TEXT NOT NULL, metadata_offset INTEGER NOT NULL, "
            "metadata_length INTEGER NOT NULL, num_turns INTEGER NOT NULL, "
            "channels INTEGER NOT NULL, frames INTEGER NOT NULL, dtype TEXT NOT NULL)"
        )
        connection.executemany(
            "INSERT INTO families VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows
        )
    (root / "READY").write_text(
        json.dumps({"families": family_count}) + "\n", encoding="utf-8"
    )


class SpatialFamilyRankSelectionTests(unittest.TestCase):
    def test_selection_preserves_requested_rank_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _create_store(root)
            dataset = SpatialFamilyDataset(
                [{"path": root, "family_ranks": [2, 0]}]
            )
            self.assertEqual(len(dataset), 2)
            first, first_info = dataset[0]
            second, second_info = dataset[1]
            self.assertTrue(torch.equal(first, torch.full((1, 2, 3), 2.0)))
            self.assertEqual(first_info["family_id"], "family_2")
            self.assertTrue(torch.equal(second, torch.zeros(1, 2, 3)))
            self.assertEqual(second_info["family_id"], "family_0")
            for family_root in dataset.roots:
                family_root.close()

    def test_invalid_rank_selections_fail_closed(self):
        cases = (
            ([0, 0], ValueError, "duplicates"),
            ([], ValueError, "empty"),
            ([-1], ValueError, "negative"),
            ([True], TypeError, "boolean"),
            ([1.5], TypeError, "non-integer"),
            ([3], IndexError, "missing"),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _create_store(root)
            for ranks, exception, label in cases:
                with self.subTest(label=label), self.assertRaises(exception):
                    SpatialFamilyDataset([{"path": root, "family_ranks": ranks}])


if __name__ == "__main__":
    unittest.main()
