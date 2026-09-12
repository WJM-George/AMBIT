from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from stable_audio_tools.data.dataset import (
    LatentDatasetConfig,
    PreEncodedDataset,
    _local_dataloader_kwargs,
    _normalize_padding_mask,
    _validate_finalized_latent_cache,
    get_audio_filenames,
)


class _OffsetTokenizer:
    def __call__(self, text, **kwargs):
        max_length = kwargs["max_length"]
        ids = torch.zeros(1, max_length, dtype=torch.long)
        mask = torch.zeros(1, max_length, dtype=torch.long)
        offsets = torch.zeros(1, max_length, 2, dtype=torch.long)
        ids[0, :2] = torch.tensor([10, 11])
        mask[0, :2] = 1
        offsets[0, 0] = torch.tensor([0, 8])
        offsets[0, 1] = torch.tensor([9, len(text)])
        result = {"input_ids": ids, "attention_mask": mask}
        if kwargs.get("return_offsets_mapping"):
            result["offset_mapping"] = offsets
        return result


def _write_latent_pair(root: Path) -> tuple[Path, Path]:
    latent_path = root / "sample.npy"
    metadata_path = root / "sample.json"
    np.save(latent_path, np.arange(10, dtype=np.float32).reshape(2, 5))
    metadata_path.write_text(
        json.dumps(
            {
                "prompt": 'says "hello"',
                "seconds_start": 2.0,
                "seconds_total": 10.0,
                "timestamps": [10.0, 20.0],
                "padding_mask": [1, 1, 1, 1, 1],
            }
        ),
        encoding="utf-8",
    )
    return latent_path, metadata_path


class ManifestTests(unittest.TestCase):
    def test_audio_manifest_preserves_absolute_entries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            absolute = root / "absolute.wav"
            relative = root / "relative.wav"
            (root / "filelist.txt").write_text(
                f"{absolute}\n{relative.name}\n\n",
                encoding="utf-8",
            )
            self.assertEqual(
                get_audio_filenames(str(root)),
                [str(absolute), str(relative)],
            )

    def test_ready_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "READY").write_text(
                json.dumps(
                    {
                        "schema": "stable_audio_tools.preencoded_ready",
                        "entries": 1,
                    }
                ),
                encoding="utf-8",
            )
            _validate_finalized_latent_cache(str(root), 1)
            with self.assertRaisesRegex(RuntimeError, "count mismatch"):
                _validate_finalized_latent_cache(str(root), 2)


class PreEncodedDatasetTests(unittest.TestCase):
    def test_no_crop_flattens_mask_and_infers_duration_from_timestamps(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, metadata_path = _write_latent_pair(root)
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata.pop("seconds_total")
            metadata["padding_mask"] = [[1, 1, 1, 0, 0]]
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

            dataset = PreEncodedDataset(
                [LatentDatasetConfig(id="test", path=str(root))]
            )
            latent, result = dataset[0]

            self.assertEqual(tuple(latent.shape), (2, 5))
            self.assertEqual(tuple(result["padding_mask"][0].shape), (5,))
            self.assertEqual(result["padding_mask"][0].dtype, torch.bool)
            self.assertEqual(result["seconds_total"], 10.0)

    def test_crop_updates_temporal_metadata_and_worker_tokens(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_latent_pair(root)
            dataset = PreEncodedDataset(
                [
                    LatentDatasetConfig(
                        id="test",
                        path=str(root),
                        latent_extension="npy",
                    )
                ],
                latent_crop_length=3,
                random_crop=True,
                latent_downsampling_ratio=10,
                sample_rate=10,
                tokenizers={
                    "prompt": (
                        _OffsetTokenizer(),
                        8,
                        {"enabled": True, "strategy": "speech_quote_v1"},
                    )
                },
            )
            with mock.patch(
                "stable_audio_tools.data.dataset.random.randint",
                return_value=2,
            ):
                latent, metadata = dataset[0]

            self.assertEqual(tuple(latent.shape), (2, 3))
            self.assertEqual(metadata["latent_crop_start"], 2)
            self.assertEqual(metadata["seconds_start"], 4.0)
            self.assertEqual(metadata["seconds_total"], 3.0)
            self.assertEqual(metadata["latent_source_seconds_total"], 10.0)
            self.assertEqual(metadata["timestamps"], [14.0, 20.0])
            self.assertEqual(metadata["prompt_text"], 'says "hello"')
            self.assertEqual(tuple(metadata["prompt"]["input_ids"].shape), (8,))
            self.assertIn("region_ids", metadata["prompt"])

    def test_num_workers_zero_omits_worker_only_options(self):
        kwargs = _local_dataloader_kwargs({}, 0)
        self.assertNotIn("persistent_workers", kwargs)
        self.assertNotIn("prefetch_factor", kwargs)

    def test_padding_mask_normalization_is_flat_and_length_bounded(self):
        self.assertEqual(
            _normalize_padding_mask([[1, 0]], 4),
            [True, False, False, False],
        )
        self.assertEqual(
            _normalize_padding_mask(np.array([[1, 1, 0]]), 2),
            [True, True],
        )
        with self.assertRaisesRegex(ValueError, "contiguous valid prefix"):
            _normalize_padding_mask([1, 0, 1], 3)


if __name__ == "__main__":
    unittest.main()
