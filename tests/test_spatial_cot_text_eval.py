from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from scripts.t2a.eval.evaluate_spatial_cot_text import _canonical_device


class SpatialCotTextEvalTests(unittest.TestCase):
    def test_cpu_device_is_unchanged(self):
        self.assertEqual(_canonical_device("cpu"), torch.device("cpu"))

    def test_unindexed_cuda_alias_uses_process_local_device(self):
        with patch.object(torch.cuda, "current_device", return_value=3):
            self.assertEqual(_canonical_device("cuda"), torch.device("cuda:3"))

    def test_explicit_cuda_index_is_unchanged(self):
        with patch.object(torch.cuda, "current_device") as current_device:
            self.assertEqual(_canonical_device("cuda:2"), torch.device("cuda:2"))
            current_device.assert_not_called()


if __name__ == "__main__":
    unittest.main()
