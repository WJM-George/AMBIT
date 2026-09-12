#!/usr/bin/env python3
"""Regression tests for pair-balanced P11 full-curriculum augmentation."""

from __future__ import annotations

import sys
from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.t2a.data.build_sceneplan_p11_v4_full_curriculum import (  # noqa: E402
    _augmentation_selected,
    _augmentations_before,
    _g_queue_start,
    _part_expected_counts,
)
from scripts.t2a.test.validate_sceneplan_p11_v4_full_curriculum import (  # noqa: E402
    _augmentation_selected as validator_augmentation_selected,
)


class FullAugmentationSelectorTests(unittest.TestCase):
    def test_exactly_one_member_of_every_adjacent_pair(self):
        selected = [
            position
            for position in range(320)
            if _augmentation_selected(position, seed=42)
        ]
        self.assertEqual(len(selected), 160)
        self.assertEqual({position // 2 for position in selected}, set(range(160)))
        # This specifically rejects the retired fixed-even selector.
        self.assertGreater(sum(position % 2 == 0 for position in selected), 0)
        self.assertGreater(sum(position % 2 == 1 for position in selected), 0)

    def test_queue_indices_remain_contiguous(self):
        generation_indices: list[int] = []
        editing_pair_indices: list[int] = []
        for position in range(320):
            start = _g_queue_start(position, seed=42)
            generation_indices.append(start)
            if _augmentation_selected(position, seed=42):
                generation_indices.extend((start + 1, start + 2))
                pair_start = 2 * _augmentations_before(position, seed=42)
                editing_pair_indices.extend((pair_start, pair_start + 1))
        self.assertEqual(generation_indices, list(range(640)))
        self.assertEqual(editing_pair_indices, list(range(320)))

    def test_part_counts_sum_to_full_contract(self):
        totals = {
            "generation": 0,
            "understanding": 0,
            "editing_exact": 0,
            "editing_pair": 0,
        }
        for part in range(8):
            counts = _part_expected_counts(
                40 * part, 40 * (part + 1), seed=42
            )
            for key, value in counts.items():
                totals[key] += value
        self.assertEqual(
            totals,
            {
                "generation": 640,
                "understanding": 640,
                "editing_exact": 320,
                "editing_pair": 320,
            },
        )

    def test_builder_and_validator_implement_same_selector(self):
        for position in range(4096):
            self.assertEqual(
                _augmentation_selected(position, seed=42),
                validator_augmentation_selected(position, seed=42),
            )


if __name__ == "__main__":
    unittest.main()
