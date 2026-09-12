#!/usr/bin/env python3
"""Executable tests for the explicit spoken-language boundary."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from classify_spoken_language import classify_description


CASES = (
    Path(__file__).with_name("tests")
    / "spoken_language_policy_cases_v1.jsonl"
)


class SpokenLanguagePolicyTest(unittest.TestCase):
    def test_frozen_cases(self) -> None:
        for line in CASES.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            actual, _ = classify_description(row["source_description"])
            self.assertEqual(row["expected"], actual, row["id"])

    def test_singing_and_rap_are_not_spoken_language(self) -> None:
        for description in (
            "A woman sings an operatic melody over strings.",
            "A rapper performs over a heavy hip-hop beat.",
        ):
            actual, _ = classify_description(description)
            self.assertFalse(actual)

    def test_explicit_speaking_is_detected(self) -> None:
        actual, evidence = classify_description(
            "A vocalist sings over music, then a woman speaks a short sentence."
        )
        self.assertTrue(actual)
        self.assertIn("speaks", evidence)


if __name__ == "__main__":
    unittest.main()
