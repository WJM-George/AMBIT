#!/usr/bin/env python3
"""Tests for the deliberately minimal source-description contract."""

from __future__ import annotations

import unittest

from description_contract import compact_whitespace, validate_source_description


class DescriptionContractTest(unittest.TestCase):
    def test_accepts_normal_english_audio_description(self) -> None:
        qc = validate_source_description(
            "A bright acoustic guitar strums a relaxed folk melody while soft hand percussion keeps a steady rhythm."
        )
        self.assertTrue(qc.hard_gates_ok)

    def test_allows_recording_and_listener_words(self) -> None:
        qc = validate_source_description(
            "The audio clip lets the listener hear a buzzing insect flying close to the microphone with rapidly vibrating wings."
        )
        self.assertTrue(qc.hard_gates_ok)

    def test_allows_generic_or_semantic_speech_description(self) -> None:
        qc = validate_source_description(
            "A person whispers instructions while a dog pants softly and moves through dry leaves nearby."
        )
        self.assertTrue(qc.hard_gates_ok)

    def test_rejects_non_english_output(self) -> None:
        qc = validate_source_description("持续的金属嗡鸣声逐渐增强，随后突然停止。")
        self.assertIn("not_predominantly_english", qc.hard_flags)

    def test_word_target_is_soft(self) -> None:
        qc = validate_source_description("A bell rings clearly.")
        self.assertTrue(qc.hard_gates_ok)
        self.assertIn("word_count_outside_soft_target", qc.soft_flags)

    def test_compacts_whitespace_without_rewriting(self) -> None:
        raw = "A piano plays   slow chords\nunder a warm string melody."
        self.assertEqual(
            "A piano plays slow chords under a warm string melody.",
            compact_whitespace(raw),
        )


if __name__ == "__main__":
    unittest.main()
