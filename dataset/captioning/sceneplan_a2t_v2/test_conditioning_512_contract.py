#!/usr/bin/env python3
"""Synchronized checks for the revised 512-token conditioning envelope."""

from __future__ import annotations

import json
from pathlib import Path
import unittest


REPO = Path(__file__).resolve().parents[3]


class Conditioning512ContractTest(unittest.TestCase):
    def test_amendment_and_compiler_schema_agree(self) -> None:
        amendment = json.loads(
            (
                REPO
                / "docs/sceneplan_v2/sceneplan_conditioning_amendment_v2_512.json"
            ).read_text()
        )
        schema = json.loads(
            (
                REPO
                / "docs/sceneplan_v2/sceneplan_compiler_output_v3.schema.json"
            ).read_text()
        )
        self.assertEqual(2, amendment["conditioning_contract_revision"])
        self.assertEqual(512, amendment["caption"]["max_tokens"])
        self.assertEqual(384, amendment["caption"]["p99_target_tokens"])
        self.assertEqual(512, schema["properties"]["sequence_length"]["maximum"])
        self.assertEqual(512, schema["properties"]["tokenizer"]["properties"]["max_tokens"]["const"])
        self.assertFalse(schema["properties"]["tokenizer"]["properties"]["truncated"]["const"])

    def test_model_and_dataset_configs_agree(self) -> None:
        model = json.loads(
            (
                REPO
                / "stable_audio_tools/configs/model_configs/txt2audio/t2a/dit/"
                "qwen35_0p8b_300m_model_sceneplan_44_preflight_candidate.json"
            ).read_text()
        )
        self.assertEqual(
            512,
            model["model"]["conditioning"]["configs"][0]["config"]["max_length"],
        )
        for split in ("train", "validation", "test"):
            dataset = json.loads(
                (
                    REPO
                    / f"stable_audio_tools/configs/dataset_configs/sceneplan_v2_{split}.json"
                ).read_text()
            )
            self.assertEqual(512, dataset["caption_max_tokens"])


if __name__ == "__main__":
    unittest.main()
