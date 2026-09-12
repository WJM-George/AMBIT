#!/usr/bin/env python3
"""Minimal contract for a direct English semantic audio description."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass


TARGET_MIN_WORDS = 20
TARGET_MAX_WORDS = 30
WORD_RE = re.compile(r"[A-Za-z]+(?:[-'][A-Za-z]+)*")
SENTENCE_END_RE = re.compile(r"[.!?](?:[\"'”’]+)?(?=\s|$)")
PROMPT_TEMPLATE_VERSION = "qwen3_omni_system_instruction_audio_user_v1"
PROMPT_USER_TRIGGER = "Produce the requested source description now."


@dataclass(frozen=True)
class DescriptionQC:
    text: str
    word_count: int
    sentence_count: int
    hard_flags: tuple[str, ...]
    soft_flags: tuple[str, ...]
    informational_flags: tuple[str, ...] = ()

    @property
    def hard_gates_ok(self) -> bool:
        return not self.hard_flags


def compact_whitespace(text: str) -> str:
    """The only permitted cleanup; no words are added, removed, or rewritten."""
    return " ".join(str(text).split())


def predominantly_english(text: str) -> bool:
    letters = [character for character in text if character.isalpha()]
    if not letters or len(WORD_RE.findall(text)) < 3:
        return False
    ascii_letters = sum(character.isascii() for character in letters)
    return ascii_letters / len(letters) >= 0.85


def prompt_contract_sha256(system_instruction: str) -> str:
    payload = {
        "template_version": PROMPT_TEMPLATE_VERSION,
        "messages": [
            {"role": "system", "content": system_instruction},
            {
                "role": "user",
                "content": ["<audio>", PROMPT_USER_TRIGGER],
            },
        ],
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def validate_source_description(raw_text: str) -> DescriptionQC:
    """Check only that output is non-empty English; length is soft."""
    text = compact_whitespace(raw_text)
    words = len(WORD_RE.findall(text))
    hard: list[str] = []
    soft: list[str] = []
    if not text:
        hard.append("empty_response")
    elif not predominantly_english(text):
        hard.append("not_predominantly_english")
    if not TARGET_MIN_WORDS <= words <= TARGET_MAX_WORDS:
        soft.append("word_count_outside_soft_target")
    return DescriptionQC(
        text=text,
        word_count=words,
        sentence_count=len(SENTENCE_END_RE.findall(text)),
        hard_flags=tuple(hard),
        soft_flags=tuple(soft),
    )
