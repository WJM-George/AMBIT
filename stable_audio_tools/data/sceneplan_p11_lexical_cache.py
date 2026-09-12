"""Read-only lexical evidence produced by a frozen ASR model for P11-U.

This cache is deliberately separate from the ScenePlan target manifest.  The
planner may consume only an ASR hypothesis generated from the input FOA; the
authoritative target transcript must never be copied into the evidence path.
"""

from __future__ import annotations

import math
import sqlite3
from pathlib import Path
from typing import Any, Mapping


P11_LEXICAL_CACHE_SCHEMA = "stable_audio_tools.p11_lexical_cache"
P11_LEXICAL_CACHE_VERSION = 2
P11_LEXICAL_EVIDENCE_CONTRACT = "frozen_asr_input_foa_confidence_v2"
P11_LEXICAL_CONFIDENCE_CONTRACT = (
    "geometric_mean_language_token_speech_v1"
)
P11_LEXICAL_AUTHORITY_CONTRACT = "frozen_asr_reliable_lexical_authority_v1"
P11_LEXICAL_ASSEMBLER_POLICY = "deterministic_reliable_speech_assembler_v1"
P11_LEXICAL_TOKEN_CONCAT_POLICY = "optional_reliable_speech_only_v1"


def parse_reliable_lexical_authority(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the input-only ASR authority object shared by every arm."""

    if not isinstance(value, Mapping):
        raise ValueError(
            "deterministic lexical assembly requires structured ASR evidence"
        )
    authority = value.get("lexical_authority")
    if not isinstance(authority, Mapping):
        raise ValueError("reliable ASR evidence lacks lexical authority metadata")
    if authority.get("contract") != P11_LEXICAL_AUTHORITY_CONTRACT:
        raise ValueError("reliable ASR lexical authority contract changed")
    transcript = " ".join(str(authority.get("transcript") or "").split())
    if not transcript:
        raise ValueError("reliable ASR lexical authority has an empty transcript")
    confidence = float(authority.get("confidence", float("nan")))
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise ValueError("reliable ASR lexical confidence is invalid")
    if authority.get("target_transcript_access") is not False:
        raise ValueError("lexical authority must forbid target transcript access")
    return {
        "contract": P11_LEXICAL_AUTHORITY_CONTRACT,
        "transcript": transcript,
        "confidence": confidence,
        "language": str(authority.get("language") or "unknown"),
    }


class P11LexicalEvidenceCache:
    """Strict process-safe reader for one frozen-ASR hypothesis per scene."""

    def __init__(
        self,
        path: str | Path,
        *,
        source_manifest: str | Path,
        source_index: str | Path,
        encoder_revision: str,
        expected_ordinals: list[int],
    ) -> None:
        self.path = Path(path).expanduser().resolve(strict=True)
        self.source_manifest = Path(source_manifest).expanduser().resolve(strict=True)
        self.source_index = Path(source_index).expanduser().resolve(strict=True)
        self.encoder_revision = str(encoder_revision)
        if not self.encoder_revision:
            raise ValueError("P11 lexical cache requires a frozen ASR revision")
        self._connection: sqlite3.Connection | None = None

        connection = self._open()
        try:
            metadata = dict(connection.execute("SELECT key, value FROM metadata"))
            required = {
                "schema": P11_LEXICAL_CACHE_SCHEMA,
                "schema_version": str(P11_LEXICAL_CACHE_VERSION),
                "contract": P11_LEXICAL_EVIDENCE_CONTRACT,
                "source_manifest": str(self.source_manifest),
                "source_index": str(self.source_index),
                "encoder_revision": self.encoder_revision,
                "source": "input_foa_only",
                "target_transcript_access": "forbidden",
                "confidence_contract": P11_LEXICAL_CONFIDENCE_CONTRACT,
            }
            for key, expected in required.items():
                if metadata.get(key) != expected:
                    raise RuntimeError(
                        f"P11 lexical cache {key}={metadata.get(key)!r}, "
                        f"expected {expected!r}"
                    )
            columns = [
                str(row[1])
                for row in connection.execute("PRAGMA table_info(hypotheses)")
            ]
            if columns != [
                "ordinal",
                "text",
                "has_speech",
                "confidence",
                "language",
                "language_probability",
                "mean_average_log_probability",
                "mean_no_speech_probability",
                "speech_seconds",
                "segment_count",
            ]:
                raise RuntimeError("P11 lexical cache hypothesis columns changed")
            cached = [
                int(row[0])
                for row in connection.execute(
                    "SELECT ordinal FROM hypotheses ORDER BY ordinal"
                )
            ]
            if cached != [int(value) for value in expected_ordinals]:
                raise RuntimeError(
                    "P11 lexical cache does not exactly cover the manifest scenes"
                )
            invalid = int(
                connection.execute(
                    """
                    SELECT COUNT(*) FROM hypotheses
                    WHERE has_speech NOT IN (0,1)
                       OR confidence < 0.0 OR confidence > 1.0
                       OR confidence IS NULL
                       OR language IS NULL OR trim(language) = ''
                       OR language_probability < 0.0 OR language_probability > 1.0
                       OR language_probability IS NULL
                       OR speech_seconds < 0.0 OR speech_seconds IS NULL
                       OR segment_count < 0 OR segment_count IS NULL
                       OR (segment_count = 0 AND (
                              mean_average_log_probability IS NOT NULL
                           OR mean_no_speech_probability IS NOT NULL
                       ))
                       OR (segment_count > 0 AND (
                              mean_average_log_probability IS NULL
                           OR mean_no_speech_probability IS NULL
                           OR mean_no_speech_probability < 0.0
                           OR mean_no_speech_probability > 1.0
                       ))
                       OR (has_speech = 1 AND (text IS NULL OR trim(text) = ''))
                       OR (has_speech = 1 AND segment_count = 0)
                       OR (has_speech = 0 AND (text IS NULL OR trim(text) != ''))
                    """
                ).fetchone()[0]
            )
            if invalid:
                raise RuntimeError(
                    f"P11 lexical cache contains {invalid} invalid hypotheses"
                )
        finally:
            connection.close()

    def _open(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            f"file:{self.path}?mode=ro&immutable=1",
            uri=True,
            check_same_thread=False,
        )
        connection.execute("PRAGMA query_only=ON")
        return connection

    def _db(self) -> sqlite3.Connection:
        if self._connection is None:
            self._connection = self._open()
        return self._connection

    def row(self, ordinal: int) -> dict[str, Any]:
        value = self._db().execute(
            """
            SELECT text, has_speech, confidence, language,
                   language_probability, mean_average_log_probability,
                   mean_no_speech_probability, speech_seconds, segment_count
            FROM hypotheses WHERE ordinal = ?
            """,
            (int(ordinal),),
        ).fetchone()
        if value is None:
            raise RuntimeError(f"P11 lexical cache lacks ordinal {ordinal}")
        text = " ".join(str(value[0] or "").split())
        has_speech = bool(int(value[1]))
        confidence = float(value[2])
        language = str(value[3])
        if has_speech and not text:
            raise RuntimeError("P11 lexical speech hypothesis is empty")
        if not 0.0 <= confidence <= 1.0:
            raise RuntimeError("P11 lexical confidence is outside [0,1]")
        return {
            "contract": P11_LEXICAL_EVIDENCE_CONTRACT,
            "text": text,
            "has_speech": has_speech,
            "confidence": confidence,
            "language": language,
            "language_probability": float(value[4]),
            "mean_average_log_probability": (
                None if value[5] is None else float(value[5])
            ),
            "mean_no_speech_probability": (
                None if value[6] is None else float(value[6])
            ),
            "speech_seconds": float(value[7]),
            "segment_count": int(value[8]),
            "target_transcript_access": False,
        }

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_connection"] = None
        return state

    def __del__(self) -> None:
        self.close()


__all__ = [
    "P11_LEXICAL_ASSEMBLER_POLICY",
    "P11_LEXICAL_AUTHORITY_CONTRACT",
    "P11_LEXICAL_CACHE_SCHEMA",
    "P11_LEXICAL_CACHE_VERSION",
    "P11_LEXICAL_CONFIDENCE_CONTRACT",
    "P11_LEXICAL_EVIDENCE_CONTRACT",
    "P11_LEXICAL_TOKEN_CONCAT_POLICY",
    "P11LexicalEvidenceCache",
    "parse_reliable_lexical_authority",
]
