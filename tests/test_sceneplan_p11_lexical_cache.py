from __future__ import annotations

import sqlite3

from stable_audio_tools.data.sceneplan_p11_lexical_cache import (
    P11_LEXICAL_CACHE_SCHEMA,
    P11_LEXICAL_CACHE_VERSION,
    P11_LEXICAL_CONFIDENCE_CONTRACT,
    P11_LEXICAL_EVIDENCE_CONTRACT,
    P11LexicalEvidenceCache,
)


def test_input_only_frozen_asr_cache_round_trip(tmp_path):
    manifest = tmp_path / "manifest.sqlite"
    index = tmp_path / "index.sqlite"
    manifest.touch()
    index.touch()
    path = tmp_path / "lexical.sqlite"
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE hypotheses (
            ordinal INTEGER PRIMARY KEY,
            text TEXT NOT NULL,
            has_speech INTEGER NOT NULL,
            confidence REAL NOT NULL,
            language TEXT NOT NULL,
            language_probability REAL NOT NULL,
            mean_average_log_probability REAL,
            mean_no_speech_probability REAL,
            speech_seconds REAL NOT NULL,
            segment_count INTEGER NOT NULL
        );
        """
    )
    metadata = {
        "schema": P11_LEXICAL_CACHE_SCHEMA,
        "schema_version": str(P11_LEXICAL_CACHE_VERSION),
        "contract": P11_LEXICAL_EVIDENCE_CONTRACT,
        "source_manifest": str(manifest.resolve()),
        "source_index": str(index.resolve()),
        "encoder_revision": "frozen-asr-test",
        "source": "input_foa_only",
        "target_transcript_access": "forbidden",
        "confidence_contract": P11_LEXICAL_CONFIDENCE_CONTRACT,
    }
    connection.executemany(
        "INSERT INTO metadata VALUES (?,?)", metadata.items()
    )
    connection.executemany(
        "INSERT INTO hypotheses VALUES (?,?,?,?,?,?,?,?,?,?)",
        [
            (1, "hello spatial world", 1, 0.8, "en", 0.99, -0.2, 0.1, 1.2, 1),
            (2, "", 0, 0.0, "en", 0.99, None, None, 0.0, 0),
        ],
    )
    connection.commit()
    connection.close()

    cache = P11LexicalEvidenceCache(
        path,
        source_manifest=manifest,
        source_index=index,
        encoder_revision="frozen-asr-test",
        expected_ordinals=[1, 2],
    )
    assert cache.row(1)["text"] == "hello spatial world"
    assert cache.row(1)["target_transcript_access"] is False
    assert cache.row(2)["has_speech"] is False
    assert cache.row(2)["text"] == ""
