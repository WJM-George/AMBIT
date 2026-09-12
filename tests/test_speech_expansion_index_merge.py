from __future__ import annotations

from pathlib import Path
import sqlite3
import zlib

from scripts.t2a.data.build_model_sceneplan_training_index_v1 import create_schema
from scripts.t2a.data.build_speech_expansion_training_index_v6 import merge_split


def _source_index(path: Path, sample_id: str, latent_path: str) -> None:
    connection = sqlite3.connect(path)
    create_schema(connection)
    connection.execute(
        "INSERT INTO latent_shards(id,path,sha256) VALUES (1,?,?)",
        (latent_path, "a" * 64),
    )
    payload = sqlite3.Binary(zlib.compress(b"{}"))
    connection.execute(
        """
        INSERT INTO samples(
            ordinal,sample_id,model_num_samples,latent_frames_valid,
            renderer_caption_zlib,scene_plan_zlib,model_sceneplan_sha256,
            model_sceneplan_path,model_sceneplan_row,
            model_sceneplan_byte_offset,model_sceneplan_byte_length,
            compiled_conditioning_ref,latent_shard_id,latent_key,
            latent_tensor_sha256,renderer_caption_sha256
        ) VALUES (0,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            sample_id,
            44_100,
            44,
            payload,
            payload,
            "b" * 64,
            f"/{sample_id}.jsonl",
            0,
            0,
            3,
            f"/{sample_id}.conditioning#0:3",
            1,
            sample_id,
            "c" * 64,
            "d" * 64,
        ),
    )
    metadata = {
        "schema": "stable_audio_tools.sceneplan_v2_training_index",
        "schema_version": "3",
        "contract_revision": "5",
        "split": "validation",
        "rows": "1",
    }
    connection.executemany(
        "INSERT INTO metadata(key,value) VALUES (?,?)", metadata.items()
    )
    connection.commit()
    connection.close()


def test_revision6_merge_rebases_ordinals_and_deduplicates_latent_shards(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.sqlite"
    second = tmp_path / "second.sqlite"
    shared_latent = "/mnt/sdb/audio_dataset/shared.safetensors"
    _source_index(first, "base_validation", shared_latent)
    _source_index(second, "delta_validation", shared_latent)

    summary = merge_split(
        split="validation",
        sources=[first, second],
        output_root=tmp_path / "merged",
        expected_rows=2,
    )
    assert summary["rows"] == 2
    assert summary["latent_shards"] == 1

    connection = sqlite3.connect(summary["path"])
    try:
        assert connection.execute(
            "SELECT ordinal,sample_id FROM samples ORDER BY ordinal"
        ).fetchall() == [(0, "base_validation"), (1, "delta_validation")]
        metadata = dict(connection.execute("SELECT key,value FROM metadata"))
        assert metadata["contract_revision"] == "6"
        assert metadata["model_sceneplan_schema_version"] == "2"
        assert metadata["caption_compiler_version"] == "5"
        assert metadata["max_latent_frames"] == "648"
    finally:
        connection.close()

