from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from scripts.t2a.data.materialize_model_sceneplan_v1_shard import public_render_result
from scripts.t2a.data.render_tts_v2_pilot import write_pcm24


def test_pcm24_flac_bytes_do_not_depend_on_storage_root(tmp_path: Path) -> None:
    rng = np.random.default_rng(20260818)
    audio = rng.uniform(-0.2, 0.2, size=(4, 44_100)).astype(np.float32)
    left = tmp_path / "disk" / "foa_WYZX_SN3D.flac"
    right = Path("/dev/shm") / f"sceneplan-flac-parity-{left.parent.name}.flac"
    try:
        write_pcm24(left, audio)
        write_pcm24(right, audio)
        assert hashlib.sha256(left.read_bytes()).digest() == hashlib.sha256(
            right.read_bytes()
        ).digest()
    finally:
        right.unlink(missing_ok=True)


def test_transient_tmpfs_path_is_not_exposed_in_the_manifest_result() -> None:
    result = {
        "sample_id": "sample",
        "foa_path": "/dev/shm/p8/train/work-00012/sample/foa_WYZX_SN3D.flac",
        "foa_sha256": "a" * 64,
    }
    public = public_render_result(
        result,
        cleanup_foa=True,
        logical_render_root=Path("/mnt/sdb/dataset/materialized/renders"),
        split="train",
        shard=12,
        sample_id="sample",
    )
    assert public["foa_path"] == (
        "/mnt/sdb/dataset/materialized/renders/train/work-00012/"
        "sample/foa_WYZX_SN3D.flac"
    )
    assert result["foa_path"].startswith("/dev/shm/")
