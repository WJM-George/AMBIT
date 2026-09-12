from __future__ import annotations

from pathlib import Path

from scripts.t2a.data.materialize_sceneplan_v2_shard import (
    remove_interrupted_render_temporaries,
)


def test_resume_cleanup_removes_only_known_atomic_write_remnants(tmp_path: Path) -> None:
    stale_foa = tmp_path / ".foa_WYZX_SN3D.deadbeef.flac"
    stale_stem = tmp_path / ".stem_source_2_WYZX_SN3D.deadbeef.flac"
    stale_result = tmp_path / "render_result.json.tmp.1234"
    canonical_foa = tmp_path / "foa_WYZX_SN3D.flac"
    canonical_result = tmp_path / "render_result.json"
    unrelated = tmp_path / "notes.txt"
    for path in (
        stale_foa,
        stale_stem,
        stale_result,
        canonical_foa,
        canonical_result,
        unrelated,
    ):
        path.write_bytes(b"test")

    assert remove_interrupted_render_temporaries(tmp_path) == 3
    assert not stale_foa.exists()
    assert not stale_stem.exists()
    assert not stale_result.exists()
    assert canonical_foa.is_file()
    assert canonical_result.is_file()
    assert unrelated.is_file()
