import concurrent.futures
import json
import sys
import threading

import pytest

from scripts.t2a.data import materialize_sceneplan_transfusion_editing_worker as worker


@pytest.mark.parametrize("prefetch", [False, True])
@pytest.mark.parametrize("failed_shard", [None, 1])
@pytest.mark.parametrize("control_switch", [False, True])
def test_render_overlap_preserves_order_and_failed_shards_are_not_published(
    monkeypatch, tmp_path, prefetch, failed_shard, control_switch
):
    index = tmp_path / "pairs.sqlite"
    index.touch()
    argv = ["worker", "--pair-index", str(index), "--gpu", "3",
            "--shard-start", "0", "--shard-stop", "3", "--jobs", "2"]
    if prefetch:
        argv.append("--prefetch-render")
    control = tmp_path / "prefetch.json"
    if control_switch:
        control.write_text(json.dumps({"prefetch_gpus": [3] if prefetch else []}))
        argv.extend(["--prefetch-control", str(control)])
    monkeypatch.setattr(sys, "argv", argv)
    metadata = {"editing_ar_input_contract": worker.EDITING_AR_INPUT_CONTRACT,
                "editing_ar_old_sceneplan_input": "false",
                "target_root": "/mnt/sdb/codex-worker-test-no-files-written"}
    monkeypatch.setattr(worker, "_load_rows", lambda path, shard: (
        metadata,
        [{"split": "train", "pair_id": f"{shard}:{row}", "shard": shard}
         for row in range(2)],
    ))
    monkeypatch.setattr(worker, "sha256_file", lambda path: "a" * 64)
    monkeypatch.setattr(worker, "_validate_physical_gpu", lambda gpu: "test-device")
    monkeypatch.setattr(worker, "_completed", lambda *args: False)
    model_loads = []
    model = object()
    monkeypatch.setattr(worker, "load_vae", lambda device: model_loads.append(device) or model)
    started = [threading.Event() for _ in range(3)]
    second_row_finished = [threading.Event() for _ in range(3)]

    def render(row, *args):
        shard = row["shard"]
        started[shard].set()
        # Deliberately finish each shard's second row first.
        if row["pair_id"].endswith(":0"):
            assert second_row_finished[shard].wait(5)
        else:
            second_row_finished[shard].set()
        return {"pair_id": row["pair_id"],
                "status": "error" if shard == failed_shard else "ok",
                "source_parity_verified": False}

    monkeypatch.setattr(worker, "_render_one", render)
    monkeypatch.setattr(worker.concurrent.futures, "ProcessPoolExecutor",
                        lambda max_workers, mp_context: concurrent.futures.ThreadPoolExecutor(
                            max_workers=max_workers))
    encoded = []

    def encode(rows, rendered, **kwargs):
        shard = rows[0]["shard"]
        assert [r["pair_id"] for r in rendered] == [r["pair_id"] for r in rows]
        assert kwargs["model"] is model
        assert kwargs["batch_size"] == 8
        assert shard != failed_shard
        overlap_expected = (not prefetch) if control_switch and shard > 0 else prefetch
        if shard < 2:
            if overlap_expected:
                assert started[shard + 1].wait(5)
            else:
                assert not started[shard + 1].is_set()
        if shard == 0:
            assert not started[2].is_set(), "at most one following shard may be submitted"
            if control_switch:
                control.write_text(json.dumps({"prefetch_gpus": [] if prefetch else [3]}))
        encoded.append(shard)
        return tmp_path / f"{shard}.parquet"

    monkeypatch.setattr(worker, "_encode", encode)
    published = []
    monkeypatch.setattr(worker, "_atomic_json", lambda path, value: published.append(value))
    if failed_shard is None:
        assert worker.main() == 0
        assert encoded == [0, 1, 2]
    else:
        with pytest.raises(RuntimeError, match="render failures"):
            worker.main()
        assert encoded == [0]
        assert any("failures" in record for record in published)
    assert [record["work_shard"] for record in published if record.get("status") == "ok"] == encoded
    assert model_loads == ["test-device"]
