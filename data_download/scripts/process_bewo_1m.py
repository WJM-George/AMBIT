#!/usr/bin/env python3
"""Extract BEWO-1M from snapshot chunks into a training-ready layout under bewo_1m/.

Keeps ``snapshot/`` as the immutable HF download. Writes processed assets next to it:

  ${AMBIT_DATA_ROOT}/datasets/bewo_1m/
    snapshot/              # raw chunks + small tars (do not delete)
    extracted/             # merged .tar.gz + untar'd audio trees
      BEWO_SS_Audio_v1/
      BEWO_DS_Audio_v1/
      ...
    annotations/           # jsonl from *Annotation* tars
    manifests/             # unified jsonl for dataset_4ch (id, path, caption, spatial_format)

BEWO audio is **binaural stereo** (2ch). Our 4ch VAE uses layout [L,R,0,0] with spatial_format=binaural.
Captions are already in the official jsonl (no Qwen re-caption needed).

Examples
--------
# 1) Extract annotations only (fast, ~500MB):
python3 scripts/process_bewo_1m.py --steps annotations

# 2) Merge+extract ONE subset (SS ~130GB compressed chunks):
python3 scripts/process_bewo_1m.py --subset ss --steps merge,extract

# 3) Build train manifest for extracted SS audio:
python3 scripts/process_bewo_1m.py --subset ss --steps manifest --split train

# 4) Full pipeline for SS (merge -> extract -> manifest):
python3 scripts/process_bewo_1m.py --subset ss --steps all --split train
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
from pathlib import Path

DEFAULT_ROOT = Path(os.environ.get("AUDIO_DATASET_SECONDARY_ROOT", os.environ.get("AMBIT_DATA_ROOT", "data")))
DEFAULT_BEWO = DEFAULT_ROOT / "datasets" / "bewo_1m"
DEFAULT_SNAPSHOT = DEFAULT_BEWO / "snapshot"

# Audio subset key -> (chunk glob prefix, inner tar folder name, annotation tar member folder)
SUBSETS = {
    "ss": {
        "chunks": "BEWO_SS_Audio_v1_chunk_*",
        "merged": "BEWO_SS_Audio_v1.tar.gz",
        "audio_dir": "BEWO_SS_Audio_v1",
        "ann_tar": "BEWO_SS_Annotation_v1.tar.gz",
        "ann_dir": "BEWO_SS_Annotation_v1",
        "splits": {
            "train": "audiocaps_single_train.jsonl",
            "val": "audiocaps_single_val.jsonl",
            "test": "audiocaps_single_test.jsonl",
            "full": "full_single.jsonl",
        },
    },
    "sd": {
        "chunks": "BEWO_SD_Audio_v1_chunk_*",
        "merged": "BEWO_SD_Audio_v1.tar.gz",
        "audio_dir": "BEWO_SD_Audio_v1",
        "ann_tar": "BEWO_SD_Annotation_v1.tar.gz",
        "ann_dir": "BEWO_SD_Annotation_v1",
        "splits": {
            "train": "audiocaps_move_train.jsonl",
            "val": "audiocaps_move_val.jsonl",
            "test": "audiocaps_move_test.jsonl",
            "full": "full_move.jsonl",
        },
    },
    "ds": {
        "chunks": "BEWO_DS_Audio_v1_chunk_*",
        "merged": "BEWO_DS_Audio_v1.tar.gz",
        "audio_dir": "BEWO_DS_Audio_v1",
        "ann_tar": "BEWO_DS_Annotation_v1.tar.gz",
        "ann_dir": "BEWO_DS_Annotation_v1",
        "splits": {
            "train": "audiocaps_double_train.jsonl",
            "val": "audiocaps_double_val.jsonl",
            "test": "audiocaps_double_test.jsonl",
            "full": "full_double.jsonl",
        },
    },
    "mix": {
        "chunks": "BEWO_Mix_Audio_v1_chunk_*",
        "merged": "BEWO_Mix_Audio_v1.tar.gz",
        "audio_dir": "BEWO_Mix_Audio_v1",
        "ann_tar": "BEWO_Mix_Annotation_v1.tar.gz",
        "ann_dir": "BEWO_Mix_Annotation_v1",
        "splits": {
            "train": "audiocaps_train_mix.jsonl",
            "val": "audiocaps_val_mix.jsonl",
            "test": "audiocaps_mix_test.jsonl",
            "full": "full_mix.jsonl",
        },
    },
}


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def merge_chunks(snapshot: Path, out_tar: Path, chunk_glob: str) -> None:
    chunks = sorted(snapshot.glob(chunk_glob))
    if not chunks:
        raise FileNotFoundError(f"No chunks matching {chunk_glob} under {snapshot}")
    out_tar.parent.mkdir(parents=True, exist_ok=True)
    if out_tar.exists() and out_tar.stat().st_size > 0:
        print(f"[bewo] merged tar exists, skip: {out_tar}")
        return
    print(f"[bewo] merging {len(chunks)} chunks -> {out_tar}")
    with open(out_tar, "wb") as out:
        for ch in chunks:
            print(f"  cat {ch.name}")
            with open(ch, "rb") as inp:
                while True:
                    buf = inp.read(1024 * 1024)
                    if not buf:
                        break
                    out.write(buf)


def extract_tar(tar_path: Path, dest: Path, inner_dir: str) -> None:
    marker = dest / inner_dir / ".extracted_ok"
    if marker.exists():
        print(f"[bewo] already extracted: {dest / inner_dir}")
        return
    dest.mkdir(parents=True, exist_ok=True)
    run(["tar", "-xzf", str(tar_path), "-C", str(dest)])
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("ok\n")


def extract_annotations(snapshot: Path, ann_root: Path) -> None:
    ann_root.mkdir(parents=True, exist_ok=True)
    for tar_name in (
        "BEWO_SS_Annotation_v1.tar.gz",
        "BEWO_SD_Annotation_v1.tar.gz",
        "BEWO_DS_Annotation_v1.tar.gz",
        "BEWO_Mix_Annotation_v1.tar.gz",
        "BEWO_RW_Annotation_v1.tar.gz",
    ):
        tar_path = snapshot / tar_name
        if not tar_path.exists():
            continue
        sub = ann_root / tar_name.replace(".tar.gz", "")
        if (sub / ".extracted_ok").exists():
            print(f"[bewo] annotations skip {sub}")
            continue
        run(["tar", "-xzf", str(tar_path), "-C", str(ann_root)])
        (sub / ".extracted_ok").write_text("ok\n")


def index_wav_by_stem(audio_root: Path) -> dict[str, Path]:
    """Map filename stem -> absolute wav path."""
    idx: dict[str, Path] = {}
    for wav in audio_root.rglob("*.wav"):
        stem = wav.stem
        idx[stem] = wav.resolve()
    return idx


def build_manifest(
    subset_key: str,
    split: str,
    bewo_root: Path,
    ann_root: Path,
    audio_extracted: Path,
) -> Path:
    cfg = SUBSETS[subset_key]
    ann_jsonl = ann_root / cfg["ann_dir"] / cfg["splits"][split]
    if not ann_jsonl.exists():
        raise FileNotFoundError(f"Annotation file missing: {ann_jsonl}")

    audio_tree = audio_extracted / cfg["audio_dir"]
    if not audio_tree.exists():
        raise FileNotFoundError(
            f"Audio tree missing: {audio_tree}. Run --steps merge,extract first."
        )

    wav_index = index_wav_by_stem(audio_tree)
    out_path = bewo_root / "manifests" / f"bewo_{subset_key}_{split}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_ok = n_miss = 0
    with open(ann_jsonl, "r", encoding="utf-8") as src, open(out_path, "w", encoding="utf-8") as sink:
        for line in src:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            aid = row.get("audio_name") or row.get("id")
            if not aid:
                continue
            path = wav_index.get(str(aid))
            if path is None:
                n_miss += 1
                continue
            cap = (row.get("caption") or row.get("text") or "").strip()
            rec = {
                "id": str(aid),
                "caption": cap,
                "text": cap,
                "prompt": cap,
                "path": str(path),
                "spatial_format": "binaural",
                "subset": subset_key,
                "split": split,
                "meta": row.get("meta"),
            }
            sink.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_ok += 1

    print(f"[bewo] manifest {out_path}: {n_ok} rows, {n_miss} missing audio")
    return out_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bewo-root", type=Path, default=DEFAULT_BEWO)
    p.add_argument("--snapshot", type=Path, default=None)
    p.add_argument("--subset", choices=list(SUBSETS.keys()) + ["all"], default="ss",
                   help="Audio subset: ss=single static, sd=move, ds=double, mix=mixed")
    p.add_argument("--split", choices=["train", "val", "test", "full"], default="train")
    p.add_argument("--steps", default="all",
                   help="Comma-separated: annotations, merge, extract, manifest, all")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    bewo_root = args.bewo_root
    snapshot = args.snapshot or (bewo_root / "snapshot")
    extracted = bewo_root / "extracted"
    annotations = bewo_root / "annotations"
    steps = {s.strip() for s in args.steps.split(",")}
    if "all" in steps:
        steps = {"annotations", "merge", "extract", "manifest"}

    if "annotations" in steps:
        extract_annotations(snapshot, annotations)

    subsets = list(SUBSETS.keys()) if args.subset == "all" else [args.subset]
    for key in subsets:
        cfg = SUBSETS[key]
        merged_tar = extracted / cfg["merged"]
        audio_dest = extracted

        if "merge" in steps:
            merge_chunks(snapshot, merged_tar, cfg["chunks"])
        if "extract" in steps:
            if not merged_tar.exists():
                sys.exit(f"Missing merged tar {merged_tar}; run --steps merge first.")
            extract_tar(merged_tar, audio_dest, cfg["audio_dir"])
        if "manifest" in steps:
            build_manifest(key, args.split, bewo_root, annotations, audio_dest)

    print(f"[bewo] done. Root layout: {bewo_root}")
    print("  snapshot/     raw HF download")
    print("  extracted/    audio trees")
    print("  annotations/  official jsonl")
    print("  manifests/    dataset_4ch-ready jsonl")


if __name__ == "__main__":
    main()
