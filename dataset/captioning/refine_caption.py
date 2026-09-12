#!/usr/bin/env python3
"""Refine spatial-audio captions from the pyroom synthesis manifest.

This is the TEXT counterpart of the original image refiner: instead of an image +
long caption, it takes the *structured spatial facts* recorded by
``dataset/synthesis/build_spatial_dataset.py`` (content label per source + azimuth /
elevation / distance / motion + room) and rewrites them into ONE natural English
caption that states BOTH the sound content AND where it is in space (and how it
moves). These captions become the Stage-2 text condition for the 4ch spatial DiT
and a richer label for the VAE corpus.

Input : the synthesis manifest jsonl (rows from build_spatial_dataset.py with
        ``sources`` + ``room`` + ``spatial_caption``).
Output: a captions jsonl of {"id","foa_path","caption"} that
        ``dataset_4ch.load_caption_map`` consumes during pre-encode (keyed by id
        and foa_path). A deterministic template is always written; with an LLM it
        is rewritten into fluent prose.

Model: a TEXT LLM (default Qwen2.5-7B-Instruct). Pass --no-llm to emit only the
deterministic template (no GPU needed).

Env (dedicated venv recommended):
    uv venv /home/tanhe/dataset_storage/.venv-qwen --python 3.10
    source .venv-qwen/bin/activate
    pip install "transformers>=4.44" accelerate torch

Run (single GPU or CPU template-only):
    uv run python dataset/captioning/refine_caption.py \
        --manifest /mnt/sdc/audio_dataset_tmp/spatial_foa/manifest.jsonl \
        --out /mnt/sdc/audio_dataset_tmp/spatial_foa/captions.jsonl \
        --batch_size 16

    uv run python dataset/captioning/refine_caption.py --manifest ... --out ... --no-llm

Multi-GPU (data-parallel shards via spawn):
    uv run python dataset/captioning/refine_caption.py --manifest ... --out ... --num_gpus 8
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

# Qwen3.5 27B Instruct
# 一开始只训练基础的自然语言，只学模板，后面可以分多阶段学习，学习更复杂的自然语言。

QWEN_MODEL_PATH = os.environ.get("CAPTION_LLM", "Qwen/Qwen2.5-7B-Instruct")

SYSTEM_PROMPT = """You are an expert spatial-audio captioner.
You are given structured facts about ONE first-order-ambisonics (FOA) audio clip:
the sound content of each source, its position (direction, elevation, distance),
whether it is static or moving, and the room acoustics.

Write ONE vivid English caption (2-3 sentences, 40-60 words) that describes:
1. WHAT is heard (the sound content of every source, blended naturally).
2. WHERE it is in space: use concrete words - front, behind, left, right,
   front-left, etc.; above / below / at ear level; near / far.
3. HOW it moves, if a source is moving (e.g. "panning from the left to behind").
4. The room character briefly (dry / reverberant) at the end.

Rules:
- Do NOT invent content that is not in the facts. Keep the listed directions.
- A source label may contain its OWN location words (e.g. "from the front",
  "X meters away"): IGNORE those; use ONLY the position facts given below them.
- No markdown, no lists, no quotes. Output ONLY the caption sentence(s).
"""


# ----------------------------------------------------------------- fact sheet

def _source_phrase(s: dict) -> str:
    label = (s.get("label") or s.get("category") or "a sound").strip().rstrip(".")
    st = s.get("start", {})
    where = f"{st.get('dir', 'front')}"
    if st.get("elev") and st["elev"] != "level":
        where += f", {st['elev']}"
    where += f", {st.get('dist_word', 'nearby')}"
    if s.get("motion") == "dynamic" and s.get("move"):
        where = f"moving from {s['move']['from']} to {s['move']['to']}"
    return f'- "{label}"  [{s.get("category","")}]  position: {where}'


def build_user_prompt(row: dict) -> str:
    sources = row.get("sources", [])
    room = row.get("room", {})
    lines = ["Sound sources:"]
    lines += [_source_phrase(s) for s in sources]
    room_desc = room.get("desc") or room.get("reverb", "a room")
    lines.append(f"Room: {room_desc}, {room.get('reverb', '')} (RT60 {room.get('rt60', '?')}s).")
    lines.append(f"Mix: {row.get('mix_type','single')} ({row.get('n_sources',1)} source(s)).")
    if row.get("spatial_caption"):
        lines.append(f"Draft: {row['spatial_caption']}")
    lines.append("Write the final caption now.")
    return "\n".join(lines)


def template_caption(row: dict) -> str:
    return row.get("spatial_caption") or "a sound in a room."


# --------------------------------------------------------------------- LLM

def load_model(device: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"  Loading {QWEN_MODEL_PATH} on {device} ...")
    tok = AutoTokenizer.from_pretrained(QWEN_MODEL_PATH)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        QWEN_MODEL_PATH, torch_dtype=torch.bfloat16,
    ).to(device).eval()
    return model, tok


def refine_batch(model, tok, rows: list[dict], device: str) -> list[str]:
    import torch
    texts = []
    for row in rows:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(row)},
        ]
        texts.append(tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
    inputs = tok(texts, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=96, do_sample=True,
                             temperature=0.7, top_p=0.9, pad_token_id=tok.pad_token_id)
    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, gen)]
    out = tok.batch_decode(trimmed, skip_special_tokens=True)
    return [o.strip().replace("\n", " ") for o in out]


# ------------------------------------------------------------------ pipeline

def load_manifest(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("status", "ok") == "ok" and r.get("foa_path"):
                rows.append(r)
    return rows


def load_done(out_path: Path) -> set[str]:
    done = set()
    if out_path.exists():
        with out_path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["id"])
                except Exception:  # noqa: BLE001
                    continue
    return done


def process_shard(gpu_id: int, rows: list[dict], out_path: Path, batch_size: int,
                  use_llm: bool) -> int:
    device = f"cuda:{gpu_id}" if use_llm else "cpu"
    model = tok = None
    if use_llm:
        model, tok = load_model(device)

    shard_out = out_path if gpu_id == 0 else out_path.with_name(
        f"{out_path.stem}.gpu{gpu_id}{out_path.suffix}")
    done = load_done(shard_out)
    rows = [r for r in rows if r["id"] not in done]
    print(f"[GPU {gpu_id}] {len(rows)} rows to caption -> {shard_out}")

    processed = 0
    with shard_out.open("a", encoding="utf-8") as sink:
        for i in range(0, len(rows), batch_size):
            batch = rows[i:i + batch_size]
            if use_llm:
                try:
                    caps = refine_batch(model, tok, batch, device)
                except Exception as e:  # noqa: BLE001
                    print(f"[GPU {gpu_id}] batch error: {e}; using templates")
                    caps = [template_caption(r) for r in batch]
            else:
                caps = [template_caption(r) for r in batch]
            for r, cap in zip(batch, caps):
                if not cap or len(cap) < 5:
                    cap = template_caption(r)
                sink.write(json.dumps({"id": r["id"], "foa_path": r["foa_path"],
                                       "caption": cap}, ensure_ascii=False) + "\n")
                processed += 1
            sink.flush()
            if (i // batch_size) % 20 == 0:
                print(f"[GPU {gpu_id}] {processed}/{len(rows)}")
    print(f"[GPU {gpu_id}] done, {processed} captions")
    return processed


def _spawn_worker(gpu_id, shards, out_path, batch_size, use_llm):
    process_shard(gpu_id, shards[gpu_id], out_path, batch_size, use_llm)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_gpus", type=int, default=None, help="Default: auto-detect.")
    ap.add_argument("--no-llm", action="store_true", help="Template only (no GPU).")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    rows = load_manifest(args.manifest)
    if args.limit:
        rows = rows[: args.limit]
    print(f"manifest rows: {len(rows)}")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    use_llm = not args.no_llm
    num_gpus = 0
    if use_llm:
        try:
            import torch
            num_gpus = args.num_gpus if args.num_gpus is not None else torch.cuda.device_count()
        except Exception:  # noqa: BLE001
            num_gpus = 0
        if num_gpus == 0:
            print("No GPU available; falling back to --no-llm template mode.")
            use_llm = False

    if not use_llm or num_gpus <= 1:
        process_shard(0, rows, args.out, args.batch_size, use_llm)
    else:
        from huggingface_hub import snapshot_download
        try:
            snapshot_download(QWEN_MODEL_PATH)
        except Exception:  # noqa: BLE001
            pass
        shard_size = (len(rows) + num_gpus - 1) // num_gpus
        shards = [rows[i * shard_size:(i + 1) * shard_size] for i in range(num_gpus)]
        from torch.multiprocessing import spawn
        spawn(_spawn_worker, args=(shards, args.out, args.batch_size, use_llm),
              nprocs=num_gpus, join=True)
        # merge shard files into out
        with args.out.open("a", encoding="utf-8") as sink:
            for g in range(1, num_gpus):
                sp = args.out.with_name(f"{args.out.stem}.gpu{g}{args.out.suffix}")
                if sp.exists():
                    sink.write(sp.read_text(encoding="utf-8"))
                    sp.unlink()
    print(f"Done -> {args.out}")


if __name__ == "__main__":
    start = time.time()
    main()
    print(f"Total time: {time.time() - start:.1f}s")
