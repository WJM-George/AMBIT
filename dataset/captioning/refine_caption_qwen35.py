#!/usr/bin/env python3
"""Refine spatial-audio captions from the pyroom synthesis manifest.

This is the TEXT counterpart of the original image refiner: instead of an image +
long caption, it takes the *structured spatial facts* recorded by
``dataset/synthesis/build_spatial_dataset.py`` (content label per source + azimuth /
elevation / distance / motion + room) and rewrites them into ONE natural English
caption that states BOTH the sound content AND where it is in space (and how it
moves). These captions become the Stage-2 text condition for the 4ch spatial DiT
and a richer label for the VAE corpus.

**Curriculum stages**

- **Default (no ``--stage``):** manifest ``spatial_caption`` is the draft; Qwen
  rewrites structured facts + draft into fluent prose.
- **Stage 7 (``--stage 7``):** Stage 6 template captions from
  ``captions_stage6.jsonl`` (``--draft-captions``) plus manifest structured facts
  are passed to Qwen for the final natural-language caption
  (``captions_stage7.jsonl``). Real stage-7 output comes from this script, not
  from ``apply_spatial_templates.py`` (that stage-7 jsonl is a template placeholder).

Input : synthesis manifest jsonl (``sources`` + ``room`` + ``spatial_caption``).
        Optional ``--draft-captions`` jsonl with stage-6 rows (``id``, ``caption``,
        optional ``draft``).
Output: captions jsonl keyed by ``id`` / ``foa_path`` for ``dataset_4ch.load_caption_map``.
        Stage 7 rows include ``draft_stage6``, ``stage``, and ``model`` for traceability.

Model: Qwen3.5-27B text-only (``Qwen3_5ForCausalLM``, skips the vision encoder).
Pass ``--no-llm`` or ``--template-only`` to emit drafts without GPU.

Qwen3.5 thinks by default; this script disables thinking (instruct mode) and uses
the recommended sampling params for short caption generation.

Env (``.venv-qwen`` is already OK: Py3.10 + transformers 5.10 dev + Qwen3_5ForCausalLM):
    source .venv/bin/activate

First run — download weights (~56 GB, once) to ${AMBIT_CKPT_ROOT} (not ~/.cache):
    hf download Qwen/Qwen3.5-27B --local-dir ${AMBIT_CKPT_ROOT}/Qwen/Qwen3.5-27B

Preflight (no GPU load):
    python dataset/captioning/refine_caption_qwen35.py --preflight

Run (pick *idle* GPUs via CUDA_VISIBLE_DEVICES; 27B bf16 needs ~2×48GB):
    cd ./stable-audio-tools
    CUDA_VISIBLE_DEVICES=6,7 uv run python dataset/captioning/refine_caption_qwen35.py \\
        --manifest ${AMBIT_CACHE_ROOT}/spatial_foa/manifest.jsonl \\
        --out ${AMBIT_CACHE_ROOT}/spatial_foa/captions_qwen35.jsonl \\
        --batch_size 4

    uv run python dataset/captioning/refine_caption_qwen35.py --manifest ... --out ... --no-llm

Stage 6 → Stage 7 (when GPU free):
    # CUDA_VISIBLE_DEVICES=6,7 uv run python dataset/captioning/refine_caption_qwen35.py \\
    #     --manifest ${AMBIT_DATA_ROOT}/spatial_foa/manifest.jsonl \\
    #     --draft-captions ${AMBIT_DATA_ROOT}/spatial_foa/captions_stage6.jsonl \\
    #     --stage 7 \\
    #     --out ${AMBIT_DATA_ROOT}/spatial_foa/captions_stage7.jsonl

    # Template-only smoke test (no GPU):
    # uv run python dataset/captioning/refine_caption_qwen35.py \\
    #     --manifest ${AMBIT_DATA_ROOT}/spatial_foa/manifest.jsonl \\
    #     --draft-captions ${AMBIT_DATA_ROOT}/spatial_foa/captions_stage6.jsonl \\
    #     --stage 7 --template-only

Alternative (vLLM, higher throughput; run on idle GPUs):
    vllm serve Qwen/Qwen3.5-27B --port 8000 --tensor-parallel-size 2 \\
        --max-model-len 8192 --reasoning-parser qwen3 --language-model-only
    # then point a small OpenAI-client wrapper at http://localhost:8000/v1
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path

# Qwen3.5-27B — text-only caption refinement.
# Stage-1: natural template-following captions; later stages can use richer prompts.
QWEN_MODEL_PATH = os.environ.get("CAPTION_LLM", os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/Qwen/Qwen3.5-27B")

# Qwen3.5 instruct (non-thinking) sampling for general tasks.
GEN_TEMPERATURE = 0.7
GEN_TOP_P = 0.8
GEN_TOP_K = 20
GEN_MAX_NEW_TOKENS = 128

_THINKING_BLOCK_RE = re.compile(
    r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE,
)
_FINAL_LABEL_RE = re.compile(
    r"(?:^|\n|\s)(?:final\s+caption|caption)\s*:\s*",
    re.IGNORECASE,
)
_PROMPT_LEAK_RE = re.compile(
    r"thinking process|analy[sz]e the request|structured manifest|"
    r"structured source facts|source of truth|write one concise|"
    r"no markdown|bullet list|authoritative manifest metadata|"
    r"expert spatial-audio captioner|\*\*role:\*\*|\*\*task:\*\*|"
    r"\*\*constraints:\*\*",
    re.IGNORECASE,
)

_DYNAMIC_TEMPLATE_DIR = Path(__file__).resolve().parent / "dynamic_spatial_corpus"
if str(_DYNAMIC_TEMPLATE_DIR) not in sys.path:
    sys.path.insert(0, str(_DYNAMIC_TEMPLATE_DIR))
try:
    from spatial_template_corpus import render_stage_caption
except Exception:  # noqa: BLE001
    render_stage_caption = None

SYSTEM_PROMPT = """You are an expert spatial-audio captioner for a constructed
first-order-ambisonics (FOA) pyroom dataset.

You receive authoritative manifest metadata for ONE synthetic FOA audio clip:
source content labels, source count and mix type, direction, elevation, distance,
static or dynamic motion paths, and room / reverberation facts. Write ONE concise,
natural English caption suitable for training a spatial audio model directly from
Qwen3.5-refined manifest facts, without relying on a staged curriculum.

Describe what is heard and where it is in space using listener-centered language:
front / behind / left / right / front-left, above / below / ear level, near / far,
fixed positions for static sources, motion paths for dynamic sources, and a brief
natural phrase for the room or reverb character.

Rules:
- The structured manifest facts are the source of truth. Do NOT hallucinate,
  embellish, infer unseen causes, or add sources, events, emotions, instruments,
  distances, directions, motion, or room details not present in the facts.
- If a source label contains embedded spatial words such as "from the front",
  "behind", or "X meters away", ignore those words and use only the structured
  direction, elevation, distance, and motion fields.
- Cover every listed source and preserve the stated mix relationships.
- Output only the final caption: fluent English, 2-3 sentences, about 40-70 words,
  no markdown, no bullet list, no quotation marks.
"""

STAGE7_SYSTEM_PROMPT = """You are an expert spatial-audio captioner writing the
final curriculum caption for Stage 7, after Stages 1-6 have already introduced
content, direction, elevation, distance, room / reverb, motion, and multi-source
phrasing.

You receive two inputs: (1) authoritative structured manifest facts from the
constructed FOA pyroom dataset, and (2) a Stage 6 template draft that already
encodes the curriculum semantics. Rewrite the Stage 6 draft into the final Stage 7
caption while preserving all semantics from Stage 6 and resolving any ambiguity
with the structured facts.

Preserve every source's content, listener-centered direction, elevation, distance,
static or dynamic status, dynamic path if present, fixed position if static, room /
reverb character, mix type, source count, and multi-source relationships. For
dynamic sources, describe the path naturally; for static sources, describe the
fixed position naturally. Mention the room or reverberation as a natural phrase or
sentence, not as a metadata dump.

Rules:
- Do NOT invent, drop, merge, or reorder facts in a way that changes the scene.
- If a source label contains embedded spatial words, ignore them and trust the
  structured direction, elevation, distance, and motion facts.
- Keep FOA / spatial language precise, but avoid technical jargon in the final
  caption unless it helps naturalness.
- Output only the final caption: fluent English, 2-3 sentences, about 40-70 words.
  If there are many sources, slightly longer is acceptable, but stay concise. No
  markdown, no bullet list, no quotation marks.
"""


# ----------------------------------------------------------------- fact sheet

def _placement_phrase(p: dict) -> str:
    where = f"{p.get('dir', 'front')}"
    if p.get("elev") and p["elev"] != "level":
        where += f", {p['elev']}"
    else:
        where += ", at ear level"
    where += f", {p.get('dist_word', 'nearby')}"
    return where


def _source_phrase(s: dict) -> str:
    label = (s.get("label") or s.get("category") or "a sound").strip().rstrip(".")
    st = s.get("start", {})
    category = s.get("category", "")
    if s.get("motion") == "dynamic":
        end = s.get("end") or {}
        move = s.get("move") or {}
        move_from = move.get("from") or st.get("dir", "front")
        move_to = move.get("to") or end.get("dir", "front")
        return (
            f'- "{label}"  [{category}]  motion: dynamic; path: '
            f'{_placement_phrase(st)} -> {_placement_phrase(end)} '
            f'({move_from} to {move_to})'
        )
    return (
        f'- "{label}"  [{category}]  motion: static; fixed position: '
        f'{_placement_phrase(st)}'
    )


def resolve_draft_caption(row: dict, draft_map: dict[str, str] | None) -> str | None:
    """Return the draft caption for a row: stage-6 jsonl entry, else manifest spatial_caption."""
    rid = row.get("id")
    if draft_map and rid and rid in draft_map:
        return draft_map[rid]
    sc = row.get("spatial_caption")
    return sc.strip() if sc else None


def build_user_prompt(row: dict, draft_caption: str | None = None,
                      *, stage: int | None = None) -> str:
    sources = row.get("sources", [])
    room = row.get("room", {})
    lines = ["Structured source facts:"]
    lines += [_source_phrase(s) for s in sources]
    room_desc = room.get("desc") or room.get("reverb", "a room")
    lines.append(
        f"Room facts: {room_desc}, {room.get('reverb', '')} "
        f"(RT60 {room.get('rt60', '?')}s)."
    )
    lines.append(
        f"Mix facts: {row.get('mix_type','single')} mix with "
        f"{row.get('n_sources',1)} source(s)."
    )
    draft = draft_caption or row.get("spatial_caption")
    if draft:
        if stage == 7:
            lines.append(f"Draft (Stage 6 template): {draft}")
            lines.append(
                "Rewrite this as the final Stage 7 caption. Preserve every structured "
                "fact and every Stage 6 semantic detail: content, direction, elevation, "
                "distance, static positions, dynamic paths, room/reverb, mix type, "
                "source count, and multi-source relationships."
            )
        else:
            lines.append(f"Draft (manifest/template caption): {draft}")
            lines.append(
                "Use the draft only as a wording aid; the structured manifest facts "
                "above are authoritative for the final direct Qwen3.5 caption."
            )
    lines.append("Write the final caption now.")
    return "\n".join(lines)


def template_caption(row: dict, draft_caption: str | None = None) -> str:
    if render_stage_caption is not None:
        try:
            rng = random.Random(hash(row.get("id", "")) & 0xFFFFFFFF)
            caption, _ = render_stage_caption(7, row, rng=rng)
            return caption.strip()
        except Exception:  # noqa: BLE001
            pass
    return (draft_caption or row.get("spatial_caption") or "a sound in a room.").strip()


def _strip_thinking(text: str) -> str:
    text = _THINKING_BLOCK_RE.sub("", text).strip()
    if text.lower().startswith("<think>"):
        end = text.lower().find("</think>")
        if end != -1:
            text = text[end + len("</think>"):].strip()
    matches = list(_FINAL_LABEL_RE.finditer(text))
    if matches:
        text = text[matches[-1].end():].strip()
    return text


def _is_bad_caption(text: str) -> bool:
    text = (text or "").strip()
    if len(text) < 8:
        return True
    if _PROMPT_LEAK_RE.search(text):
        return True
    if len(text.split()) > 120:
        return True
    return False


def _model_device(model) -> str:
    return str(next(model.parameters()).device)


# --------------------------------------------------------------------- LLM

def load_model(device_map: str):
    import torch
    from transformers import AutoTokenizer, Qwen3_5ForCausalLM

    print(f"  Loading {QWEN_MODEL_PATH} (text-only, device_map={device_map}) ...")
    tok = AutoTokenizer.from_pretrained(QWEN_MODEL_PATH)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = Qwen3_5ForCausalLM.from_pretrained(
        QWEN_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
    ).eval()
    print(f"  Model ready on {_model_device(model)}")
    return model, tok


def refine_batch(
    model,
    tok,
    rows: list[dict],
    *,
    draft_map: dict[str, str] | None = None,
    system_prompt: str = SYSTEM_PROMPT,
    stage: int | None = None,
) -> list[str]:
    import torch

    texts = []
    for row in rows:
        draft = resolve_draft_caption(row, draft_map)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": build_user_prompt(row, draft, stage=stage)},
        ]
        try:
            rendered = tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            rendered = tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                chat_template_kwargs={"enable_thinking": False},
            )
        texts.append(rendered)
    device = _model_device(model)
    inputs = tok(texts, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        gen = model.generate(
            **inputs,
            max_new_tokens=GEN_MAX_NEW_TOKENS,
            do_sample=True,
            temperature=GEN_TEMPERATURE,
            top_p=GEN_TOP_P,
            top_k=GEN_TOP_K,
            pad_token_id=tok.pad_token_id,
        )
    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, gen)]
    out = tok.batch_decode(trimmed, skip_special_tokens=True)
    return [_strip_thinking(o).replace("\n", " ").strip() for o in out]


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


def load_draft_captions(path: Path) -> dict[str, str]:
    """Load ``id -> caption`` from a stage-6 (or other draft) captions jsonl."""
    drafts: dict[str, str] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            rid = row.get("id")
            if not rid:
                continue
            cap = (row.get("draft") or row.get("caption") or "").strip()
            if cap:
                drafts[rid] = cap
    return drafts


def default_output_path(manifest: Path, stage: int | None) -> Path:
    if stage == 7:
        return manifest.parent / "captions_stage7.jsonl"
    return manifest.parent / "captions_qwen35.jsonl"


def _output_record(
    row: dict,
    caption: str,
    *,
    stage: int | None,
    draft_caption: str | None,
    model_name: str | None,
) -> dict:
    rec: dict = {"id": row["id"], "foa_path": row["foa_path"], "caption": caption}
    if stage == 7:
        rec["draft_stage6"] = draft_caption or row.get("spatial_caption") or ""
        rec["stage"] = 7
        rec["model"] = model_name or QWEN_MODEL_PATH
    return rec


def process_rows(
    rows: list[dict],
    out_path: Path,
    batch_size: int,
    use_llm: bool,
    device_map: str,
    *,
    draft_map: dict[str, str] | None = None,
    system_prompt: str = SYSTEM_PROMPT,
    stage: int | None = None,
    model_name: str | None = None,
) -> int:
    model = tok = None
    if use_llm:
        model, tok = load_model(device_map)

    done = load_done(out_path)
    rows = [r for r in rows if r["id"] not in done]
    stage_label = f" (stage {stage})" if stage else ""
    print(f"{len(rows)} rows to caption{stage_label} -> {out_path}")
    if draft_map:
        missing = sum(1 for r in rows if r["id"] not in draft_map)
        if missing:
            print(f"  warning: {missing}/{len(rows)} manifest ids missing from --draft-captions")

    processed = 0
    with out_path.open("a", encoding="utf-8") as sink:
        for i in range(0, len(rows), batch_size):
            batch = rows[i:i + batch_size]
            if use_llm:
                try:
                    caps = refine_batch(
                        model, tok, batch,
                        draft_map=draft_map,
                        system_prompt=system_prompt,
                        stage=stage,
                    )
                except Exception as e:  # noqa: BLE001
                    print(f"batch error: {e}; using templates")
                    caps = [
                        template_caption(r, resolve_draft_caption(r, draft_map))
                        for r in batch
                    ]
            else:
                caps = [
                    template_caption(r, resolve_draft_caption(r, draft_map))
                    for r in batch
                ]
            for r, cap in zip(batch, caps):
                draft = resolve_draft_caption(r, draft_map)
                if _is_bad_caption(cap):
                    print(f"bad caption for {r.get('id')}; using stage-7 template fallback")
                    cap = template_caption(r, draft)
                rec = _output_record(
                    r, cap, stage=stage, draft_caption=draft, model_name=model_name,
                )
                sink.write(json.dumps(rec, ensure_ascii=False) + "\n")
                processed += 1
            sink.flush()
            if (i // batch_size) % 20 == 0:
                print(f"{processed}/{len(rows)}")
    print(f"done, {processed} captions")
    return processed


def preflight() -> int:
    """Check venv imports, hub cache, and GPU headroom without loading the model."""
    ok = True
    try:
        import torch
        from transformers import Qwen3_5ForCausalLM  # noqa: F401
        print(f"torch {torch.__version__}, cuda={torch.cuda.is_available()}, "
              f"gpus={torch.cuda.device_count() if torch.cuda.is_available() else 0}")
    except Exception as e:  # noqa: BLE001
        print(f"import check FAILED: {e}")
        return 1

    try:
        import json
        from pathlib import Path as P
        from huggingface_hub import try_to_load_from_cache

        def _shard_report(weights_dir: P) -> tuple[int, int, float]:
            idx_path = weights_dir / "model.safetensors.index.json"
            meta = json.loads(idx_path.read_text(encoding="utf-8"))
            need = sorted(set(meta["weight_map"].values()))
            have = sum(1 for s in need if (weights_dir / s).is_file())
            size_gb = meta["metadata"]["total_size"] / 1e9
            return have, len(need), size_gb

        hub_id = "Qwen/Qwen3.5-27B"
        local_default = P(os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/Qwen/Qwen3.5-27B")
        model_p = P(QWEN_MODEL_PATH)
        if model_p.is_dir():
            if not (model_p / "model.safetensors.index.json").is_file():
                print(f"weights: missing at {model_p} — run: hf download {hub_id} --local-dir {model_p}")
                ok = False
            else:
                have, n_need, size_gb = _shard_report(model_p)
                print(f"weights ({model_p}): {have}/{n_need} shards present (~{size_gb:.0f} GB total)")
                if have < n_need:
                    print(f"  incomplete — run: hf download {hub_id} --local-dir {model_p}")
                    ok = False
        else:
            idx = try_to_load_from_cache(QWEN_MODEL_PATH, "model.safetensors.index.json")
            if not idx:
                print(f"weights: NOT cached — run: hf download {hub_id} --local-dir {local_default}")
                ok = False
            else:
                have, n_need, size_gb = _shard_report(P(idx).parent)
                print(f"weights (hub cache): {have}/{n_need} shards present (~{size_gb:.0f} GB total)")
                if have < n_need:
                    print(f"  incomplete — run: hf download {hub_id} --local-dir {local_default}")
                    ok = False
    except Exception as e:  # noqa: BLE001
        print(f"weights check skipped: {e}")

    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.free,memory.total,utilization.gpu",
             "--format=csv,noheader"],
            text=True,
        )
        free_gpus = []
        for line in out.strip().splitlines():
            idx, free, total, util = [x.strip() for x in line.split(",")]
            free_mib = int(free.split()[0])
            print(f"  GPU {idx}: {free} free / {total}, util {util}")
            if free_mib >= 40000 and util.split()[0] == "0":
                free_gpus.append(idx)
        if free_gpus:
            print(f"suggested CUDA_VISIBLE_DEVICES={','.join(free_gpus[:2])}")
        else:
            print("no fully idle 49GB GPUs right now — wait or use --no-llm")
            ok = False
    except Exception as e:  # noqa: BLE001
        print(f"nvidia-smi skipped: {e}")

    return 0 if ok else 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preflight", action="store_true",
                    help="Check imports, weight cache, GPU headroom; exit without running.")
    ap.add_argument("--manifest", type=Path, required=False)
    ap.add_argument("--out", type=Path, required=False,
                    help="Output jsonl. Default: captions_stage7.jsonl when --stage 7, else "
                         "captions_qwen35.jsonl beside manifest.")
    ap.add_argument("--draft-captions", type=Path, default=None,
                    help="Draft captions jsonl (e.g. captions_stage6.jsonl: id, caption, draft). "
                         "Without this, manifest spatial_caption is used as the draft.")
    stage_grp = ap.add_mutually_exclusive_group()
    stage_grp.add_argument("--stage", type=int, default=None, choices=[7],
                           help="Curriculum stage mode (7 = Qwen refine of stage-6 drafts).")
    stage_grp.add_argument("--curriculum-stage", type=int, default=None, choices=[7],
                           dest="stage", help="Alias for --stage.")
    ap.add_argument("--batch_size", type=int, default=4,
                    help="Keep small for 27B (default: 4).")
    ap.add_argument("--device-map", type=str, default="auto",
                    help="HF device_map for model parallelism (default: auto).")
    ap.add_argument("--no-llm", action="store_true", help="Template only (no GPU).")
    ap.add_argument("--template-only", action="store_true",
                    help="Copy draft captions to output without LLM (stage-6 or spatial_caption).")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    if args.preflight:
        raise SystemExit(preflight())
    if not args.manifest:
        ap.error("--manifest is required unless --preflight is set")

    out_path = args.out or default_output_path(args.manifest, args.stage)

    draft_map: dict[str, str] | None = None
    if args.draft_captions:
        draft_map = load_draft_captions(args.draft_captions)
        print(f"draft captions: {len(draft_map)} from {args.draft_captions}")
    elif args.stage == 7:
        print("warning: --stage 7 without --draft-captions; using manifest spatial_caption as draft")

    system_prompt = STAGE7_SYSTEM_PROMPT if args.stage == 7 else SYSTEM_PROMPT

    rows = load_manifest(args.manifest)
    if args.limit:
        rows = rows[: args.limit]
    print(f"manifest rows: {len(rows)}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    use_llm = not (args.no_llm or args.template_only)
    if use_llm:
        try:
            import torch
            if torch.cuda.device_count() == 0:
                print("No GPU available; falling back to template-only mode.")
                use_llm = False
        except Exception:  # noqa: BLE001
            print("torch unavailable; falling back to template-only mode.")
            use_llm = False

    if use_llm:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "all")
        print(f"CUDA_VISIBLE_DEVICES={visible}")
    elif args.template_only:
        print("template-only: copying drafts to output (no LLM)")

    process_rows(
        rows,
        out_path,
        args.batch_size,
        use_llm,
        args.device_map,
        draft_map=draft_map,
        system_prompt=system_prompt,
        stage=args.stage,
        model_name=QWEN_MODEL_PATH if use_llm else None,
    )
    print(f"Done -> {out_path}")


if __name__ == "__main__":
    start = time.time()
    main()
    print(f"Total time: {time.time() - start:.1f}s")
