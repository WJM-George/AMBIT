#!/usr/bin/env python3
"""Build the AMBIT project listening page from the listening-page shortlist.

Uses six test examples (3 generation + 3 editing) that passed the archived
quality gates and an audio-preview waveform/mel screen: matched duration, no
leading silence or hard cutoff, and a GT-like envelope. Native FOA is decoded
with the paper KEMAR binaural renderer when available, otherwise the in-repo
±30° virtual-stereo preview. One shared peak gain is applied per example.
Selected examples are qualitative; they are not average-case results.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.t2a.eval.sceneplan_44_eval_common import virtual_stereo
from stable_audio_tools.paths import editing_bench

DEFAULT_SELECTION = Path(os.environ.get("AMBIT_DEMO_SELECTION", str(REPO / "docs" / "demo")))
DEFAULT_SELECTION_FILE = "LISTENING_SELECTED.json"
DEFAULT_OUT = REPO / "docs" / "demo"
KEMAR_ROOT = editing_bench()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path):
    return json.loads(path.read_text())


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def plan_from(path: Path) -> dict:
    data = load(path)
    return data.get("p10_plan") or data.get("raw_plan") or data


def pose_text(pose: dict) -> str:
    return (
        f"{pose['azimuth_deg']:.1f}° az, "
        f"{pose['elevation_deg']:.1f}° el, "
        f"{pose['distance_m']:.2f} m"
    )


def trajectory_text(traj: dict) -> str:
    if traj.get("type") == "static":
        return "static at " + pose_text(traj["position"])
    start, end = traj["start"], traj["end"]
    return f"linear {pose_text(start)} → {pose_text(end)}"


def source_summary(plan: dict) -> str:
    rows = []
    for source in plan.get("sources", []):
        activity = source.get("activity", {})
        content = source.get("transcript") or source.get("description") or ""
        rows.append(
            f"{source.get('source_id', '?')} · {source.get('kind', '?')} · "
            f"{activity.get('onset_sec', 0):.2f}–{activity.get('offset_sec', 0):.2f}s · "
            f"{trajectory_text(source.get('trajectory', {}))} · {content}"
        )
    return "\n".join(rows)


def load_foa(path: Path, digest: str) -> tuple[np.ndarray, int]:
    path = Path(path)
    assert path.exists(), path
    assert sha256(path) == digest, (path, digest)
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    assert rate == 44100 and audio.shape[1] == 4, (path, audio.shape, rate)
    assert np.isfinite(audio).all(), path
    return audio.T.copy(), rate


def make_decoder():
    bench = KEMAR_ROOT
    module = bench / "benchmark_audio_v1.py"
    if module.exists():
        sys.path.insert(0, str(bench))
        from benchmark_audio_v1 import KemarFoaDecoder

        decoder = KemarFoaDecoder()
        return {
            "name": "MIT KEMAR 8-virtual-speaker FOA binaural",
            "id": "MIT_KEMAR_8virtual_ACN_SN3D_basic_44100_v1",
            "note": (
                "Headphones recommended. 0° is front, positive azimuth is left, "
                "negative is right. This is a fixed first-order binaural preview, "
                "not native FOA playback."
            ),
            "fn": lambda foa: decoder.decode(np.asarray(foa, dtype=np.float64)),
        }

    def virtual(foa):
        stereo, _ = virtual_stereo(__import__("torch").from_numpy(np.asarray(foa)))
        return stereo.numpy().astype(np.float64)

    return {
        "name": "WYZX ±30° virtual stereo",
        "id": "virtual_stereo_pm30",
        "note": (
            "Headphones recommended. This page uses a fixed ±30° stereo preview, "
            "not HRTF binaural or native FOA playback."
        ),
        "fn": virtual,
    }


def shared_peak_wavs(waveforms: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    peak = max(float(np.max(np.abs(wave))) for wave in waveforms.values())
    assert peak > 0
    gain = (10 ** (-1 / 20)) / peak
    return {name: wave * gain for name, wave in waveforms.items()}, gain


def preview_figure(waves: dict[str, np.ndarray], dest: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from scipy import signal

    rate = 44100
    nfft, hop, nmel = 2048, 512, 80
    frequencies = np.fft.rfftfreq(nfft, 1 / rate)
    hz = lambda mel: 700 * (10 ** (mel / 2595) - 1)
    edges = hz(np.linspace(0, 2595 * np.log10(1 + (rate / 2) / 700), nmel + 2))
    basis = np.zeros((nmel, len(frequencies)), np.float32)
    for i in range(nmel):
        left, center, right = edges[i], edges[i + 1], edges[i + 2]
        basis[i] = np.maximum(
            0,
            np.minimum(
                (frequencies - left) / max(center - left, 1e-12),
                (right - frequencies) / max(right - center, 1e-12),
            ),
        )
    basis /= np.maximum(basis.sum(axis=1, keepdims=True), 1e-12)

    def stft_db(wave):
        freq, times, z = signal.stft(
            wave, fs=rate, window="hann", nperseg=nfft, noverlap=nfft - hop,
            boundary="zeros", padded=True, scaling="spectrum",
        )
        amplitude = np.abs(z)
        amplitude[1:-1] *= 2
        return freq, times, 20 * np.log10(np.maximum(amplitude, 10 ** (-90 / 20)))

    def logmel(wave):
        _, times, z = signal.stft(
            wave, fs=rate, window="hann", nperseg=nfft, noverlap=nfft - hop,
            boundary="zeros", padded=True, scaling="spectrum",
        )
        return times, 10 * np.log10(np.maximum(basis @ (np.abs(z) ** 2), 1e-9))

    roles = list(waves)
    duration = max(len(wave) for wave in waves.values()) / rate
    fig, axes = plt.subplots(
        3, len(roles), figsize=(5.2 * len(roles), 7.8), layout="constrained", squeeze=False
    )
    peak = max(float(np.max(np.abs(wave))) for wave in waves.values()) or 1.0
    for col, role in enumerate(roles):
        wave = waves[role]
        axes[0, col].plot(np.arange(len(wave)) / rate, wave, color="#1f4b6e", lw=0.35)
        axes[0, col].set_xlim(0, duration)
        axes[0, col].set_ylim(-peak, peak)
        axes[0, col].set_title(role, fontweight="semibold")
        axes[0, col].set_ylabel("Waveform" if col == 0 else "")
        freq, times, db = stft_db(wave)
        axes[1, col].pcolormesh(
            times, freq, db, shading="auto", cmap="magma", norm=Normalize(-90, 0), rasterized=True
        )
        axes[1, col].set_yscale("symlog", linthresh=100, linscale=0.55)
        axes[1, col].set_ylim(0, rate / 2)
        axes[1, col].set_xlim(0, duration)
        axes[1, col].set_ylabel("Hz" if col == 0 else "")
        times, mel = logmel(wave)
        axes[2, col].pcolormesh(times, np.arange(nmel), mel, shading="auto", cmap="magma", rasterized=True)
        axes[2, col].set_xlim(0, duration)
        axes[2, col].set_xlabel("Time (s)")
        axes[2, col].set_ylabel("Mel" if col == 0 else "")
    fig.suptitle(title, fontsize=12)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=140)
    plt.close(fig)


CASES = [
    {
        "code": "G1",
        "folder": "G1_0002223",
        "id": "spv2_test_no_speech_1_0000223",
        "title": "Solo acoustic guitar, stationary behind the listener",
        "listen": "The guitar should sit behind you for the full clip. The late swell near 8s should line up with GT.",
    },
    {
        "code": "G2",
        "folder": "G2_0003591",
        "id": "spv2_test_no_speech_3_0000191",
        "title": "Chiptune, a passing train, and a left-side drone",
        "listen": "Chiptune stays back-right; the train should travel right to left; the drone stays left.",
    },
    {
        "code": "G3",
        "folder": "G3_0003463",
        "id": "spv2_test_no_speech_3_0000063",
        "title": "Three overlapping glitch and noise textures",
        "listen": "A left-stationary glitch bed, a left-to-front-right modulator, and a front-to-back receding noise.",
    },
    {
        "code": "E1",
        "folder": "E1_0002551",
        "id": "speditv1_test_2e336fbb88da900f18cdb7ed",
        "title": "Relocate the ambient drone to the front",
        "listen": "Compare reference (behind) with AMBIT and GT (front, 0°).",
    },
    {
        "code": "E2",
        "folder": "E2_0002538",
        "id": "speditv1_test_8df78e38274782be01e14563",
        "title": "Relocate the retro organ to the right (−90°)",
        "listen": "The organ should jump from its original seat to the right ear.",
    },
    {
        "code": "E3",
        "folder": "E3_0002511",
        "id": "speditv1_test_8a3f12a6641adc3c062ef2e3",
        "title": "Relocate the synth drone to the right (−90°)",
        "listen": "The drone should move to the right while the timbre stays put.",
    },
]


ABSTRACT = (
    "Generating and editing multi-source spatial audio requires fine-grained "
    "control over content, timing, and 3D trajectories, while preserving "
    "unmodified sources. AMBIT (AMBIsonic Transfusion) is an end-to-end "
    "framework for native first-order Ambisonic generation and editing that "
    "models spatial planning as an executable ScenePlan."
)


def case_record(spec: dict, selected: dict) -> dict:
    quality = selected["quality"]
    spectral = selected["spectral_score"]
    if selected["task"] == "generation":
        plan = plan_from(Path(selected["generated_sceneplan_path"]))
        model = "AMBIT AR → DiT150k"
    else:
        plan = selected.get("generated_sceneplan") or {}
        model = "AMBIT joint AR+DiT50k"
    record = {
        **spec,
        "task": selected["task"],
        "id": selected["id"],
        "ordinal": selected["ordinal"],
        "operation": selected.get("operation") or "generation",
        "request": selected["request"],
        "source_count": selected.get("source_count") or selected.get("target_count"),
        "source_kinds": selected.get("source_kinds"),
        "doa_deg": quality.get("DOA_deg"),
        "activity_iou": quality.get("activity_iou"),
        "change_iou": quality.get("change_iou"),
        "logmel_mae_db": spectral.get("logmel_mae_db"),
        "nmse": quality.get("NMSE_to_codec_GT"),
        "model": model,
        "plan": plan,
        "plan_summary": source_summary(plan) if plan else "",
        "audio_src": {
            role: {
                "native_copy": {
                    "path": path,
                    "sha256": selected["expected_hashes"][role],
                }
            }
            for role, path in selected["paths"].items()
        },
        "swanweave": None,
    }
    outputs = selected.get("original_manifest_row", {}).get("outputs", {})
    swan = outputs.get("SwanWeave", {})
    if swan.get("output_path") and Path(swan["output_path"]).exists():
        record["swanweave"] = {
            "path": swan["output_path"],
            "sha256": swan["output_sha256"],
        }
    return record


def decode_case(record: dict, decoder: dict, dest: Path) -> dict:
    dest.mkdir(parents=True, exist_ok=True)
    audio_src = record["audio_src"]
    waves = {}
    w_preview = {}
    labels = []
    if record["task"] == "generation":
        order = [("gt", "Ground truth"), ("ours", "AMBIT")]
    else:
        order = [("reference", "Reference"), ("ours", "AMBIT"), ("gt", "Ground truth")]
        if record["swanweave"]:
            order.append(("swanweave", "SwanWeave"))
    for key, label in order:
        if key == "swanweave":
            foa, _ = load_foa(Path(record["swanweave"]["path"]), record["swanweave"]["sha256"])
        else:
            item = audio_src[key]
            foa, _ = load_foa(Path(item["native_copy"]["path"]), item["native_copy"]["sha256"])
        stereo = np.asarray(decoder["fn"](foa), dtype=np.float64)
        if stereo.shape[0] != 2:
            raise ValueError((key, stereo.shape))
        waves[key] = stereo
        w_preview[label] = np.asarray(foa[0], dtype=np.float32)
        labels.append((key, label))
    scaled, gain = shared_peak_wavs(waves)
    players = []
    for key, label in labels:
        out = dest / f"{key}.wav"
        payload = scaled[key].T
        sf.write(out, payload, 44100, subtype="PCM_16")
        players.append(
            {
                "key": key,
                "label": label,
                "src": f"assets/{dest.name}/{out.name}",
                "seconds": payload.shape[0] / 44100,
                "sha256": sha256(out),
            }
        )
    preview_path = dest / "waveform_mel.png"
    preview_figure(w_preview, preview_path, f"{record['code']}  ·  {record['id']}  ·  FOA W")
    if record["plan"]:
        write_json(dest / "sceneplan.json", record["plan"])
    return {
        **{k: record[k] for k in (
            "code", "title", "listen", "task", "id", "ordinal", "operation",
            "request", "doa_deg", "activity_iou", "change_iou", "logmel_mae_db",
            "nmse", "model", "plan_summary",
        )},
        "players": players,
        "spectrogram": f"assets/{dest.name}/{preview_path.name}",
        "sceneplan": f"assets/{dest.name}/sceneplan.json" if record["plan"] else None,
        "preview_gain": gain,
    }


def render_page(cases: list[dict], decoder: dict) -> str:
    def esc(text) -> str:
        return html.escape("" if text is None else str(text))

    def metric_bits(case: dict) -> str:
        bits = [f"DOA {case['doa_deg']:.2f}°"]
        if case["activity_iou"] is not None:
            bits.append(f"IoU {case['activity_iou']:.3f}")
        if case["change_iou"] is not None:
            bits.append(f"change IoU {case['change_iou']:.3f}")
        if case["logmel_mae_db"] is not None:
            bits.append(f"W log-mel MAE {case['logmel_mae_db']:.2f} dB")
        if case["nmse"] is not None:
            bits.append(f"NMSE {case['nmse']:.4f}")
        return " · ".join(bits)

    def players(case: dict) -> str:
        blocks = []
        for player in case["players"]:
            ident = f"{case['code']}-{player['key']}"
            blocks.append(
                f'<div class="player"><strong>{esc(player["label"])}</strong>'
                f'<audio id="{esc(ident)}" controls preload="metadata" '
                f'src="{esc(player["src"])}"></audio>'
                f'<button type="button" data-audio="{esc(ident)}">'
                f"Switch at this time</button></div>"
            )
        return "".join(blocks)

    def article(case: dict) -> str:
        image = (
            f'<img src="{esc(case["spectrogram"])}" alt="W-channel waveform, spectrum, and mel of {esc(case["code"])}">'
            if case["spectrogram"]
            else ""
        )
        plan = (
            f'<details><summary>Predicted ScenePlan</summary><pre>{esc(case["plan_summary"])}</pre>'
            + (
                f'<p><a href="{esc(case["sceneplan"])}">sceneplan.json</a></p>'
                if case["sceneplan"]
                else ""
            )
            + "</details>"
        )
        return f"""
<article class="case" id="{esc(case["code"])}" data-task="{esc(case["task"])}">
  <header>
    <p class="kicker">{esc(case["code"])} · {esc(case["task"])} · {esc(case["operation"])} · {esc(case["id"])}</p>
    <h3>{esc(case["title"])}</h3>
    <p class="listen">{esc(case["listen"])}</p>
    <p class="metrics">{esc(metric_bits(case))}</p>
  </header>
  <blockquote>{esc(case["request"])}</blockquote>
  <div class="grid">{players(case)}</div>
  {plan}
  {image}
</article>
"""

    generation = "".join(article(case) for case in cases if case["task"] == "generation")
    editing = "".join(article(case) for case in cases if case["task"] == "editing")
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AMBIT · Listening examples</title>
  <style>
    :root {{
      --ink: #16202b;
      --muted: #5b6876;
      --line: #d7dee6;
      --paper: #f6f3ec;
      --card: #fffdf8;
      --accent: #1f4b6e;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      background: var(--paper);
      font: 17px/1.6 "Iowan Old Style", "Palatino Linotype", Palatino, serif;
    }}
    main {{ max-width: 1080px; margin: 0 auto; padding: 40px 22px 80px; }}
    h1, h2, h3 {{ font-family: "Source Serif 4", "Iowan Old Style", Palatino, serif; letter-spacing: -0.02em; }}
    h1 {{ font-size: 42px; line-height: 1.15; margin: 12px 0 8px; }}
    h2 {{ font-size: 26px; margin: 42px 0 12px; }}
    h3 {{ font-size: 22px; margin: 0 0 8px; }}
    .kicker, .metrics, .note, nav a {{ font-family: "IBM Plex Sans", "Helvetica Neue", sans-serif; }}
    .kicker {{ text-transform: uppercase; letter-spacing: 0.08em; font-size: 12px; color: var(--accent); margin: 0; }}
    .lead {{ font-size: 19px; max-width: 72ch; }}
    .note, .metrics, .listen {{ color: var(--muted); font-size: 14px; }}
    nav {{ display: flex; gap: 16px; flex-wrap: wrap; margin: 22px 0 8px; }}
    nav a {{ color: var(--accent); }}
    blockquote {{
      margin: 14px 0;
      padding: 12px 16px;
      background: #efe8d8;
      border-left: 3px solid var(--accent);
      white-space: pre-wrap;
    }}
    .case {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 22px;
      margin: 22px 0 28px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
      gap: 14px;
      margin: 16px 0;
    }}
    .player {{ min-width: 0; }}
    audio {{ width: 100%; display: block; margin: 8px 0; }}
    button {{
      font: 13px/1.3 "IBM Plex Sans", sans-serif;
      border: 1px solid var(--line);
      background: white;
      border-radius: 8px;
      padding: 6px 10px;
      cursor: pointer;
    }}
    img {{ width: 100%; border-radius: 8px; margin-top: 12px; background: #111; }}
    details {{ margin-top: 10px; }}
    pre {{ white-space: pre-wrap; font-size: 13px; background: #f3f0e7; padding: 12px; border-radius: 8px; }}
    footer {{ color: var(--muted); font-size: 14px; margin-top: 48px; max-width: 72ch; }}
  </style>
</head>
<body>
<main>
  <p class="kicker">Listening examples</p>
  <h1>AMBIT</h1>
  <p class="lead"><strong>AMBIsonic Transfusion</strong> — native first-order Ambisonic generation and instruction-guided editing with executable ScenePlans.</p>
  <p>{esc(ABSTRACT)}</p>
  <p class="note">{esc(decoder["note"])} Playback uses <strong>{esc(decoder["name"])}</strong>. One shared peak gain is applied inside each example. These six clips passed the archived quality gates, then an audio-preview screen for matched duration, no leading silence or hard cutoff, and a waveform/mel that follows GT. They are showcase examples, not an estimate of average test performance.</p>
  <nav>
    <a href="#generation">Generation</a>
    <a href="#editing">Editing</a>
    <a href="https://github.com/WJM-George/AMBIT">Code</a>
  </nav>

  <h2 id="generation">Generation</h2>
  <p>English request → ScenePlan AR → compiled control → FOA DiT. Each row compares the real test FOA with the full AMBIT AR→DiT150k output.</p>
  {generation}

  <h2 id="editing">Editing</h2>
  <p>Reference FOA + instruction → ScenePlan AR → FOA DiT conditioned on the reference latent. Each row keeps the source recording, AMBIT AR+DiT50k, and the real edited GT. SwanWeave is included when the archived official output exists for that pair.</p>
  {editing}

  <footer>
    <p>Azimuth convention: 0° front, +90° left, −90° right. Generation DOA is execution error versus the predicted ScenePlan; editing DOA is common-frame error versus the real target. Figures are FOA W-channel waveform, linear spectrum, and mel.</p>
  </footer>
</main>
<script>
document.querySelectorAll("button[data-audio]").forEach((button) => {{
  button.addEventListener("click", () => {{
    const next = document.getElementById(button.dataset.audio);
    const article = button.closest("article");
    const current = article.querySelector("audio.playing") || article.querySelector("audio");
    const time = current ? current.currentTime : 0;
    article.querySelectorAll("audio").forEach((node) => {{
      node.pause();
      node.classList.remove("playing");
    }});
    next.currentTime = time;
    next.classList.add("playing");
    next.play();
  }});
}});
document.querySelectorAll("audio").forEach((node) => {{
  node.addEventListener("play", () => {{
    const article = node.closest("article");
    article.querySelectorAll("audio").forEach((other) => {{
      if (other !== node) other.pause();
      other.classList.toggle("playing", other === node);
    }});
  }});
}});
</script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-root", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    selection = load(args.selection_root / DEFAULT_SELECTION_FILE)
    decoder = make_decoder()
    out = args.out.resolve()
    assets = out / "assets"
    if assets.exists():
        shutil.rmtree(assets)
    assets.mkdir(parents=True)
    rendered = []
    for spec in CASES:
        selected = next(row for row in selection["samples"] if row["id"] == spec["id"])
        record = case_record(spec, selected)
        assert record["id"] == selected["id"]
        rendered.append(decode_case(record, decoder, assets / spec["folder"]))
    page = render_page(rendered, decoder)
    (out / "index.html").write_text(page)
    write_json(
        out / "MANIFEST.json",
        {
            "status": "COMPLETE",
            "decoder": {k: decoder[k] for k in ("name", "id", "note")},
            "selection": "listening shortlist",
            "cases": rendered,
            "policy": "Qualitative shortlist after quality gates and audio-preview waveform/mel screen; not average-case.",
        },
    )
    (out / "README.md").write_text(
        "\n".join(
            [
                "# AMBIT listening page",
                "",
                "Open `index.html` locally (a static file server is enough).",
                "",
                f"Decoder: {decoder['name']}.",
                "",
                "Samples are a listening-page shortlist:",
                "3 generation + 3 editing examples after quality gates and an audio-preview",
                "waveform/mel screen (no hard cutoff, GT-like envelope).",
                "They are showcase clips, not a test-set average.",
                "",
                "Rebuild:",
                "",
                "```bash",
                "python scripts/t2a/eval/build_ambit_demo_page.py",
                "```",
                "",
            ]
        )
    )
    print(out / "index.html")


if __name__ == "__main__":
    main()
