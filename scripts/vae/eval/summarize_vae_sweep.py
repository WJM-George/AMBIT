#!/usr/bin/env python3
"""Aggregate VAE sweep results (auto-detects output format).

Modes (or auto from files under --eval-root):
  heldout   — step_*/heldout_summary.json  (ours vs baseline)
  preencode — step_*/preencode_summary.json (pre-encode pipeline)

Run:
  uv run python scripts/vae/eval/summarize_vae_sweep.py
  uv run python scripts/vae/eval/summarize_vae_sweep.py --eval-root ${AMBIT_CKPT_ROOT}/eval_metric --mode heldout
"""
from __future__ import annotations
import os

import argparse
import json
from pathlib import Path

METRICS = [
    ("si_sdr_db", "SI-SDR", True),
    ("w_si_sdr_db", "W-SI-SDR", True),
    ("lsd_db", "LSD", False),
    ("pesq_w", "PESQ@W", True),
    ("doa_az_err_deg", "DoA az", False),
    ("doa_el_err_deg", "DoA el", False),
    ("gt_doa_az_err_deg", "GT DoA az", False),
    ("ild_err_db", "ILD err", False),
    ("itd_err_us", "ITD err (us)", False),
    ("seld_doa_err_deg", "SELD DoA", False),
    ("dir_energy_ratio_err", "Dir energy", False),
    ("ic_corr_err", "IC corr", False),
]


def detect_mode(root: Path) -> str | None:
    if list(root.glob("step_*/heldout_summary.json")):
        return "heldout"
    if list(root.glob("step_*/preencode_summary.json")):
        return "preencode"
    return None


def load_heldout(root: Path) -> list[tuple[int, dict]]:
    out = []
    for p in sorted(root.glob("step_*/heldout_summary.json")):
        step = int(p.parent.name.replace("step_", ""))
        with p.open() as f:
            out.append((step, json.load(f)))
    return sorted(out, key=lambda x: x[0])


def load_preencode(root: Path) -> list[tuple[int, dict]]:
    out = []
    for p in sorted(root.glob("step_*/preencode_summary.json")):
        step = int(p.parent.name.replace("step_", ""))
        with p.open() as f:
            out.append((step, json.load(f)))
    return sorted(out, key=lambda x: x[0])


def med_heldout(rep: dict, method: str, key: str) -> float | None:
    v = rep.get("aggregate", {}).get(method, {}).get(key, {}).get("median")
    return float(v) if v is not None else None


def med_preencode(rep: dict, key: str) -> float | None:
    tag = rep.get("tag", "ours_preencoded")
    v = rep.get("aggregate", {}).get(tag, {}).get(key, {}).get("median")
    return float(v) if v is not None else None


def summarize_heldout(root: Path, out_path: Path) -> None:
    reports = load_heldout(root)
    if not reports:
        print(f"No heldout summaries under {root}")
        return

    baseline: dict[str, float | None] = {}
    for _, rep in reports:
        for key, _, _ in METRICS:
            if key not in baseline:
                baseline[key] = med_heldout(rep, "baseline_2x_stereo", key)

    lines = [
        "# VAE sweep (heldout, leakage-free)",
        f"Root: `{root}` | steps: {', '.join(str(s) for s, _ in reports)}",
        "",
        "| metric | baseline |" + "".join(f" {s} |" for s, _ in reports) + " best |",
        "|---|---|" + "---|" * (len(reports) + 1),
    ]

    best_per: dict[str, tuple[int, float]] = {}
    for key, label, higher in METRICS:
        row = f"| {label} ({'↑' if higher else '↓'}) |"
        b = baseline.get(key)
        row += f" {b:.3f} |" if b is not None else " n/a |"
        for step, rep in reports:
            v = med_heldout(rep, "ours_4ch_native", key)
            if v is None:
                row += " n/a |"
                continue
            prev = best_per.get(key)
            if prev is None or ((v > prev[1]) if higher else (v < prev[1])):
                best_per[key] = (step, v)
            mark = ""
            if b is not None and ((v > b) if higher else (v < b)):
                mark = "**"
            row += f" {mark}{v:.3f}{mark} |"
        best_s = best_per.get(key, (None,))[0]
        row += f" {best_s} |" if best_s else " n/a |"
        lines.append(row)

    lines += ["", _best_overall(best_per, [s for s, _ in reports])]
    text = "\n".join(lines) + "\n"
    out_path.write_text(text)
    print(text)
    print(f"Wrote -> {out_path}")


def summarize_preencode(root: Path, out_path: Path) -> None:
    reports = load_preencode(root)
    if not reports:
        print(f"No preencode summaries under {root}")
        return

    lines = [
        "# VAE sweep (pre-encode -> decode -> eval)",
        f"Root: `{root}` | steps: {', '.join(str(s) for s, _ in reports)}",
        "",
        "| step |" + "".join(f" {label} |" for _, label, _ in METRICS) + " wins |",
        "|---|" + "---|" * (len(METRICS) + 1),
    ]

    best_per: dict[str, tuple[int, float]] = {}
    for step, rep in reports:
        row = f"| {step} |"
        for key, _, higher in METRICS:
            v = med_preencode(rep, key)
            if v is None:
                row += " n/a |"
                continue
            prev = best_per.get(key)
            if prev is None or ((v > prev[1]) if higher else (v < prev[1])):
                best_per[key] = (step, v)
            row += f" {v:.3f} |"
        lines.append(row)

    win_count: dict[int, int] = {s: 0 for s, _ in reports}
    for key in best_per:
        win_count[best_per[key][0]] += 1
    best_overall = max(win_count, key=lambda s: win_count[s])
    lines += [
        "",
        "## Best per metric",
        "",
        "| metric | dir | step | median |",
        "|---|---|---|---|",
    ]
    for key, label, higher in METRICS:
        if key not in best_per:
            continue
        s, v = best_per[key]
        lines.append(f"| {label} | {'↑' if higher else '↓'} | **{s}** | {v:.3f} |")
    lines.append("")
    lines.append(f"**Overall: step {best_overall} ({win_count[best_overall]}/{len(METRICS)} metrics)**")

    text = "\n".join(lines) + "\n"
    out_path.write_text(text)
    print(text)
    print(f"Wrote -> {out_path}")


def _best_overall(best_per: dict, steps: list[int]) -> str:
    if not best_per:
        return ""
    win_count = {s: 0 for s in steps}
    for key in best_per:
        win_count[best_per[key][0]] += 1
    best = max(win_count, key=lambda s: win_count[s])
    return f"**Overall best (ours): step {best} ({win_count[best]} metrics)**"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-root", type=Path, default=Path(os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/eval_metric"))
    ap.add_argument("--mode", choices=("heldout", "preencode", "auto"), default="auto")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    mode = args.mode if args.mode != "auto" else detect_mode(args.eval_root)
    if mode is None:
        print(f"No sweep results under {args.eval_root}")
        return

    out_path = args.out or (args.eval_root / f"sweep_summary_{mode}.md")
    if mode == "heldout":
        summarize_heldout(args.eval_root, out_path)
    else:
        summarize_preencode(args.eval_root, out_path)


if __name__ == "__main__":
    main()
