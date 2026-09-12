#!/usr/bin/env python3
import argparse
import json
import math
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import median

PARTS = [
    {
        "name": "sdb",
        "root": Path("/mnt/sdb/audio_dataset/datasets/spatial_speech_foa_tts_v1_part_sdb"),
    },
    {
        "name": "sdc",
        "root": Path("/mnt/sdc/speech_dataset/spatial_speech_foa_tts_v1_part_sdc"),
    },
]

BAD_REASONS = {
    "missing_audio",
    "ffprobe_error",
    "bad_format",
    "bad_duration",
    "bad_peak",
    "clipping",
    "near_silent",
    "mostly_silent_W",
    "spatial_inactive",
    "dc_offset",
}

# Conservative but not overzealous thresholds. These match the prior probe spirit:
# reject real structural/audio issues and the mild DC-offset cases the user wants out.
MIN_DURATION = 3.50
MAX_DURATION = 10.50
MIN_RMS_DBFS = -50.0
MIN_W_RMS_DBFS = -55.0
MAX_PEAK_DBFS = -0.05
MIN_PEAK_DBFS = -40.0
MAX_DC_ABS = 0.0025
MIN_YZX_TO_W_DB = -18.0
EXPECTED_SR = 48000
EXPECTED_CHANNELS = 4
EXPECTED_BITS = 24


def read_jsonl(path):
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def dump_jsonl(path, rows):
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        for row in rows:
            row.pop('license', None)
            row.pop('lisence', None)
            f.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')
    os.replace(tmp, path)


def ffprobe_stream(path):
    cmd = [
        'ffprobe', '-v', 'error', '-select_streams', 'a:0',
        '-show_entries', 'stream=sample_rate,channels,bits_per_raw_sample,bits_per_sample,duration',
        '-of', 'json', str(path),
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        return None, p.stderr.strip()[:500]
    try:
        data = json.loads(p.stdout)
        streams = data.get('streams') or []
        if not streams:
            return None, 'no audio stream'
        return streams[0], None
    except Exception as e:
        return None, repr(e)


def ffmpeg_stats(path):
    # astats gives per-channel and overall peak/RMS/DC without decoding to temp files.
    cmd = [
        'ffmpeg', '-hide_banner', '-nostats', '-v', 'info', '-i', str(path),
        '-af', 'astats=metadata=1:reset=0', '-f', 'null', '-'
    ]
    p = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    # ffmpeg may return non-zero on some FLAC/WAV edge cases even after printing
    # complete astats output. Treat parseable astats as authoritative.
    stderr = p.stderr.splitlines()
    chans = []
    cur = None
    overall = {}
    in_overall = False
    for line in stderr:
        s = line.strip()
        if 'Parsed_astats' not in s:
            continue
        if 'Channel:' in s:
            try:
                ch = int(s.rsplit('Channel:', 1)[1].strip())
            except Exception:
                ch = len(chans) + 1
            cur = {}
            chans.append(cur)
            in_overall = False
            continue
        if 'Overall' in s:
            cur = None
            in_overall = True
            continue
        target = overall if in_overall else cur
        if target is None or ':' not in s:
            continue
        key, val = s.rsplit(':', 1)
        key = key.split(']', 1)[-1].strip()
        val = val.strip()
        try:
            target[key] = float(val)
        except Exception:
            pass
    if len(chans) < 4:
        err = p.stderr.strip()[:800] if p.returncode != 0 else 'astats missing channels'
        return None, err or 'astats missing channels'
    return {"channels": chans[:4], "overall": overall}, None


def db_from_rms(rms):
    if rms is None or rms <= 0:
        return -999.0
    return 20.0 * math.log10(rms)


def probe_row(row):
    audio_path = Path(row.get('audio_path') or row.get('foa_path') or row.get('path') or '')
    reasons = []
    metrics = {"id": row.get('id'), "audio_path": str(audio_path)}
    if not audio_path.exists():
        reasons.append('missing_audio')
        return row.get('id'), reasons, metrics

    stream, err = ffprobe_stream(audio_path)
    if err:
        reasons.append('ffprobe_error')
        metrics['ffprobe_error'] = err
        return row.get('id'), reasons, metrics

    try:
        sr = int(stream.get('sample_rate') or 0)
    except Exception:
        sr = 0
    channels = int(stream.get('channels') or 0)
    bits = int(stream.get('bits_per_raw_sample') or stream.get('bits_per_sample') or 0)
    duration = float(stream.get('duration') or row.get('duration_sec') or 0.0)
    metrics.update({"sample_rate": sr, "channels": channels, "bits": bits, "duration_sec": duration})
    if sr != EXPECTED_SR or channels != EXPECTED_CHANNELS or bits != EXPECTED_BITS:
        reasons.append('bad_format')
    if duration < MIN_DURATION or duration > MAX_DURATION:
        reasons.append('bad_duration')

    stats, err = ffmpeg_stats(audio_path)
    if err:
        reasons.append('ffprobe_error')
        metrics['ffmpeg_error'] = err
        return row.get('id'), reasons, metrics

    ch = stats['channels']
    rms_levels = [c.get('RMS level dB', -999.0) for c in ch]
    peaks = [c.get('Peak level dB', -999.0) for c in ch]
    dc_offsets = [abs(c.get('DC offset', 0.0)) for c in ch]
    flat_factors = [c.get('Flat factor', 0.0) for c in ch]
    w_rms = rms_levels[0]
    yzx_rms_lin = []
    for level in rms_levels[1:4]:
        yzx_rms_lin.append(0.0 if level <= -900 else 10 ** (level / 20.0))
    w_lin = 0.0 if w_rms <= -900 else 10 ** (w_rms / 20.0)
    yzx_mean_lin = sum(yzx_rms_lin) / 3.0
    yzx_to_w_db = 20.0 * math.log10((yzx_mean_lin + 1e-12) / (w_lin + 1e-12))
    peak_dbfs = max(peaks)
    rms_dbfs = stats.get('overall', {}).get('RMS level dB', median(rms_levels))
    dc_abs_max = max(dc_offsets)
    metrics.update({
        "peak_dbfs": peak_dbfs,
        "rms_dbfs": rms_dbfs,
        "w_rms_dbfs": w_rms,
        "yzx_to_w_db": yzx_to_w_db,
        "dc_abs_max": dc_abs_max,
        "rms_channels_dbfs": rms_levels,
        "peak_channels_dbfs": peaks,
    })

    if peak_dbfs > MAX_PEAK_DBFS:
        reasons.append('clipping')
    if peak_dbfs < MIN_PEAK_DBFS or rms_dbfs < MIN_RMS_DBFS:
        reasons.append('near_silent')
    if w_rms < MIN_W_RMS_DBFS:
        reasons.append('mostly_silent_W')
    if yzx_to_w_db < MIN_YZX_TO_W_DB:
        reasons.append('spatial_inactive')
    if dc_abs_max > MAX_DC_ABS:
        reasons.append('dc_offset')
    if any(x > 0 for x in flat_factors):
        # astats flat factor can catch hard clipping-like plateaus; keep as clipping family.
        reasons.append('clipping')

    return row.get('id'), sorted(set(reasons)), metrics


def clean_part(part, workers):
    root = part['root']
    manifest_path = root / 'manifests' / 'render_manifest.jsonl'
    captions_path = root / 'caption_jsonl' / 'captions_rendered_template.jsonl'
    clean_manifest_path = root / 'manifests' / 'render_manifest_qc_clean.jsonl'
    reject_manifest_path = root / 'manifests' / 'render_manifest_qc_rejected.jsonl'
    clean_captions_path = root / 'caption_jsonl' / 'captions_rendered_template_qc_clean.jsonl'
    reject_captions_path = root / 'caption_jsonl' / 'captions_rendered_template_qc_rejected.jsonl'
    report_path = root / 'manifests' / 'qc_clean_report.json'

    rows = list(read_jsonl(manifest_path))
    captions = {r.get('id'): r for r in read_jsonl(captions_path)}
    print(f"[{part['name']}] loaded manifest={len(rows)} captions={len(captions)}", flush=True)

    rejects = {}
    metrics_examples = []
    checked = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(probe_row, row) for row in rows]
        for fut in as_completed(futs):
            rid, reasons, metrics = fut.result()
            checked += 1
            if reasons:
                rejects[rid] = {"id": rid, "reasons": reasons, "metrics": metrics}
                if len(metrics_examples) < 50:
                    metrics_examples.append(rejects[rid])
            if checked % 5000 == 0:
                print(f"[{part['name']}] checked={checked}/{len(rows)} rejects={len(rejects)}", flush=True)

    clean_rows = []
    reject_rows = []
    for row in rows:
        rid = row.get('id')
        row.pop('license', None)
        row.pop('lisence', None)
        if rid in rejects:
            out = dict(row)
            out['qc_reject_reasons'] = rejects[rid]['reasons']
            reject_rows.append(out)
        else:
            clean_rows.append(row)

    clean_caps = []
    reject_caps = []
    for row in read_jsonl(captions_path):
        rid = row.get('id')
        row.pop('license', None)
        row.pop('lisence', None)
        if rid in rejects:
            out = dict(row)
            out['qc_reject_reasons'] = rejects[rid]['reasons']
            reject_caps.append(out)
        else:
            clean_caps.append(row)

    dump_jsonl(clean_manifest_path, clean_rows)
    dump_jsonl(reject_manifest_path, reject_rows)
    dump_jsonl(clean_captions_path, clean_caps)
    dump_jsonl(reject_captions_path, reject_caps)

    reason_counts = {}
    for item in rejects.values():
        for r in item['reasons']:
            reason_counts[r] = reason_counts.get(r, 0) + 1
    report = {
        "partition": part['name'],
        "root": str(root),
        "source_manifest": str(manifest_path),
        "source_captions": str(captions_path),
        "clean_manifest": str(clean_manifest_path),
        "rejected_manifest": str(reject_manifest_path),
        "clean_captions": str(clean_captions_path),
        "rejected_captions": str(reject_captions_path),
        "input_rows": len(rows),
        "clean_rows": len(clean_rows),
        "rejected_rows": len(reject_rows),
        "reject_rate": (len(reject_rows) / len(rows)) if rows else 0.0,
        "reason_counts": reason_counts,
        "thresholds": {
            "MIN_DURATION": MIN_DURATION,
            "MAX_DURATION": MAX_DURATION,
            "MIN_RMS_DBFS": MIN_RMS_DBFS,
            "MIN_W_RMS_DBFS": MIN_W_RMS_DBFS,
            "MAX_PEAK_DBFS": MAX_PEAK_DBFS,
            "MIN_PEAK_DBFS": MIN_PEAK_DBFS,
            "MAX_DC_ABS": MAX_DC_ABS,
            "MIN_YZX_TO_W_DB": MIN_YZX_TO_W_DB,
        },
        "reject_examples": metrics_examples,
    }
    tmp = report_path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(tmp, report_path)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--parts', nargs='*', choices=['sdb', 'sdc'], default=['sdb', 'sdc'])
    args = ap.parse_args()
    selected = [p for p in PARTS if p['name'] in args.parts]
    reports = [clean_part(p, args.workers) for p in selected]
    combined = {
        "input_rows": sum(r['input_rows'] for r in reports),
        "clean_rows": sum(r['clean_rows'] for r in reports),
        "rejected_rows": sum(r['rejected_rows'] for r in reports),
        "reject_rate": (sum(r['rejected_rows'] for r in reports) / sum(r['input_rows'] for r in reports)) if reports else 0,
        "partitions": reports,
    }
    out = Path('/mnt/sdc/speech_dataset/spatial_speech_foa_tts_v1_work/quality_reports/qc_clean_combined_report.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(combined, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(tmp, out)
    print('[combined]', json.dumps(combined, ensure_ascii=False, indent=2), flush=True)

if __name__ == '__main__':
    main()
