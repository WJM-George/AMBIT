#!/usr/bin/env python3
import argparse
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import soundfile as sf

PARTS = [
    {
        "name": "sdb",
        "root": Path(os.environ.get("AMBIT_DATA_ROOT", "data") + "/datasets/spatial_speech_foa_tts_v1_part_sdb"),
    },
    {
        "name": "sdc",
        "root": Path(os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/speech_dataset/spatial_speech_foa_tts_v1_part_sdc"),
    },
]

EXPECTED_SR = 48000
EXPECTED_CHANNELS = 4
EXPECTED_SUBTYPE = "PCM_24"
MIN_DURATION = 3.50
MAX_DURATION = 10.50
MAX_PEAK_DBFS = -0.05
MIN_PEAK_DBFS = -40.0
MIN_RMS_DBFS = -50.0
MIN_W_RMS_DBFS = -55.0
MIN_YZX_TO_W_DB = -18.0
MAX_DC_ABS = 0.0025
EPS = 1e-12


def read_jsonl(path):
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        for row in rows:
            row.pop('license', None)
            row.pop('lisence', None)
            f.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')
    os.replace(tmp, path)


def audio_path(row):
    return Path(row.get('audio_path') or row.get('foa_path') or row.get('path') or '')


def db(x):
    return 20.0 * math.log10(max(float(x), EPS))


def probe_row(row):
    rid = row.get('id')
    ap = audio_path(row)
    reasons = []
    metrics = {'id': rid, 'audio_path': str(ap)}
    if not ap.exists():
        return rid, ['missing_audio'], metrics
    try:
        info = sf.info(str(ap))
    except Exception as e:
        metrics['error'] = repr(e)[:500]
        return rid, ['soundfile_info_error'], metrics

    duration = float(info.frames) / float(info.samplerate) if info.samplerate else 0.0
    metrics.update({
        'sample_rate': int(info.samplerate),
        'channels': int(info.channels),
        'subtype': str(info.subtype),
        'frames': int(info.frames),
        'duration_sec': duration,
    })
    if info.samplerate != EXPECTED_SR or info.channels != EXPECTED_CHANNELS or info.subtype != EXPECTED_SUBTYPE:
        reasons.append('bad_format')
    if duration < MIN_DURATION or duration > MAX_DURATION:
        reasons.append('bad_duration')

    try:
        data, sr = sf.read(str(ap), dtype='float32', always_2d=True)
    except Exception as e:
        metrics['error'] = repr(e)[:500]
        reasons.append('soundfile_read_error')
        return rid, sorted(set(reasons)), metrics

    if data.ndim != 2 or data.shape[1] != EXPECTED_CHANNELS:
        reasons.append('bad_format')
        return rid, sorted(set(reasons)), metrics

    abs_data = np.abs(data)
    peak = float(np.max(abs_data)) if data.size else 0.0
    rms_overall = float(np.sqrt(np.mean(np.square(data, dtype=np.float64)))) if data.size else 0.0
    rms_ch = np.sqrt(np.mean(np.square(data, dtype=np.float64), axis=0)) if data.size else np.zeros(EXPECTED_CHANNELS)
    dc_ch = np.mean(data, axis=0) if data.size else np.zeros(EXPECTED_CHANNELS)
    peak_dbfs = db(peak)
    rms_dbfs = db(rms_overall)
    rms_ch_dbfs = [db(x) for x in rms_ch]
    w_rms = float(rms_ch[0])
    yzx_mean = float(np.mean(rms_ch[1:4]))
    yzx_to_w_db = db(yzx_mean / max(w_rms, EPS))
    dc_abs_max = float(np.max(np.abs(dc_ch)))

    metrics.update({
        'peak_dbfs': peak_dbfs,
        'rms_dbfs': rms_dbfs,
        'w_rms_dbfs': rms_ch_dbfs[0],
        'rms_channels_dbfs': rms_ch_dbfs,
        'yzx_to_w_db': yzx_to_w_db,
        'dc_abs_max': dc_abs_max,
    })

    if peak_dbfs > MAX_PEAK_DBFS:
        reasons.append('clipping')
    if peak_dbfs < MIN_PEAK_DBFS or rms_dbfs < MIN_RMS_DBFS:
        reasons.append('near_silent')
    if rms_ch_dbfs[0] < MIN_W_RMS_DBFS:
        reasons.append('mostly_silent_W')
    if yzx_to_w_db < MIN_YZX_TO_W_DB:
        reasons.append('spatial_inactive')
    if dc_abs_max > MAX_DC_ABS:
        reasons.append('dc_offset')

    return rid, sorted(set(reasons)), metrics


def process_rows(rows, workers, chunk_size, part_name):
    rejects = {}
    examples = []
    checked = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for start in range(0, len(rows), chunk_size):
            chunk = rows[start:start + chunk_size]
            futs = [ex.submit(probe_row, row) for row in chunk]
            for fut in as_completed(futs):
                rid, reasons, metrics = fut.result()
                if reasons:
                    rejects[rid] = {'reasons': reasons, 'metrics': metrics}
                    if len(examples) < 50:
                        examples.append({'id': rid, 'reasons': reasons, 'metrics': metrics})
            checked += len(chunk)
            elapsed = max(time.time() - t0, 1e-6)
            rate = checked / elapsed
            print(f"[{part_name}] checked={checked}/{len(rows)} rejects={len(rejects)} rate={rate:.1f}/s", flush=True)
    return rejects, examples


def clean_part(part, workers, chunk_size, sample_per_part, write_outputs):
    root = part['root']
    manifest_path = root / 'manifests' / 'render_manifest.jsonl'
    captions_path = root / 'caption_jsonl' / 'captions_rendered_template.jsonl'
    clean_manifest_path = root / 'manifests' / 'render_manifest_qc_clean.jsonl'
    reject_manifest_path = root / 'manifests' / 'render_manifest_qc_rejected.jsonl'
    clean_captions_path = root / 'caption_jsonl' / 'captions_rendered_template_qc_clean.jsonl'
    reject_captions_path = root / 'caption_jsonl' / 'captions_rendered_template_qc_rejected.jsonl'
    report_path = root / 'manifests' / 'qc_clean_report.json'

    rows = list(read_jsonl(manifest_path))
    if sample_per_part:
        rows = rows[:sample_per_part]
    print(f"[{part['name']}] loaded rows={len(rows)} write_outputs={write_outputs}", flush=True)
    rejects, examples = process_rows(rows, workers, chunk_size, part['name'])

    clean_rows, reject_rows = [], []
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

    reason_counts = {}
    for item in rejects.values():
        for reason in item['reasons']:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

    report = {
        'partition': part['name'],
        'root': str(root),
        'source_manifest': str(manifest_path),
        'source_captions': str(captions_path),
        'clean_manifest': str(clean_manifest_path),
        'rejected_manifest': str(reject_manifest_path),
        'clean_captions': str(clean_captions_path),
        'rejected_captions': str(reject_captions_path),
        'input_rows': len(rows),
        'clean_rows': len(clean_rows),
        'rejected_rows': len(reject_rows),
        'reject_rate': len(reject_rows) / len(rows) if rows else 0.0,
        'reason_counts': reason_counts,
        'thresholds': {
            'EXPECTED_SR': EXPECTED_SR,
            'EXPECTED_CHANNELS': EXPECTED_CHANNELS,
            'EXPECTED_SUBTYPE': EXPECTED_SUBTYPE,
            'MIN_DURATION': MIN_DURATION,
            'MAX_DURATION': MAX_DURATION,
            'MAX_PEAK_DBFS': MAX_PEAK_DBFS,
            'MIN_PEAK_DBFS': MIN_PEAK_DBFS,
            'MIN_RMS_DBFS': MIN_RMS_DBFS,
            'MIN_W_RMS_DBFS': MIN_W_RMS_DBFS,
            'MIN_YZX_TO_W_DB': MIN_YZX_TO_W_DB,
            'MAX_DC_ABS': MAX_DC_ABS,
        },
        'reject_examples': examples,
    }

    if write_outputs:
        caps = {r.get('id'): r for r in read_jsonl(captions_path)}
        clean_caps, reject_caps = [], []
        for row in rows:
            cap = caps.get(row.get('id'))
            if not cap:
                continue
            cap.pop('license', None)
            cap.pop('lisence', None)
            if row.get('id') in rejects:
                out = dict(cap)
                out['qc_reject_reasons'] = rejects[row.get('id')]['reasons']
                reject_caps.append(out)
            else:
                clean_caps.append(cap)
        write_jsonl(clean_manifest_path, clean_rows)
        write_jsonl(reject_manifest_path, reject_rows)
        write_jsonl(clean_captions_path, clean_caps)
        write_jsonl(reject_captions_path, reject_caps)
        tmp = report_path.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        os.replace(tmp, report_path)

    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--chunk-size', type=int, default=2000)
    ap.add_argument('--parts', nargs='*', choices=['sdb', 'sdc'], default=['sdb', 'sdc'])
    ap.add_argument('--sample-per-part', type=int, default=0)
    ap.add_argument('--no-write', action='store_true')
    args = ap.parse_args()

    selected = [p for p in PARTS if p['name'] in args.parts]
    reports = [clean_part(p, args.workers, args.chunk_size, args.sample_per_part, not args.no_write and not args.sample_per_part) for p in selected]
    combined = {
        'input_rows': sum(r['input_rows'] for r in reports),
        'clean_rows': sum(r['clean_rows'] for r in reports),
        'rejected_rows': sum(r['rejected_rows'] for r in reports),
        'reject_rate': (sum(r['rejected_rows'] for r in reports) / sum(r['input_rows'] for r in reports)) if reports else 0.0,
        'partitions': reports,
    }
    out = Path(os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/speech_dataset/spatial_speech_foa_tts_v1_work/quality_reports/qc_clean_combined_report.json")
    if not args.no_write and not args.sample_per_part:
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(combined, ensure_ascii=False, indent=2), encoding='utf-8')
        os.replace(tmp, out)
    print('[combined]')
    print(json.dumps(combined, ensure_ascii=False, indent=2), flush=True)

if __name__ == '__main__':
    main()
