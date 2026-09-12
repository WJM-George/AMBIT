#!/usr/bin/env python3
import json
import os
import re
import subprocess
from pathlib import Path

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

EXPECTED_SR = 48000
EXPECTED_CHANNELS = 4
EXPECTED_BITS = 24
MIN_DURATION = 3.50
MAX_DURATION = 10.50
MAX_DC_ABS = 0.0025

STREAMINFO_RE = {
    "sample_rate": re.compile(r"sample_rate:\s+(\d+)\s+Hz"),
    "channels": re.compile(r"channels:\s+(\d+)"),
    "bits": re.compile(r"bits-per-sample:\s+(\d+)"),
    "total_samples": re.compile(r"total samples:\s+(\d+)"),
}


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


def metaflac_info(path):
    p = subprocess.run(
        ['metaflac', '--list', '--block-type=STREAMINFO', str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if p.returncode != 0:
        return None, p.stderr.strip()[:500]
    out = p.stdout
    vals = {}
    for k, rx in STREAMINFO_RE.items():
        m = rx.search(out)
        if not m:
            return None, f'missing_{k}'
        vals[k] = int(m.group(1))
    vals['duration_sec'] = vals['total_samples'] / vals['sample_rate'] if vals['sample_rate'] else 0.0
    return vals, None


def dc_offset_probe(path):
    # Decode one file and parse astats DC offset only. This is used only for a targeted pass.
    p = subprocess.run(
        ['ffmpeg', '-hide_banner', '-nostats', '-v', 'info', '-i', str(path), '-af', 'astats=metadata=1:reset=0', '-f', 'null', '-'],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    dc = []
    for line in p.stderr.splitlines():
        if 'DC offset:' in line:
            try:
                dc.append(abs(float(line.rsplit(':', 1)[1].strip())))
            except Exception:
                pass
    if len(dc) < 4:
        return None, 'astats_missing_dc_offset'
    return max(dc[:4]), None


def clean_part(part):
    root = part['root']
    manifest_path = root / 'manifests' / 'render_manifest.jsonl'
    captions_path = root / 'caption_jsonl' / 'captions_rendered_template.jsonl'
    clean_manifest_path = root / 'manifests' / 'render_manifest_qc_clean.jsonl'
    reject_manifest_path = root / 'manifests' / 'render_manifest_qc_rejected.jsonl'
    clean_captions_path = root / 'caption_jsonl' / 'captions_rendered_template_qc_clean.jsonl'
    reject_captions_path = root / 'caption_jsonl' / 'captions_rendered_template_qc_rejected.jsonl'
    report_path = root / 'manifests' / 'qc_clean_report.json'

    rows = list(read_jsonl(manifest_path))
    rejects = {}
    checked = 0
    for row in rows:
        rid = row.get('id')
        reasons = []
        ap = audio_path(row)
        metrics = {'audio_path': str(ap)}
        if not ap.exists():
            reasons.append('missing_audio')
        else:
            info, err = metaflac_info(ap)
            if err:
                reasons.append('metaflac_error')
                metrics['error'] = err
            else:
                metrics.update(info)
                if info['sample_rate'] != EXPECTED_SR or info['channels'] != EXPECTED_CHANNELS or info['bits'] != EXPECTED_BITS:
                    reasons.append('bad_format')
                if info['duration_sec'] < MIN_DURATION or info['duration_sec'] > MAX_DURATION:
                    reasons.append('bad_duration')

        # Targeted DC offset check: prior sampling showed this is the main issue. Running it full
        # is slower, but still acceptable with one process and no temporary cache.
        if not reasons:
            dc, err = dc_offset_probe(ap)
            if err:
                reasons.append('astats_error')
                metrics['error'] = err
            else:
                metrics['dc_abs_max'] = dc
                if dc > MAX_DC_ABS:
                    reasons.append('dc_offset')

        if reasons:
            rejects[rid] = {'reasons': sorted(set(reasons)), 'metrics': metrics}
        checked += 1
        if checked % 5000 == 0:
            print(f"[{part['name']}] checked={checked}/{len(rows)} rejects={len(rejects)}", flush=True)

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

    clean_caps, reject_caps = [], []
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

    write_jsonl(clean_manifest_path, clean_rows)
    write_jsonl(reject_manifest_path, reject_rows)
    write_jsonl(clean_captions_path, clean_caps)
    write_jsonl(reject_captions_path, reject_caps)

    reason_counts = {}
    examples = []
    for rid, item in rejects.items():
        for reason in item['reasons']:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        if len(examples) < 50:
            examples.append({'id': rid, **item})
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
        'reject_rate': len(reject_rows) / len(rows) if rows else 0,
        'reason_counts': reason_counts,
        'thresholds': {
            'EXPECTED_SR': EXPECTED_SR,
            'EXPECTED_CHANNELS': EXPECTED_CHANNELS,
            'EXPECTED_BITS': EXPECTED_BITS,
            'MIN_DURATION': MIN_DURATION,
            'MAX_DURATION': MAX_DURATION,
            'MAX_DC_ABS': MAX_DC_ABS,
        },
        'reject_examples': examples,
    }
    tmp = report_path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(tmp, report_path)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def main():
    reports = []
    for part in PARTS:
        reports.append(clean_part(part))
    combined = {
        'input_rows': sum(r['input_rows'] for r in reports),
        'clean_rows': sum(r['clean_rows'] for r in reports),
        'rejected_rows': sum(r['rejected_rows'] for r in reports),
        'reject_rate': sum(r['rejected_rows'] for r in reports) / sum(r['input_rows'] for r in reports),
        'partitions': reports,
    }
    out = Path('/mnt/sdc/speech_dataset/spatial_speech_foa_tts_v1_work/quality_reports/qc_clean_combined_report.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(combined, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(tmp, out)
    print('[combined]')
    print(json.dumps(combined, ensure_ascii=False, indent=2), flush=True)

if __name__ == '__main__':
    main()
