"""Retire explicitly reviewed OPSD states after preserving selected model weights.

No directory-wide deletion or selection by age. The input plan lists exact
inode identities, selected exports, protected paths, and retained evidence.
Model-only exports cannot resume Adam, RNG, or sampler state.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def write(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temp.replace(path)


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def model_digest(named):
    import torch
    h = hashlib.sha256()
    for name, value in sorted(named.items()):
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def check_identity(item):
    path = Path(item['path'])
    if path.is_symlink():
        raise ValueError('Do not retire symlinks: ' + str(path))
    stat = path.stat()
    actual = dict(size=stat.st_size, inode=stat.st_ino, device=stat.st_dev, mtime_ns=stat.st_mtime_ns)
    if any(actual[k] != item[k] for k in actual):
        raise ValueError('Reviewed file changed: ' + str(path))
    return stat


def assert_not_open(identities):
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            fds = list((proc / 'fd').iterdir())
        except (FileNotFoundError, PermissionError):
            continue
        for fd in fds:
            try:
                stat = fd.stat()
            except (FileNotFoundError, PermissionError):
                continue
            if (stat.st_dev, stat.st_ino) in identities:
                raise RuntimeError('A reviewed checkpoint is open in process ' + proc.name)
        try:
            maps = (proc / 'maps').read_text()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        for line in maps.splitlines():
            fields = line.split(maxsplit=5)
            if len(fields) >= 5 and ':' in fields[3]:
                major, minor = (int(x, 16) for x in fields[3].split(':'))
                if (os.makedev(major, minor), int(fields[4])) in identities:
                    raise RuntimeError('A reviewed checkpoint is memory-mapped in process ' + proc.name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    root, report = Path(plan['root']).resolve(), args.plan.parent
    deletes = plan['deletions']
    identities = set()
    protected = [Path(p).resolve() for p in plan['protected_paths']]
    if len({d['path'] for d in deletes}) != len(deletes):
        raise ValueError('Repeated deletion path.')
    for item in deletes:
        path = Path(item['path']).resolve()
        if not path.is_relative_to(root) or path.suffix != '.pt':
            raise ValueError('Only reviewed .pt states in this OPSD root may be retired.')
        if any(path == p or path.is_relative_to(p) for p in protected):
            raise ValueError('Protected path in deletion list: ' + str(path))
        st = check_identity(item)
        identities.add((st.st_dev, st.st_ino))
    for item in plan['evidence']:
        if digest_file(item['path']) != item['sha256']:
            raise ValueError('Reviewed evidence changed: ' + item['path'])
    assert_not_open(identities)
    if not args.apply:
        print(json.dumps(dict(validated=True, deletion_paths=len(deletes), unique_inodes=len(identities),
                              exports=len(plan['exports']), files_deleted=0)))
        return
    receipt_path = report / 'EXECUTION.json'
    if receipt_path.exists():
        raise ValueError('An execution receipt already exists; do not repeat a retirement.')
    import torch
    torch.set_num_threads(4)
    before = os.statvfs(root)
    receipt = dict(phase='EXPORTING', started_unix=time.time(), plan_sha256=digest_file(args.plan),
                   free_before=before.f_bavail * before.f_frsize, exports=[], deletions=[], extents={})
    write(receipt_path, receipt)
    for entry in plan['exports']:
        source, destination = Path(entry['source']), Path(entry['destination'])
        if destination.exists():
            raise ValueError('Never overwrite a retained model: ' + str(destination))
        state = torch.load(source, map_location='cpu', weights_only=False, mmap=True)
        fingerprint = model_digest(state['model'])
        if fingerprint != state['model_sha256'] or state['step'] != entry['step']:
            raise ValueError('Checkpoint model/step fingerprint mismatch: ' + str(source))
        selected = {key: state[key] for key in ('model', 'model_sha256', 'step', 'original_checkpoint',
                    'initial_overlay', 'config_sha256', 'training_source_sha256') if key in state}
        selected.update(schema='editing_opsd_model_only_v1', resume_supported=False,
                        source_checkpoint=str(source), source_checkpoint_sha256=digest_file(source))
        destination.parent.mkdir(exist_ok=True, parents=True)
        temp = destination.with_suffix('.tmp.pt')
        torch.save(selected, temp)
        reloaded = torch.load(temp, map_location='cpu', weights_only=False, mmap=True)
        if model_digest(reloaded['model']) != fingerprint or 'optimizer' in reloaded:
            raise ValueError('Model-only export failed verification.')
        del reloaded, selected, state
        gc.collect()
        temp.replace(destination)
        record = dict(**entry, model_sha256=fingerprint, sha256=digest_file(destination),
                      bytes=destination.stat().st_size, supports_inference=True, supports_optimizer_resume=False)
        receipt['exports'].append(record)
        write(receipt_path, receipt)
        print('Verified model-only export:', destination, flush=True)
    # Verify again after export, before any unlink. Preserve an extent record
    # per inode; hardlink aliases do not multiply the disk-space accounting.
    assert_not_open(identities)
    for item in deletes:
        stat = check_identity(item)
        key = f'{stat.st_dev}:{stat.st_ino}'
        if key not in receipt['extents']:
            fiemap = subprocess.run(['xfs_io', '-r', '-c', 'fiemap -v', item['path']],
                                    check=True, capture_output=True, text=True).stdout
            receipt['extents'][key] = dict(path=item['path'], allocated=stat.st_blocks * 512,
                                          nlink=stat.st_nlink, fiemap=fiemap,
                                          sha256=digest_file(item['path']))
    receipt['phase'] = 'RETIRING'; write(receipt_path, receipt)
    for item in deletes:
        check_identity(item)
        Path(item['path']).unlink()
        receipt['deletions'].append(dict(path=item['path'], reason=item['reason'],
                                        logical_bytes=item['size'], inode=item['inode']))
        write(receipt_path, receipt)
    os.sync()
    after = os.statvfs(root)
    receipt.update(phase='COMPLETE', completed_unix=time.time(),
                   free_after=after.f_bavail * after.f_frsize,
                   filesystem_free_increase=after.f_bavail * after.f_frsize - receipt['free_before'],
                   logical_unlinked_bytes=sum(x['size'] for x in deletes),
                   unique_inode_allocated_bytes=sum(x['allocated'] for x in receipt['extents'].values()))
    write(receipt_path, receipt)
    print(json.dumps({k:receipt[k] for k in ('phase','filesystem_free_increase','logical_unlinked_bytes')},indent=2))


if __name__ == '__main__':
    main()
