"""Shared artifact serialization and immutable-index access (standard library only)."""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import zlib


def now():
    return datetime.now().astimezone().isoformat()


def canonical(value, *, allow_nan=False):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=allow_nan)


def digest(value, *, allow_nan=False):
    return hashlib.sha256(canonical(value, allow_nan=allow_nan).encode('utf-8')).hexdigest()


def sha(path):
    checksum = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            checksum.update(block)
    return checksum.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    """Preserve the research workflows' flushed, atomic JSON replacement."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.tmp.{os.getpid()}')
    with temporary.open('w') as handle:
        handle.write(canonical(value) + '\n')
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def pack(value):
    return zlib.compress(canonical(value).encode(), 6)


def unpack(value):
    return json.loads(zlib.decompress(value))


def db(path):
    """Only use with finalized databases; immutable mode ignores WAL updates."""
    connection = sqlite3.connect(f'file:{Path(path)}?mode=ro&immutable=1', uri=True)
    connection.row_factory = sqlite3.Row
    return connection
