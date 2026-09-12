"""Serialize short reservation transactions, outside all render/encode work."""
from contextlib import contextmanager
from pathlib import Path
import fcntl
import os
import sqlite3
import threading
import time


_local = threading.local()


def _connection(root):
    # A connection per process/thread keeps WAL open between rows. No transaction
    # remains open while rendering; this also avoids a last-close checkpoint for
    # every individual signature update.
    key = (os.getpid(), str(Path(root).resolve()))
    if getattr(_local, 'key', None) != key:
        old = getattr(_local, 'con', None)
        if old is not None:
            old.close()
        con = sqlite3.connect(Path(root) / 'RESERVATIONS.sqlite',
                              timeout=1, isolation_level=None)
        con.execute('PRAGMA synchronous=FULL')
        _local.key, _local.con = key, con
    return _local.con


def _busy_retry(call, deadline):
    delay = .02
    while True:
        try:
            return call()
        except sqlite3.OperationalError as exc:
            code = getattr(exc, 'sqlite_errorcode', None)
            busy = ((code & 255) in (5, 6)) if code is not None else (
                str(exc) in ('database is locked', 'database table is locked'))
            if not busy or time.monotonic() >= deadline:
                raise
            time.sleep(delay)
            delay = min(.5, delay * 2)


@contextmanager
def transaction(root, *, write=True, busy_seconds=120):
    # All reservation users take this OS lock before opening a SQLite
    # transaction. Contenders wait here rather than repeatedly taking SQLite's
    # sole writer slot. The kernel releases the lock if a worker dies.
    with (Path(root) / 'RESERVATIONS.writer.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        deadline = time.monotonic() + busy_seconds
        con = _busy_retry(lambda: _connection(root), deadline)
        assert not con.in_transaction
        _busy_retry(lambda: con.execute('BEGIN IMMEDIATE' if write else 'BEGIN'), deadline)
        try:
            yield con
            _busy_retry(con.commit, deadline)
        except BaseException:
            con.rollback()
            raise


def source_ordinal(root, split, ordinal):
    with transaction(root, write=False) as con:
        row = con.execute('SELECT source_pair_ordinal FROM assignments WHERE split=? AND ordinal=?',
                          (split, ordinal)).fetchone()
        assert row is not None, (split, ordinal)
        return row[0]


def reserve_source(root, split, ordinal, source, sample_id):
    with transaction(root) as con:
        count = con.execute('SELECT count(*) FROM assignments WHERE split=? AND source_pair_ordinal=?',
                            (split, source)).fetchone()[0]
        if count >= 2:
            return False
        changed = con.execute('UPDATE assignments SET source_pair_ordinal=?,source_sample_id=?,signature=NULL '
                              'WHERE split=? AND ordinal=?', (source, sample_id, split, ordinal))
        assert changed.rowcount == 1, (split, ordinal)
        return True


def reserve_signature(root, split, ordinal, source, signature):
    try:
        with transaction(root) as con:
            row = con.execute('SELECT source_pair_ordinal FROM assignments WHERE split=? AND ordinal=?',
                              (split, ordinal)).fetchone()
            assert row is not None and row[0] == source, (split, ordinal, source, row)
            changed = con.execute('UPDATE assignments SET signature=? WHERE split=? AND ordinal=?',
                                  (signature, split, ordinal))
            assert changed.rowcount == 1
        return True
    except sqlite3.IntegrityError:
        return False
