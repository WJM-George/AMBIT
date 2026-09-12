"""Verified bounded memory reuse; no changes to renderer arithmetic or VAE RNG."""
from collections import OrderedDict
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import io
import os
from .artifact_io import canonical


@contextmanager
def memoized_sources(enabled=True):
    from scripts.t2a.data import materialize_sceneplan_transfusion_editing_targets as target
    import numpy as np
    if not enabled:
        yield {"load_hits": 0, "render_hits": 0}
        return
    load_original = target.load_complete_source
    render_original = target.render_complete_mono_source
    cache = OrderedDict()
    stored_bytes = 0
    limit = 96 * 1024**2
    stats = {"load_hits": 0, "render_hits": 0}

    def reuse(key, compute, counter):
        nonlocal stored_bytes
        if key in cache:
            value = cache.pop(key)
            cache[key] = value
            stats[counter] += 1
            return value[0].copy(), copy.deepcopy(value[1])
        value = compute()
        size = value[0].nbytes
        while cache and stored_bytes + size > limit:
            _, previous = cache.popitem(last=False)
            stored_bytes -= previous[0].nbytes
        if size <= limit:
            cache[key] = (value[0].copy(), copy.deepcopy(value[1]))
            stored_bytes += size
        return value

    def load(source, **kwargs):
        key = ("load", canonical([source, kwargs]))
        return reuse(key, lambda: load_original(source, **kwargs), "load_hits")

    def render(mono, **kwargs):
        value = np.ascontiguousarray(mono)
        key = ("render", value.dtype.str, value.shape,
               hashlib.sha256(memoryview(value).cast("B")).digest(), canonical(kwargs))
        return reuse(key, lambda: render_original(mono, **kwargs), "render_hits")

    target.load_complete_source = load
    target.render_complete_mono_source = render
    try:
        yield stats
    finally:
        target.load_complete_source = load_original
        target.render_complete_mono_source = render_original
        cache.clear()



@contextmanager
def memory_audio_reader(results):
    """Remove private byte payloads before the frozen manifest serializer runs."""
    from scripts.t2a.data import materialize_sceneplan_transfusion_editing_targets as target
    blobs = {}
    for result in results:
        blob = result.pop("_memory_foa_flac", None)
        if blob is not None:
            assert hashlib.sha256(blob).hexdigest() == result["target_foa_sha256"]
            blobs[result["target_foa_path"]] = blob
    original_read = target.sf.read
    paths = iter(blobs)
    pending = {}
    pool = ThreadPoolExecutor(max_workers=1,thread_name_prefix='editing_flac_predecode')

    def submit_one():
        path = next(paths,None)
        if path is not None:
            pending[path] = pool.submit(original_read,io.BytesIO(blobs[path]),
                                        dtype='float32',always_2d=True)

    # At most two batch-8 groups are prepared ahead. Decoded arrays remain on
    # CPU and follow the original row order; no VAE padding or RNG changes.
    for _ in range(min(16,len(blobs))):submit_one()

    def read(file, *args, **kwargs):
        if isinstance(file, (str, os.PathLike)) and str(file) in blobs:
            key=str(file)
            if not args and kwargs=={'dtype':'float32','always_2d':True} and key in pending:
                future=pending.pop(key)
                value=future.result()
                submit_one()
                return value
            file = io.BytesIO(blobs[str(file)])
        return original_read(file, *args, **kwargs)

    target.sf.read = read
    try:
        yield
    finally:
        target.sf.read = original_read
        pool.shutdown(wait=True,cancel_futures=True)
        pending.clear()
        blobs.clear()


