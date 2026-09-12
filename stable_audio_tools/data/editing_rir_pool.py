"""Load a frozen RIR extension without changing logical reduction order."""
from contextlib import contextmanager
import importlib.util
from pathlib import Path
import sys

from .artifact_io import read, sha


class FrozenRIRPool:
    """Keep extension identity and cache scoped to the caller's predecessor."""

    def __init__(self, predecessor):
        self.root = Path(predecessor)
        self._native = None

    def native_module(self):
        if self._native is None:
            build = read(self.root / 'native/rir_pool1/BUILD.json')
            if sha(build['extension']) != build['extension_sha256']:
                raise RuntimeError('Frozen RIR extension checksum changed')
            spec = importlib.util.spec_from_file_location('_editing_rir_pool1', build['extension'])
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            self._native = module
        return self._native

    @contextmanager
    def bounded_rir_pool(self, enabled=True):
        if not enabled:
            yield
            return
        import pyroomacoustics as pra
        if pra.constants.get('num_threads') != 128:
            raise RuntimeError('Frozen source arithmetic requires 128 logical RIR reduction blocks')
        native = self.native_module()
        names = ['rir_builder', 'delay_sum', 'fractional_delay']
        originals = {name: getattr(pra.libroom, name) for name in names}
        # num_threads is a logical reduction count, not a physical thread limit.
        for name in names:
            setattr(pra.libroom, name, getattr(native, name))
        try:
            yield
        finally:
            for name in names:
                setattr(pra.libroom, name, originals[name])
