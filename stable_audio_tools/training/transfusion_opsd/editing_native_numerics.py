"""Use the Editing training contract's frozen-Qwen numerical policy in OPSD.

This is a process-local use of the native training implementation. It leaves
weights and installed dependencies untouched and fails on uncatalogued keys.
"""
import hashlib
import importlib
import json
from pathlib import Path


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def install_editing_native_numerics(recipe, output_directory, *, reference_rank='0'):
    """Return (receipt, finish); call before any frozen-Qwen forward.

    Reference rank is a declared policy choice, not the physical GPU number.
    Rank 0 is used for single-process experiments on either permitted GPU.
    """
    from scripts.t2a.experiments.ar_source_grounding_v1.numerics import install_autotune_observer
    rank=str(reference_rank)
    if rank not in recipe.get('numerics',{}):
        raise ValueError('The Editing run contract lacks the declared native numerical rank')
    spec=recipe['numerics'][rank]
    directory=Path(spec['directory']).resolve(strict=True)
    path=directory/'AUTOTUNE_CATALOG.json'
    if _sha(path)!=spec['sha256']:
        raise ValueError('The native Editing numerical catalog changed')
    catalog=json.loads(path.read_text())
    if catalog['mode']!='capture' or not catalog['records']:
        raise ValueError('A captured native numerical catalog is required')
    checked={}
    for record in catalog['records'].values():
        module_name=record['function'].rsplit('.',1)[0]
        if module_name in checked:
            if checked[module_name]['sha256']!=record['source_sha256']:
                raise ValueError('Conflicting source versions in native numerical catalog')
            continue
        module=importlib.import_module(module_name)
        source=Path(module.__file__).resolve(strict=True)
        actual=_sha(source)
        if actual!=record['source_sha256']:
            raise ValueError(f'Installed FLA source differs from native training: {module_name}')
        checked[module_name]=dict(path=str(source),sha256=actual)
    out=Path(output_directory)
    out.mkdir(parents=True,exist_ok=True)
    finish=install_autotune_observer(dict(name='native_numerics',mode='pin',reference_case=str(directory)),out)
    receipt=dict(contract='editing_OPSD_native_training_numerics_v1',reference_rank=rank,
        catalog=dict(path=str(path),sha256=spec['sha256']),catalog_keys=len(catalog['records']),
        installed_sources=checked,scope='Process-local native Autotuner hook; every encountered FLA key must match the captured configuration. Checkpoint parameters and installed packages unchanged.',
        implementation='scripts.t2a.experiments.ar_source_grounding_v1.numerics.install_autotune_observer',
        causal_claim='Fixes an omitted declared runtime dependency. It does not by itself prove that this omission caused every historical discrepancy or that OPSD improves audio.')
    return receipt,finish
