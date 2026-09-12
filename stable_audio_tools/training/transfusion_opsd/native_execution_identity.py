"""Prove unchanged execution without claiming that its content is correct.

ASR intervals from the very same observation are coupled, not independent
uncertain measurements. This wrapper only resolves that exact-identity case.
Changed audio keeps the original guard result, including uncertainty.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path


def _file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def identity_aware_execution_protection(after, before, original_guard):
    """Resolve only ASR self-comparison ambiguity using verified exact bytes.

    The caller supplies the ordinary guard from its versioned observer. Same
    request, plan and every scored observation must agree. Both audio paths
    must match their declared SHA256. Real failures, different observations
    and nonidentical files do not gain any exception.
    """
    result = copy.deepcopy(original_guard)
    if (result['passed'] or result['failures']
            or result['uncertain'] != ['asr_observed_bounds']):
        return result
    fields = ('sample_id', 'plan', 'plan_admissible', 'costs', 'coarse', 'content', 'independent_ctc')
    if any(key not in after or key not in before or after[key] != before[key] for key in fields):
        return result
    if not after['plan_admissible']:
        return result
    a, b = after.get('audio', {}), before.get('audio', {})
    digest = a.get('sha256')
    if (not isinstance(digest, str) or len(digest) != 64
            or any(c not in '0123456789abcdef' for c in digest)
            or b.get('sha256') != digest or not a.get('path') or not b.get('path')):
        return result
    for path in {a['path'], b['path']}:
        if _file_digest(path) != digest:
            raise ValueError('Audio evidence digest mismatch; identity is not established.')
    observation = json.dumps({key: after[key] for key in fields}, sort_keys=True,
                             separators=(',', ':'), allow_nan=False)
    result.update(passed=True, failures=[], uncertain=[], original_guard=copy.deepcopy(original_guard),
        identity_nonregression=dict(contract='same_audio_and_observation_nonregression_v1',
            audio_sha256=digest, verified_paths=sorted({a['path'], b['path']}),
            observation_sha256=hashlib.sha256(observation.encode()).hexdigest(),
            content_correctness_certified=False,
            scope='Exact same execution has not regressed. Original ASR ambiguity and incorrect-word '
                  'observations remain; this does not establish a successful output or a repair.'))
    return result
