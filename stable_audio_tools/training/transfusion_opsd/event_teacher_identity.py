"""Bind a common teacher to the exact current EVENT training generation.

The immutable capability reference is separate from the evolving teacher.
Version zero remains readable, but a resumed student must recollect its own
prefixes and repairs from its complete checkpoint, never reuse an old pool.
"""
from pathlib import Path


def teacher_model_version(protocol):
    version = protocol.get('teacher_model_version', 0)
    if type(version) is not int or version < 0:
        raise ValueError('declare a nonnegative integer teacher model version')
    parent = protocol.get('teacher_parent')
    if parent is None:
        if version != 0:
            raise ValueError('a refreshed teacher requires its complete parent candidate')
        return version
    if (version == 0 or parent.get('model_version') != version
            or parent.get('model_fingerprint') != protocol['initial_model_fingerprint']
            or protocol.get('teacher_policy_kind') != 'candidate_response'
            or protocol.get('fresh_native_prefix_queries') is not True
            or not protocol.get('initialization_model_fingerprint')
            or not parent.get('dit_fingerprint')):
        raise ValueError('refreshed teacher lineage must identify the actual current native student')
    reference = protocol.get('original_native_reference')
    if not reference:
        raise ValueError('refreshed teachers must retain the immutable original reference')
    for artifact in (parent, reference):
        if not any(Path(entry['path']).resolve() == Path(artifact['path']).resolve()
                   and entry['sha256'] == artifact['sha256'] for entry in protocol['input_files']):
            raise ValueError('pin the teacher parent and original reference in protocol inputs')
    return version


def validate_teacher_binding(protocol, *, model_version, model_fingerprint,
                             parent_candidate=None, original_native_reference=None):
    """Reject stale pools even if an optimizer version was accidentally reused."""
    version = teacher_model_version(protocol)
    if version != model_version or protocol['initial_model_fingerprint'] != model_fingerprint:
        raise ValueError('cached teacher belongs to a different student version or model')
    teacher_parent = protocol.get('teacher_parent')
    if bool(teacher_parent) != bool(parent_candidate):
        raise ValueError('cached teacher and resumed student must share the same parent')
    if teacher_parent:
        keys = ('path', 'sha256', 'model_fingerprint', 'dit_fingerprint', 'model_version')
        if any(teacher_parent.get(key) != parent_candidate.get(key) for key in keys):
            raise ValueError('cached teacher and resumed student parent identity differs')
        if protocol['original_native_reference'] != original_native_reference:
            raise ValueError('teacher refresh cannot replace the original capability reference')
    return version


def validate_teacher_query(payload, protocol):
    version = teacher_model_version(protocol)
    if (payload.get('model_version') != version
            or payload.get('initial_model_fingerprint') != protocol['initial_model_fingerprint']):
        raise ValueError('cached prefix belongs to a different executor version')


def validate_student_feedback_action(payload, *, observed_action, model_version):
    # Older version-zero pools explicitly used the original action. New pools
    # record the actual decision before evaluating any future outcome.
    recorded = payload.get('student_feedback_action', 0 if model_version == 0 else None)
    if type(recorded) is not int or not 0 <= recorded < 7 or recorded != observed_action:
        raise ValueError('current student feedback decision differs from the recorded prefix')


def restore_teacher_parent(policy, protocol):
    """Restore full training state for collection, without taking an update."""
    import torch
    from .event_experiment import state_fingerprint
    from .event_trainer import EventFitConfig, EventRefinementTrainer
    from .provenance import sha256_file

    version = teacher_model_version(protocol)
    parent = protocol.get('teacher_parent')
    initializer = protocol.get('initialization_model_fingerprint', protocol['initial_model_fingerprint'])
    if state_fingerprint(policy) != initializer:
        raise ValueError('teacher complete initialization changed before parent restoration')
    if parent is None:
        return dict(model_version=version, complete_parent_restored=False)
    if sha256_file(parent['path']) != parent['sha256']:
        raise ValueError('teacher parent candidate changed')
    payload = torch.load(parent['path'], weights_only=True, map_location='cpu', mmap=True)
    collection = payload['collection_state']
    if (payload['commits'] != version or collection.get('next_collection_version') != version
            or collection.get('original_native_reference') != protocol['original_native_reference']):
        raise ValueError('teacher parent version or immutable reference differs before restore')
    trainer = EventRefinementTrainer(policy, identity=payload['identity'], config=EventFitConfig(**payload['config']))
    restored = trainer.restore_candidate(parent['path'])
    if (restored != collection or trainer.commits != version
            or state_fingerprint(policy) != parent['model_fingerprint']
            or state_fingerprint(policy.bundle.diffusion) != parent['dit_fingerprint']):
        raise ValueError('complete teacher parent failed to restore exactly')
    return dict(model_version=version, complete_parent_restored=True,
        parent_candidate=parent, optimizer_entries=len(trainer.optimizer.state),
        fit_attempts=trainer.fit_attempts, original_native_reference=collection['original_native_reference'])
