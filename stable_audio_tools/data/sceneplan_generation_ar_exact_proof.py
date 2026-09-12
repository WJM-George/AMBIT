"""Sufficient positive proofs that avoid irrelevant semantic-judge calls.

Exact text is only a sufficient shortcut. Every unresolved scene still uses
the same semantic judge, including synonyms and paraphrases. No nonexact pair
is labeled false by this optimization.
"""
from stable_audio_tools.data.sceneplan_generation_ar_natural_constraints import evaluate_natural_request


def exact_core_labels(requirements, prediction):
    labels = {}
    for ref in requirements['sources']:
        for source in (prediction or {}).get('sources', []):
            if source['kind'] != ref['kind']: continue
            field = 'speaker_description' if ref['kind'] == 'speech' else 'description'
            if ' '.join(ref['core'].split()) == ' '.join(source[field].split()):
                labels[ref['key'], source['source_id']] = True
    return labels


def prove_exact_satisfaction(request, requirements, prediction):
    """Return a fully satisfied coupled assignment, or leave the scene open.

    Its score already attains the maximum possible number of semantic matches,
    satisfied sources, and field/relation checks. Extra positive semantic edges
    can only produce equally complete assignments, so all reported Boolean
    request metrics remain unchanged. Missing/extra-source and partly wrong
    scenes deliberately stay open for the normal matching and semantic review.
    """
    labels = exact_core_labels(requirements, prediction)
    score = evaluate_natural_request(request, requirements, prediction, labels)
    if not score['request_constraints_joint']: return None
    assert score['valid'] and score['count_correct'] and not score['semantic_pending']
    return {'status': 'PROVEN_REQUEST_CONSTRAINTS_SATISFIED', 'assignment': score['assignment'],
        'source_count': score['requested_count'], 'completion_review': 'SEPARATE',
        'reason': 'One coupled assignment already satisfies all requested semantic, count, field, scene and relation constraints.'}
