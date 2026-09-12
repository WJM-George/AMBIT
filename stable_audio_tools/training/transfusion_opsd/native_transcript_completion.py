"""Paired transcript completions under the student's actual structural prefix.

Only the prefix before the first text mismatch is student visited. Later text
prefixes are explicitly teacher forced from the same source's paired transcript.
These targets are paired supervision, not a self-teacher or an audio reward.
The caller establishes sample/instruction/latent provenance and source binding.
"""
from __future__ import annotations

from .native_prefix_supervision import _sites


def build_native_transcript_completions(codec, native_tokens, paired_plan, allowed_fn):
    """Correct a complete text suffix without labeling the wrong continuation.

Each returned sequence changes exactly one source-bound transcript. Structural
tokens, other texts, and the original suffix after text_end remain unchanged.
Loss applies from the first differing token through the paired text_end; callers
average within each transcript and then across transcripts, as for frontier CE.
"""
    old=list(map(int,native_tokens))
    truth=codec.encode(paired_plan)['input_ids'].tolist()
    old_atoms,old_texts=_sites(codec,old)
    truth_atoms,truth_texts=_sites(codec,truth)
    result=[]
    for field,positions in old_texts.items():
        if field[1]!='<transcript>' or field not in truth_texts:
            continue
        source=field[0]
        bindings=[(source,'identity'),(source,'<kind>')]
        if any(key not in old_atoms or key not in truth_atoms
               or old[old_atoms[key]]!=truth[truth_atoms[key]] for key in bindings):
            raise ValueError('Transcript completion requires the same paired source identity and kind.')
        desired=[truth[pos] for pos in truth_texts[field]]
        actual=[old[pos] for pos in positions]
        if desired==actual:
            continue
        common=0
        while common<min(len(actual),len(desired)) and actual[common]==desired[common]:
            common+=1
        start,end=positions[0],positions[-1]+1
        corrected=old[:start]+desired+old[end:]
        first=start+common
        assert corrected[:first]==old[:first]
        targets=[]
        for pos in range(1,len(corrected)):
            allowed=sorted(allowed_fn(corrected[:pos]))
            if corrected[pos] not in allowed:
                raise ValueError('Paired transcript is incompatible with the native structural prefix or suffix.')
            if first<=pos<start+len(desired) and len(allowed)>1:
                targets.append(dict(position=pos,token_id=corrected[pos],allowed_ids=allowed,
                    field='/'.join(field),role='paired_transcript_completion',
                    prefix_origin='student_visited' if pos==first else 'paired_teacher_forced'))
        if targets:
            result.append(dict(field='/'.join(field),tokens=corrected,targets=targets,
                first_divergence=first,original_text_tokens=actual,paired_text_tokens=desired,
                unchanged_prefix_tokens=first,original_suffix_start=end,
                corrected_suffix_start=start+len(desired),
                scope='Same-source paired transcript under the actual student structural prefix. Only the first corrective prefix is student visited; subsequent prefixes are paired teacher forcing. No supervision on unrelated fields.'))
    return result
