"""Locate explicitly numbered request sources without consulting target plans.

This extracts only text spans and ordinal identities. It does not parse/copy
descriptions, spatial values, event times, or a reference ScenePlan. Ambiguous
or unnumbered requests return no spans and can retain ordinary global attention.
"""
from __future__ import annotations
import re
from typing import NamedTuple, Sequence

_LEAD = re.compile(
    r'\b(?:source\s*#?\s*(?P<number>[1-4])|(?P<ordinal>first|second|third|fourth)\s+source)'
    r'\s*:\s*(?P<kind>sound|music|speech)\s*;', re.IGNORECASE)
_ORDINALS = {'first':1,'second':2,'third':3,'fourth':4}


class RequestSourceSpan(NamedTuple):
    source_number: int
    kind: str
    start: int
    end: int


def request_source_spans(text: str) -> tuple[RequestSourceSpan, ...]:
    """Accept complete, contiguous explicit source clauses outside quotations."""
    if not isinstance(text,str) or not text:
        return ()
    # The existing request surface wraps free text in curly quotation marks;
    # quoted occurrences of 'Source 2: sound;' must never become source headers.
    quote_depth=0
    outside=[]
    for character in text:
        outside.append(quote_depth==0)
        if character=='“':quote_depth+=1
        elif character=='”':
            if quote_depth==0:return ()
            quote_depth-=1
    if quote_depth:return ()
    matches=[match for match in _LEAD.finditer(text) if outside[match.start()]]
    if not 1<=len(matches)<=4:return ()
    numbers=[int(m['number']) if m['number'] else _ORDINALS[m['ordinal'].lower()] for m in matches]
    if numbers!=list(range(1,len(matches)+1)):return ()
    spans=[]
    for index,match in enumerate(matches):
        stop=matches[index+1].start() if index+1<len(matches) else len(text)
        parentheses=0
        end=None
        for position in range(match.end(),stop):
            if not outside[position]:continue
            char=text[position]
            if char=='(':parentheses+=1
            elif char==')':
                parentheses-=1
                if parentheses<0:return ()
            elif char=='.' and parentheses==0:
                decimal=position>0 and position+1<len(text) and text[position-1].isdigit() and text[position+1].isdigit()
                if not decimal:
                    end=position+1
                    break
        if end is None or parentheses!=0:return ()
        spans.append(RequestSourceSpan(numbers[index],match['kind'].lower(),match.start(),end))
    return tuple(spans)


def source_ids_for_offsets(offsets: Sequence[Sequence[int]], spans: Sequence[RequestSourceSpan]) -> list[int]:
    """Map tokenizer character offsets to 0=global or 1..4=explicit source.

    Boundary tokens receive the source with greatest positive character overlap;
    padding/special (0,0) offsets remain global. No target data is accepted.
    """
    result=[]
    for start,end in offsets:
        start,end=int(start),int(end)
        if start<0 or end<start:raise ValueError('invalid tokenizer character offset')
        overlap=[(max(0,min(end,span.end)-max(start,span.start)),span.source_number) for span in spans]
        best=max(overlap,default=(0,0))
        result.append(best[1] if best[0]>0 else 0)
    return result
