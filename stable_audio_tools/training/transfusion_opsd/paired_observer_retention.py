"""Compare matching observer views; different views are not interchangeable."""
from __future__ import annotations

import math
from collections.abc import Mapping


def paired_error_retention(before: Mapping[str, float],
                           after: Mapping[str, float], *, tolerance=1e-12) -> dict:
    """Require every explicitly identified observer view to avoid regression.

    The caller must match the request, reference transcript, execution noise
    and observer configuration. View-to-view variability must not reject an
    identical observation. Passing is finite observer evidence, not a proof
    that all actual speech content is correct.
    """
    if (not before or set(before) != set(after)
            or any(not isinstance(key, str) or not key for key in before)
            or not math.isfinite(tolerance) or tolerance < 0):
        raise ValueError('Require matching named views and a finite tolerance.')
    rows=[]
    for key in sorted(before):
        old,new=float(before[key]),float(after[key])
        if not math.isfinite(old) or not math.isfinite(new) or old < 0 or new < 0:
            raise ValueError('Observer errors must be finite and nonnegative.')
        rows.append(dict(view=key,before=old,after=new,change=new-old,
                         passed=new<=old+tolerance))
    return dict(passed=all(row['passed'] for row in rows),rows=rows,
                scope='Per-view paired error retention; no averaging away a failed view.')
