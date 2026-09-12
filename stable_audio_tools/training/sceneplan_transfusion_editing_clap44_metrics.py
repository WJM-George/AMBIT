"""Bounded-memory retrieval and binding diagnostics, never a promotion gate."""
from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import torch
from torch.nn import functional as F


@torch.no_grad()
def retrieval_metrics(queries: torch.Tensor, candidates: torch.Tensor, query_keys: Sequence[str], candidate_keys: Sequence[str], *, chunk_size: int = 128):
    if queries.ndim != 2 or candidates.ndim != 2 or queries.shape[-1] != candidates.shape[-1] or len(query_keys) != len(queries) or len(candidate_keys) != len(candidates) or not len(queries) or chunk_size < 1:
        raise ValueError("invalid CLAP44 retrieval inputs")
    if not bool(torch.isfinite(queries).all() and torch.isfinite(candidates).all()):
        raise ValueError("non-finite CLAP44 retrieval embeddings")
    if bool((queries.norm(dim=-1) < 1e-8).any() or (candidates.norm(dim=-1) < 1e-8).any()):
        raise ValueError("zero CLAP44 retrieval embedding")
    groups = defaultdict(list)
    for i, key in enumerate(candidate_keys): groups[key].append(i)
    if any(key not in groups for key in query_keys):
        raise ValueError("retrieval query has no positive candidate")
    candidates = F.normalize(candidates.float(), dim=-1).to(queries.device)
    ranks = []
    for start in range(0, len(queries), chunk_size):
        scores = F.normalize(queries[start:start + chunk_size].float(), dim=-1) @ candidates.T
        for i, key in enumerate(query_keys[start:start + chunk_size]):
            positive_indices = groups[key]
            best = scores[i, positive_indices].max()
            # Pessimistic ties: a collapsed encoder must not score R@1=1.
            tied_or_better = scores[i] >= best
            tied_or_better[positive_indices] = False
            ranks.append(1 + int(tied_or_better.sum()))
    rank = torch.tensor(ranks, dtype=torch.float64)
    return {"queries": len(query_keys), "candidates": len(candidate_keys), "unique_candidate_keys": len(groups), "r_at_1": float((rank <= 1).double().mean()), "r_at_5": float((rank <= 5).double().mean()), "r_at_10": float((rank <= 10).double().mean()), "mrr": float(rank.reciprocal().mean()), "mean_rank": float(rank.mean()), "tie_policy": "pessimistic_nonpositive_ties", "positive_policy": "all_equal_supervision_keys", "unrelated_candidate_filtering": False}


def margin_summary(margins: Sequence[float]):
    if not margins:
        return {"n": 0, "accuracy": None, "wilson95": None, "mean_margin": None}
    x = torch.tensor(margins, dtype=torch.float64)
    if not bool(torch.isfinite(x).all()): raise ValueError("non-finite discrimination margin")
    n = len(x); p = float((x > 0).double().mean()); z = 1.959963984540054
    denominator = 1 + z*z/n
    center = (p + z*z/(2*n))/denominator
    radius = z * ((p*(1-p)/n + z*z/(4*n*n)) ** 0.5)/denominator
    return {"n": n, "accuracy": p, "wilson95": [center-radius, center+radius], "mean_margin": float(x.mean()), "ties_count_as_correct": False}
