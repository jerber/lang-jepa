"""Small statistics utilities used by evaluation and monitoring.

Kept dependency-light (only torch) so tests don't need to transitively pull in
the full evaluation stack (`datasets`, `rouge_score`, etc.).
"""
from __future__ import annotations

import torch


def spearman_rho(a: list[float], b: list[float]) -> float:
    """Spearman ρ via Pearson correlation on ranks.

    Degenerate case: if either input has zero variance (all values equal), ρ
    is undefined — we return 0.0. Ties are broken by argsort-of-argsort; for
    data with heavy ties, prefer scipy.stats.spearmanr.
    """
    at = torch.tensor(a, dtype=torch.float64)
    bt = torch.tensor(b, dtype=torch.float64)
    if at.numel() < 2 or at.std() == 0 or bt.std() == 0:
        return 0.0
    a_rank = torch.argsort(torch.argsort(at)).double()
    b_rank = torch.argsort(torch.argsort(bt)).double()
    a_rank -= a_rank.mean()
    b_rank -= b_rank.mean()
    denom = (a_rank.norm() * b_rank.norm()).item()
    if denom == 0:
        return 0.0
    return float((a_rank @ b_rank).item() / denom)
