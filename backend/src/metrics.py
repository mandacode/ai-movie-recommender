"""Ranking metrics for top-K recommendation.

All three metrics compare an *ordered* list of recommended movieIds against a
*set* of items the user actually found relevant in the held-out test period.

Terminology
-----------
* **relevant**: an item the user genuinely liked in the test set. We treat a
  test rating >= `relevance_threshold` (default 4.0) as relevant/positive.
* **top-K**: the K highest-ranked items our recommender returned.

Why these three?
    Precision@K — of what we showed, how much was good? (user's screen is finite)
    Recall@K    — of everything good, how much did we surface? (coverage)
    NDCG@K      — same, but rewards putting the good items *near the top*.
"""
from __future__ import annotations

import math
from collections.abc import Sequence


def precision_at_k(recommended: Sequence[int], relevant: set[int], k: int) -> float:
    """Fraction of the top-K recommendations that are relevant.

    P@K = |top-K ∩ relevant| / K
    """
    if k == 0:
        return 0.0
    top_k = recommended[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / k


def recall_at_k(recommended: Sequence[int], relevant: set[int], k: int) -> float:
    """Fraction of all relevant items that appear in the top-K.

    R@K = |top-K ∩ relevant| / |relevant|
    """
    if not relevant:
        return 0.0
    top_k = recommended[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / len(relevant)


def ndcg_at_k(recommended: Sequence[int], relevant: set[int], k: int) -> float:
    """Normalised Discounted Cumulative Gain with binary relevance.

    DCG@K  = Σ_{i=1..K}  rel_i / log2(i + 1)      # rel_i ∈ {0, 1}
    IDCG@K = DCG of the ideal ranking (all relevant items first)
    NDCG@K = DCG@K / IDCG@K   ∈ [0, 1]

    The 1/log2(i+1) term is the "discount": a relevant hit at rank 1 is worth
    more than the same hit at rank 10. This is what makes NDCG rank-aware,
    unlike precision/recall which only care about set membership.
    """
    top_k = recommended[:k]
    dcg = sum(
        1.0 / math.log2(rank + 2)  # rank is 0-indexed → position i = rank+1 → log2(i+1)=log2(rank+2)
        for rank, item in enumerate(top_k)
        if item in relevant
    )
    # Ideal DCG: as many leading 1s as there are relevant items (capped at k).
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0
