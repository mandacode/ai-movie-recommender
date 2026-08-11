"""Model-agnostic evaluation harness.

Given any `Recommender`, this loops over every test user, asks the model for
top-K recommendations, and averages the ranking metrics across users
(so-called *macro* averaging — every user counts equally). Because it only
calls `recommend`, the exact same harness evaluates the popularity baseline
today and collaborative filtering / XGBoost tomorrow.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from . import metrics
from .recommenders.base import Recommender


@dataclass
class EvalResult:
    model: str
    k: int
    precision: float
    recall: float
    ndcg: float
    n_users: int

    def __str__(self) -> str:
        return (
            f"{self.model:<20} @{self.k:<3} "
            f"P={self.precision:.4f}  R={self.recall:.4f}  NDCG={self.ndcg:.4f}  "
            f"(users={self.n_users})"
        )

    def to_dict(self) -> dict[str, float]:
        """Serialise to the persisted JSON shape (keys carry the K cutoff)."""
        return {
            "k": self.k,
            f"precision@{self.k}": round(self.precision, 4),
            f"recall@{self.k}": round(self.recall, 4),
            f"ndcg@{self.k}": round(self.ndcg, 4),
        }


def evaluate(
    model: Recommender,
    train: pd.DataFrame,
    test: pd.DataFrame,
    k: int = 10,
    relevance_threshold: float = 4.0,
) -> EvalResult:
    """Evaluate a fitted model on the test set.

    Args:
        model: a *fitted* recommender.
        train: training ratings (used to exclude already-seen items).
        test: held-out ratings.
        k: cutoff for the top-K metrics.
        relevance_threshold: a test rating >= this counts as relevant.
    """
    # Items each user already saw in training — never re-recommend these.
    seen = train.groupby("userId")["movieId"].agg(set)

    # Relevant (liked) test items per user = the ground truth we score against.
    liked = test[test["rating"] >= relevance_threshold]
    relevant_by_user = liked.groupby("userId")["movieId"].agg(set)

    p_sum = r_sum = n_sum = 0.0
    n_users = 0

    for user_id, relevant in relevant_by_user.items():
        if not relevant:
            continue  # user has no positive items in test → nothing to measure
        recs = model.recommend(user_id, k=k, exclude=seen.get(user_id, set()))
        p_sum += metrics.precision_at_k(recs, relevant, k)
        r_sum += metrics.recall_at_k(recs, relevant, k)
        n_sum += metrics.ndcg_at_k(recs, relevant, k)
        n_users += 1

    if n_users == 0:
        return EvalResult(model.name, k, 0.0, 0.0, 0.0, 0)

    return EvalResult(
        model=model.name,
        k=k,
        precision=p_sum / n_users,
        recall=r_sum / n_users,
        ndcg=n_sum / n_users,
        n_users=n_users,
    )
