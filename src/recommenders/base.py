"""The pluggable recommender interface.

Every model — the popularity baseline today, collaborative filtering and
XGBoost later — implements this same two-method contract. The evaluation
harness only ever talks to `fit` and `recommend`, so swapping models never
touches the split, metrics, or evaluation code.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable

import pandas as pd


class Recommender(ABC):
    """Abstract base class for all recommenders."""

    #: Human-readable name used in reports.
    name: str = "base"

    @abstractmethod
    def fit(self, train: pd.DataFrame) -> "Recommender":
        """Learn from the training ratings. Returns ``self`` for chaining.

        Args:
            train: rating events with at least ``userId``, ``movieId``,
                ``rating`` columns.
        """
        raise NotImplementedError

    @abstractmethod
    def recommend(
        self,
        user_id: int,
        k: int,
        exclude: Iterable[int] | None = None,
    ) -> list[int]:
        """Return the top-``k`` recommended ``movieId``s for a user, ranked.

        Args:
            user_id: the user to recommend for.
            k: number of items to return.
            exclude: movieIds to omit (e.g. already seen in training).
        """
        raise NotImplementedError
