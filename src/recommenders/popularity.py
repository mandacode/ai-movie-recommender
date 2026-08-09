"""Popularity baseline recommender.

Recommends the same globally-popular movies to everyone (minus items the user
has already seen). Simple, but a genuinely strong baseline that personalised
models must beat.

Scoring modes:
    * ``count``    — rank by number of ratings (raw popularity).
    * ``mean``     — rank by average rating (quality, noisy for rare movies).
    * ``weighted`` — IMDB-style Bayesian average that blends quality with
                     volume; the sensible default.
"""
from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from .base import Recommender


class PopularityRecommender(Recommender):
    name = "popularity"

    def __init__(self, scoring: str = "weighted", min_votes_quantile: float = 0.90):
        if scoring not in {"count", "mean", "weighted"}:
            raise ValueError(f"unknown scoring mode: {scoring!r}")
        self.scoring = scoring
        self.min_votes_quantile = min_votes_quantile
        self._ranking: list[int] = []  # movieIds, best first

    def fit(self, train: pd.DataFrame) -> "PopularityRecommender":
        stats = train.groupby("movieId")["rating"].agg(count="count", mean="mean")

        if self.scoring == "count":
            stats["score"] = stats["count"]
        elif self.scoring == "mean":
            stats["score"] = stats["mean"]
        else:  # weighted (Bayesian average)
            C = train["rating"].mean()               # global mean rating
            m = stats["count"].quantile(self.min_votes_quantile)  # prior strength
            v = stats["count"]
            R = stats["mean"]
            stats["score"] = (v / (v + m)) * R + (m / (v + m)) * C

        self._ranking = (
            stats.sort_values("score", ascending=False).index.tolist()
        )
        return self

    def recommend(
        self,
        user_id: int,
        k: int,
        exclude: Iterable[int] | None = None,
    ) -> list[int]:
        exclude = set(exclude or ())
        out: list[int] = []
        for movie_id in self._ranking:
            if movie_id in exclude:
                continue
            out.append(movie_id)
            if len(out) == k:
                break
        return out
