"""Movie tools — the *grounded* functions the LLM agent is allowed to call.

These are plain Python (no LLM). They are the single source of truth: the chat
agent can only ever surface movies these functions return, which stops the LLM
from hallucinating titles or ratings.

Each public method has a matching JSON schema in ``TOOL_SCHEMAS`` so it can be
exposed to OpenAI function calling. The same methods are reusable by the CLI or
tests without any LLM involved.
"""
from __future__ import annotations

import re
from typing import Any

from .data import MovieLens
from .recommenders.base import Recommender

LIKE_THRESHOLD = 4.0  # a rating >= this means the user "liked" the movie


class MovieTools:
    """Grounded lookups + recommendations over a MovieLens dataset."""

    def __init__(self, ml: MovieLens, model: Recommender, semantic=None):
        self.ml = ml
        self.model = model
        self.semantic = semantic  # optional SemanticSearch backend (may be None)
        self._title = ml.movies.set_index("movieId")["title"]
        self._genres = ml.movies.set_index("movieId")["genres"]
        self._n_ratings = ml.ratings.groupby("movieId").size()

    # --- helpers -----------------------------------------------------------

    def _movie_row(self, movie_id: int, **extra: Any) -> dict[str, Any]:
        return {
            "movieId": int(movie_id),
            "title": self._title.get(movie_id, str(movie_id)),
            "genres": list(self._genres.get(movie_id, [])),
            **extra,
        }

    # --- tools -------------------------------------------------------------

    def search_movies(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Find catalogue movies whose title matches ``query`` (case-insensitive).

        Ordered by number of ratings so the best-known match comes first.
        """
        mask = self.ml.movies["title"].str.contains(re.escape(query), case=False, na=False)
        hits = self.ml.movies[mask].copy()
        hits["n_ratings"] = hits["movieId"].map(self._n_ratings).fillna(0).astype(int)
        hits = hits.sort_values("n_ratings", ascending=False).head(limit)
        return [
            self._movie_row(row.movieId, year=(int(row.year) if row.year == row.year else None),
                            n_ratings=int(row.n_ratings))
            for row in hits.itertuples()
        ]

    def recommend_for_user(self, user_id: int, k: int = 10) -> list[dict[str, Any]]:
        """Top-K recommendations for an existing user (excludes movies they've seen)."""
        seen = set(self.ml.ratings.loc[self.ml.ratings["userId"] == user_id, "movieId"])
        rec_ids = self.model.recommend(user_id, k=k, exclude=seen)
        return [self._movie_row(mid) for mid in rec_ids]

    def similar_movies(self, movie_id: int, k: int = 10) -> list[dict[str, Any]]:
        """Movies frequently loved by the same people who loved ``movie_id``.

        Lightweight item-item co-occurrence: take everyone who rated the seed
        movie >= 4.0, count what *else* they rated >= 4.0, and return the most
        common. (Placeholder for a proper collaborative-filtering model.)
        """
        r = self.ml.ratings
        fans = set(r[(r["movieId"] == movie_id) & (r["rating"] >= LIKE_THRESHOLD)]["userId"])
        if not fans:
            return []
        others = r[
            r["userId"].isin(fans)
            & (r["movieId"] != movie_id)
            & (r["rating"] >= LIKE_THRESHOLD)
        ]
        counts = others.groupby("movieId").size().sort_values(ascending=False).head(k)
        return [self._movie_row(mid, co_fans=int(c)) for mid, c in counts.items()]

    def search_by_description(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        """Find movies whose plot/theme matches a free-text description.

        Uses semantic (embedding) search over TMDB overviews — answers queries
        collaborative filtering can't, e.g. "a heist with a clever twist".
        """
        if self.semantic is None:
            return [{"error": "semantic search unavailable (no description embeddings)"}]
        return self.semantic.search(query, k=k)

    # --- dispatch ----------------------------------------------------------

    def call(self, name: str, arguments: dict[str, Any]) -> Any:
        """Dispatch a tool call by name (used by the agent loop)."""
        if not hasattr(self, name) or name.startswith("_"):
            raise ValueError(f"unknown tool: {name!r}")
        return getattr(self, name)(**arguments)


# OpenAI function-calling schemas for the three tools above.
TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_movies",
            "description": "Find movies in the catalogue by (partial) title. Call this first to resolve any movie the user mentions into a movieId.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Title or part of a title, e.g. 'Harry Potter'."},
                    "limit": {"type": "integer", "description": "Max results (default 5).", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "similar_movies",
            "description": "Given a movieId, return movies that people who loved it also loved. Use this for 'I like X, what's similar?' requests.",
            "parameters": {
                "type": "object",
                "properties": {
                    "movie_id": {"type": "integer", "description": "The seed movieId (get it from search_movies)."},
                    "k": {"type": "integer", "description": "How many to return (default 10).", "default": 10},
                },
                "required": ["movie_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recommend_for_user",
            "description": "Top-K personalised recommendations for an existing userId (excludes what they've already seen).",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {"type": "integer", "description": "An existing userId."},
                    "k": {"type": "integer", "description": "How many to return (default 10).", "default": 10},
                },
                "required": ["user_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_by_description",
            "description": "Semantic search over movie plots/themes for free-text ideas like 'a heist with a clever twist' or 'lonely robot finds friendship'. Use when the user describes a vibe or plot rather than naming a movie.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Free-text description of the desired movie."},
                    "k": {"type": "integer", "description": "How many to return (default 5).", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
]
