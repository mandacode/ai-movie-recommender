"""Semantic search over TMDB description embeddings.

Enables free-text queries ("a heist with a clever twist") that collaborative
filtering cannot answer — it matches the query embedding against precomputed
movie-description embeddings. Loads lazily and degrades gracefully if the
embedding artifacts aren't present (they're regenerable via
scripts/fetch_tmdb.py + scripts/embed_descriptions.py).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_EMB = Path("datasets/ml-32m-sample/desc_emb.npz")
DEFAULT_MOVIES = Path("datasets/ml-32m-sample/movies.csv")
DEFAULT_MODEL = "all-MiniLM-L6-v2"


class SemanticSearch:
    """Nearest-neighbour search over description embeddings."""

    def __init__(
        self,
        emb_path: Path = DEFAULT_EMB,
        movies_path: Path = DEFAULT_MOVIES,
        model_name: str = DEFAULT_MODEL,
    ):
        data = np.load(emb_path)
        self._ids = data["movie_ids"]
        self._emb = data["embeddings"]  # already L2-normalised
        self._title = pd.read_csv(movies_path).set_index("movieId")["title"]
        self._model_name = model_name
        self._model = None  # sentence-transformer loaded on first query

    @classmethod
    def load_if_available(cls, **kwargs) -> "SemanticSearch | None":
        """Return an instance if the embedding file exists, else None."""
        try:
            if DEFAULT_EMB.exists():
                return cls(**kwargs)
        except Exception:
            pass
        return None

    def _encode(self, query: str) -> np.ndarray:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name)
        return self._model.encode([query], normalize_embeddings=True)[0]

    def search(self, query: str, k: int = 5) -> list[dict]:
        """Return the ``k`` movies whose descriptions best match ``query``."""
        scores = self._emb @ self._encode(query)  # cosine (all normalised)
        top = np.argsort(-scores)[:k]
        return [
            {
                "movieId": int(self._ids[i]),
                "title": str(self._title.get(int(self._ids[i]), self._ids[i])),
                "score": round(float(scores[i]), 3),
            }
            for i in top
        ]
