"""Item-based collaborative filtering recommender.

Idea: a movie is a vector of the ratings it received from every user. Two movies
are *similar* if the same people rate them similarly (cosine similarity between
those vectors). To recommend, we score each candidate movie by how similar it is
to the movies the user already liked, weighted by how much they liked them.

Two standard refinements are applied:

* **Adjusted cosine** — each rating is centred by that user's mean before
  computing similarity, so a generous 5-star rater and a stingy 3-star rater are
  put on the same footing ("above *my* average" rather than "a high number").
* **No giant matrix** — instead of materialising the 9.7k×9.7k item-item
  similarity matrix, we exploit ``scores = Xn @ (Xnᵀ @ s_u)``: two sparse
  matrix-vector products per user, computed on the fly.

Cold items (no ratings in train) get a zero vector and are simply never
recommended — an honest reflection of the cold-start limitation.
"""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy import sparse

from .base import Recommender


class ItemCFRecommender(Recommender):
    name = "item_cf"

    def __init__(
        self,
        min_item_ratings: int = 1,
        center: bool = True,
        positive_only: bool = False,
        like_threshold: float = 4.0,
    ):
        """
        Args:
            min_item_ratings: ignore items rated fewer than this many times when
                building similarities (noise control for the long tail).
            center: subtract each user's mean rating (adjusted cosine). Good for
                rating prediction; can hurt top-N ranking — worth benchmarking.
            positive_only: keep only "liked" ratings (>= ``like_threshold``) as a
                binary signal. Implicit-feedback style, often better for top-N.
            like_threshold: rating at/above which an item counts as liked.
        """
        self.min_item_ratings = min_item_ratings
        self.center = center
        self.positive_only = positive_only
        self.like_threshold = like_threshold
        self._Xn: sparse.csr_matrix | None = None      # items × users, L2-normalised
        self._Rc: sparse.csr_matrix | None = None      # users × items, signal values
        self._item_ids: np.ndarray | None = None       # index → movieId
        self._user_pos: dict[int, int] = {}            # userId → row index
        self._item_pos: dict[int, int] = {}            # movieId → col index

    def fit(self, train: pd.DataFrame) -> "ItemCFRecommender":
        # Optionally drop rarely-rated items to reduce noise.
        counts = train.groupby("movieId").size()
        keep = set(counts[counts >= self.min_item_ratings].index)
        df = train[train["movieId"].isin(keep)]

        # Implicit mode: keep only liked interactions as a binary signal.
        if self.positive_only:
            df = df[df["rating"] >= self.like_threshold]

        users = df["userId"].unique()
        items = df["movieId"].unique()
        self._user_pos = {u: i for i, u in enumerate(users)}
        self._item_pos = {m: i for i, m in enumerate(items)}
        self._item_ids = items

        u_idx = df["userId"].map(self._user_pos).to_numpy()
        i_idx = df["movieId"].map(self._item_pos).to_numpy()

        # Build the rating signal: binary (implicit), centred (adjusted cosine),
        # or raw ratings.
        if self.positive_only:
            centred = np.ones(len(df), dtype=float)
        elif self.center:
            user_mean = df.groupby("userId")["rating"].transform("mean")
            centred = (df["rating"] - user_mean).to_numpy()
        else:
            centred = df["rating"].to_numpy(dtype=float)

        n_users, n_items = len(users), len(items)
        # Users × items, centred (used to build each user's rating vector s_u).
        self._Rc = sparse.csr_matrix((centred, (u_idx, i_idx)), shape=(n_users, n_items))

        # Items × users, then L2-normalise each item row so a dot product
        # between two rows equals their cosine similarity.
        X = self._Rc.T.tocsr()
        norms = np.sqrt(X.multiply(X).sum(axis=1)).A1  # per-item L2 norm
        norms[norms == 0] = 1.0                        # avoid divide-by-zero
        inv = sparse.diags(1.0 / norms)
        self._Xn = (inv @ X).tocsr()
        return self

    def recommend(
        self,
        user_id: int,
        k: int,
        exclude: Iterable[int] | None = None,
    ) -> list[int]:
        if self._Xn is None or user_id not in self._user_pos:
            return []  # unknown user → no CF signal (caller may fall back)

        # s_u: this user's centred ratings as a (1 × items) sparse row.
        s_u = self._Rc[self._user_pos[user_id]]

        # scores = Xn @ (Xnᵀ @ s_uᵀ) — the two-step sparse product from the docstring.
        profile = self._Xn.T @ s_u.T        # users × 1 : the user's taste profile
        scores = np.asarray((self._Xn @ profile).todense()).ravel()  # items × 1

        # Never recommend items the user already has (or an explicit exclude set).
        exclude = set(exclude or ())
        for movie_id in exclude:
            pos = self._item_pos.get(movie_id)
            if pos is not None:
                scores[pos] = -np.inf

        # Top-k by score (only positive scores are meaningful recommendations).
        n = min(k, np.sum(np.isfinite(scores) & (scores > 0)))
        if n <= 0:
            return []
        top = np.argpartition(-scores, n - 1)[:n]
        top = top[np.argsort(-scores[top])]
        return [int(self._item_ids[i]) for i in top]
