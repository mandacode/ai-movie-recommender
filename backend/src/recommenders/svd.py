"""Matrix-factorization recommender (truncated SVD).

Factorises the sparse user-item matrix ``R`` into low-rank latent vectors::

    R  ≈  P · Qᵀ         P: users × k,   Q: items × k

Each user and each movie becomes a ``k``-dimensional latent vector; their dot
product estimates affinity. The ``k`` latent dimensions are *learned* from the
rating patterns (not hand-defined), which lets the model generalise across the
98%-sparse matrix better than explicit item-item similarity.

As with our CF model, a **positive-only** (implicit) signal — factorising the
binary "liked" matrix — tends to rank better for top-N than raw star ratings.
"""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD

from .base import Recommender

LIKE_THRESHOLD = 4.0


class SVDRecommender(Recommender):
    name = "svd"

    def __init__(
        self,
        n_factors: int = 50,
        positive_only: bool = True,
        like_threshold: float = LIKE_THRESHOLD,
        center: bool = False,
        random_state: int = 0,
    ):
        """
        Args:
            n_factors: number of latent dimensions ``k``.
            positive_only: factorise the binary liked matrix (implicit feedback).
            like_threshold: rating at/above which an item counts as liked.
            center: subtract each user's mean rating (only used if not positive_only).
            random_state: seed for the randomized SVD solver.
        """
        self.n_factors = n_factors
        self.positive_only = positive_only
        self.like_threshold = like_threshold
        self.center = center
        self.random_state = random_state
        self.svd: TruncatedSVD | None = None
        self._user_factors: np.ndarray | None = None   # users × k
        self._item_factors: np.ndarray | None = None    # k × items
        self._item_ids: np.ndarray | None = None
        self._user_pos: dict[int, int] = {}
        self._item_pos: dict[int, int] = {}

    def fit(self, train: pd.DataFrame) -> "SVDRecommender":
        df = train
        if self.positive_only:
            df = df[df["rating"] >= self.like_threshold]

        users = df["userId"].unique()
        items = df["movieId"].unique()
        self._user_pos = {u: i for i, u in enumerate(users)}
        self._item_pos = {m: i for i, m in enumerate(items)}
        self._item_ids = items

        u_idx = df["userId"].map(self._user_pos).to_numpy()
        i_idx = df["movieId"].map(self._item_pos).to_numpy()
        if self.positive_only:
            values = np.ones(len(df), dtype=float)
        elif self.center:
            values = (df["rating"] - df.groupby("userId")["rating"].transform("mean")).to_numpy()
        else:
            values = df["rating"].to_numpy(dtype=float)

        R = sparse.csr_matrix((values, (u_idx, i_idx)), shape=(len(users), len(items)))

        # TruncatedSVD needs k < n_items; keep it safe for tiny catalogues.
        k = min(self.n_factors, min(R.shape) - 1)
        self.svd = TruncatedSVD(n_components=k, random_state=self.random_state)
        self._user_factors = self.svd.fit_transform(R)   # users × k  (= P)
        self._item_factors = self.svd.components_          # k × items (= Qᵀ)
        return self

    def score_vector(self, user_id: int) -> np.ndarray | None:
        """Latent affinity score for every item (aligned with ``item_ids``)."""
        if self._user_factors is None or user_id not in self._user_pos:
            return None
        u = self._user_factors[self._user_pos[user_id]]
        return u @ self._item_factors

    def fold_in_scores(self, liked_movie_ids) -> np.ndarray | None:
        """Score every item for an ad-hoc user defined by their liked movies.

        Folds the liked items' latent vectors into a user vector (item factors
        stay fixed) — the standard way to serve *new users* and *fresh likes*
        in real time without retraining. Returns ``None`` if none of the liked
        movies are in the model (caller should fall back, e.g. to popularity).
        """
        if self._item_factors is None:
            return None
        pos = [self._item_pos[m] for m in liked_movie_ids if m in self._item_pos]
        if not pos:
            return None
        user_vec = self._item_factors[:, pos].mean(axis=1)  # k-dim, fold-in
        return user_vec @ self._item_factors                 # score per item

    @property
    def item_ids(self) -> np.ndarray | None:
        return self._item_ids

    def recommend(self, user_id: int, k: int, exclude: Iterable[int] | None = None) -> list[int]:
        scores = self.score_vector(user_id)
        if scores is None:
            return []
        scores = scores.copy()
        for movie_id in set(exclude or ()):
            pos = self._item_pos.get(movie_id)
            if pos is not None:
                scores[pos] = -np.inf
        n = min(k, int(np.sum(np.isfinite(scores))))
        if n <= 0:
            return []
        top = np.argpartition(-scores, n - 1)[:n]
        top = top[np.argsort(-scores[top])]
        return [int(self._item_ids[i]) for i in top]
