"""XGBoost learning-to-rank recommender (two-stage retrieval + ranking).

This is a *supervised* recommender, unlike popularity/CF. It reframes
recommendation as tabular classification:

    features(user, movie)  →  P(user will like movie)

and ranks candidates by that probability.

Pipeline
--------
1. **Retrieval** — a cheap model (popularity ∪ item-CF) proposes a few hundred
   candidate movies per user, instead of scoring all ~9.7k.
2. **Feature engineering** — each (user, movie) pair becomes a feature row:
   user stats, movie stats, the **CF score** (stacking!), and genre match.
3. **Labels + negative sampling** — liked (rating >= 4) pairs are positives;
   disliked and randomly-sampled unseen pairs are negatives.
4. **Ranking** — XGBoost predicts P(like); we sort candidates by it.

The CF model is used both as a candidate source and as a feature, so XGBoost
*combines* collaborative and content signals rather than replacing them.
"""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRanker

from ..split import per_user_leave_last_n
from .base import Recommender
from .item_cf import ItemCFRecommender
from .svd import SVDRecommender

LIKE_THRESHOLD = 4.0
FEATURES = [
    "user_mean", "user_count", "user_like_rate",
    "item_mean", "item_count", "item_year", "item_like_rate",
    "cf_score", "svd_score", "genre_match",
    # content/interaction features added in the feature-engineering pass:
    "era_affinity",   # how close the item's year is to the user's liked era
    "genre_cosine",   # cosine(user genre taste, item genres) — stronger than genre_match
    "taste_breadth",  # genre entropy of the user (eclectic vs focused)
    "tag_cosine",     # cosine(user tag taste, item tags) — richer content signal than genres
    "desc_cosine",    # cosine(user description-embedding profile, item description embedding)
]


class XGBRankerRecommender(Recommender):
    name = "xgboost"

    def __init__(
        self,
        cf_model: ItemCFRecommender | None = None,
        svd_model: SVDRecommender | None = None,
        movie_embeddings: dict[int, np.ndarray] | None = None,
        n_negatives: int = 4,
        n_candidates: int = 300,
        n_label: int = 10,
        use_ranker: bool = True,
        random_state: int = 0,
    ):
        """
        Args:
            cf_model: item-CF model for candidates + the cf_score feature.
            svd_model: matrix-factorization model for candidates + svd_score.
                Both default to their positive-only variants if not given.
            n_negatives: random unseen negatives sampled per user for training.
            n_candidates: candidates per source (popularity, CF, SVD) at inference.
            use_ranker: if True, train LambdaMART (`rank:ndcg`, grouped per user)
                which optimises ranking directly; if False, the older pointwise
                classifier (`P(like)`). LambdaMART is the sound choice for top-N.
            random_state: seed for negative sampling and XGBoost.

        Content features (genres, year) are read from optional ``genres`` /
        ``year`` columns on the ``train`` frame passed to ``fit`` — every
        candidate item appears in train, so no separate catalogue is needed.
        """
        self.cf = cf_model or ItemCFRecommender(min_item_ratings=5, positive_only=True)
        self.svd = svd_model or SVDRecommender(n_factors=50, positive_only=True)
        self._movie_emb = movie_embeddings or {}   # movieId → L2-normalised description vector
        self._user_desc_profile: dict[int, np.ndarray] = {}
        self.n_negatives = n_negatives
        self.n_candidates = n_candidates
        self.n_label = n_label
        self.use_ranker = use_ranker
        self.rng = np.random.default_rng(random_state)
        common = dict(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            random_state=random_state, n_jobs=-1,
        )
        if use_ranker:
            # LambdaMART: learns to order items *within each user's group*,
            # weighting each pair by how much swapping it would change NDCG.
            self.model = XGBRanker(objective="rank:ndcg", eval_metric="ndcg@10", **common)
        else:
            self.model = XGBClassifier(eval_metric="logloss", **common)
        # Learned-at-fit lookups (all derived from train only).
        self._user_stats: pd.DataFrame | None = None
        self._item_stats: pd.DataFrame | None = None
        self._user_genre_pref: dict[int, dict[str, float]] = {}
        self._user_pref_norm: dict[int, float] = {}   # L2 norm of each user's genre-pref vector
        self._movie_genres: dict[int, list[str]] = {}
        self._movie_tags: dict[int, list[str]] = {}
        self._user_tag_pref: dict[int, dict[str, float]] = {}
        self._user_tag_norm: dict[int, float] = {}    # L2 norm of each user's tag-pref vector
        self._all_items: np.ndarray | None = None
        self._pop_order: list[int] = []
        self._cf_cache: dict[int, dict[int, float]] = {}
        self._svd_cache: dict[int, dict[int, float]] = {}

    # --- fit ---------------------------------------------------------------

    def fit(self, train: pd.DataFrame) -> "XGBRankerRecommender":
        # --- Phase 1: build LEAK-FREE training data ---
        # Split each user's history in time: earlier slice `hist` builds the
        # features, later slice `lab` supplies the labels. A label item is never
        # part of the history that computed its own features → no leakage.
        hist, lab = per_user_leave_last_n(train, n=self.n_label)
        self.cf.fit(hist)
        self.svd.fit(hist)
        self._build_lookups(hist)
        hist_users = set(hist["userId"].unique())

        pos = lab[lab["rating"] >= LIKE_THRESHOLD][["userId", "movieId"]].copy()
        pos["label"] = 1
        neg = lab[lab["rating"] < LIKE_THRESHOLD][["userId", "movieId"]].copy()
        neg["label"] = 0
        sampled = self._sample_negatives(train)  # implicit negatives (unseen), full-history seen

        data = pd.concat([pos, neg, sampled], ignore_index=True)
        data = data[data["userId"].isin(hist_users)]  # need a history to build features from
        if self.use_ranker:
            # LambdaMART needs rows grouped by query (user) — sort so each user's
            # positives+negatives are contiguous, then pass group ids via `qid`.
            data = data.sort_values("userId", kind="stable").reset_index(drop=True)

        X = self._build_features(data[["userId", "movieId"]])
        y = data["label"].to_numpy()
        if self.use_ranker:
            self.model.fit(X, y, qid=data["userId"].to_numpy())
        else:
            self.model.fit(X, y)

        # --- Phase 2: refit feature generators on FULL history for inference ---
        # At serving time every signal should use the user's complete history,
        # so rebuild CF/SVD/profiles on all of `train`.
        self.cf.fit(train)
        self.svd.fit(train)
        self._build_lookups(train)
        self._cf_cache.clear()
        self._svd_cache.clear()
        return self

    def _build_lookups(self, train: pd.DataFrame) -> None:
        # Reset — this runs twice (leak-free hist view, then full-history view).
        self._user_genre_pref, self._user_pref_norm = {}, {}
        self._user_tag_pref, self._user_tag_norm = {}, {}
        self._movie_genres, self._movie_tags = {}, {}
        self._user_desc_profile = {}

        g = train.groupby("userId")["rating"]
        self._user_stats = pd.DataFrame({
            "user_mean": g.mean(), "user_count": g.count(),
            "user_like_rate": train.assign(like=train["rating"] >= LIKE_THRESHOLD)
                                    .groupby("userId")["like"].mean(),
        })
        gi = train.groupby("movieId")["rating"]
        self._item_stats = pd.DataFrame({
            "item_mean": gi.mean(), "item_count": gi.count(),
            "item_like_rate": train.assign(like=train["rating"] >= LIKE_THRESHOLD)
                                   .groupby("movieId")["like"].mean(),
        })
        self._pop_order = self._item_stats["item_count"].sort_values(ascending=False).index.tolist()
        self._all_items = self._item_stats.index.to_numpy()

        # Movie genres + year — read from the enriched train columns. Every
        # candidate item appears in train, so one row per movie is enough.
        meta = train.drop_duplicates("movieId").set_index("movieId")
        if "genres" in train.columns:
            self._movie_genres = meta["genres"].to_dict()
        if "tags" in train.columns:
            self._movie_tags = meta["tags"].to_dict()
        if "year" in train.columns:
            # `year` is a nullable Int64 → coerce NA to NaN safely, not via astype(float).
            self._item_stats["item_year"] = pd.to_numeric(
                self._item_stats.index.map(meta["year"]), errors="coerce"
            )
        else:
            self._item_stats["item_year"] = np.nan
        self._item_stats["item_year"] = self._item_stats["item_year"].fillna(
            self._item_stats["item_year"].median()
        )

        # Per-user genre preference: distribution of genres among liked movies,
        # plus the vector's L2 norm (for genre_cosine) and entropy (taste_breadth).
        liked = train[train["rating"] >= LIKE_THRESHOLD]
        breadth: dict[int, float] = {}
        for user_id, grp in liked.groupby("userId"):
            counts: dict[str, float] = {}
            for mid in grp["movieId"]:
                for genre in self._movie_genres.get(mid, []):
                    counts[genre] = counts.get(genre, 0.0) + 1.0
            total = sum(counts.values()) or 1.0
            pref = {k: v / total for k, v in counts.items()}
            self._user_genre_pref[user_id] = pref
            probs = np.array(list(pref.values()))
            self._user_pref_norm[user_id] = float(np.sqrt((probs ** 2).sum())) or 1.0
            # Shannon entropy: 0 = one-genre fan, higher = eclectic taste.
            breadth[user_id] = float(-(probs * np.log(probs + 1e-12)).sum())

        # Per-user tag preference (same construction as genres, richer vocabulary).
        if self._movie_tags:
            for user_id, grp in liked.groupby("userId"):
                counts: dict[str, float] = {}
                for mid in grp["movieId"]:
                    for tag in self._movie_tags.get(mid, []):
                        counts[tag] = counts.get(tag, 0.0) + 1.0
                total = sum(counts.values()) or 1.0
                pref = {k: v / total for k, v in counts.items()}
                self._user_tag_pref[user_id] = pref
                probs = np.array(list(pref.values()))
                self._user_tag_norm[user_id] = float(np.sqrt((probs ** 2).sum())) or 1.0

        # Per-user description profile = mean embedding of their liked movies,
        # L2-normalised. Built from `liked` only → leak-free in the training phase.
        if self._movie_emb:
            for user_id, grp in liked.groupby("userId"):
                vecs = [self._movie_emb[m] for m in grp["movieId"] if m in self._movie_emb]
                if vecs:
                    p = np.mean(vecs, axis=0)
                    norm = np.linalg.norm(p)
                    self._user_desc_profile[user_id] = p / norm if norm else p

        # User-level features derived above → attach to the user stats table.
        self._user_stats["taste_breadth"] = self._user_stats.index.map(breadth).fillna(0.0)
        liked_year = pd.to_numeric(liked["year"], errors="coerce") if "year" in liked else None
        if liked_year is not None:
            mean_year = liked_year.groupby(liked["userId"]).mean()
            self._user_stats["user_liked_year"] = self._user_stats.index.map(mean_year)
        else:
            self._user_stats["user_liked_year"] = np.nan
        self._user_stats["user_liked_year"] = self._user_stats["user_liked_year"].fillna(
            self._user_stats["user_liked_year"].median()
        )

    def _sample_negatives(self, train: pd.DataFrame) -> pd.DataFrame:
        seen_by_user = train.groupby("userId")["movieId"].agg(set)
        rows = []
        for user_id, seen in seen_by_user.items():
            pool = self._all_items
            picks = self.rng.choice(pool, size=min(self.n_negatives * 4, len(pool)), replace=False)
            fresh = [m for m in picks if m not in seen][: self.n_negatives]
            rows.extend((user_id, int(m), 0) for m in fresh)
        return pd.DataFrame(rows, columns=["userId", "movieId", "label"])

    # --- feature engineering ----------------------------------------------

    def _cf_scores_for(self, user_id: int) -> dict[int, float]:
        if user_id not in self._cf_cache:
            vec = self.cf.score_vector(user_id)
            ids = self.cf.item_ids
            self._cf_cache[user_id] = {} if vec is None else dict(zip(ids.tolist(), vec.tolist()))
        return self._cf_cache[user_id]

    def _svd_scores_for(self, user_id: int) -> dict[int, float]:
        if user_id not in self._svd_cache:
            vec = self.svd.score_vector(user_id)
            ids = self.svd.item_ids
            self._svd_cache[user_id] = {} if vec is None else dict(zip(ids.tolist(), vec.tolist()))
        return self._svd_cache[user_id]

    def _genre_match(self, user_id: int, movie_id: int) -> float:
        pref = self._user_genre_pref.get(user_id)
        genres = self._movie_genres.get(movie_id)
        if not pref or not genres:
            return 0.0
        return float(np.mean([pref.get(g, 0.0) for g in genres]))

    def _genre_cosine(self, user_id: int, movie_id: int) -> float:
        """Cosine between the user's genre-taste vector and the item's genres."""
        pref = self._user_genre_pref.get(user_id)
        genres = self._movie_genres.get(movie_id)
        if not pref or not genres:
            return 0.0
        dot = sum(pref.get(g, 0.0) for g in genres)
        norm = self._user_pref_norm.get(user_id, 1.0) * float(np.sqrt(len(genres)))
        return float(dot / norm) if norm else 0.0

    def _tag_cosine(self, user_id: int, movie_id: int) -> float:
        """Cosine between the user's tag-taste vector and the item's tags."""
        pref = self._user_tag_pref.get(user_id)
        tags = self._movie_tags.get(movie_id)
        if not pref or not tags:
            return 0.0
        dot = sum(pref.get(t, 0.0) for t in tags)
        norm = self._user_tag_norm.get(user_id, 1.0) * float(np.sqrt(len(tags)))
        return float(dot / norm) if norm else 0.0

    def _desc_cosine(self, user_id: int, movie_id: int) -> float:
        """Cosine between the user's description-embedding profile and the item's."""
        p = self._user_desc_profile.get(user_id)
        v = self._movie_emb.get(movie_id)
        if p is None or v is None:
            return 0.0
        return float(np.dot(p, v))  # both L2-normalised → dot == cosine

    def _build_features(self, pairs: pd.DataFrame) -> pd.DataFrame:
        df = pairs.merge(self._user_stats, left_on="userId", right_index=True, how="left")
        df = df.merge(self._item_stats, left_on="movieId", right_index=True, how="left")

        users = df["userId"].to_numpy()
        movies = df["movieId"].to_numpy()
        df["cf_score"] = [self._cf_scores_for(u).get(m, 0.0) for u, m in zip(users, movies)]
        df["svd_score"] = [self._svd_scores_for(u).get(m, 0.0) for u, m in zip(users, movies)]
        df["genre_match"] = [self._genre_match(u, m) for u, m in zip(users, movies)]
        df["genre_cosine"] = [self._genre_cosine(u, m) for u, m in zip(users, movies)]
        df["tag_cosine"] = [self._tag_cosine(u, m) for u, m in zip(users, movies)]
        df["desc_cosine"] = [self._desc_cosine(u, m) for u, m in zip(users, movies)]

        # Era affinity: how far the item's year is from the user's liked era.
        df["era_affinity"] = (df["item_year"] - df["user_liked_year"]).abs()

        # Coerce every feature to plain float64 — nullable Int64/Float64 dtypes
        # (from the `year` column) would otherwise reach XGBoost as `object`.
        return df[FEATURES].apply(pd.to_numeric, errors="coerce").astype("float64").fillna(0.0)

    # --- recommend ---------------------------------------------------------

    def recommend(self, user_id: int, k: int, exclude: Iterable[int] | None = None) -> list[int]:
        if self._user_stats is None or user_id not in self._user_stats.index:
            return []
        exclude = set(exclude or ())

        # Stage 1 — retrieval: popularity ∪ CF ∪ SVD candidates, minus seen.
        cf_scores = self._cf_scores_for(user_id)
        cf_top = sorted(cf_scores, key=cf_scores.get, reverse=True)[: self.n_candidates]
        svd_scores = self._svd_scores_for(user_id)
        svd_top = sorted(svd_scores, key=svd_scores.get, reverse=True)[: self.n_candidates]
        pool = self._pop_order[: self.n_candidates] + cf_top + svd_top
        candidates = [m for m in dict.fromkeys(pool) if m not in exclude]
        if not candidates:
            return []

        # Stage 2 — ranking: score candidates and sort. LambdaMART returns a
        # ranking score via predict; the classifier returns P(like).
        pairs = pd.DataFrame({"userId": user_id, "movieId": candidates})
        feats = self._build_features(pairs)
        scores = self.model.predict(feats) if self.use_ranker else self.model.predict_proba(feats)[:, 1]
        order = np.argsort(-scores)[:k]
        self._cf_cache.pop(user_id, None)   # keep caches small during eval
        self._svd_cache.pop(user_id, None)
        return [int(candidates[i]) for i in order]

    def feature_importances(self) -> dict[str, float]:
        """Map feature name → importance (for interpretability)."""
        return dict(zip(FEATURES, self.model.feature_importances_.tolist()))
