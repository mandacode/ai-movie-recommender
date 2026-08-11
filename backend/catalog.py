"""Movie catalog service — Postgres-backed data layer behind the API.

Sources, all in Postgres (pgvector): `movies` (metadata + FTS + embedding),
`ratings` (base MovieLens history), `interactions` (live likes).

Recommendations use a fitted SVD model. Serving is **fold-in**: a user's vector
is computed on the fly from their current liked set (base ratings + live likes),
so likes change the ranking immediately without retraining, and a brand-new
user (Krystian) cold-starts from popularity and personalises as they like.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import db
from search import HybridSearch
from src.recommenders import ItemCFRecommender, PopularityRecommender, SVDRecommender

LIKE = 4.0
KRYSTIAN_ID = 9999
JULIA_ID = 9998

# Fresh users (cold-start, 0 base ratings) first, then MovieLens demo profiles.
PROFILES = [
    (KRYSTIAN_ID, "Krystian"), (JULIA_ID, "Julia"),
    (414, "Ava Renn"), (599, "Milo Frost"), (474, "Nadia Voss"), (448, "Theo Marsh"),
    (274, "Iris Kane"), (68, "Leo Barnes"), (380, "Sena Ito"), (610, "Cass Okoro"),
]


class Catalog:
    def __init__(self):
        self._build()

    def _training_ratings(self) -> pd.DataFrame:
        """Base MovieLens ratings + live likes (liked → 5.0) — the retrain set."""
        base = pd.DataFrame(
            db.query("SELECT user_id, movie_id, rating FROM ratings"),
            columns=["userId", "movieId", "rating"],
        )
        likes = pd.DataFrame(
            db.query("SELECT user_id, movie_id FROM interactions WHERE liked"),
            columns=["userId", "movieId"],
        )
        if len(likes):
            likes["rating"] = 5.0
            return pd.concat([base, likes], ignore_index=True)
        return base

    def _build(self) -> None:
        ratings = self._training_ratings()
        self.model = SVDRecommender(n_factors=50, positive_only=True).fit(ratings)
        self._ratings = ratings

        self._base_likes = ratings[ratings["rating"] >= LIKE].groupby("userId")["movieId"].agg(set)
        self._seen = ratings.groupby("userId")["movieId"].agg(set)
        counts = ratings.groupby("movieId").size()
        self._popular_ids = counts.sort_values(ascending=False).index.tolist()

        self._meta = self._load_movies()
        self.genres = sorted({g for m in self._meta.values() for g in m["genres"]})
        self.engine = HybridSearch()
        for cached in ("_cmp", "_sm"):  # drop lazily-built caches on retrain
            self.__dict__.pop(cached, None)

    def retrain(self) -> dict:
        """Batch retrain: refit the model on base ratings + accumulated likes.

        Fold-in already personalises in real time; this is the scheduled
        (e.g. nightly) job that folds new interactions into the *item* factors.
        """
        self._build()
        return {"trained_on": int(len(self._ratings)), "movies": len(self._meta)}

    # --- movie metadata (cached in memory for fast shaping) ----------------

    def _load_movies(self) -> dict[int, dict]:
        rows = db.query(
            "SELECT movie_id, title, year, genres, overview, poster, backdrop, "
            "runtime, director, cast_top FROM movies"
        )
        out = {}
        for mid, title, year, genres, overview, poster, backdrop, runtime, director, cast in rows:
            out[mid] = {
                "movieId": mid, "title": title, "year": year, "genres": genres or [],
                "poster": poster, "backdrop": backdrop, "runtime": runtime,
                "director": director, "cast": cast or [], "synopsis": overview,
            }
        return out

    def _movie(self, movie_id: int, rank: int | None = None, liked: bool | None = None) -> dict:
        m = dict(self._meta.get(movie_id, {"movieId": movie_id, "title": str(movie_id),
                                           "year": None, "genres": [], "poster": None,
                                           "backdrop": None, "runtime": None, "director": None,
                                           "cast": [], "synopsis": None}))
        if rank is not None:
            m["rank"] = rank
        if liked is not None:
            m["liked"] = liked
        return m

    # --- likes / interactions ---------------------------------------------

    def liked_set(self, user_id: int) -> set[int]:
        """Current liked movies = base ratings ≥ 4 overlaid with live likes."""
        s = set(self._base_likes.get(user_id, set()))
        for mid, is_liked in db.query(
            "SELECT movie_id, liked FROM interactions WHERE user_id = %s", (user_id,)
        ):
            s.add(mid) if is_liked else s.discard(mid)
        return s

    def _seen_set(self, user_id: int) -> set[int]:
        s = set(self._seen.get(user_id, set()))
        s |= {r[0] for r in db.query("SELECT movie_id FROM interactions WHERE user_id = %s", (user_id,))}
        return s

    def set_like(self, user_id: int, movie_id: int, liked: bool) -> None:
        db.execute(
            "INSERT INTO interactions (user_id, movie_id, liked) VALUES (%s, %s, %s) "
            "ON CONFLICT (user_id, movie_id) DO UPDATE SET liked = EXCLUDED.liked, ts = now()",
            (user_id, movie_id, liked),
        )

    def likes(self, user_id: int) -> list[dict]:
        ids = self.liked_set(user_id)
        # order live likes first (most recent), then base likes
        recent = [r[0] for r in db.query(
            "SELECT movie_id FROM interactions WHERE user_id=%s AND liked ORDER BY ts DESC", (user_id,))]
        ordered = [m for m in recent if m in ids] + [m for m in ids if m not in recent]
        return [self._movie(m) for m in ordered]

    # --- API-facing methods ------------------------------------------------

    def users(self) -> list[dict]:
        out = []
        for uid, name in PROFILES:
            n = int(self._seen.get(uid, set()).__len__()) if uid in self._seen else len(self.liked_set(uid))
            out.append({"id": uid, "name": name, "ratings": n})
        return out

    def recommendations(self, user_id: int, k: int = 20, genre: str | None = None) -> list[dict]:
        liked = self.liked_set(user_id)
        scores = self.model.fold_in_scores(liked)
        if scores is None:  # cold start → popularity
            ranked = self._popular_ids
        else:
            ids = self.model.item_ids
            ranked = [int(ids[i]) for i in np.argsort(-scores)]

        exclude = self._seen_set(user_id) | liked
        out = []
        for m in ranked:
            if m in exclude:
                continue
            if genre and genre != "All" and genre not in self._meta.get(m, {}).get("genres", []):
                continue
            out.append(self._movie(m, rank=len(out) + 1))
            if len(out) == k:
                break
        return out

    def search(self, query: str, limit: int = 60) -> list[dict]:
        return [self._movie(m) for m in self.engine.search(query, k=limit)]

    def popular(self, k: int = 24) -> list[dict]:
        return [self._movie(m) for m in self._popular_ids[:k]]

    def movie(self, movie_id: int, user_id: int | None = None) -> dict | None:
        if movie_id not in self._meta:
            return None
        liked = movie_id in self.liked_set(user_id) if user_id is not None else None
        return self._movie(movie_id, liked=liked)

    def similar(self, movie_id: int, user_id: int, k: int = 7) -> list[dict]:
        base_genres = set(self._meta.get(movie_id, {}).get("genres", []))
        scores = self.model.fold_in_scores(self.liked_set(user_id))
        if scores is None:
            ranked = self._popular_ids
        else:
            ids = self.model.item_ids
            ranked = [int(ids[i]) for i in np.argsort(-scores)]
        out = [m for m in ranked if m != movie_id and base_genres & set(self._meta.get(m, {}).get("genres", []))]
        return [self._movie(m) for m in out[:k]]

    # --- insights ----------------------------------------------------------

    def _compare_models(self) -> dict:
        if not hasattr(self, "_cmp"):
            self._cmp = {
                "popularity_baseline": (PopularityRecommender("count").fit(self._ratings), False),
                "cf_positive_only": (ItemCFRecommender(min_item_ratings=5, positive_only=True).fit(self._ratings), True),
                "svd": (self.model, True),
            }
        return self._cmp

    def compare(self, user_id: int, k: int = 8) -> dict:
        seen = self._seen_set(user_id)
        out = {}
        for key, (mdl, has_scores) in self._compare_models().items():
            ids = mdl.recommend(user_id, k=k, exclude=seen)
            scores = {}
            if has_scores:
                vec = mdl.score_vector(user_id)
                if vec is not None:
                    scores = dict(zip(mdl.item_ids.tolist(), vec.tolist()))
            out[key] = [{"title": self._meta.get(m, {}).get("title", str(m)), "score": scores.get(m)}
                        for m in ids]
        return out

    def golden_metrics(self) -> dict:
        """Evaluate search on the curated golden set of semantic queries."""
        if not hasattr(self, "_gm"):
            import json
            from pathlib import Path

            from search import evaluate_golden

            golden = json.loads((Path(__file__).parent / "datasets" / "golden_queries.json").read_text())
            by_title: dict[str, list[int]] = {}
            for mid, m in self._meta.items():
                by_title.setdefault(m["title"].lower(), []).append(mid)

            def resolve(title: str) -> set[int]:
                tl = title.lower()
                if tl in by_title:
                    return set(by_title[tl])
                out: set[int] = set()
                for name, mids in by_title.items():
                    if name.startswith(tl) and (len(name) == len(tl) or name[len(tl)] in ": ("):
                        out.update(mids)
                return out

            queries = []
            for entry in golden:
                rel: set[int] = set()
                for t in entry["titles"]:
                    rel |= resolve(t)
                queries.append((entry["query"], rel))
            self._gm = evaluate_golden(self.engine, queries)
        return self._gm

    def search_metrics(self, sample: int = 150) -> dict:
        if not hasattr(self, "_sm"):
            from search import evaluate_search

            rng = np.random.default_rng(0)
            with_ov = [mid for mid, m in self._meta.items() if m.get("synopsis")]
            picks = rng.choice(with_ov, size=min(sample, len(with_ov)), replace=False)
            queries: list[tuple[str, int]] = []
            for mid in picks:
                mid = int(mid)
                title = self._meta[mid]["title"]
                if title:
                    queries.append((title, mid))
                snippet = " ".join((self._meta[mid]["synopsis"] or "").split()[:12])
                if snippet:
                    queries.append((snippet, mid))
            self._sm = evaluate_search(self.engine, queries)
        return self._sm
