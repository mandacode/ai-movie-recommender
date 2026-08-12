"""Hybrid movie search over Postgres — full-text (tsvector) + semantic (pgvector).

Two retrievers, both served by Postgres:
  * **FTS** — `tsvector` GIN index with prefix `to_tsquery`, ranked by `ts_rank`.
    Great for titles, names and exact words.
  * **Semantic** — `pgvector` cosine (`<=>`) over TMDB-description embeddings,
    HNSW-indexed. Great for vibes/paraphrases that share no keywords.

Their ranked lists are fused with Reciprocal Rank Fusion (RRF).
"""
from __future__ import annotations

import math
import re

from db import query

RRF_K = 60


def _tokens(q: str) -> list[str]:
    return re.findall(r"\w+", q.lower())


def _rrf(*weighted: tuple[list[int], float]) -> list[int]:
    """Weighted Reciprocal Rank Fusion. Each arg is ``(ranking, weight)``."""
    scores: dict[int, float] = {}
    for ranking, w in weighted:
        for rank, mid in enumerate(ranking):
            scores[mid] = scores.get(mid, 0.0) + w / (RRF_K + rank)
    return sorted(scores, key=scores.get, reverse=True)


class HybridSearch:
    def __init__(self):
        self._model = None  # sentence-transformer, loaded on first semantic query

    def _fts_search(self, q: str, n: int = 120) -> list[int]:
        toks = _tokens(q)
        if not toks:
            return []
        tsquery = " | ".join(f"{t}:*" for t in toks)  # prefix OR match
        # Rank by field-weighted text relevance × a popularity boost, so canonical
        # titles (e.g. Lord of the Rings) beat obscure same-keyword titles.
        rows = query(
            "SELECT movie_id FROM movies WHERE tsv @@ to_tsquery('english', %s) "
            "ORDER BY ts_rank(tsv, to_tsquery('english', %s)) * ln(2 + coalesce(popularity, 0)) "
            "DESC LIMIT %s",
            (tsquery, tsquery, n),
        )
        return [r[0] for r in rows]

    def _semantic_search(self, q: str, n: int = 120) -> list[int]:
        # Fail-safe: if the embedding model can't load (e.g. offline), fall back
        # to FTS-only rather than breaking search entirely.
        try:
            if self._model is None:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer("all-MiniLM-L6-v2")
            vec = self._model.encode([q], normalize_embeddings=True)[0]
            vec_str = "[" + ",".join(f"{x:.6f}" for x in vec) + "]"
            rows = query(
                "SELECT movie_id FROM movies WHERE embedding IS NOT NULL "
                "ORDER BY embedding <=> %s::vector LIMIT %s",
                (vec_str, n),
            )
            return [r[0] for r in rows]
        except Exception:
            return []

    def search(self, q: str, k: int = 60) -> list[int]:
        q = q.strip()
        if not q:
            return []
        # Short keyword/title queries ("lord") trust FTS — a single vague word is
        # meaningless to the embedder and just injects noise. Descriptive queries
        # ("lonely robot in space") get the full semantic weight.
        sem_w = 1.0 if len(_tokens(q)) >= 3 else 0.25
        return _rrf((self._fts_search(q), 2.0), (self._semantic_search(q), sem_w))[:k]


def _known_item_metrics(ranked: list[int], target: int, k: int = 10):
    rank = ranked.index(target) + 1 if target in ranked else None
    if not rank:
        return 0.0, 0.0, 0.0
    hit = rank <= k
    return 1.0 / rank, (1.0 if hit else 0.0), (1.0 / math.log2(rank + 1) if hit else 0.0)


def _multi_metrics(ranked: list[int], relevant: set[int], k: int = 10):
    """For a query with several relevant targets: (MRR, Recall@k, NDCG@k)."""
    if not relevant:
        return 0.0, 0.0, 0.0
    hit_ranks = [i + 1 for i, m in enumerate(ranked[:k]) if m in relevant]
    mrr = 1.0 / hit_ranks[0] if hit_ranks else 0.0
    recall = len(hit_ranks) / len(relevant)
    dcg = sum(1.0 / math.log2(r + 1) for r in hit_ranks)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return mrr, recall, (dcg / idcg if idcg else 0.0)


def evaluate_golden(engine: HybridSearch, queries: list[tuple[str, set[int]]], k: int = 10) -> dict:
    """Evaluate on a curated golden set (semantic queries, multiple relevant each)."""
    agg = {r: [0.0, 0.0, 0.0] for r in ("fts", "semantic", "hybrid")}
    n = 0
    for q, rel in queries:
        if not rel:
            continue
        n += 1
        rankings = {
            "fts": engine._fts_search(q, n=50),
            "semantic": engine._semantic_search(q, n=50),
            "hybrid": engine.search(q, k=50),
        }
        for name, ranked in rankings.items():
            for i, val in enumerate(_multi_metrics(ranked, rel, k)):
                agg[name][i] += val
    n = max(n, 1)
    return {name: {"mrr": v[0] / n, f"recall@{k}": v[1] / n, f"ndcg@{k}": v[2] / n}
            for name, v in agg.items()}


def evaluate_search(engine: HybridSearch, queries: list[tuple[str, int]], k: int = 10) -> dict:
    """Compare FTS-only, semantic-only and hybrid on known-item queries (MRR/Recall/NDCG)."""
    agg = {r: [0.0, 0.0, 0.0] for r in ("fts", "semantic", "hybrid")}
    for q, target in queries:
        rankings = {
            "fts": engine._fts_search(q, n=50),
            "semantic": engine._semantic_search(q, n=50),
            "hybrid": engine.search(q, k=50),
        }
        for name, ranked in rankings.items():
            for i, val in enumerate(_known_item_metrics(ranked, target, k)):
                agg[name][i] += val
    n = max(len(queries), 1)
    return {
        name: {"mrr": v[0] / n, f"recall@{k}": v[1] / n, f"ndcg@{k}": v[2] / n}
        for name, v in agg.items()
    }
