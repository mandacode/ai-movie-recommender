# 🎬 Mandaflix — AI Movie Recommender

A full-stack, streaming-style movie recommender with **two AI features**:
**personalised recommendations** and **hybrid semantic search**. Built on the
MovieLens dataset, served from a Postgres + `pgvector` store, with a React
frontend that hides all algorithm internals behind a Netflix-like UI.

**▶ Live:** https://mandaflix.krystianjarmul.dev

---

## Overview

Two independent ML systems power the app:

| Feature | What it does | How |
| --- | --- | --- |
| **Recommendations** | ranks the catalogue for the current user | matrix-factorisation (SVD) with real-time **fold-in** on likes |
| **Search** | free-text find, from titles to vibes | **hybrid**: Postgres full-text (`tsvector`) + semantic (`pgvector`), fused with RRF |

You switch between profiles, browse a personalised tile grid, like/unlike titles
(which instantly re-ranks), search by keyword *or* by plot ("a heist with a
clever twist"), and open a movie detail page. A separate **Insights** panel
exposes the model/search metrics the consumer UI deliberately hides.

## Architecture

```
React SPA (Vite)  ──►  FastAPI  ──►  Postgres + pgvector
  Mandaflix UI          /api/*        movies (FTS + embeddings)
  served by FastAPI     recommender   ratings (base history)
                        + search      interactions (live likes)
```

One origin: FastAPI serves both the API and the built frontend. A dedicated
Cloudflare Tunnel fronts it publicly.

## Recommendations

Four models were built behind one pluggable `Recommender` interface
(`fit` / `recommend`), benchmarked on a **per-user temporal split** (each user's
most-recent ratings held out), relevance = rating ≥ 4.

| Model | Precision@10 | Recall@10 | NDCG@10 |
| --- | ---: | ---: | ---: |
| Popularity baseline | 0.037 | 0.059 | 0.054 |
| Item-CF (adjusted cosine) | 0.024 | 0.044 | 0.040 |
| Item-CF (positive-only) | 0.056 | 0.096 | 0.082 |
| **SVD (matrix factorization)** | **0.059** | **0.096** | **0.086** |
| XGBoost learning-to-rank | 0.040 | 0.064 | 0.057 |

**SVD wins** and is what the app serves. See *Key findings* below for why the
"textbook" CF and XGBoost variants lose — the interesting part.

**Serving (production pattern).** The app does **not** retrain on every like.
Two speeds:

- **Real-time fold-in** — a user's latent vector is recomputed on the fly from
  their current likes against the *fixed* item factors, so a like re-ranks
  instantly and a brand-new user (0 ratings) cold-starts from popularity and
  personalises as they like. No training on the hot path.
- **Batch retrain** (`scripts/retrain.py`, scheduled) — folds accumulated likes
  back into the item factors. This is where new items/history enter the model.

## Search

Hybrid retrieval, both served by Postgres, fused with **Reciprocal Rank Fusion**:

- **Full-text** — `tsvector` + GIN, **field-weighted** (title = A, cast/genres =
  B, plot = C) so a title hit outranks a word buried in an overview, and ordered
  by `ts_rank × popularity` so canonical titles beat obscure same-keyword ones.
- **Semantic** — `pgvector` cosine over TMDB-overview embeddings
  (`all-MiniLM-L6-v2`), HNSW-indexed. Answers vibe/plot queries with no shared
  keywords.
- **Fusion** — RRF weights FTS higher, and scales the semantic term by query
  length: short keyword queries ("lord") trust FTS; descriptive queries
  ("lonely robot finds friendship in space") lean on semantics. Falls back to
  FTS-only if the embedding model is unavailable.

## Evaluation

- **Recommendations** — offline temporal split, P/R/NDCG (above). Absolute values
  are low by nature (predicting a user's specific next films from ~10k is hard);
  what matters is the *relative* comparison.
- **Search** — two complementary sets, shown in the Insights panel:
  a **synthetic** known-item set (title / plot-snippet queries) where FTS leads,
  and a **golden** set of hand-labelled semantic queries where semantic/hybrid
  lead — a reminder that metrics only compare *within* one protocol.

## Tech stack

- **Backend** — Python, FastAPI, scikit-learn (TruncatedSVD), scipy, XGBoost,
  sentence-transformers, psycopg.
- **Data** — Postgres 16 + `pgvector` (metadata, FTS, embeddings, interactions).
- **Frontend** — React + Vite, the Nocturne design system.
- **Infra** — Docker Compose, Cloudflare Tunnel; TMDB for poster/plot enrichment.

## Project structure

```
backend/
  api.py            FastAPI app (recommendations, search, likes, insights)
  catalog.py        data layer: SVD fold-in serving, likes, metadata
  search.py         hybrid FTS + pgvector + RRF, with search evaluation
  src/recommenders/ popularity · item_cf · svd · xgb_ranker (Recommender ABC)
  src/…             data loading, split, metrics, evaluation
  scripts/          seed_db · fetch_tmdb · embed_descriptions · retrain
frontend/           React SPA (Mandaflix UI)
docker-compose.yml  db (pgvector) + app + cloudflared
DEPLOY.md           deployment runbook
```

## Local setup

Requires Docker (for Postgres) and Python 3.12 / Node 22 for dev.

```bash
# 1. Postgres + pgvector
docker run -d --name mandaflix-pg -e POSTGRES_PASSWORD=mandaflix \
  -e POSTGRES_DB=mandaflix -p 5433:5432 pgvector/pgvector:pg16

# 2. Backend
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/seed_db.py          # load MovieLens + TMDB + embeddings
uvicorn api:app --reload --port 8000

# 3. Frontend (separate terminal)
cd frontend && npm install && npm run dev   # http://localhost:5173
```

Offline evaluation of the recommenders (no DB needed):

```bash
cd backend && python main.py       # → experiments/metrics.json
```

## Deployment

Docker Compose (db + app + a dedicated Cloudflare tunnel) fronts the app at
`mandaflix.krystianjarmul.dev`. Full runbook in **[DEPLOY.md](DEPLOY.md)**.

## Key engineering decisions & lessons

The measured, sometimes counter-intuitive findings that shaped the build:

- **Match the signal to the task.** For top-N *ranking*, a binary
  "liked / not" (positive-only) signal beats star-rating regression — the
  "textbook" adjusted-cosine CF, tuned for RMSE, actually *lost* to a trivial
  popularity baseline.
- **Feature leakage is the silent killer.** User-preference features computed
  from a user's full history leak into training when the positives come from
  that same history; the fix is temporal separation (profiles from an earlier
  slice, labels from a later one). This explained most of XGBoost's
  underperformance.
- **Don't add complexity that doesn't earn its keep.** With careful
  leak-free features (incl. tags and description embeddings), XGBoost still
  didn't beat SVD on this data — so **SVD ships**. Collaborative signal
  dominates content signal for warm-user ranking.
- **Content embeddings shine where CF can't** — free-text/semantic search — so
  they power *search*, not the ranker.
- **No LLM.** The recommender core is classical ML; an LLM would be the wrong,
  slower, costlier tool for ranking 10k items. (Search is FTS + embeddings, not
  an LLM either.)
- **Two-speed serving** (real-time fold-in + batch retrain) is the standard way
  to make likes feel instant without retraining per interaction.

## Credits

- Data: [MovieLens](https://grouplens.org/datasets/movielens/) `ml-latest-small`.
- Metadata & imagery: [TMDB](https://www.themoviedb.org/) (this product uses the
  TMDB API but is not endorsed or certified by TMDB).
