# AI Movie Recommendation System

A personalized movie recommendation engine using collaborative filtering, embeddings,
feature engineering, XGBoost ranking, and LLM-based explanations.

## Overview

The system combines several techniques into a single recommendation pipeline:

- **Collaborative filtering** — learns user/item preferences from the rating matrix.
- **Embeddings** — dense latent representations of users and movies for similarity search.
- **Feature engineering** — genres, tags, popularity, recency and rating statistics.
- **XGBoost ranking** — a learning-to-rank model that orders candidate movies per user.
- **LLM-based explanations** — natural-language reasons for why each movie is recommended.

## Dataset

Built on [MovieLens `ml-latest-small`](https://grouplens.org/datasets/movielens/)
(`datasets/ml-latest-small/`).

| File          | Rows    | Description                                      |
| ------------- | ------- | ------------------------------------------------ |
| `ratings.csv` | 100,836 | `userId, movieId, rating (0.5–5.0), timestamp`   |
| `movies.csv`  | 9,742   | `movieId, title (with year), genres (pipe-sep.)` |
| `tags.csv`    | 3,683   | `userId, movieId, tag, timestamp`                |
| `links.csv`   | 9,742   | `movieId, imdbId, tmdbId` (external IDs)         |

**Key statistics**

- 610 users, 9,724 rated movies, ratings from 1996 → 2018.
- Average rating **3.50** (median 3.5); distribution skews positive (4.0 is the mode).
- User–item matrix is **98.3% sparse** (density ~1.7%) — typical for recommenders.
- Each user has rated **≥ 20** movies (median 70); the most active rated 2,698.
- **48.8%** of movies have ≤ 2 ratings → significant **cold-start** challenge.
- Top genres: Drama, Comedy, Thriller, Action, Romance. 34 movies have no genre listed.

## Planned Pipeline

```
raw CSVs → feature engineering → collaborative filtering + embeddings
        → candidate generation → XGBoost ranking → LLM explanations → top-N
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Evaluation

Metrics are computed by the pipeline and stored in
[`experiments/metrics.json`](experiments/metrics.json). The table below is
generated from that file — do not edit it by hand:

```bash
python main.py                          # evaluate → experiments/metrics.json
python scripts/update_readme_metrics.py # regenerate the table below
```

Protocol: per-user leave-last-10 split; an item is *relevant* when its held-out
rating is ≥ 4.0.

<!-- EVALUATION_START -->
| Model | Precision@10 | Recall@10 | NDCG@10 |
| --- | ---: | ---: | ---: |
| Popularity Baseline | 0.037 | 0.059 | 0.054 |
<!-- EVALUATION_END -->

## Status

🚧 Early development. Data exploration complete; modeling in progress.
