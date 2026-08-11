"""End-to-end recommendation pipeline.

    MovieLens → EDA → per-user split → fit model → top-K → metrics → metrics.json

To swap the model, edit the `MODELS` registry below. Everything downstream
(split, evaluation, metrics, persistence) stays identical — that is the whole
point of the `Recommender` interface.

Run:
    python main.py                          # evaluate + write experiments/metrics.json
    python scripts/update_readme_metrics.py # refresh the README table
"""
from __future__ import annotations

from pathlib import Path

from src.data import load_movielens
from src.eda import summarize
from src.evaluation import evaluate
from src.recommenders import (
    ItemCFRecommender,
    PopularityRecommender,
    SVDRecommender,
    XGBRankerRecommender,
)
from src.recommenders.base import Recommender
from src.reporting import save_metrics
from src.split import per_user_leave_last_n

K = 10                      # top-K cutoff for recommendations & metrics
LEAVE_N = 10                # per user, hold out their N most-recent ratings
RELEVANCE_THRESHOLD = 4.0   # a held-out rating >= 4.0 counts as "liked"
METRICS_PATH = Path("experiments/metrics.json")

# Model registry: JSON key → recommender. Add XGBoost etc. here later.
# We keep BOTH CF variants to tell the story: the "textbook" adjusted-cosine one
# loses to popularity, while the positive-only (implicit) one wins — because
# ranking cares about "liked or not", not predicted star count.
MODELS: dict[str, Recommender] = {
    "popularity_baseline": PopularityRecommender(scoring="count"),
    "cf_adjusted_cosine": ItemCFRecommender(min_item_ratings=5, center=True),
    "cf_positive_only": ItemCFRecommender(min_item_ratings=5, positive_only=True),
    "svd": SVDRecommender(n_factors=50, positive_only=True),
    "xgboost": XGBRankerRecommender(),
}


def main() -> None:
    # 1. Load MovieLens
    ml = load_movielens()

    # 2. Exploratory data analysis
    summarize(ml)

    # 3. Enrich ratings with movie metadata (genres, year) so a single `train`
    #    frame carries everything models might need — popularity/CF ignore the
    #    extra columns, XGBoost uses them. Then per-user leave-last-N split.
    ratings = ml.ratings.merge(ml.movies[["movieId", "genres", "year"]], on="movieId", how="left")
    train, test = per_user_leave_last_n(ratings, n=LEAVE_N)
    print("=" * 60)
    print(f"Split: leave-last-{LEAVE_N} per user  →  train={len(train):,}  test={len(test):,}")
    print(f"Test users evaluated: {test['userId'].nunique():,}")

    # 4. Fit + evaluate each registered model with the same harness
    print("\n" + "=" * 60)
    print(f"Metrics @ K={K}  (relevance = rating >= {RELEVANCE_THRESHOLD})")
    print("=" * 60)
    results = {}
    for key, model in MODELS.items():
        model.fit(train)
        result = evaluate(model, train, test, k=K, relevance_threshold=RELEVANCE_THRESHOLD)
        result.model = key
        results[key] = result
        print(result)

    # 5. Persist results for the README generator to consume
    save_metrics(results, METRICS_PATH)
    print(f"\nWrote {len(results)} model(s) → {METRICS_PATH}")


if __name__ == "__main__":
    main()
