"""Generate top-K movie recommendations for a user from the command line.

Usage:
    python scripts/recommend.py --user 1
    python scripts/recommend.py --user 42 --k 15 --genre Comedy
    python scripts/recommend.py --user 7 --scoring weighted

Uses the same `src` modules as the pipeline and the Streamlit app, so the CLI,
dashboard and evaluation all share one implementation.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the project root importable when run as a plain script.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import load_movielens  # noqa: E402
from src.recommenders import PopularityRecommender  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate top-K movie recommendations.")
    p.add_argument("--user", type=int, required=True, help="userId to recommend for")
    p.add_argument("--k", type=int, default=10, help="number of recommendations (default: 10)")
    p.add_argument("--genre", type=str, default=None, help="restrict to a single genre")
    p.add_argument(
        "--scoring",
        choices=["count", "mean", "weighted"],
        default="count",
        help="popularity scoring mode (default: count)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    ml = load_movielens()

    if args.user not in set(ml.ratings["userId"].unique()):
        print(f"error: user {args.user} not found in the dataset", file=sys.stderr)
        return 1

    model = PopularityRecommender(scoring=args.scoring).fit(ml.ratings)

    title = ml.movies.set_index("movieId")["title"]
    genres = ml.movies.set_index("movieId")["genres"]
    seen = set(ml.ratings.loc[ml.ratings["userId"] == args.user, "movieId"])

    # Pull a generous candidate list, then optionally filter by genre.
    candidates = model.recommend(args.user, k=500, exclude=seen)
    if args.genre:
        candidates = [m for m in candidates if args.genre in genres.get(m, [])]
    top = candidates[: args.k]

    header = f"Top {len(top)} recommendations for user {args.user}"
    if args.genre:
        header += f" · genre={args.genre}"
    header += f" · scoring={args.scoring}"
    print(header)
    print("-" * len(header))
    if not top:
        print("(no candidates — try a different genre)")
        return 0
    for rank, movie_id in enumerate(top, start=1):
        print(f"{rank:>2}. {title.get(movie_id, movie_id)}  [{', '.join(genres.get(movie_id, []))}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
