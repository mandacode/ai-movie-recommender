"""Exploratory data analysis summary for MovieLens."""
from __future__ import annotations

from collections import Counter

from .data import MovieLens


def summarize(ml: MovieLens) -> None:
    """Print a compact EDA report to stdout."""
    r = ml.ratings
    print("=" * 60)
    print("EDA — MovieLens ml-latest-small")
    print("=" * 60)

    # --- Ratings ---
    density = len(r) / (ml.n_users * ml.n_movies)
    per_user = r.groupby("userId").size()
    print("\n[Ratings]")
    print(f"  interactions : {len(r):,}")
    print(f"  users        : {ml.n_users:,}")
    print(f"  rated movies : {ml.n_movies:,}")
    print(f"  avg rating   : {r['rating'].mean():.3f}  (median {r['rating'].median()})")
    print(f"  matrix density: {density * 100:.2f}%  (sparsity {100 - density * 100:.2f}%)")
    print(f"  ratings/user : min {per_user.min()}  median {int(per_user.median())}  max {per_user.max()}")
    print(f"  time span    : {r['timestamp'].min().date()} -> {r['timestamp'].max().date()}")

    # --- Movies ---
    genre_counts = Counter(g for gs in ml.movies["genres"] for g in gs)
    no_genre = (ml.movies["genres"].apply(len) == 0).sum()
    print("\n[Movies]")
    print(f"  catalogue    : {len(ml.movies):,}  (no genres: {no_genre})")
    print(f"  year range   : {int(ml.movies['year'].min())} - {int(ml.movies['year'].max())}")
    top = ", ".join(f"{g} ({c})" for g, c in genre_counts.most_common(5))
    print(f"  top genres   : {top}")

    # --- Cold start ---
    per_movie = r.groupby("movieId").size()
    cold = (per_movie <= 2).sum()
    never = len(ml.movies) - ml.n_movies
    print("\n[Cold start]")
    print(f"  movies with <=2 ratings: {cold:,} ({cold / len(per_movie) * 100:.1f}%)")
    print(f"  movies with 0 ratings  : {never}")

    # --- Most rated ---
    title = ml.movies.set_index("movieId")["title"]
    print("\n[Most rated]")
    for movie_id, count in per_movie.sort_values(ascending=False).head(5).items():
        print(f"  {count:>4}x  {title.get(movie_id, movie_id)}")
    print()
