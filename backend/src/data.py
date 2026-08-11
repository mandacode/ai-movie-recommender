"""Loading of the MovieLens `ml-latest-small` dataset."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "datasets" / "ml-latest-small"


@dataclass
class MovieLens:
    """Container for the raw MovieLens tables."""

    ratings: pd.DataFrame  # userId, movieId, rating, timestamp (datetime)
    movies: pd.DataFrame   # movieId, title, year, genres (list[str])
    tags: pd.DataFrame     # userId, movieId, tag, timestamp (datetime)
    links: pd.DataFrame    # movieId, imdbId, tmdbId

    @property
    def n_users(self) -> int:
        return self.ratings["userId"].nunique()

    @property
    def n_movies(self) -> int:
        return self.ratings["movieId"].nunique()


def load_movielens(data_dir: str | Path = DEFAULT_DATA_DIR) -> MovieLens:
    """Load and lightly clean the MovieLens tables."""
    data_dir = Path(data_dir)

    ratings = pd.read_csv(data_dir / "ratings.csv")
    ratings["timestamp"] = pd.to_datetime(ratings["timestamp"], unit="s")

    movies = pd.read_csv(data_dir / "movies.csv")
    movies["year"] = (
        movies["title"].str.extract(r"\((\d{4})\)\s*$")[0].astype("Int64")
    )
    movies["genres"] = movies["genres"].apply(
        lambda g: [] if g == "(no genres listed)" else g.split("|")
    )

    tags = pd.read_csv(data_dir / "tags.csv")
    tags["timestamp"] = pd.to_datetime(tags["timestamp"], unit="s")

    links = pd.read_csv(data_dir / "links.csv")

    return MovieLens(ratings=ratings, movies=movies, tags=tags, links=links)
