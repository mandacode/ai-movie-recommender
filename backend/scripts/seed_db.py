"""Seed Postgres (pgvector) from MovieLens + TMDB cache + embeddings.

Creates the schema and loads everything the app serves into one store:
  * movies  — metadata + FTS `tsvector` + pgvector `embedding`
  * ratings — base MovieLens interactions (read-only history)
  * interactions — live likes (starts empty)

Idempotent: drops and recreates the data tables each run.

Usage:
    python scripts/seed_db.py
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from textutil import clean_title  # noqa: E402

import os

DSN = os.environ.get("DATABASE_URL", "postgresql://postgres:mandaflix@localhost:5433/mandaflix")
DATA = ROOT / "datasets" / "ml-latest-small"
TMDB_DB = ROOT / "datasets" / "tmdb_cache.db"
IMG = "https://image.tmdb.org/t/p"

SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;
DROP TABLE IF EXISTS movies, ratings, interactions CASCADE;
CREATE TABLE movies (
    movie_id  INT PRIMARY KEY,
    title     TEXT,
    year      INT,
    genres    TEXT[],
    overview  TEXT,
    poster    TEXT,
    backdrop  TEXT,
    runtime   INT,
    director  TEXT,
    cast_top  TEXT[],
    tsv       tsvector,
    embedding vector(384)
);
CREATE TABLE ratings (
    user_id INT, movie_id INT, rating REAL, ts BIGINT
);
CREATE TABLE interactions (
    user_id INT, movie_id INT, liked BOOLEAN NOT NULL,
    ts TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (user_id, movie_id)
);
"""

INDEXES = """
CREATE INDEX ratings_user_idx ON ratings (user_id);
CREATE INDEX movies_tsv_idx ON movies USING GIN (tsv);
CREATE INDEX movies_emb_idx ON movies USING hnsw (embedding vector_cosine_ops);
"""


def load_tmdb() -> dict[int, dict]:
    if not TMDB_DB.exists():
        return {}
    con = sqlite3.connect(str(TMDB_DB))
    rows = con.execute(
        "SELECT movieId, overview, poster_path, backdrop_path, runtime, director, cast_top "
        "FROM tmdb"
    ).fetchall()
    con.close()
    out = {}
    for mid, ov, poster, backdrop, runtime, director, cast in rows:
        out[int(mid)] = dict(
            overview=ov,
            poster=f"{IMG}/w342{poster}" if poster else None,
            backdrop=f"{IMG}/w1280{backdrop}" if backdrop else None,
            runtime=int(runtime) if runtime else None,
            director=director,
            cast=[c for c in (cast.split(", ") if cast else []) if c],
        )
    return out


def main() -> None:
    movies = pd.read_csv(DATA / "movies.csv")
    ratings = pd.read_csv(DATA / "ratings.csv")
    tmdb = load_tmdb()

    emb_path = DATA / "desc_emb.npz"
    emb = {}
    if emb_path.exists():
        d = np.load(emb_path)
        emb = {int(m): v for m, v in zip(d["movie_ids"], d["embeddings"])}

    con = psycopg.connect(DSN)
    con.execute(SCHEMA)
    con.commit()

    def vec(mid):
        v = emb.get(mid)
        return "[" + ",".join(f"{x:.6f}" for x in v) + "]" if v is not None else None

    print(f"loading {len(movies):,} movies …")
    with con.cursor() as cur:
        for row in movies.itertuples():
            mid = int(row.movieId)
            meta = tmdb.get(mid, {})
            title = clean_title(row.title)
            genres = [] if row.genres == "(no genres listed)" else row.genres.split("|")
            year = int(row.title[-5:-1]) if row.title.rstrip().endswith(")") and row.title[-5:-1].isdigit() else None
            search_text = " ".join(filter(None, [
                title, " ".join(genres), meta.get("overview") or "",
                meta.get("director") or "", " ".join(meta.get("cast", [])),
            ]))
            cur.execute(
                "INSERT INTO movies (movie_id,title,year,genres,overview,poster,backdrop,"
                "runtime,director,cast_top,tsv,embedding) VALUES "
                "(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s, to_tsvector('english',%s), %s)",
                (mid, title, year, genres, meta.get("overview"), meta.get("poster"),
                 meta.get("backdrop"), meta.get("runtime"), meta.get("director"),
                 meta.get("cast", []), search_text, vec(mid)),
            )
    con.commit()

    print(f"loading {len(ratings):,} ratings …")
    with con.cursor() as cur:
        with cur.copy("COPY ratings (user_id,movie_id,rating,ts) FROM STDIN") as copy:
            for r in ratings.itertuples():
                copy.write_row((int(r.userId), int(r.movieId), float(r.rating), int(r.timestamp)))
    con.commit()

    con.execute(INDEXES)
    con.commit()
    n_emb = con.execute("SELECT count(*) FROM movies WHERE embedding IS NOT NULL").fetchone()[0]
    print(f"done: {len(movies):,} movies ({n_emb:,} with embeddings), {len(ratings):,} ratings.")
    con.close()


if __name__ == "__main__":
    main()
