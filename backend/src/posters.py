"""Movie poster URLs from the TMDB cache.

Maps movieId → poster thumbnail URL using the SQLite cache populated by
scripts/fetch_tmdb.py. movieIds are consistent across MovieLens datasets, so
this works for any loaded catalogue. Returns an empty map if the cache is
absent (posters are an optional visual enhancement).
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

DEFAULT_DB = Path("datasets/tmdb_cache.db")
IMG_BASE = "https://image.tmdb.org/t/p/w185"  # w185 = thumbnail size


def load_posters(db: Path = DEFAULT_DB) -> dict[int, str]:
    """Return {movieId: poster_url} for every cached movie that has a poster."""
    db = Path(db)
    if not db.exists():
        return {}
    con = sqlite3.connect(str(db))
    try:
        rows = con.execute(
            "SELECT movieId, poster_path FROM tmdb WHERE poster_path IS NOT NULL"
        ).fetchall()
    except sqlite3.OperationalError:
        return {}  # column missing (pre-migration cache)
    finally:
        con.close()
    return {int(m): IMG_BASE + p for m, p in rows}
